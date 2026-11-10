// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
// fifa-hybrid (0.5.1-f, BB_GAME_TRACE=1): why a game frame came late. Game side of the split:
// between two flips, how long each game thread was blocked in an event queue wait
// (sceKernelWaitEqueue: flip and vblank events) and how much CPU the game's threads used.
// A late frame with a long event wait was paced by an event; one without was computing.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

namespace BbGameTrace {

inline bool Enabled() {
    static const bool on = [] {
        const char* env = std::getenv("BB_GAME_TRACE");
        return env && env[0] == '1';
    }();
    return on;
}

inline unsigned long long NowNs() {
    return static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

struct Slot {
    std::atomic<pthread_t> thread{nullptr};
    std::atomic<unsigned long long> wait_ns{0}, longest_ns{0}, last_end_ns{0}, last_ident{0};
    std::atomic<unsigned> waits{0};
    std::atomic<int> last_filter{0}, last_count{0};
};
inline Slot g_slots[24];

/// One blocking event queue wait of the calling thread.
inline void NoteWait(unsigned long long start_ns, unsigned long long end_ns,
                     unsigned long long ident, int filter, int count) {
    const pthread_t self = pthread_self();
    for (Slot& slot : g_slots) {
        pthread_t owner = slot.thread.load(std::memory_order_relaxed);
        if (owner != self) {
            if (owner != nullptr) {
                continue;
            }
            if (!slot.thread.compare_exchange_strong(owner, self) && owner != self) {
                continue;
            }
        }
        const unsigned long long ns = end_ns - start_ns;
        slot.wait_ns.fetch_add(ns, std::memory_order_relaxed);
        if (ns > slot.longest_ns.load(std::memory_order_relaxed)) {
            slot.longest_ns.store(ns, std::memory_order_relaxed);
        }
        slot.last_end_ns.store(end_ns, std::memory_order_relaxed);
        slot.last_ident.store(ident, std::memory_order_relaxed);
        slot.last_filter.store(filter, std::memory_order_relaxed);
        slot.last_count.store(count, std::memory_order_relaxed);
        slot.waits.fetch_add(1, std::memory_order_relaxed);
        return;
    }
}

/// At each flip of the game (one caller at a time: the submission lock is held).
/// CPU figures are totals in ns (0 when the emulator does not provide them).
inline void OnFlip(unsigned long long main_cpu_ns, unsigned long long flip_cpu_ns,
                   unsigned long long all_cpu_delta_ns, unsigned long long busiest_delta_ns,
                   const char* busiest_name) {
    static unsigned long long last_ns, last_main, last_flip;
    static unsigned printed, frames, on_time;
    static double sum_interval, sum_main, sum_flip, sum_all, sum_wait[24];
    const unsigned long long now = NowNs();
    const double interval = last_ns ? double(now - last_ns) * 1e-6 : 0.0;
    const double main_ms = double(main_cpu_ns - last_main) * 1e-6;
    const double flip_ms = double(flip_cpu_ns - last_flip) * 1e-6;
    const bool late = interval > 20.8 && interval < 2000.0;
    char line[1024];
    int at = 0;
    if (late && printed < 1500) {
        at = std::snprintf(line, sizeof(line),
                           "GameLate: flip after %.1f ms | on CPU: main %.1f, flip thread %.1f, "
                           "all %.1f, busiest %s %.1f | event waits:",
                           interval, main_ms, flip_ms, double(all_cpu_delta_ns) * 1e-6,
                           busiest_name ? busiest_name : "", double(busiest_delta_ns) * 1e-6);
    }
    int index = 0;
    for (Slot& slot : g_slots) {
        const int i = index++;
        const pthread_t owner = slot.thread.load(std::memory_order_relaxed);
        if (owner == nullptr) {
            continue;
        }
        const unsigned long long wait = slot.wait_ns.exchange(0, std::memory_order_relaxed);
        const unsigned long long longest = slot.longest_ns.exchange(0, std::memory_order_relaxed);
        const unsigned waits = slot.waits.exchange(0, std::memory_order_relaxed);
        if (!late) {
            sum_wait[i] += double(wait) * 1e-6;
        }
        if (at > 0 && waits && at < int(sizeof(line)) - 160) {
            char name[32] = "";
            pthread_getname_np(owner, name, sizeof(name));
            const unsigned long long end = slot.last_end_ns.load(std::memory_order_relaxed);
            at += std::snprintf(line + at, sizeof(line) - at,
                                " [%s %.1f ms in %u, longest %.1f, last ended %.1f ms before the "
                                "flip, filter %d ident %#llx n %d]",
                                name, double(wait) * 1e-6, waits, double(longest) * 1e-6,
                                double(now - end) * 1e-6,
                                slot.last_filter.load(std::memory_order_relaxed),
                                slot.last_ident.load(std::memory_order_relaxed),
                                slot.last_count.load(std::memory_order_relaxed));
        }
    }
    if (at > 0) {
        ++printed;
        std::fprintf(stderr, "%s\n", line);
    }
    if (last_ns && !late && interval < 2000.0) {
        ++on_time;
        sum_interval += interval;
        sum_main += main_ms;
        sum_flip += flip_ms;
        sum_all += double(all_cpu_delta_ns) * 1e-6;
    }
    if (++frames >= 600) {
        if (on_time) {
            at = std::snprintf(line, sizeof(line),
                               "GameFrames: %u of %u on time, mean %.1f ms | on CPU: main %.1f, "
                               "flip thread %.1f, all %.1f | event waits:",
                               on_time, frames, sum_interval / on_time, sum_main / on_time,
                               sum_flip / on_time, sum_all / on_time);
            index = 0;
            for (Slot& slot : g_slots) {
                const int i = index++;
                const pthread_t owner = slot.thread.load(std::memory_order_relaxed);
                if (owner != nullptr && sum_wait[i] > 0.05 * on_time &&
                    at < int(sizeof(line)) - 80) {
                    char name[32] = "";
                    pthread_getname_np(owner, name, sizeof(name));
                    at += std::snprintf(line + at, sizeof(line) - at, " [%s %.1f ms]", name,
                                        sum_wait[i] / on_time);
                }
            }
            std::fprintf(stderr, "%s\n", line);
        }
        frames = on_time = 0;
        sum_interval = sum_main = sum_flip = sum_all = 0;
        std::memset(sum_wait, 0, sizeof(sum_wait));
    }
    last_ns = now;
    last_main = main_cpu_ns;
    last_flip = flip_cpu_ns;
}

} // namespace BbGameTrace
