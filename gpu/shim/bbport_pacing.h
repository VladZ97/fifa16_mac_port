// fifa-hybrid (0.5.1-f): phase lock between the host's vblank tick (which paces the game) and the
// display. The presenter measures how long each finished frame waited for its display slot and
// publishes a small correction for the next tick; the present thread adds it to the tick period.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace BbPacing {

inline std::atomic<std::int64_t> g_slew_ns{0};
inline std::atomic<std::uint64_t> g_slew_stamp_ns{0};
inline std::atomic<std::int64_t> g_slewed_fwd_ns{0};
inline std::atomic<std::int64_t> g_slewed_back_ns{0};

inline std::uint64_t NowNs() {
    return std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count());
}

/// Presenter: correction for the coming ticks (positive = later ticks = frames wait less).
inline void SetTickSlew(std::int64_t ns) {
    g_slew_ns.store(ns, std::memory_order_relaxed);
    g_slew_stamp_ns.store(NowNs(), std::memory_order_relaxed);
}

/// Present thread: correction for this tick; 0 when the presenter has gone quiet.
inline std::int64_t TickSlewNs() {
    const std::int64_t ns = g_slew_ns.load(std::memory_order_relaxed);
    if (!ns) {
        return 0;
    }
    const std::uint64_t stamp = g_slew_stamp_ns.load(std::memory_order_relaxed);
    if (NowNs() - stamp > 250'000'000ull) {
        return 0;
    }
    (ns > 0 ? g_slewed_fwd_ns : g_slewed_back_ns).fetch_add(ns > 0 ? ns : -ns,
                                                             std::memory_order_relaxed);
    return ns;
}

} // namespace BbPacing
