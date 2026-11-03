// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the two-process split of the GPU library (bbport_split.h).
#include "bbport_split.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>
#include <pthread.h>
#include <pthread/qos.h>
#include <signal.h>
#include <unistd.h>
#include "common/thread.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"

extern std::unique_ptr<AmdGpu::Liverpool> liverpool; // gnmdriver.cpp

extern "C" void runtime_memory_gpu_protect(uintptr_t address, uint64_t size, int read, int write);
extern "C" void runtime_restart(void); // bb-probe (probe.c)

namespace Libraries::VideoOut {
void BbSplitApplyEvent(const BbGpuRecord* record);
}

namespace BbSplit {
namespace {

std::atomic<Mode> g_mode{Mode::Single};
BbGpuShared* g_shared;
std::mutex g_push_mutex;  // producers on this side of each ring
std::mutex g_fault_mutex; // game: producers of the fault ring (fault handlers)
std::atomic<u64> g_gpu_idle_seq;

constexpr u64 Second = 1'000'000'000;

/// The peer process is gone: nothing can make progress any more.
[[noreturn]] void PeerLost(const char* what) {
    std::printf("GPU split: the %s process is gone; exiting\n", what);
    std::fflush(stdout);
    _exit(1);
}

bool PeerAlive(s32 pid) {
    return pid <= 0 || kill(pid, 0) == 0;
}

void PushTo(BbGpuRing& ring, const void* record, s32 peer_pid, const char* peer) {
    std::scoped_lock lock{g_push_mutex};
    for (u32 waited = 0;; ++waited) {
        if (bbgpu_ring_push(g_shared, &ring, record, Second) == 0) {
            return;
        }
        if (!PeerAlive(peer_pid)) {
            PeerLost(peer);
        }
        if (waited == 10) {
            std::printf("GPU split: the %s has not taken records for 10 s (ring full)\n", peer);
        }
    }
}

// ---- fifa-hybrid: write faults on upload-tracked pages are handled in the game ----
// The host protects pages read-only to notice CPU writes before the next upload. Relaying such
// a fault cost four cross-process hops (~0.5 ms each in a match, ~1000/s): the game now lifts
// the protection itself over the run of read-only pages around the fault and reports the range
// on the submit ring, so the host marks it CPU-modified before any later submit uses it. Pages
// the host made unreadable (pending readbacks) still relay. BB_LOCAL_WFAULT=0 disables.
constexpr u64 kLocalPageBits = 12;
constexpr u64 kLocalPage = u64{1} << kLocalPageBits;
constexpr u64 kLocalWindow = u64{256} << 10;
std::mutex g_protect_mutex;  // protection changes and the page kinds below
u8* g_page_kind = nullptr;   // per 4 KiB page of the low guest range: 1 = read-only (upload-tracked)

// BB_LOCAL_WFAULT: 0 = every write relays, 1 (default) = only buffer-only pages (kind 1) are
// handled locally, 2 = image pages too (kinds 2 and 3; left the kit/face textures stale: the host
// then re-uploaded them from guest memory the game had not finished writing).
int LocalWriteFaultLevel() {
    static const int level = [] {
        const char* env = std::getenv("BB_LOCAL_WFAULT");
        return env ? std::atoi(env) : 1;
    }();
    return level;
}

bool LocalWriteFaultsEnabled() {
    return LocalWriteFaultLevel() > 0;
}

u8* KindSlot(u64 address) {
    if (!g_page_kind || address < BBGPU_GUEST_LOW_MIN || address >= BBGPU_GUEST_LOW_MAX) {
        return nullptr;
    }
    return g_page_kind + ((address - BBGPU_GUEST_LOW_MIN) >> kLocalPageBits);
}

// The page byte keeps the host's last protection: 0x80 | read bits (AddressSpace::Protect:
// 1 readable, 2 an image write-tracks it, 4 unreadable only for droppable image readbacks) << 1
// | writable. KindOf: 1 = buffer-only read-only page (the run in the window is handled locally),
// 2 = image write-tracked page (that page only), 3 = droppable image readback page (that page
// only), 0 = relay.
u8 KindOf(u8 raw) {
    if (!(raw & 0x80)) {
        return 0;
    }
    const u32 read = (raw >> 1) & 7u;
    const bool write = raw & 1u;
    return (read & 4u) ? 3 : ((read & 1u) && !write) ? ((read & 2u) ? 2 : 1) : 0;
}

void NoteProtect(u64 address, u64 size, u32 read, bool write) { // g_protect_mutex held
    const u8 raw = 0x80 | u8((read & 7u) << 1) | (write ? 1 : 0);
    for (u64 a = address & ~(kLocalPage - 1); a < address + size; a += kLocalPage) {
        if (u8* k = KindSlot(a)) {
            *k = raw;
        }
    }
}

std::atomic<u64> g_relayed_write_states[16]; // BB_FAULT_STATS: relayed writes by page state
std::atomic<u64> g_spurious_write_faults{0}; // writes retried because the page is writable now

std::atomic<u64> g_local_write_faults{0};

bool LocalWriteFault(u64 address) {
    if (!LocalWriteFaultsEnabled()) {
        return false;
    }
    std::scoped_lock lock{g_protect_mutex};
    const u8* k = KindSlot(address);
    u8 kind = k ? KindOf(*k) : 0;
    if (kind >= 2 && LocalWriteFaultLevel() < 2) {
        return false; // image pages relay (the host downloads / re-tracks the image itself)
    }
    if (kind == 0) {
        // A page the host has meanwhile made writable (another thread's fault in the same window
        // unprotected it between this fault and now): retry the write without a relay. The
        // second fault in a row on the same page relays, in case the map is wrong.
        static thread_local u64 retried_page = 0;
        const u64 page = address & ~(kLocalPage - 1);
        if (k && (*k & 0x81) == 0x81 && retried_page != page) {
            retried_page = page;
            g_spurious_write_faults.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        retried_page = 0;
        g_relayed_write_states[k ? (*k & 0x0f) : 15].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const u64 page = address & ~(kLocalPage - 1);
    u64 start = page, end = page + kLocalPage;
    if (kind == 1) {
        // Buffer-only pages: the run of such pages in the window (as the host's ExtendWriteFault).
        const u64 window_lo = std::max<u64>(page & ~(kLocalWindow - 1), BBGPU_GUEST_LOW_MIN);
        const u64 window_hi = std::min<u64>(window_lo + kLocalWindow, BBGPU_GUEST_LOW_MAX);
        while (start > window_lo && KindOf(*KindSlot(start - kLocalPage)) == 1) {
            start -= kLocalPage;
        }
        while (end < window_hi && KindOf(*KindSlot(end)) == 1) {
            end += kLocalPage;
        }
    } // an image's page: this page only, so no other image is marked dirty
    runtime_memory_gpu_protect(start, end - start, 1, 1);
    for (u64 a = start; a < end; a += kLocalPage) {
        *KindSlot(a) = 0x80 | 0x2 | 0x1; // readable and writable now
    }
    const BbGpuCpuWrite record{{BBGPU_REC_CPU_WRITE, sizeof(BbGpuCpuWrite)}, start, end - start, address};
    PushTo(g_shared->submit, &record, g_shared->host_pid, "GPU host");
    g_local_write_faults.fetch_add(1, std::memory_order_relaxed);
    return true;
}

} // namespace

Mode GetMode() {
    return g_mode.load(std::memory_order_relaxed);
}

BbGpuShared* Shared() {
    return g_shared;
}

void Attach(BbGpuShared* shared) {
    g_shared = shared;
}

void InitRemote(BbGpuShared* shared) {
    g_shared = shared;
    g_mode = Mode::Remote;
    g_page_kind = new u8[(BBGPU_GUEST_LOW_MAX - BBGPU_GUEST_LOW_MIN) >> kLocalPageBits]();
    // M4: the host's page protection requests, applied here and acknowledged in order.
    std::thread([] {
        Common::SetCurrentThreadName("bb:GpuProtect");
        // fifa-hybrid: the host's fault thread waits for every acknowledgement.
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        BbGpuShared* s = g_shared;
        for (;;) {
            const BbGpuRecord* r = bbgpu_ring_peek(s, &s->protect, Second);
            if (!r) {
                if (!PeerAlive(s->host_pid)) {
                    PeerLost("GPU host");
                }
                continue;
            }
            const auto* p = reinterpret_cast<const BbGpuProtect*>(r);
            {
                std::scoped_lock lock{g_protect_mutex};
                runtime_memory_gpu_protect(p->address, p->size, static_cast<int>(p->read & 1u),
                                           static_cast<int>(p->write));
                NoteProtect(p->address, p->size, p->read, p->write != 0);
            }
            const u32 seq = p->seq;
            bbgpu_ring_pop(&s->protect, r);
            __atomic_store_n(&s->protect_done, seq, __ATOMIC_RELEASE);
            bbgpu_wake_word(&s->protect_done);
        }
    }).detach();
}

void InitHost(BbGpuShared* shared) {
    g_shared = shared;
    g_mode = Mode::Host;
}

void Push(const void* record) {
    PushTo(g_shared->submit, record, g_shared->host_pid, "GPU host");
}

s64 Request(void* record, u32* slot) {
    const int claimed = bbgpu_reply_claim(g_shared);
    *slot = static_cast<u32>(claimed);
    Push(record);
    s64 result = 0;
    for (u32 waited = 0; bbgpu_reply_wait(g_shared, claimed, &result, Second) != 0; ++waited) {
        if (!PeerAlive(g_shared->host_pid)) {
            PeerLost("GPU host");
        }
        if (waited == 30) {
            std::printf("GPU split: request %u unanswered for 30 s\n",
                        static_cast<const BbGpuRecord*>(record)->type);
        }
    }
    return result;
}

bool RelayFault(u64 address, bool write) {
    if (!bbgpu_guest_range_ok(address, address + 1)) {
        return false;
    }
    if (write && LocalWriteFault(address)) {
        return true;
    }
    const auto start = std::chrono::steady_clock::now();
    const int slot = bbgpu_reply_claim(g_shared);
    const BbGpuFault fault{{BBGPU_REC_FAULT, sizeof(BbGpuFault)}, address, static_cast<u32>(slot),
                           write ? 1u : 0u};
    {
        std::scoped_lock lock{g_fault_mutex};
        while (bbgpu_ring_push(g_shared, &g_shared->faults, &fault, Second) != 0) {
            if (!PeerAlive(g_shared->host_pid)) {
                PeerLost("GPU host");
            }
        }
    }
    s64 handled = 0;
    while (bbgpu_reply_wait(g_shared, slot, &handled, Second) != 0) {
        if (!PeerAlive(g_shared->host_pid)) {
            PeerLost("GPU host");
        }
    }
    const auto waited = std::chrono::steady_clock::now() - start;
    { // fifa-hybrid diagnostics: BB_FAULT_STATS=1 prints the relayed fault rate every 2 s.
        static const bool stats = std::getenv("BB_FAULT_STATS") != nullptr;
        if (stats) {
            static std::atomic<u64> n_write{0}, n_read{0}, wait_ns{0};
            static std::atomic<s64> window_start{0};
            (write ? n_write : n_read).fetch_add(1, std::memory_order_relaxed);
            wait_ns.fetch_add(u64(std::chrono::duration_cast<std::chrono::nanoseconds>(waited).count()),
                              std::memory_order_relaxed);
            const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
            s64 ws = window_start.load(std::memory_order_relaxed);
            if (ws == 0) {
                window_start.compare_exchange_strong(ws, now);
            } else if (now - ws >= 2000000000 &&
                       window_start.compare_exchange_strong(ws, now)) {
                const u64 w = n_write.exchange(0), r = n_read.exchange(0), ns = wait_ns.exchange(0);
                char states[160];
                int pos = 0;
                for (int i = 0; i < 16; ++i) {
                    if (const u64 c = g_relayed_write_states[i].exchange(0)) {
                        pos += std::snprintf(states + pos, sizeof states - pos, " s%x=%llu", i,
                                             (unsigned long long)c);
                    }
                }
                std::printf("[FAULTS] %.1fs: write=%llu read=%llu wait=%.1fms (%.1fus avg) local=%llu"
                            " spurious=%llu relayed-write states:%s\n",
                            (now - ws) / 1e9, (unsigned long long)w, (unsigned long long)r,
                            ns / 1e6, (w + r) ? ns / 1e3 / double(w + r) : 0.0,
                            (unsigned long long)g_local_write_faults.exchange(0),
                            (unsigned long long)g_spurious_write_faults.exchange(0), states);
                std::fflush(stdout);
            }
        }
    }
    __atomic_add_fetch(&g_shared->game_faults, 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&g_shared->game_fault_ns,
                       u64(std::chrono::duration_cast<std::chrono::nanoseconds>(waited).count()),
                       __ATOMIC_RELAXED);
    return handled != 0;
}

u64 GpuIdleSeq() {
    return g_gpu_idle_seq.load(std::memory_order_acquire);
}

void StartEventDispatcher() {
    std::thread([] {
        Common::SetCurrentThreadName("bb:GpuEvents");
        BbGpuShared* s = g_shared;
        for (;;) {
            const BbGpuRecord* r = bbgpu_ring_peek(s, &s->events, Second);
            if (!r) {
                if (!PeerAlive(s->host_pid)) {
                    PeerLost("GPU host");
                }
                continue;
            }
            switch (r->type) {
            case BBGPU_EV_IRQ:
                if (reinterpret_cast<const BbGpuIrq*>(r)->id ==
                    static_cast<u32>(Platform::InterruptId::GpuIdle)) {
                    g_gpu_idle_seq.store(reinterpret_cast<const BbGpuIrq*>(r)->seq,
                                         std::memory_order_release);
                }
                // The game's own registrations (sceGnmAddEqEvent, the submission lock) run here
                // as they ran on the GPU thread in a single process.
                Platform::IrqC::Instance()->Signal(
                    static_cast<Platform::InterruptId>(reinterpret_cast<const BbGpuIrq*>(r)->id));
                break;
            case BBGPU_EV_FLIP:
            case BBGPU_EV_EOP_FLIP_QUEUED:
            case BBGPU_EV_VBLANK:
                Libraries::VideoOut::BbSplitApplyEvent(r);
                break;
            case BBGPU_EV_RESTART:
                runtime_restart(); // execs run.sh; the host follows bb-probe's exec and exits
                break;
            default:
                std::printf("GPU split: unknown event %u\n", r->type);
            }
            bbgpu_ring_pop(&s->events, r);
        }
    }).detach();
}

void Post(const void* record) {
    PushTo(g_shared->events, record, g_shared->game_pid, "game");
}

void ForwardInterrupts() {
    using Platform::InterruptId;
    static constexpr InterruptId ids[] = {
        InterruptId::Compute0RelMem, InterruptId::Compute1RelMem, InterruptId::Compute2RelMem,
        InterruptId::Compute3RelMem, InterruptId::Compute4RelMem, InterruptId::Compute5RelMem,
        InterruptId::Compute6RelMem, InterruptId::GfxEop,         InterruptId::GpuIdle,
    };
    static int uid; // one registration per interrupt
    for (const InterruptId id : ids) {
        Platform::IrqC::Instance()->Register(
            id,
            [](InterruptId irq) {
                // GpuIdle carries the SubmitDone it follows (gnmdriver ResetSubmissionLock).
                const u32 seq = irq == InterruptId::GpuIdle && liverpool
                                    ? static_cast<u32>(liverpool->SubmitDonesHandled())
                                    : 0;
                const BbGpuIrq event{{BBGPU_EV_IRQ, sizeof(BbGpuIrq)}, static_cast<u32>(irq), seq};
                Post(&event);
            },
            &uid);
    }
}

} // namespace BbSplit
