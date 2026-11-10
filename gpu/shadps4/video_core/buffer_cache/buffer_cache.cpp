// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <array>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <pthread.h>
#include <cstring>
#include <bit>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <magic_enum/magic_enum.hpp>
#ifdef __APPLE__
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>
#endif
#include "bbport_arena_runs.h"
#include "bbport_copy.h"
#include "bbport_toggles.h"
#include "common/alignment.h"
#include "common/path_util.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h" // fifa-hybrid: DrainDrawPipe
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderer_metal/mtl_buffer.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_encoder.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {
void VbNoteWrite(VAddr address, u64 size, char tag); // BB_VB_LOG debug
}

namespace VideoCore {

std::atomic<u64> g_bb_bufwatch{[] {
    const char* env = std::getenv("BB_BUFWATCH");
    return env ? u64(std::strtoull(env, nullptr, 16)) : u64{0};
}()};

using Vulkan::VbNoteWrite;

// bbport: which build wrote a layout fallback: this library's size and modification time
// (a new app version or rebuild has another).
static std::string LayoutBuildStamp() {
    Dl_info info{};
    struct stat st{};
    if (!dladdr(reinterpret_cast<const void*>(&LayoutBuildStamp), &info) || !info.dli_fname ||
        stat(info.dli_fname, &st) != 0) {
        return "unknown";
    }
    return fmt::format("build {} {}", static_cast<long long>(st.st_size),
                       static_cast<long long>(st.st_mtime));
}

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;

BufferCache::BufferCache(Metal::Device& device_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : device{device_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{device, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{device, MemoryType::Stream, 0, GDS_BUFFER_SIZE, "GDS Buffer"},
      write_sink{device, MemoryType::DeviceLocal, 0, 64_KB, "Write Sink"} {
    // bbport: one fully backed arena, so the page table granularity is a free choice; 16 KiB is
    // what MoltenVK's sparse alignment gave (the recompiler and the fault shader read it).
    block_size = MIN_BLOCK_SIZE;
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;

    // bbport: the page table and the fault buffer serve direct memory access only
    // (BB_DIRECT_MEMORY_ACCESS=1: shaders reach guest memory by address). Without it no shader
    // reads them (CollectShaderInfoPass clears uses_dma, and LoadShaderMeta rejects a cache entry
    // compiled with it on), yet they held 520 MiB of resident memory on every Mac, the page
    // table private and committed in full. They exist only with the flag now; the texture
    // budget keeps counting them (TextureCache::GarbageCollectImages, BbStats::legacy_resident).
    const u64 bda_pagetable_size = (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(u64);
    const u64 fault_buffer_size = FaultManager::BufferSize(blocks_per_arena_page * NUM_ARENA_PAGES);
    u64 dma_bytes = 0;
    if (EmulatorSettings.IsDirectMemoryAccessEnabled()) {
        fault_manager = std::make_unique<FaultManager>(device, scheduler, *this, block_shift,
                                                       blocks_per_arena_page * NUM_ARENA_PAGES);
        bda_pagetable_buffer = std::make_unique<Buffer>(device, MemoryType::DeviceLocal, 0,
                                                        bda_pagetable_size, "BDA Page Table Buffer");
        // Its updates and the shaders' reads (direct memory access) are ordered by the Runtime.
        runtime.RegisterOrdered(bda_pagetable_buffer.get(), bda_pagetable_size);
        runtime.FillBuffer(*bda_pagetable_buffer, 0u, bda_pagetable_size, 0u);
        // Shaders reach the page table by address.
        device.MakeResident(bda_pagetable_buffer->Handle());
        dma_bytes = bda_pagetable_size + fault_buffer_size;
    }

    // bbport: Metal has no sparse buffers. One ordinary arena covers the start of the guest
    // GPU range instead (runtime_memory.c places guest memory from USER_MIN = 0x7000000000 on
    // macOS); Metal commits its memory as it is touched.
    // Requests outside it stop in GetChunk (docs/MACOS_BACKLOG.md: per-page arenas).
    // The arena is sized to the guest memory the game can map, not to the Mac: Metal wires the
    // arena's memory for the GPU again and again, and the bigger it is the worse that gets. With
    // 16 GiB on a 36 GB Mac (M5 Max, macOS 27) the wiring never settled (Metal System Trace:
    // 254 GiB wired in 8 s, 224 GiB of it arena pages; 7 FPS); 12 GiB: 60 FPS with ~110 ms
    // stalls; 8 GiB: 60 FPS. Lazy commit (mtl_buffer.mm, BB_ARENA_LAZY) does not change that.
    // On 16 GB Macs 8 GiB was still too much: the GPU process sat at Metal's working-set limit
    // (11.53 of 11.55 GB on an M1 Pro, issue #19) at 4-9 FPS.
    // The game's direct and flexible memory sit together from MacArenaBase (runtime_memory.c):
    // with the retail 5056 MiB of direct memory, every log so far ends its highest mapping at
    // 4.80 GiB. The arena covers direct memory plus 1 GiB (flexible memory, 448 MiB, and room),
    // rounded up to 512 MiB: 6 GiB, or 10 GiB with BB_DMEM_MB=9152 (run.sh's startup patch).
    // Past it the game stops in GetChunk, and "the game reached N GiB" is logged as it grows.
    // BB_ARENA_GIB (at least 4) overrides: smaller arenas for tools that snapshot every buffer
    // (Metal frame capture fails on large ones), larger ones for testing.
    const char* reason = "the game's direct memory plus 1 GiB";
    const u64 wanted_bytes = [&] {
        if (const char* env = std::getenv("BB_ARENA_GIB")) {
            if (const u64 gib = std::strtoull(env, nullptr, 10); gib >= 4) {
                reason = "BB_ARENA_GIB";
                return std::min<u64>(gib, 24) << 30;
            }
        }
        // runtime_memory.c pool_size_bytes(): the same variable, the same bounds.
        u64 dmem_mb = 0;
        if (const char* env = std::getenv("BB_DMEM_MB")) {
            dmem_mb = std::strtoull(env, nullptr, 10);
        }
        if (dmem_mb < 5056 || dmem_mb > 16384) {
            dmem_mb = 5056;
        }
        return Common::AlignUp((dmem_mb << 20) + 1_GB, 512_MB);
    }();
    // One Metal buffer is limited to maxBufferLength, about 60% of the Mac's memory (28 GiB with
    // 48 GB; ~9 GiB with 16 GB). BB_MAX_BUFFER_GIB pretends a lower limit (testing).
    u64 max_buffer = device.GetLimits().max_buffer_length;
    if (const char* env = std::getenv("BB_MAX_BUFFER_GIB")) {
        max_buffer = std::min<u64>(max_buffer, std::strtoull(env, nullptr, 10) << 30);
    }
    const u64 arena_bytes = std::min<u64>(wanted_bytes, max_buffer & ~(256_MB - 1));
    if (arena_bytes < wanted_bytes) {
        reason = "the largest Metal buffer";
    }
    ASSERT_MSG(arena_bytes >= 5_GB,
               "This Mac's GPU allows buffers of {:.1f} GiB; the game needs 5 GiB (a Mac with "
               "at least 16 GB of memory)",
               double(max_buffer) / double(1_GB));
    u64 memsize = 0;
    size_t memsize_size = sizeof memsize;
    if (sysctlbyname("hw.memsize", &memsize, &memsize_size, nullptr, 0) != 0) {
        memsize = 0;
    }
    std::printf("GPU arena: %.1f GiB (%s; the Mac has %.0f GB of memory, its largest Metal buffer "
                "is %.1f GiB)\n",
                double(arena_bytes) / double(1_GB), reason, double(memsize) / 1e9,
                double(max_buffer) / double(1_GB));
    // bbport: the layout of guest memory in GPU buffers. legacy: one arena from MacArenaBase, as
    // above. span: one buffer from the first byte the game maps for the GPU to the last. runs:
    // one buffer per run of memory mapped for the GPU (~3.3 GiB of the arena's 6). The driver
    // wires each buffer by its length (tests/test_arena_accounting.mm), and offsets stay below
    // 4 GiB in a run (no index copies). span and runs follow the game's maps (BbArenaRuns).
    // BB_ARENA_GUARD_MB (64): bytes past each buffer, for shaders reading past a binding (Metal
    // does not bounds-check).
    // Default: runs on Macs with 24 GB or less, or whose Metal working set is 18 GiB or less
    // (iogpu.wired_limit_mb); legacy on the others. On an 18 GB M3 Pro (macOS 27, issue #69) the
    // legacy arena's pinned memory pushed the Mac into swap: 54 FPS falling to 10-20 within a
    // minute; runs held 60 (FINDINGS 2026-10-26). BB_ARENA_LAYOUT=legacy|span|runs chooses.
    legacy_bytes = arena_bytes;
    mac_arena_end = MacArenaBase + arena_bytes;
    const char* layout_name = std::getenv("BB_ARENA_LAYOUT");
    const bool small_mac = (memsize != 0 && memsize <= (24ULL << 30)) ||
                           device.GetLimits().recommended_working_set <= (18ULL << 30);
    const std::string_view wanted_layout = layout_name ? layout_name
                                           : small_mac ? "runs"
                                                       : "legacy";
    layout = wanted_layout == "runs"   ? Layout::Runs
             : wanted_layout == "span" ? Layout::Span
                                       : Layout::Legacy;
    const auto name_of = [](Layout l) {
        return l == Layout::Runs ? "runs" : l == Layout::Span ? "span" : "legacy";
    };
    if (layout != Layout::Legacy) {
        const char* forced = nullptr;
        if (EmulatorSettings.IsDirectMemoryAccessEnabled()) {
            forced = "BB_DIRECT_MEMORY_ACCESS reads any guest memory by address";
        } else if (std::getenv("BB_ARENA_GIB") || std::getenv("BB_ARENA_MIB")) {
            forced = "BB_ARENA_GIB sets the arena";
        }
        // The previous launch's fallback (MonitorLayout): one step back from the layout it names.
        // It holds for this build only: a file another build wrote is removed (an update tries
        // the layout again instead of keeping a small Mac on the legacy arena for good).
        const auto fallback = Common::FS::GetUserPath(Common::FS::PathType::UserDir) /
                              "gpu_memory_layout_fallback";
        std::string failed;
        if (!forced) {
            if (std::FILE* f = std::fopen(fallback.string().c_str(), "r")) {
                char line[512] = {}, stamp[128] = {};
                if (std::fgets(line, sizeof line, f)) {
                    failed = line;
                }
                const bool has_stamp = std::fgets(stamp, sizeof stamp, f) != nullptr;
                std::fclose(f);
                std::string_view written{stamp};
                if (written.ends_with('\n')) {
                    written.remove_suffix(1);
                }
                if (!has_stamp || written != LayoutBuildStamp()) {
                    std::printf("GPU arena: the layout fallback from another build (%s) no longer "
                                "applies\n",
                                failed.empty() ? "empty" : failed.substr(0, failed.find('\n')).c_str());
                    std::remove(fallback.string().c_str());
                    failed.clear();
                }
            }
        }
        if (forced) {
            std::printf("GPU arena: layout legacy, not %s (%s)\n", name_of(layout), forced);
            layout = Layout::Legacy;
        } else if (!failed.empty() && (failed.starts_with("span") ||
                                       (failed.starts_with("runs") && layout == Layout::Runs))) {
            const Layout back = failed.starts_with("span") ? Layout::Legacy : Layout::Span;
            if (!failed.empty() && failed.back() == '\n') {
                failed.pop_back();
            }
            std::printf("GPU arena: layout %s, not %s: the last session had %s (delete %s to try "
                        "it again)\n",
                        name_of(back), name_of(layout), failed.c_str(), fallback.string().c_str());
            layout = back;
        }
    }
    if (layout != Layout::Legacy) {
        static const u64 guard_mb = [] {
            const char* env = std::getenv("BB_ARENA_GUARD_MB");
            return env ? std::strtoull(env, nullptr, 10) : 64;
        }();
        guard_bytes = guard_mb << 20;
    }
    // bbport: what this layout makes resident (the driver wires it by its length) and Metal's
    // working set, for the logs of 16 and 24 GB Macs.
    const u64 arena_now = layout == Layout::Legacy ? arena_bytes : 0;
    std::printf("GPU memory at start: %.0f MiB resident (arena %.0f MiB%s, direct memory access "
                "tables %.0f MiB%s); Metal working set %.0f MiB\n",
                double(arena_now + dma_bytes) / double(1_MB), double(arena_now) / double(1_MB),
                layout == Layout::Legacy ? "" : ": guest memory buffers follow the game's GPU maps",
                double(dma_bytes) / double(1_MB),
                dma_bytes ? "" : ", not created: BB_DIRECT_MEMORY_ACCESS is off",
                double(device.GetLimits().recommended_working_set) / double(1_MB));
    BbStats::layout_resident_bytes.store(arena_now + dma_bytes, std::memory_order_relaxed);
    BbStats::legacy_resident_bytes.store(arena_bytes + bda_pagetable_size + fault_buffer_size,
                                         std::memory_order_relaxed);
    BbArenaRuns::SetArena(MacArenaBase, mac_arena_end);
    if (layout == Layout::Legacy) {
        SetUpChunk(MacArenaBase, arena_bytes, "Guest Arena");
        return;
    }
    std::printf("GPU arena: layout %s (%s): buffers for the game's GPU memory as it "
                "is mapped, %llu MiB past each, at most %.1f GiB in all\n",
                name_of(layout), layout_name ? "BB_ARENA_LAYOUT" : "a Mac with 24 GB or less",
                static_cast<unsigned long long>(guard_bytes >> 20),
                double(legacy_bytes) / double(1_GB));
    BbArenaRuns::SetListener(&BufferCache::OnRunsChanged, this);
    MaterializeRuns(); // the runs mapped before the renderer started (none yet: the first record)
    layout_monitor = std::jthread{[this](std::stop_token stop) { MonitorLayout(stop); }};
}

const Buffer* BufferCache::SetUpChunk(VAddr start, u64 bytes, const std::string& name,
                                      u64 guard) {
    const auto* buffer = &arenas.emplace_back(device, MemoryType::DeviceLocal, start,
                                              bytes + guard, name, layout != Layout::Legacy);
    ASSERT_MSG(chunks.Add(start, start + bytes, buffer),
               "Guest memory buffer {:#x}+{:#x} overlaps another", start, bytes);
    runtime.RegisterHoistTarget(buffer, bytes + guard);
    // Shaders reach guest memory by address (page table): resident, and declared on every encoder
    // (the declaration survives Submit). BB_ARENA_DECLARE=0 leaves the declaration out (an A/B
    // of its CPU cost per encoder; residency alone may be enough without direct memory access).
    device.MakeResident(buffer->Handle());
    static const bool declare = [] {
        const char* env = std::getenv("BB_ARENA_DECLARE");
        const bool on = !env || env[0] != '0';
        if (!on) {
            std::printf("GPU arena: not declared on encoders (BB_ARENA_DECLARE=0)\n");
        }
        return on;
    }();
    if (declare) {
        scheduler.Record(
            [handle = buffer->Handle()](Metal::Encoder& e) { e.DeclareIndirect(handle); });
    }
    return buffer;
}

void BufferCache::OnRunsChanged(void* self) {
    // The ring thread, before it dispatches the record after the maps: the GPU command thread
    // runs this before its next packet (Liverpool::ProcessCommands, the draw pipe drained), so
    // every packet issued after a map finds its chunk.
    auto* cache = static_cast<BufferCache*>(self);
    cache->liverpool->SendCommand<false>([cache] { cache->MaterializeRuns(); });
}

void BufferCache::MaterializeRuns() {
    std::scoped_lock lock{materialize_mutex};
    const auto runs = BbArenaRuns::Runs();
    if (runs.empty()) {
        return;
    }
    std::vector<BbArenaRuns::Range> wanted;
    if (layout == Layout::Span) {
        wanted.push_back({runs.front().start, runs.back().end});
    } else {
        wanted = runs;
    }
    // Chunks never move, shrink or go away during a session (bindings recorded with them may be
    // in flight): a run that grew, or a span that widened, gets another chunk for the new part.
    bool synced = false;
    for (const auto& w : wanted) {
        for (VAddr p = w.start; p < w.end;) {
            if (const auto* c = chunks.FindContaining(p)) {
                p = c->end;
                continue;
            }
            const auto* next = chunks.NextAfter(p);
            const VAddr end = std::min<VAddr>(w.end, next ? next->start : w.end);
            const u64 bytes = end - p;
            if (chunk_bytes.load() + bytes + guard_bytes > legacy_bytes) {
                std::printf("GPU arena: no buffer for %#llx-%#llx: the buffers would pass the "
                            "legacy arena's %.1f GiB (requests there take the slow route)\n",
                            static_cast<unsigned long long>(p),
                            static_cast<unsigned long long>(end),
                            double(legacy_bytes) / double(1_GB));
                p = end;
                continue;
            }
            if (!synced) {
                // The recording thread may hold commands that use the registries (ordering,
                // hoisting) a chunk joins: let it finish them first.
                scheduler.SyncRecording();
                synced = true;
            }
            const auto t0 = std::chrono::steady_clock::now();
            SetUpChunk(p, bytes, fmt::format("Guest Memory {:#x}", p), guard_bytes);
            const u64 total = chunk_bytes.fetch_add(bytes + guard_bytes) + bytes + guard_bytes;
            BbStats::layout_resident_bytes.fetch_add(bytes + guard_bytes,
                                                     std::memory_order_relaxed);
            std::printf("GPU arena: buffer %#llx-%#llx, %.3f MiB + %llu MiB guard, in %.1f ms; "
                        "guest memory buffers now %.1f MiB\n",
                        static_cast<unsigned long long>(p), static_cast<unsigned long long>(end),
                        double(bytes) / double(1_MB),
                        static_cast<unsigned long long>(guard_bytes >> 20),
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count(),
                        double(total) / double(1_MB));
            p = end;
        }
    }
}

void BufferCache::WriteLayoutFallback(const std::string& reason) {
    const auto path =
        Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "gpu_memory_layout_fallback";
    if (std::FILE* f = std::fopen(path.string().c_str(), "w")) {
        std::fprintf(f, "%s\n%s\n", reason.c_str(), LayoutBuildStamp().c_str());
        std::fclose(f);
    }
    std::printf("GPU arena: %s; the next launch uses the previous layout (%s)\n", reason.c_str(),
                path.string().c_str());
}

void BufferCache::MonitorLayout(std::stop_token stop) {
    Common::SetCurrentThreadName("bb:GuestLayout");
    const char* name = layout == Layout::Runs ? "runs" : "span";
    std::mutex m;
    std::condition_variable_any cv;
    // Only GPU page faults (an address no resident buffer maps) step back: the one error a
    // layout can cause. Other command buffer errors (out of memory, timeouts) are more likely
    // on the small Macs this layout is for, and stepping back on them kept those Macs slow.
    u64 last_faults = device.PageFaultCount();
    bool guard_logged = false;
    for (u32 round = 0;; ++round) {
        {
            std::unique_lock lock{m};
            cv.wait_for(lock, stop, std::chrono::seconds(10), [] { return false; });
        }
        if (stop.stop_requested()) {
            return;
        }
        const u64 faults = device.PageFaultCount();
        if (faults > last_faults) {
            WriteLayoutFallback(fmt::format("{}: {} GPU page faults", name, faults));
            return;
        }
        last_faults = faults;
        // The guards hold zeros unless something wrote past a run (every 30 s): logged once, as
        // a diagnostic. Writes there are expected (ObtainMiss binds writes ending in a guard,
        // image copies may fill to a buffer's end), so they do not step back. Only the
        // first MiB past each run, and only its pages the process has (mincore): reading the
        // rest gave the process 3 x 64 MiB of zero pages (~170 MiB more footprint).
        if (guard_logged || round % 3 != 2) {
            continue;
        }
        const u64 page = static_cast<u64>(getpagesize());
        for (const auto& c : chunks.Chunks()) {
            const u8* view = c.buffer->buffer.HostView();
            const u64 run = c.end - c.start;
            if (!view || c.buffer->SizeBytes() <= run) {
                continue;
            }
            const u64 scan = std::min<u64>(c.buffer->SizeBytes() - run, 1_MB);
            const u8* guard = view + run;
            std::array<char, 1_MB / 4096> resident{};
            if (mincore(const_cast<u8*>(guard), scan, resident.data()) != 0) {
                continue;
            }
            for (u64 p = 0; p < scan && !guard_logged; p += page) {
                if (!(resident[p / page] & MINCORE_INCORE)) {
                    continue;
                }
                const u64* words = reinterpret_cast<const u64*>(guard + p);
                for (u64 i = 0; i < std::min(page, scan - p) / sizeof(u64); ++i) {
                    if (words[i] != 0) {
                        std::printf("GPU arena: bytes written %llu past the buffer at %#llx (its "
                                    "guard; %s layout kept)\n",
                                    static_cast<unsigned long long>(p + i * sizeof(u64)),
                                    static_cast<unsigned long long>(c.start), name);
                        guard_logged = true;
                        break;
                    }
                }
            }
        }
    }
}

BufferCache::~BufferCache() {
    if (layout != Layout::Legacy) {
        BbArenaRuns::SetListener(nullptr, nullptr);
    }
}

// fifa-hybrid diagnostics ([WATCH]): the watched table's byte range should stay in
// gpu_modified_ranges as long as its pages are GPU-modified; report the first loss.
static void WatchCheckLoss(BufferCache& bc, RangeSet& ranges, const char* where) {
    const u64 lo = g_watch_lo.load(std::memory_order_relaxed), hi = g_watch_hi.load(std::memory_order_relaxed);
    if (hi == 0) {
        return;
    }
    static u64 reported_for = 0;
    if (reported_for == lo) {
        return;
    }
    const bool pages = bc.IsRegionGpuModified(lo, hi - lo);
    const bool bytes = ranges.Intersects(lo, hi - lo);
    if (pages && !bytes) {
        reported_for = lo;
        std::fprintf(stderr, "[WATCH] LOST byte range %#llx+%llu at %s (tid %#llx)\n", (unsigned long long)lo,
                     (unsigned long long)(hi - lo), where, (unsigned long long)(uintptr_t)pthread_self());
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    WatchCheckLoss(*this, gpu_modified_ranges, "InvalidateMemory");
    if (WatchOverlaps(device_addr, size)) {
        std::fprintf(stderr, "[WATCH] invalidate %#llx+%llu gpu_modified=%d\n", (unsigned long long)device_addr,
                     (unsigned long long)size, int(memory_tracker->IsRegionGpuModified(device_addr, size)));
    }
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        write_flushes.fetch_add(1, std::memory_order_relaxed);
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

// bbport: the game fills its per-frame buffers (constants, skinning output) sequentially and
// every 4 KiB page cost a protection fault (~110k/s in Hunter's Nightmare, a fifth of each
// render worker's time in the kernel). A fault unprotects the aligned window around it instead;
// pages marked CPU-modified without being written only cost an upload when bound.
// Item: context = BufferCache, source = guest address, destination = host pointer, size,
// extra = the Buffer whose mapping holds the destination (shared storage: nothing to flush).
void BufferCache::RunGuestCopy(const BbCopy::Item& item) {
    auto* cache = static_cast<BufferCache*>(item.context);
    auto* dst = reinterpret_cast<u8*>(item.destination);
    cache->memory->CopySparseMemory(item.source, dst, item.size);
}

// Small guest copies run on the recording thread (it spins for work: no wakeup, and it is
// idle most of the time); PoolSmallCopies (toggle 524288) batches them for the copy threads.
void BufferCache::SmallGuestCopy(const BbCopy::Item& item) {
    if (scheduler.IsRecordingDeferred() && !BbToggle::Disabled(BbToggle::PoolSmallCopies)) {
        scheduler.RecordHostCopy([item] { item.run(item); });
        return;
    }
    BbCopy::QueueCopy(item);
}

void BufferCache::NoteReadFault(VAddr addr) {
    static const u32 threshold = [] {
        const char* env = std::getenv("BB_HOT_READ");
        return env ? u32(std::strtoul(env, nullptr, 10)) : 8u;
    }();
    if (threshold == 0) {
        return;
    }
    const VAddr page = addr & ~VAddr{0xFFF};
    if (oneshot_ranges.Intersects(page, 0x1000)) {
        // The page holds a GPU result that is not a steady rewrite: its reads stay exact.
        return;
    }
    std::scoped_lock lk{hot_read_mutex};
    if (++read_fault_counts[page] != threshold) {
        return;
    }
    hot_read_pages.push_back(page);
    memory_tracker->SetHotRead(page, 0x1000);
    std::fprintf(stderr, "[HOTREAD] page %#llx (%zu hot pages)\n", (unsigned long long)page,
                 hot_read_pages.size());
}

void BufferCache::DropHotRead(VAddr page, const char* why) {
    std::scoped_lock lk{hot_read_mutex};
    const auto it = std::find(hot_read_pages.begin(), hot_read_pages.end(), page);
    if (it == hot_read_pages.end()) {
        return;
    }
    hot_read_pages.erase(it);
    read_fault_counts[page] = 0;
    memory_tracker->ClearHotRead(page, 0x1000);
    static std::atomic<int> n{0};
    if (n++ < 40) {
        std::fprintf(stderr, "[HOTREAD] page %#llx dropped: %s (%zu hot pages)\n",
                     (unsigned long long)page, why, hot_read_pages.size());
    }
}

void BufferCache::DropHotReadIn(VAddr addr, u64 size, const char* why) {
    boost::container::small_vector<VAddr, 4> hot;
    {
        std::scoped_lock lk{hot_read_mutex};
        if (hot_read_pages.empty()) {
            return;
        }
        for (const VAddr page : hot_read_pages) {
            if (page < addr + size && addr < page + 0x1000) {
                hot.push_back(page);
            }
        }
    }
    for (const VAddr page : hot) {
        DropHotRead(page, why);
    }
}

// fifa-hybrid: a GPU write to guest memory takes an asynchronous write-back path (hot read pages,
// tiny written buffers) only when it is a steady rewrite of a tiny buffer: at most 64 bytes
// (BB_ASYNC_MAX_BYTES) bound as written in several frames in a row (a per-frame histogram slot,
// an occlusion counter). A CPU read that comes before such a write-back sees the previous
// frame's value, which that kind of reader is built for. Everything else is a result: its pages
// stay read-protected until the CPU reads them, and that read waits for the GPU.
// FIFA computes its 35 pitch light probes on the GPU while the match loads, one 112-byte record a
// frame, and reads each exactly once; it binds the whole 3920-byte table as written every frame,
// so "written every frame" alone does not make a write steady. On a hot read page the game read
// a record before its write-back had landed and kept zeros: every player near that probe was
// drawn dark for the whole match.
void BufferCache::NoteHostWrite(VAddr addr, u64 size) {
    if (size == 0 || memory_tracker->IsRegionGpuModified(addr, size)) {
        return; // GPU data pending there: the callers take the GPU path for those
    }
    memory_tracker->MarkRegionAsCpuModified(addr, size);
}

u64 BufferCache::DebugArenaDiff(VAddr addr, u64 size, u64* clean, u64* first_diff) {
    const Buffer* arena = FindChunk(addr, size);
    if (clean) {
        *clean = 0;
    }
    if (!arena || size == 0) {
        return ~0ull;
    }
    struct Piece {
        VAddr start;
        u64 size;
        u64 staging_offset;
    };
    std::vector<Piece> pieces;
    boost::container::small_vector<BufferCopy, 16> copies;
    u64 total = 0;
    const VAddr end = addr + size;
    for (VAddr page = addr & ~0xfffull; page < end; page += 0x1000) {
        if (memory_tracker->IsRegionCpuModified(page, 0x1000) ||
            memory_tracker->IsRegionGpuModified(page, 0x1000)) {
            continue;
        }
        const VAddr lo = Common::AlignDown(std::max(page, addr), 4);
        const VAddr hi = Common::AlignUp(std::min<VAddr>(page + 0x1000, end), 4);
        copies.push_back(BufferCopy{
            .src_offset = arena->Offset(lo),
            .dst_offset = total,
            .size = hi - lo,
        });
        pieces.push_back({lo, hi - lo, total});
        total += Common::AlignUp(hi - lo, 64);
    }
    if (total == 0) {
        return 0;
    }
    const auto download = staging_pool.Request(total, VideoCore::MemoryType::HostCached, 64);
    for (auto& copy : copies) {
        copy.dst_offset += download.offset;
    }
    runtime.CopyBuffer(*arena, *download.buffer, copies);
    scheduler.Finish();
    u64 diff = 0;
    for (const auto& piece : pieces) {
        const u8* guest = std::bit_cast<const u8*>(piece.start);
        const u8* gpu = download.mapped + piece.staging_offset;
        for (u64 i = 0; i < piece.size; ++i) {
            if (guest[i] != gpu[i]) {
                if (diff == 0 && first_diff) {
                    *first_diff = piece.start + i;
                }
                ++diff;
            }
        }
        if (clean) {
            *clean += piece.size;
        }
    }
    return diff;
}

void BufferCache::NoteGpuWrite(VAddr addr, u64 size) {
    static const u32 need = [] {
        const char* env = std::getenv("BB_ONESHOT_WRITES"); // 0 = off (every write is steady)
        return env ? u32(std::strtoul(env, nullptr, 10)) : 4u;
    }();
    static const u64 max_bytes = [] {
        const char* env = std::getenv("BB_ASYNC_MAX_BYTES");
        return env ? u64(std::strtoull(env, nullptr, 10)) : 64ull;
    }();
    if (need == 0 || size == 0) {
        return;
    }
    static const bool log = std::getenv("BB_ONESHOT_LOG") != nullptr;
    if (size > max_bytes) {
        if (!oneshot_ranges.Contains(addr, size)) {
            oneshot_ranges.Add(addr, size);
            static std::atomic<int> logged_big{0};
            if (log && logged_big++ < 20000) {
                std::fprintf(stderr, "[ONESHOT] gpu write %#llx+%llu (result)\n",
                             (unsigned long long)addr, (unsigned long long)size);
            }
        }
        DropHotReadIn(addr, size, "holds a GPU result");
        return;
    }
    constexpr u64 SameFrameNs = 10'000'000;   // more writes within one frame count once
    constexpr u64 MaxGapNs = 2'000'000'000;   // a longer pause starts over
    const u64 now = u64(std::chrono::steady_clock::now().time_since_epoch().count());
    auto& h = gpu_write_history[addr];
    if (h.last_ns == 0 || now - h.last_ns > MaxGapNs) {
        h.count = 1;
        h.last_ns = now;
    } else if (now - h.last_ns >= SameFrameNs) {
        h.count = std::min<u32>(h.count + 1, 1u << 20);
        h.last_ns = now;
    }
    if (h.count > need) {
        if (oneshot_ranges.Intersects(addr, size)) {
            oneshot_ranges.Subtract(addr, size);
        }
        return;
    }
    oneshot_ranges.Add(addr, size);
    static std::atomic<int> logged{0};
    if (log && logged++ < 4000) {
        std::fprintf(stderr, "[ONESHOT] gpu write %#llx+%llu (write %u in a row)\n",
                     (unsigned long long)addr, (unsigned long long)size, h.count);
    }
    DropHotReadIn(addr, size, "one-shot GPU write");
    if (gpu_write_history.size() > 16384) {
        std::erase_if(gpu_write_history,
                      [now](const auto& e) { return now - e.second.last_ns > MaxGapNs; });
    }
}

void BufferCache::FlushHotReadPages() {
    WatchCheckLoss(*this, gpu_modified_ranges, "FlushHotReadPages (submit)");
    // fifa-hybrid diagnostics: BB_HOT_PAGE=<hex> forces a page onto the hot list (its GPU writes are
    // written back every submit, so BB_BUFWATCH can print them).
    static bool forced = false;
    if (!forced) {
        forced = true;
        if (const char* env = std::getenv("BB_HOT_PAGE")) {
            const VAddr page = VAddr(std::strtoull(env, nullptr, 16)) & ~VAddr{0xFFF};
            std::scoped_lock lk{hot_read_mutex};
            hot_read_pages.push_back(page);
            memory_tracker->SetHotRead(page, 0x1000);
            std::fprintf(stderr, "[HOTREAD] forced page %#llx\n", (unsigned long long)page);
        }
    }
    boost::container::small_vector<VAddr, 8> pages;
    {
        std::scoped_lock lk{hot_read_mutex};
        pages.assign(hot_read_pages.begin(), hot_read_pages.end());
    }
    for (const VAddr page : pages) {
        if (oneshot_ranges.Intersects(page, 0x1000)) {
            DropHotRead(page, "holds a one-shot GPU write");
            continue;
        }
        if (memory_tracker->IsRegionGpuModified(page, 0x1000)) {
            DownloadSmallAsync(page, 0x1000);
        }
    }
}

void BufferCache::OnCpuWrite(VAddr addr, u64 size, VAddr fault) {
    WatchCheckLoss(*this, gpu_modified_ranges, "OnCpuWrite");
    if (WatchOverlaps(addr, size)) {
        std::fprintf(stderr, "[WATCH] cpu write fault %#llx (window %#llx+%llu) gpu_modified=%d\n",
                     (unsigned long long)fault, (unsigned long long)addr, (unsigned long long)size,
                     int(memory_tracker->IsRegionGpuModified(fault & ~VAddr{0xFFF}, 0x1000)));
    }
    memory_tracker->ExtendWriteFault(addr, size);
    const VAddr page = fault & ~VAddr{0xFFF};
    if (memory_tracker->IsRegionGpuModified(page, 0x1000)) {
        // The written page also holds GPU-written words not yet written back (a polled label
        // page, say). The pending write-backs stay: they cover only the GPU-written bytes
        // (gpu_modified_ranges), and dropping them lost the word the game was polling (match
        // frozen at 60 fps). The page is CPU-modified too, so the next upload takes the CPU's bytes.
        static const bool stats = std::getenv("BB_FAULT_STATS") != nullptr;
        const u64 n = cpu_write_wins.fetch_add(1, std::memory_order_relaxed);
        if (stats && n < 20) {
            std::fprintf(stderr, "[CPUW] CPU write into a page with pending GPU writes at %#llx\n",
                         (unsigned long long)fault);
        }
        memory_tracker->MarkRegionAsCpuModified(page, 0x1000);
    }
}

void BufferCache::ExtendWriteFault(VAddr device_addr) {
    static const u64 window = [] {
        const char* env = std::getenv("BB_FAULT_WINDOW");
        const u64 kib = env ? std::strtoull(env, nullptr, 10) : 256;
        return std::bit_ceil(std::clamp<u64>(kib, 4, 1024)) * 1024;
    }();
    if (window <= TRACKER_BYTES_PER_PAGE || BbToggle::Disabled(BbToggle::FaultWindow)) {
        return;
    }
    memory_tracker->ExtendWriteFault(Common::AlignDown(device_addr, window), window);
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    // Counted on the command thread (inside flush_request): it looks at the GPU write ranges.
    const bool gpu_read =
        !is_write && !assume_locks && memory_tracker->IsRegionGpuModified(device_addr, size);
    // bbport: guest memory the arena does not cover holds no GPU data of this cache (every
    // GPU-modified range was bound through the arena): nothing to read back. A fault there
    // stopped the game in GetChunk before (texture watchers protect pages anywhere).
    // The chunk is looked up by the faulting address alone: a request ending past a run's end
    // (a read fault in its last bytes) was declined and left the page protected, so the guest
    // faulted on it forever. The window below stops at the chunk's end.
    if (!FindChunk(device_addr, 1)) {
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
        BbStats::uncovered_readbacks.fetch_add(1, std::memory_order_relaxed);
        BbArenaRuns::Note(BbArenaRuns::Caller::ReadBack, device_addr, size);
        return;
    }
    const auto flush_request = [this, device_addr, size, is_write, gpu_read] {
        if (gpu_read) {
            NoteReadFault(device_addr);
        }
        // fifa-hybrid: the download must see every submitted draw and dispatch, including the
        // ones still queued in the draw pipe (recorded by other threads): drain it first, or a
        // guest polling a GPU-written counter reads a value from before the dispatch, the page
        // is unprotected, and it spins forever (FIFA at the title screen).
        if (auto* r = liverpool->GetRasterizer()) {
            r->DrainDrawPipe();
        }
        const auto* arena = GetChunk(device_addr, 1);
        BbArenaRuns::Note(BbArenaRuns::Caller::ReadBack, device_addr, size);

        // GPU-modified ranges come as many small scattered islands,
        // so the download is widened to a window around the request.
        // It ends with the chunk's guest memory, not its guard (another run may follow there).
        constexpr u64 WindowSize = 512_KB;
        const auto* chunk = chunks.Find(device_addr, 1);
        const VAddr arena_end = chunk ? chunk->end : arena->cpu_addr + arena->size_bytes;
        const VAddr window_start =
            std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
        const VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
        if (auto* r = liverpool->GetRasterizer(); r && r->HasDeferredDispatches() &&
            (r->NoteDeferredConflictSite("readback"),
             r->DeferredDispatchTouches(window_start, window_end - window_start, false))) {
            r->FlushDeferredDispatches(); // fifa-hybrid: a deferred dispatch may write what is read
        }
        DownloadMemory(arena, window_start, window_end - window_start);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        flush_request();
    } else {
        liverpool->SendCommand<true>(std::move(flush_request));
    }
}

void BufferCache::DownloadSmallAsync(VAddr addr, u64 size) {
    if (WatchOverlaps(addr, size)) {
        std::fprintf(stderr, "[WATCH] small async write-back %#llx+%llu\n", (unsigned long long)addr,
                     (unsigned long long)size);
    }
    const VAddr page = addr & ~VAddr(0xFFF);
    const VAddr end = Common::AlignUp(addr + size, 0x1000);
    if (oneshot_ranges.Intersects(page, end - page)) {
        // The pages stay GPU-modified (read-protected): the CPU's read waits for the data.
        return;
    }
    const auto* chunk = FindChunk(page, 1);
    if (!chunk) {
        return;
    }
    if (auto* r = liverpool->GetRasterizer(); r && r->HasDeferredDispatches() &&
        (r->NoteDeferredConflictSite("small-async"), r->DeferredDispatchTouches(page, end - page, false))) {
        r->FlushDeferredDispatches(); // fifa-hybrid
    }
    struct Download {
        VAddr address;
        u64 size;
        u64 staging_offset;
    };
    boost::container::small_vector<BufferCopy, 2> copies;
    boost::container::small_vector<Download, 2> downloads;
    u64 total_size_bytes = 0;
    const VAddr arena_base = chunk->cpu_addr;
    const VAddr range_end = std::min<VAddr>(end, chunk->cpu_addr + chunk->size_bytes);
    memory_tracker->ForEachDownloadRange<false>(page, range_end - page, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr stop) {
            const VAddr word_start = Common::AlignDown(start, 4);
            const VAddr word_end = Common::AlignUp(stop, 4);
            copies.push_back(BufferCopy{
                .src_offset = word_start - arena_base,
                .dst_offset = total_size_bytes,
                .size = word_end - word_start,
            });
            downloads.push_back({start, stop - start, total_size_bytes + (start - word_start)});
            constexpr u64 align = 64ULL;
            total_size_bytes += (word_end - word_start + align - 1) & ~(align - 1ULL);
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
        if (WatchOverlaps(address, size)) {
            std::fprintf(stderr, "[WATCH]   subtract (DownloadSmallAsync) %#llx+%llu tid %#llx\n", (unsigned long long)address,
                         (unsigned long long)size, (unsigned long long)(uintptr_t)pthread_self());
        }
        gpu_modified_ranges.Subtract(address, size);
        oneshot_ranges.Subtract(address, size);
    });
    if (total_size_bytes == 0) {
        return;
    }
    const auto download =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 64, true);
    for (auto& copy : copies) {
        copy.dst_offset += download.offset;
    }
    runtime.CopyBuffer(*chunk, *download.buffer, copies);
    scheduler.DeferPriorityOperation([this, download, downloads = std::move(downloads)] {
        const u64 watch = g_bb_bufwatch.load(std::memory_order_relaxed);
        for (const auto& d : downloads) {
            if (watch != 0 && (d.address & ~VAddr{0xFFF}) == (watch & ~VAddr{0xFFF})) {
                static std::atomic<int> n{0};
                static const int want_n = [] {
                    const char* env = std::getenv("BB_BUFWATCH_N");
                    return env ? std::atoi(env) : 0;
                }();
                if (want_n > 0 && n < 400) {
                    // BB_BUFWATCH_N=<n>: also print n floats from the watched offset inside the download.
                    const u64 off = watch > d.address ? watch - d.address : 0;
                    if (off + 4 <= d.size) {
                        const u8* src = download.mapped + d.staging_offset + off;
                        const int cnt = int(std::min<u64>(u64(want_n), (d.size - off) / 4));
                        std::string line = fmt::format("[BUFWATCH] floats @{:#x}:", (unsigned long long)(d.address + off));
                        for (int i = 0; i < cnt; ++i) {
                            float v;
                            std::memcpy(&v, src + i * 4, 4);
                            line += fmt::format(" {:g}", v);
                        }
                        std::fprintf(stderr, "%s\n", line.c_str());
                    }
                }
                if (n++ < 60) {
                    const u8* src = download.mapped + d.staging_offset;
                    float f[4] = {};
                    std::memcpy(f, src, std::min<u64>(d.size, sizeof f));
                    u32 u[4] = {};
                    std::memcpy(u, src, std::min<u64>(d.size, sizeof u));
                    std::fprintf(stderr, "[BUFWATCH] writeback %#llx+%llu: %g %g %g %g (%#x %#x %#x %#x)\n",
                                 (unsigned long long)d.address, (unsigned long long)d.size, f[0], f[1],
                                 f[2], f[3], u[0], u[1], u[2], u[3]);
                }
            }
            memory->TryWriteBacking(std::bit_cast<u8*>(d.address),
                                    download.mapped + d.staging_offset, d.size);
            // The result is in guest memory now: an image the cache still holds over it is stale.
            texture_cache.NoteLanding(d.address, d.size);
        }
        staging_pool.FreeDeferred(download);
    });
    memory_tracker->UnmarkRegionAsGpuModified(page, range_end - page);
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    WatchCheckLoss(*this, gpu_modified_ranges, "DownloadMemory entry");
    if (WatchOverlaps(device_addr, size)) {
        std::fprintf(stderr, "[WATCH] download (write-back) %#llx+%llu\n", (unsigned long long)device_addr,
                     (unsigned long long)size);
    }
    // Metal copies whole 32-bit words (offsets and sizes multiples of 4): each range is copied
    // widened to words, and only its own bytes go back to guest memory.
    struct Download {
        VAddr address;
        u64 size;
        u64 staging_offset; ///< of `address` in the download, relative to the request
    };
    boost::container::small_vector<BufferCopy, 1> copies;
    boost::container::small_vector<Download, 1> downloads;
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    const bool watched_dl = WatchOverlaps(device_addr, size);
    if (watched_dl) {
        gpu_modified_ranges.ForEachOverlappingLinear(device_addr, size, [&](VAddr a, VAddr b) {
            std::fprintf(stderr, "[WATCH]   set interval in window (linear): %#llx..%#llx\n", (unsigned long long)a,
                         (unsigned long long)b);
        });
    }
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        if (watched_dl) {
            std::fprintf(stderr, "[WATCH]   gpu-modified page run %#llx+%llu\n", (unsigned long long)address,
                         (unsigned long long)size);
            gpu_modified_ranges.ForEachOverlappingLinear(address, size, [&](VAddr a, VAddr b) {
                std::fprintf(stderr, "[WATCH]     linear overlap %#llx..%#llx\n", (unsigned long long)a,
                             (unsigned long long)b);
            });
        }
        const auto add_download = [&](VAddr start, VAddr end) {
            if (watched_dl) {
                std::fprintf(stderr, "[WATCH]     byte range %#llx..%#llx\n", (unsigned long long)start,
                             (unsigned long long)end);
            }
            const VAddr word_start = Common::AlignDown(start, 4);
            const VAddr word_end = Common::AlignUp(end, 4);
            copies.push_back(BufferCopy{
                .src_offset = word_start - arena_base,
                .dst_offset = total_size_bytes,
                .size = word_end - word_start,
            });
            downloads.push_back({start, end - start, total_size_bytes + (start - word_start)});
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (word_end - word_start + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
        if (WatchOverlaps(address, size)) {
            std::fprintf(stderr, "[WATCH]   subtract (DownloadMemory) %#llx+%llu tid %#llx\n", (unsigned long long)address,
                         (unsigned long long)size, (unsigned long long)(uintptr_t)pthread_self());
        }
        gpu_modified_ranges.Subtract(address, size);
        oneshot_ranges.Subtract(address, size);
    });
    if (total_size_bytes == 0) {
        return;
    }
    const auto download =
        staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 64);
    for (auto& copy : copies) {
        copy.dst_offset += download.offset;
    }
    runtime.CopyBuffer(*arena, *download.buffer, copies);
    // Shared storage is coherent: the data is readable once the copy completed.
    scheduler.Finish();

    const bool watched = WatchOverlaps(device_addr, size);
    u64 watched_bytes = 0;
    // BB_CLOB_LOG: write-backs that change guest bytes on pages the CPU wrote after the GPU write
    // (stale GPU ranges landing over newer CPU data), and those that land on a CPU-filled image.
    if (static const bool clob = std::getenv("BB_CLOB_LOG") != nullptr; clob) {
        static u64 n_all = 0, n_cm = 0, n_img = 0, b_all = 0, b_cm = 0;
        static u32 lines = 0;
        for (const auto& d : downloads) {
            const u8* guest = std::bit_cast<const u8*>(d.address);
            const u8* gpu = download.mapped + d.staging_offset;
            u64 diff = 0, first = 0;
            for (u64 i = 0; i < d.size; ++i) {
                if (guest[i] != gpu[i]) {
                    if (!diff) first = i;
                    ++diff;
                }
            }
            ++n_all;
            b_all += d.size;
            if (!diff) {
                continue;
            }
            const bool cm = memory_tracker->IsRegionCpuModified(d.address, d.size);
            const ImageId img = texture_cache.FindImageFromRange(d.address, d.size, false);
            const bool cpu_image = img && False(texture_cache.GetImage(img).flags & ImageFlagBits::GpuModified);
            n_cm += cm;
            b_cm += cm ? diff : 0;
            n_img += cpu_image;
            if ((cm || cpu_image) && (lines < 600 || (cpu_image && lines < 3000))) {
                ++lines;
                u32 was = 0, now = 0;
                std::memcpy(&was, guest + (first & ~3ull), std::min<u64>(4, d.size - (first & ~3ull)));
                std::memcpy(&now, gpu + (first & ~3ull), std::min<u64>(4, d.size - (first & ~3ull)));
                std::fprintf(stderr, "[CLOB] write-back %#llx+%llu (request %#llx+%llu): %llu bytes change, cpu-modified %d, "
                             "cpu image %d, first +%llu %08x -> %08x\n",
                             (unsigned long long)d.address, (unsigned long long)d.size,
                             (unsigned long long)device_addr, (unsigned long long)size,
                             (unsigned long long)diff, int(cm), int(cpu_image), (unsigned long long)first, was, now);
            }
        }
        if ((n_all & 0x3ff) == 0) {
            std::fprintf(stderr, "[CLOB] totals: %llu write-backs %.1f MB, %llu changed CPU-modified pages (%.2f MB), %llu on CPU images\n",
                         (unsigned long long)n_all, b_all / 1e6, (unsigned long long)n_cm, b_cm / 1e6,
                         (unsigned long long)n_img);
        }
    }
    for (const auto& d : downloads) {
        const bool ok = memory->TryWriteBacking(std::bit_cast<u8*>(d.address), download.mapped + d.staging_offset,
                                                d.size);
        texture_cache.NoteLanding(d.address, d.size);
        if (watched && WatchOverlaps(d.address, d.size)) {
            watched_bytes += d.size;
            const u64 lo = g_watch_lo.load(std::memory_order_relaxed);
            if (d.address <= lo && lo + 16 <= d.address + d.size) {
                const u32* w = reinterpret_cast<const u32*>(download.mapped + d.staging_offset + (lo - d.address));
                std::fprintf(stderr, "[WATCH]   write-back covers the table start: dw %08x %08x %08x %08x ok=%d\n",
                             w[0], w[1], w[2], w[3], int(ok));
            }
        }
    }
    if (watched) {
        std::fprintf(stderr, "[WATCH]   write-back: %llu bytes inside the watched range (%zu ranges in the window)\n",
                     (unsigned long long)watched_bytes, downloads.size());
    }
    if (watched) {
        std::fprintf(stderr, "[WATCH]   unmark gpu-modified %#llx+%llu\n", (unsigned long long)device_addr,
                     (unsigned long long)size);
    }
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
}

namespace {
// bbport: BB_BUFFER_STATS=1 — how buffer bindings reach the GPU, by guest region (256 MiB),
// printed every 5 s: small read-only copies into the stream buffer, arena bindings, and the
// bytes those re-upload after CPU writes. Input for engine-level short paths.
struct BufferStats {
    struct Region {
        u64 stream_count{}, stream_bytes{}, arena_count{}, arena_bytes{}, upload_bytes{};
    };
    std::map<u64, Region> regions;
    /// Hot window: frame of the last binding per 64 KiB block, and the distribution of the
    /// frames between uses (1, 2, 3, 4, 5-8, 9-16, 17+): the ring's reuse distance.
    std::unordered_map<u64, u64> last_use;
    std::array<u64, 7> reuse{};
    std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
    u64 calls = 0;
};
/// 256 MiB regions; 1 MiB inside the hot 0x104xxxxxxx window (the engine's frame data).
BufferStats& Stats();
void NoteHotUse(VAddr address, u32 size) {
    if ((address >> 28) != 0x104) {
        return;
    }
    auto& st = Stats();
    const u64 frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
    for (u64 block = address >> 16; block <= (address + size - 1) >> 16; ++block) {
        auto [it, inserted] = st.last_use.try_emplace(block, frame);
        if (!inserted && it->second != frame) {
            const u64 d = frame - it->second;
            ++st.reuse[d <= 4 ? d - 1 : d <= 8 ? 4 : d <= 16 ? 5 : 6];
            it->second = frame;
        }
    }
}
u64 RegionKey(VAddr address) {
    return (address >> 28) == 0x104 ? (address >> 20) | (1ull << 40) : address >> 28;
}
bool BufferStatsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("BB_BUFFER_STATS");
        return value && value[0] == '1';
    }();
    return enabled;
}
BufferStats& Stats() {
    static BufferStats stats;
    return stats;
}
void PrintBufferStats() {
    auto& st = Stats();
    if ((++st.calls & 4095) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - st.window).count();
    if (seconds < 5.0) {
        return;
    }
    std::printf("Buffer stats (%.1f s, per second): region, stream copies/bytes, arena bindings, "
                "re-uploaded bytes\n", seconds);
    std::printf("  hot window reuse distance in frames (1,2,3,4,5-8,9-16,17+): %llu %llu %llu %llu "
                "%llu %llu %llu\n",
                (unsigned long long)st.reuse[0], (unsigned long long)st.reuse[1],
                (unsigned long long)st.reuse[2], (unsigned long long)st.reuse[3],
                (unsigned long long)st.reuse[4], (unsigned long long)st.reuse[5],
                (unsigned long long)st.reuse[6]);
    st.reuse = {};
    for (const auto& [region, r] : st.regions) {
        const u64 base = (region >> 40) ? (region & ((1ull << 40) - 1)) << 20 : region << 28;
        if ((region >> 40) && r.stream_bytes + r.upload_bytes < 1e6 * seconds) {
            continue; // quiet MiB of the hot window
        }
        std::printf("  %#012llx: %8.0f copies %8.2f MB, %8.0f arena %8.2f MB bound, %8.2f MB uploads\n",
                    static_cast<unsigned long long>(base), r.stream_count / seconds,
                    r.stream_bytes / seconds / 1e6, r.arena_count / seconds,
                    r.arena_bytes / seconds / 1e6,
                    r.upload_bytes / seconds / 1e6);
    }
    st.regions.clear();
    st.window = now;
}
} // namespace


std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer,
                                                        bool note_use, BbArenaRuns::Caller caller) {
    if (is_written) {
        VbNoteWrite(device_addr, size, 'O');
    }
    WatchCheckLoss(*this, gpu_modified_ranges, "ObtainBuffer entry");
    if (WatchOverlaps(device_addr, size)) {
        static std::atomic<int> n{0};
        if (n.fetch_add(1) < 400) {
            std::fprintf(stderr, "[WATCH] bind %#llx+%u written=%d cpu_modified=%d gpu_modified=%d\n",
                         (unsigned long long)device_addr, size, int(is_written),
                         int(memory_tracker->IsRegionCpuModified(device_addr, size)),
                         int(memory_tracker->IsRegionGpuModified(device_addr, size)));
        }
    }
    // fifa-hybrid: a GPU read of memory an on-demand image still owns (the guest reuses render
    // targets as vertex/constant data) would see stale RAM: download the image first.
    if (!is_written && texture_cache.HasOndemandIn(device_addr, size) &&
        texture_cache.HasOndemandOverlapping(device_addr, size)) {
        static const bool vb_log = std::getenv("BB_VB_LOG") != nullptr;
        static int n = 0;
        if (vb_log && n++ < 2000) {
            std::fprintf(stderr, "[RBQ] bufread addr=%#llx size=%#x caller=%d\n",
                         (unsigned long long)device_addr, size, int(caller));
        }
        static const bool off = std::getenv("BB_RB_NOBUFRESOLVE") != nullptr;
        // fifa-hybrid: BB_ONDEMAND_GPU_BUFREAD (default on): the image bytes go to the buffer on
        // the GPU (tiled into staging, copied into the arena range as a GPU write) instead of a
        // synchronous download to RAM and an upload back (a full GPU drain per read: FIFA reads a
        // 93x75 RGBA32F image as a buffer twice a frame -> 2.5 ms of GPU bubble a frame). The
        // bytes still land in guest memory, later, for the CPU (pages stay protected until then).
        static const bool gpu_read = [] {
            const char* e = std::getenv("BB_ONDEMAND_GPU_BUFREAD");
            return !(e && e[0] == '0');
        }();
        if (!off && gpu_read) {
            TextureCache::CpuReadWork work;
            texture_cache.DownloadImagesForCpuRead(device_addr, size, false, false, &work);
            if (!work.copies.empty()) {
                const auto [dst, dst_offset] =
                    ObtainBuffer(device_addr, size, true, is_texel_buffer, note_use, caller);
                for (const auto& c : work.copies) {
                    const VAddr b = std::max<VAddr>(c.addr, device_addr);
                    const VAddr e = std::min<VAddr>(c.addr + c.size, device_addr + size);
                    if (e <= b) {
                        continue;
                    }
                    const BufferCopy copy{.src_offset = c.download.offset + (b - c.addr),
                                          .dst_offset = dst_offset + (b - device_addr),
                                          .size = e - b};
                    runtime.CopyBuffer(*c.download.buffer, *dst, std::span{&copy, 1});
                }
                BbStats::ondemand_gpu_bufreads.fetch_add(1, std::memory_order_relaxed);
                work.tick = scheduler.CurrentTick();
                scheduler.DeferPriorityOperation(
                    [this, work = std::move(work)]() mutable { texture_cache.FinishCpuReadWork(work); });
                return {dst, dst_offset};
            }
            if (!work.unprotect.empty()) {
                texture_cache.FinishCpuReadWork(work);
            }
        } else if (!off) {
            texture_cache.DownloadImagesForCpuRead(device_addr, size, false, false);
            memory_tracker->InvalidateRegion(device_addr, size, [] {});
        }
    }
    const bool stats = BufferStatsEnabled();
    if (stats) {
        PrintBufferStats();
        NoteHotUse(device_addr, size);
    }
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (WatchOverlaps(device_addr, size) && !is_written && size <= STREAM_THRESHOLD &&
        !IsRegionGpuModified(device_addr, size)) {
        std::fprintf(stderr, "[WATCH] bind %#llx+%u read through the stream buffer (not gpu-modified)\n",
                     (unsigned long long)device_addr, size);
    }
    if (!is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size)) {
        if (stats) {
            auto& region = Stats().regions[RegionKey(device_addr)];
            ++region.stream_count;
            region.stream_bytes += size;
        }
        // bbport: the guest data is copied on a copy thread, started now; submission and
        // guest-visible fences wait for it (Scheduler::WaitHostCopies).
        if (!stream_buffer.mapped_data.empty() &&
            !BbToggle::Disabled(BbToggle::DeferredStreamCopies)) {
            if (const auto offset = stream_buffer.Reserve(size, device.GetLimits().uniform_alignment)) {
                SmallGuestCopy({
                    .run = &RunGuestCopy,
                    .context = this,
                    .source = device_addr,
                    .destination = reinterpret_cast<u64>(stream_buffer.mapped_data.data() + *offset),
                    .size = size,
                    .extra = reinterpret_cast<u64>(static_cast<Buffer*>(&stream_buffer)),
                });
                return {&stream_buffer, *offset};
            }
        }
        const auto [data, offset] = stream_buffer.Map(size, device.GetLimits().uniform_alignment);
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        return {&stream_buffer, offset};
    }
    BbArenaRuns::Note(caller, device_addr, size);
    if (!FindChunk(device_addr, size)) {
        // bbport: a size-0 request outside the guest memory buffers stopped the game in GetChunk.
        // It reaches no memory: a sink.
        if (size == 0) {
            BbStats::size0_sinks.fetch_add(1, std::memory_order_relaxed);
            return {&write_sink, 0};
        }
        // The legacy arena stops the game here (GetChunk): its end is past all guest memory.
        if (layout != Layout::Legacy) {
            return ObtainMiss(device_addr, size, is_written, is_texel_buffer, note_use);
        }
    }
    const auto* arena = GetChunk(device_addr, size);
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + std::max<u32>(size, 1) - 1) >> block_shift;
    EnsureResident(arena, first_block, last_block);
    const u64 uploaded_before = BbStats::buffer_upload_bytes.load(std::memory_order_relaxed);
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    if (stats) {
        auto& region = Stats().regions[RegionKey(device_addr)];
        ++region.arena_count;
        region.arena_bytes += size;
        region.upload_bytes +=
            BbStats::buffer_upload_bytes.load(std::memory_order_relaxed) - uploaded_before;
    }
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
        NoteGpuWrite(device_addr, size);
        if (WatchOverlaps(device_addr, size)) {
            u64 covered = 0;
            gpu_modified_ranges.ForEachInRange(device_addr, size, [&](VAddr a, VAddr b) { covered += b - a; });
            std::fprintf(stderr, "[WATCH] registered gpu write %#llx+%u: ranges cover %llu bytes, first page gpu=%d last page gpu=%d tid %#llx\n",
                         (unsigned long long)device_addr, size, (unsigned long long)covered, (unsigned long long)(uintptr_t)pthread_self(),
                         int(memory_tracker->IsRegionGpuModified(device_addr, 4)),
                         int(memory_tracker->IsRegionGpuModified(device_addr + size - 4, 4)));
        }
    }
    // bbport: index and vertex reads are not tracked as accesses; every use is (upload hoisting).
    if (note_use) {
        runtime.NoteSubmissionUse(arena, arena->Offset(device_addr), size);
    }
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainIndexBuffer(VAddr device_addr, u32 size) {
    return LowOffsetRange(device_addr, size, BbArenaRuns::Caller::Index);
}

std::pair<const Buffer*, u64> BufferCache::ObtainDrawArgumentBuffer(VAddr device_addr, u32 size) {
    return LowOffsetRange(device_addr, size, BbArenaRuns::Caller::Indirect);
}

std::pair<const Buffer*, u64> BufferCache::LowOffsetRange(VAddr device_addr, u32 size,
                                                          BbArenaRuns::Caller caller) {
    const auto [buffer, offset] = ObtainBuffer(device_addr, size, false, false, true, caller);
    // BB_DRAW_OFFSET_LIMIT_MB (tests): a lower limit, so that the copies below run for the
    // retail game's index and argument ranges (all of them below 4.8 GiB).
    static const u64 limit = [] {
        const char* env = std::getenv("BB_DRAW_OFFSET_LIMIT_MB");
        const u64 mb = env ? std::strtoull(env, nullptr, 10) : 0;
        return mb ? std::min<u64>(mb << 20, u64{1} << 32) : u64{1} << 32;
    }();
    const bool in_arena = FindChunk(device_addr, size) == buffer;
    if (!in_arena || (offset < limit && size <= limit - offset)) {
        return {buffer, offset};
    }
    // bbport: with BB_ARENA_WINDOWS=1 the encoder draws through a window of the arena at a low
    // offset (Metal::LowOffset) instead; off by default (Metal::CanLowerOffsets, #42).
    if (Metal::CanLowerOffsets(buffer->Ref().buffer)) {
        return {buffer, offset};
    }
    const bool index = caller == BbArenaRuns::Caller::Index;
    // The bytes in the arena are correct, but drawIndexedPrimitives fetches from the
    // wrapped offset (reproduced on M2). Small indices already use the stream buffer;
    // this is why a prop's details can draw while its larger body disappears.
    // bbport: indirectBufferOffset wraps the same way (32 bits).
    if (!IsRegionGpuModified(device_addr, size)) {
        (index ? BbStats::index_copies_cpu : BbStats::indirect_copies_cpu)
            .fetch_add(1, std::memory_order_relaxed);
        return ObtainBufferForImage(device_addr, size);
    }
    // A CPU copy would discard GPU-produced indices. Copy in command order instead,
    // using the staging pool's retirement ticks to keep the range alive through the draw.
    (index ? BbStats::index_copies_gpu : BbStats::indirect_copies_gpu)
        .fetch_add(1, std::memory_order_relaxed);
    const auto staging = staging_pool.Request(size, MemoryType::DeviceLocal, 16);
    const BufferCopy copy{offset, staging.offset, size};
    runtime.CopyBuffer(*buffer, *staging.buffer, std::span{&copy, 1});
    return {staging.buffer, staging.offset};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false, false, true, BbArenaRuns::Caller::Image);
    }
    return StagingCopy(device_addr, size);
}

std::pair<const Buffer*, u64> BufferCache::ObtainMiss(VAddr device_addr, u32 size, bool is_written,
                                                      bool is_texel_buffer, bool note_use) {
    const u64 n = misses.fetch_add(1, std::memory_order_relaxed);
    const VAddr end = device_addr + size;
    const auto* chunk = chunks.FindContaining(device_addr);
    const auto* next = chunks.NextAfter(device_addr);
    if (chunk && end <= chunk->start + chunk->buffer->SizeBytes() && (!next || end <= next->start)) {
        // Past the run, the guard: zeros, as memory the PS4's GPU has not mapped (only the run's
        // part is synchronized and marked GPU-written).
        if (n < 16) {
            std::printf("GPU arena: request %#llx + %u bytes ends %llu bytes past its buffer's run "
                        "(%s): bound with the guard\n",
                        static_cast<unsigned long long>(device_addr), size,
                        static_cast<unsigned long long>(end - chunk->end),
                        is_written ? "written" : "read");
        }
        const auto* arena = chunk->buffer;
        const u32 in_run = static_cast<u32>(chunk->end - device_addr);
        EnsureResident(arena, device_addr >> block_shift,
                       (device_addr + in_run - 1) >> block_shift);
        SynchronizeMemory(arena, device_addr, in_run, is_written, is_texel_buffer);
        if (is_written) {
            if (WatchOverlaps(device_addr, in_run)) {
                std::fprintf(stderr, "[WATCH] registered gpu write (miss path, in-run) %#llx+%u\n",
                             (unsigned long long)device_addr, in_run);
            }
            gpu_modified_ranges.Add(device_addr, in_run);
            NoteGpuWrite(device_addr, in_run);
        }
        if (note_use) {
            runtime.NoteSubmissionUse(arena, arena->Offset(device_addr), size);
        }
        return {arena, arena->Offset(device_addr)};
    }
    if (n < 16) {
        std::printf("GPU arena: request %#llx + %u bytes (%s) outside the guest memory buffers: "
                    "%s\n",
                    static_cast<unsigned long long>(device_addr), size,
                    is_written ? "written" : "read",
                    is_written ? "a scratch range" : "a copy of the guest bytes");
    }
    if (is_written) {
        const auto scratch = staging_pool.Request(size, MemoryType::DeviceLocal,
                                                  device.GetLimits().storage_alignment);
        return {scratch.buffer, scratch.offset};
    }
    return StagingCopy(device_addr, size);
}

std::pair<const Buffer*, u64> BufferCache::StagingCopy(VAddr device_addr, u32 size) {
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              device.GetLimits().storage_alignment);
    if (!BbToggle::Disabled(BbToggle::DeferredUploads)) {
        // bbport: texture data is copied on the copy threads (streaming: 100+ MB per frame);
        // the upload reads the staging only after submission, which waits for the copies.
        constexpr u64 Chunk = 1_MB;
        for (u64 offset = 0; offset < staging.size; offset += Chunk) {
            const BbCopy::Item item{
                .run = &RunGuestCopy,
                .context = this,
                .source = device_addr + offset,
                .destination = reinterpret_cast<u64>(staging.mapped + offset),
                .size = std::min(Chunk, staging.size - offset),
                .extra = reinterpret_cast<u64>(staging.buffer),
            };
            if (staging.size < Chunk) {
                SmallGuestCopy(item);
            } else {
                BbCopy::Async([item] { item.run(item); });
            }
        }
        return {staging.buffer, staging.offset};
    }
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::ProcessFaultBuffer() {
    if (fault_manager) {
        fault_manager->ProcessFaultBuffer();
    }
}

void BufferCache::SynchronizeDmaBuffers() {
    if (!bda_pagetable_buffer) {
        return; // BB_DIRECT_MEMORY_ACCESS is off: no shader reaches memory by address
    }
    for (const auto& range : resident_ranges) {
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(GetChunk(device_addr, size), device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetChunk(VAddr device_addr, u64 size) {
    const auto* chunk = chunks.Find(device_addr, size);
    ASSERT_MSG(chunk,
               "GPU buffer {:#x}..{:#x} outside the macOS guest arena (ends at {:#x}; limited by "
               "this Mac's largest Metal buffer)",
               device_addr, device_addr + std::max<u64>(size, 1) - 1, mac_arena_end);
    // bbport: how much of the arena the game reaches (the arena's sizing).
    const u64 end = Common::AlignUp(device_addr + std::max<u64>(size, 1), u64{block_size}) -
                    MacArenaBase;
    const u64 steps = (end + (256_MB - 1)) / 256_MB;
    u64 seen = arena_high_steps.load(std::memory_order_relaxed);
    while (steps > seen &&
           !arena_high_steps.compare_exchange_weak(seen, steps, std::memory_order_relaxed)) {
    }
    if (steps > seen) {
        std::printf("GPU arena: the game reached %.2f GiB of the %.1f GiB arena\n",
                    double(steps * 256_MB) / double(1_GB),
                    double(mac_arena_end - MacArenaBase) / double(1_GB));
    }
    return chunk->buffer;
}

const Buffer* BufferCache::FindChunk(VAddr device_addr, u64 size) const {
    const auto* chunk = chunks.Find(device_addr, size);
    return chunk ? chunk->buffer : nullptr;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    if (!bda_pagetable_buffer) {
        return; // the arena is fully backed; only the page table (direct memory access) is filled
    }
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }
    BbStats::Timer timer{BbStats::t_resident};
    // The arena is fully backed: residency only fills the BDA page table.
    boost::container::small_vector<BufferCopy, 8> copies;
    const auto staging = staging_pool.Request(resident_blocks * sizeof(u64),
                                              MemoryType::HostUncached, sizeof(u64));
    auto* bda_addrs = reinterpret_cast<u64*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        resident_ranges.Add({range.start, range.end});
        const u64 resource_offset = (range.start << block_shift) - arena->cpu_addr;
        for (u64 block = range.start; block < range.end; ++block) {
            *(bda_addrs++) =
                arena->GpuAddress() + resource_offset + ((block - range.start) << block_shift);
        }
        const u64 copy_size = (range.end - range.start) * sizeof(u64);
        copies.emplace_back(offset, range.start * sizeof(u64), copy_size);
        offset += copy_size;
    }
    runtime.CopyBuffer(*staging.buffer, *bda_pagetable_buffer, copies);
    BbStats::dma_first_touches.fetch_add(1, std::memory_order_relaxed);
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<BufferCopy, 4> copies;
    size_t total_size_bytes{};
    const Buffer* src_buffer{};
    auto* deferred_rasterizer = liverpool->GetRasterizer(); // fifa-hybrid: BB_DEFER_CS
    bool deferred_conflict = false;
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 addr, u64 size) {
            if (WatchOverlaps(addr, size)) {
                const u64 lo = g_watch_lo.load(std::memory_order_relaxed);
                const u64 at = std::max<u64>(addr, lo);
                const u32* w = reinterpret_cast<const u32*>(at);
                std::fprintf(stderr, "[WATCH] upload %#llx+%llu (bind %#llx+%u w%d) guest dw at %#llx: %08x %08x %08x %08x\n",
                             (unsigned long long)addr, (unsigned long long)size,
                             (unsigned long long)device_addr, size, int(is_written), (unsigned long long)at,
                             w[0], w[1], w[2], w[3]);
            }
            copies.emplace_back(total_size_bytes, addr, size);
            total_size_bytes += size;
            if (deferred_rasterizer && deferred_rasterizer->HasDeferredDispatches()) {
                deferred_rasterizer->NoteDeferredConflictSite("upload");
                deferred_conflict = deferred_conflict ||
                                    deferred_rasterizer->DeferredDispatchTouches(addr, size, true);
            }
        },
        [&] { src_buffer = UploadCopies(arena, copies, total_size_bytes); });

    if (deferred_conflict) {
        deferred_rasterizer->FlushDeferredDispatches(); // the upload lands after them
    }
    if (src_buffer && !runtime.HoistCopy(src_buffer, arena, copies)) {
        runtime.CopyBuffer(*src_buffer, *arena, copies);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

const Buffer* BufferCache::UploadCopies(const Buffer* arena, std::span<BufferCopy> copies,
                                        size_t total_size_bytes) {
    if (copies.empty()) {
        return nullptr;
    }
    BbStats::buffer_upload_bytes.fetch_add(total_size_bytes, std::memory_order_relaxed);
    const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached,
                                              device.GetLimits().storage_alignment);
    // bbport: the guest memory is copied into staging on the copy threads, started now in
    // groups of about 1 MiB; submission and guest-visible fences wait for them
    // (Scheduler::WaitHostCopies).
    if (!BbToggle::Disabled(BbToggle::DeferredUploads) && total_size_bytes < 1_MB) {
        // Small uploads join the calling thread's batch.
        for (auto& copy : copies) {
            SmallGuestCopy({
                .run = &RunGuestCopy,
                .context = this,
                .source = copy.dst_offset,
                .destination = reinterpret_cast<u64>(staging.mapped + copy.src_offset),
                .size = copy.size,
                .extra = reinterpret_cast<u64>(staging.buffer),
            });
            copy.src_offset += staging.offset;
            copy.dst_offset -= arena->cpu_addr;
        }
        return staging.buffer;
    }
    if (!BbToggle::Disabled(BbToggle::DeferredUploads)) {
        struct HostCopy {
            VAddr source;
            u8* destination;
            u64 size;
        };
        using Group = boost::container::small_vector<HostCopy, 8>;
        auto group = std::make_shared<Group>();
        u64 group_bytes = 0;
        const auto launch = [&] {
            BbCopy::Async([group, memory = memory] {
                for (const auto& copy : *group) {
                    memory->CopySparseMemory(copy.source, copy.destination, copy.size);
                }
            });
        };
        for (auto& copy : copies) {
            group->push_back({copy.dst_offset, staging.mapped + copy.src_offset, copy.size});
            group_bytes += copy.size;
            copy.src_offset += staging.offset;
            copy.dst_offset -= arena->cpu_addr;
            if (group_bytes >= 1_MB) {
                launch();
                group = std::make_shared<Group>();
                group_bytes = 0;
            }
        }
        if (!group->empty()) {
            launch();
        }
        return staging.buffer;
    }
    const auto copy = [&](std::size_t i) {
        memory->CopySparseMemory(copies[i].dst_offset, staging.mapped + copies[i].src_offset,
                                 copies[i].size);
    };
    if (total_size_bytes >= 2_MB && copies.size() > 1) {
        BbCopy::ParallelFor(copies.size(), copy);
    } else {
        for (std::size_t i = 0; i < copies.size(); ++i) {
            copy(i);
        }
    }
    for (auto& copy : copies) {
        copy.src_offset += staging.offset;
        copy.dst_offset -= arena->cpu_addr;
    }
    return staging.buffer;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(*arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    // bbport: most texel buffers alias no image; remember misses until images change.
    const u64 generation = texture_cache.RegistryGeneration();
    auto& miss = image_miss_cache[((device_addr >> 6) ^ size * 0x9E3779B1u) % image_miss_cache.size()];
    if (miss.address == device_addr && miss.size == size && miss.generation == generation &&
        !BbToggle::Disabled(BbToggle::TextureBindingMemo)) {
        return false;
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        miss = {device_addr, size, generation};
        return false;
    }
    // bbport: the lookups above only read what the texture binding helper leaves alone; the
    // copy below changes image state, so the helper finishes first.
    runtime.BeforeImageAccess();
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

} // namespace VideoCore
