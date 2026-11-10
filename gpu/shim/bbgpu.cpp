// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
#include "bbport_write_log.h"
#include "bbport_arena_runs.h"
// bbport: glue between the C loader and the vendored shadPS4 video core.
#include "bbport_overlay.h"
#include "bbport_settings.h"
#include <sys/resource.h>
#include "bbport_toggles.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <boost/asio/io_context.hpp>
#include "common/polyfill_thread.h"
#include "video_core/renderdoc.h"
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <memory>
#include <mutex>
#include <string>
#include <chrono>
#include <thread>
#include "common/thread.h"
#include "video_core/amdgpu/liverpool.h"
#include <vector>
#include <SDL3/SDL.h>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#endif
#include "../bbgpu.h"
#include "bbport_split.h"
#include "common/elf_info.h"
#include "common/path_util.h"
#include <filesystem>
#include "common/logging/log.h"
#include "common/rdtsc.h"
#include "common/signal_context.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/libs.h"
#include "core/memory.h"
#include "core/signals.h"
#include "sdl_window.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

extern "C" {
// runtime_memory.c
int runtime_memory_is_mapped(uintptr_t address, uint64_t size);
int runtime_memory_write_backing(uintptr_t address, const void* data, uint64_t size);
int runtime_memory_read_backing(uintptr_t address, void* out, uint64_t size);
uint64_t runtime_memory_clamp(uintptr_t address, uint64_t size);
int runtime_memory_region(uintptr_t address, uintptr_t* start, uintptr_t* end, int* mapped);
void runtime_memory_gpu_protect(uintptr_t address, uint64_t size, int read, int write);
// 0.5.1-b: provided by the GPU host's runtime only (host_runtime.c); the game-side library runs
// without it (its protects are in-process mprotects, nothing to batch).
void runtime_memory_gpu_protect_batch(int begin) __attribute__((weak_import));
typedef void (*RuntimeGpuRange)(uintptr_t address, uint64_t size);
// runtime_kernel.c: one clock for guest and GPU timestamps
uint64_t runtime_process_time_us(void);
uint64_t runtime_process_time_counter(void);
uint64_t runtime_tsc_frequency(void);
int32_t* runtime_errno(void);
void runtime_memory_set_gpu_hooks(RuntimeGpuRange map, RuntimeGpuRange unmap, RuntimeGpuRange invalidate);
}

Frontend::WindowSDL* g_window = nullptr;
extern std::unique_ptr<Vulkan::Presenter> presenter; // gnmdriver.cpp

namespace Libraries::GnmDriver {
void RegisterLib(Core::Loader::SymbolsResolver* sym);
bool BbSplitHost(const BbGpuRecord* record);
} // namespace Libraries::GnmDriver
namespace Libraries::AvPlayer { void RegisterLib(Core::Loader::SymbolsResolver* sym); }
namespace Libraries::VideoOut {
void RegisterLib(Core::Loader::SymbolsResolver* sym);
bool BbSplitHost(const BbGpuRecord* record);
u64 BbSplitLabelAddress();
} // namespace Libraries::VideoOut
namespace Libraries::Kernel { void RegisterEventQueue(Core::Loader::SymbolsResolver* sym); }

namespace {
struct Symbol {
    std::string nid, library, module;
    u64 address;
};
std::vector<Symbol> g_symbols;
u32 g_sdk_version;
} // namespace

namespace Core::Loader {
void SymbolsResolver::AddSymbol(const char* nid, const char* library, const char* module, SymbolType,
                                u64 address) {
    g_symbols.push_back({nid, library, module, address});
}
} // namespace Core::Loader

// ElfInfo's fields are private to Core::Emulator; the port fills them here.
namespace Vulkan { extern std::atomic<bool> g_bb_diag_window; } // vk_rasterizer.cpp
namespace Core {
class Emulator {
public:
    static void FillElfInfo(const BbGpuConfig& config) {
        auto& info = Common::ElfInfo::Instance();
        info.initialized = true;
        info.game_serial = config.serial ? config.serial : "UNKNOWN";
        info.title = config.title ? config.title : "";
        info.sdk_ver = config.sdk_version;
        info.psf_attributes.raw = config.psf_attributes;
    }
};

void MemoryManager::SetRasterizer(Vulkan::Rasterizer* rasterizer_) {
    rasterizer = rasterizer_;
    // Existing guest mappings are replayed by the runtime when hooks are installed.
    // bbport: the invalidate hook is the view table's protection change (host_view_apply); the
    // GPU runs follow all three (bbport_arena_runs.h, telemetry).
    runtime_memory_set_gpu_hooks(
        [](uintptr_t address, uint64_t size) {
            BbArenaRuns::OnMap(address, size);
            auto* r = Memory::Instance()->GetRasterizer();
            r->RegisterMemory(address, size);
            r->MapMemory(address, size);
        },
        [](uintptr_t address, uint64_t size) {
            BbArenaRuns::OnUnmap(address, size);
            Memory::Instance()->GetRasterizer()->UnmapMemory(address, size);
        },
        [](uintptr_t address, uint64_t size) {
            BbArenaRuns::OnProtect(address, size);
            Memory::Instance()->GetRasterizer()->InvalidateMemory(address, size);
        });
}
void MemoryManager::InvalidateMemory(VAddr address, u64 size) {
    // bbport: in the split the copies to invalidate are the host's (AvPlayer frames, design P8).
    if (BbSplit::Remote()) {
        const BbGpuRange record{{BBGPU_REC_INVALIDATE, sizeof(BbGpuRange)}, address, size};
        BbSplit::Push(&record);
        return;
    }
    if (rasterizer) rasterizer->InvalidateMemory(address, size);
}
u64 MemoryManager::ClampRangeSize(VAddr virtual_addr, u64 size) {
    return runtime_memory_clamp(virtual_addr, size);
}
static void CopySparseSerial(VAddr source, u8* dest, u64 size) {
    while (size) {
        uintptr_t start = 0, end = 0;
        int mapped = 0;
        if (!runtime_memory_region(source, &start, &end, &mapped)) {
            end = source + size;
        }
        const u64 n = std::min<u64>(size, end - source);
        // fifa-hybrid: read through the unprotected backing alias when the runtime has one, so a
        // renderer thread never takes a read fault on a page the precise readback mode protects
        // (the fault handler would wait for this very thread: deadlock).
        if (!mapped) std::memset(dest, 0, n);
        else if (!runtime_memory_read_backing(source, dest, n)) std::memcpy(dest, reinterpret_cast<const void*>(source), n);
        source += n; dest += n; size -= n;
    }
}
void MemoryManager::CopySparseMemory(VAddr source, u8* dest, u64 size) {
    // bbport: large uploads (streaming) are split across the copy threads.
    constexpr u64 Chunk = 512 * 1024;
    if (size < 4 * Chunk || !BbCopy::Enabled()) {
        return CopySparseSerial(source, dest, size);
    }
    BbCopy::ParallelFor((size + Chunk - 1) / Chunk, [&](std::size_t i) {
        const u64 offset = i * Chunk;
        CopySparseSerial(source + offset, dest + offset, std::min(Chunk, size - offset));
    });
}
bool MemoryManager::TryWriteBacking(void* address, const void* data, u64 size) {
    BbWriteLog::Note(reinterpret_cast<uintptr_t>(address), data, size, BbWriteLog::Backing);
    // diag (0.5.1-g): BB_WB_LOG=1 lists every GPU-to-guest write-back inside a BB_DRAWLOG_TRIGGER
    // window with how many guest bytes it changes; BB_WB_SKIP=<addr>[,<addr>] drops write-backs
    // that start there; BB_WB_DUMP=<addr> saves the new and the previous guest bytes.
    static const bool wb_log = std::getenv("BB_WB_LOG") != nullptr;
    static const std::string wb_skip = [] {
        const char* env = std::getenv("BB_WB_SKIP");
        return env ? "," + std::string(env) + "," : std::string{};
    }();
    static const u64 wb_dump = [] {
        const char* env = std::getenv("BB_WB_DUMP");
        return env ? std::strtoull(env, nullptr, 0) : 0ull;
    }();
    const u64 a = reinterpret_cast<uintptr_t>(address);
    if (!wb_skip.empty()) {
        char key[32];
        std::snprintf(key, sizeof(key), ",%#llx,", (unsigned long long)a);
        if (wb_skip.find(key) != std::string::npos) {
            return true;
        }
    }
    if ((wb_log || wb_dump) && ::Vulkan::g_bb_diag_window.load(std::memory_order_relaxed)) {
        std::vector<u8> old(size);
        const bool have = runtime_memory_read_backing(a, old.data(), size) != 0;
        u64 changed = 0, first = 0, last = 0;
        if (have) {
            const u8* nd = static_cast<const u8*>(data);
            for (u64 i = 0; i < size; ++i) {
                if (old[i] != nd[i]) {
                    if (!changed) first = i;
                    last = i;
                    ++changed;
                }
            }
        }
        if (wb_log) {
            std::fprintf(stderr, "[WB] %#llx+%llu have=%d changed=%llu first=%llu last=%llu\n",
                         (unsigned long long)a, (unsigned long long)size, int(have),
                         (unsigned long long)changed, (unsigned long long)first,
                         (unsigned long long)last);
        }
        static std::atomic<int> dumps{0};
        if (wb_dump && a == wb_dump && have) {
            const int n = dumps.fetch_add(1);
            const char* dir = std::getenv("BB_DUMP_DIR");
            if (n < 12 && dir) {
                for (int k = 0; k < 2; ++k) {
                    char name[512];
                    std::snprintf(name, sizeof(name), "%s/wb_%llx_%d.%s", dir,
                                  (unsigned long long)a, n, k ? "old" : "new");
                    if (FILE* f = std::fopen(name, "wb")) {
                        std::fwrite(k ? old.data() : static_cast<const u8*>(data), 1, size, f);
                        std::fclose(f);
                    }
                }
            }
        }
    }
    return runtime_memory_write_backing(reinterpret_cast<uintptr_t>(address), data, size) != 0;
}
// 0.5.1-b: see memory_tracker.h ForEachUploadRange and host_runtime.c runtime_memory_gpu_protect_batch.
void ProtectBatch(bool begin) {
    if (runtime_memory_gpu_protect_batch) {
        BbStats::Timer timer{BbStats::t_protect}; // the batch's single wait counts as protect time
        runtime_memory_gpu_protect_batch(begin ? 1 : 0);
    }
}
void AddressSpace::Protect(VAddr virtual_addr, u64 size, MemoryPermission perms, u32 flags) {
    BbStats::Timer timer{BbStats::t_protect};
    const u64 pages = (size + 4095) / 4096;
    BbStats::protect_calls.fetch_add(1, std::memory_order_relaxed);
    BbStats::protect_pages.fetch_add(pages, std::memory_order_relaxed);
    if (!True(perms & MemoryPermission::Write)) {
        BbStats::protect_revoke_calls.fetch_add(1, std::memory_order_relaxed);
        BbStats::protect_revoke_pages.fetch_add(pages, std::memory_order_relaxed);
    }
    // fifa-hybrid: `read` bit 1 = an image write-tracks the pages, bit 2 = unreadable only for
    // image readbacks the host drops on a write (bbport_split.cpp decides local handling).
    runtime_memory_gpu_protect(virtual_addr, size,
                               int(True(perms & MemoryPermission::Read)) | int(flags << 1),
                               True(perms & MemoryPermission::Write));
}
boost::icl::interval_set<VAddr> AddressSpace::GetUsableRegions() {
    boost::icl::interval_set<VAddr> set;
    set += boost::icl::interval<VAddr>::right_open(0x400000ULL, 0xfc'0000'0000ULL) /* fifa-hybrid: shadPS4's whole guest range */;
    return set;
}
} // namespace Core

// Kernel services the vendored libraries call directly.
namespace Libraries::Kernel {
u64 PS4_SYSV_ABI sceKernelGetTscFrequency() { return runtime_tsc_frequency(); }
u64 PS4_SYSV_ABI sceKernelReadTsc() { return Common::FencedRDTSC(); }
u64 PS4_SYSV_ABI sceKernelGetProcessTime() { return runtime_process_time_us(); }
u64 PS4_SYSV_ABI sceKernelGetProcessTimeCounter() { return runtime_process_time_counter(); }
u64 PS4_SYSV_ABI sceKernelGetProcessTimeCounterFrequency() { return 1'000'000'000; }
s32 PS4_SYSV_ABI sceKernelIsNeoMode() { return 0; } // base PS4
s32 PS4_SYSV_ABI sceKernelGetCompiledSdkVersion(s32* ver) {
    if (!ver) return ORBIS_KERNEL_ERROR_EINVAL;
    *ver = s32(g_sdk_version);
    return ORBIS_OK;
}
s32 PS4_SYSV_ABI sceKernelUsleep(u32 microseconds) {
    std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
    return ORBIS_OK;
}
} // namespace Libraries::Kernel

namespace {
// SDL video must be driven from one thread: the window lives on its own host thread.
std::thread g_window_thread;
std::mutex g_window_mutex;
std::condition_variable g_window_cv;
bool g_window_ready;
} // namespace

u32 BbDisplayRefreshHz() {
    static const u32 hz = [] {
        const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(Frontend::ChosenDisplay());
        const u32 rate = mode && mode->refresh_rate > 0 ? u32(mode->refresh_rate + 0.5f) : 60;
        std::printf("GPU: vblank follows the display refresh rate, %u Hz\n", rate);
        return std::max<u32>(rate, 60);
    }();
    return hz;
}

// fifa-hybrid (0.5.1-g): the highest refresh rate among the active displays' current modes
// (CoreGraphics: usable before the window exists). 0 when it cannot be read.
u32 BbMaxDisplayRefreshHz() {
    static const u32 hz = [] {
        CGDirectDisplayID displays[16];
        uint32_t count = 0;
        if (CGGetActiveDisplayList(16, displays, &count) != kCGErrorSuccess) {
            return 0u;
        }
        double best = 0;
        for (uint32_t i = 0; i < count; ++i) {
            if (CGDisplayModeRef mode = CGDisplayCopyDisplayMode(displays[i])) {
                best = std::max(best, CGDisplayModeGetRefreshRate(mode));
                CGDisplayModeRelease(mode);
            }
        }
        return u32(best + 0.5);
    }();
    return hz;
}

#ifdef BB_PGO_GENERATE
extern "C" void __gcov_dump(void);
extern "C" void __gcov_reset(void);
// Instrumented build (BB_PGO=generate): the game often ends through _exit (watchdog, guest
// exit), which skips the profile write at exit. Write every 30 s and reset: the files sum the
// intervals.
static void StartProfileWriter() {
    std::thread([] {
        Common::SetCurrentThreadName("bb:pgo");
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            __gcov_dump();
            __gcov_reset();
            std::printf("PGO: profile written\n");
        }
    }).detach();
}
#endif

static void OpenWindow(const BbGpuConfig* config);

// fifa-hybrid: the system's memory pressure. Under it the GPU's own memory is paged and frames
// take tens to hundreds of milliseconds (the "slow GPU" state; a 22 GiB hog on a 32 GB Mac made
// the first tick take over 2 s). The level is logged when it changes and the texture cache
// evicts earlier (BbStats::system_memory_pressure). BB_MEMORY_PRESSURE=0 leaves it unwatched.
static void WatchMemoryPressure() {
    if (const char* e = std::getenv("BB_MEMORY_PRESSURE"); e && e[0] == '0') {
        return;
    }
    static dispatch_source_t source = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
        dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
    if (!source) {
        return;
    }
    dispatch_source_set_event_handler(source, ^{
        const unsigned long data = dispatch_source_get_data(source);
        const int level = (data & DISPATCH_MEMORYPRESSURE_CRITICAL) ? 2
                          : (data & DISPATCH_MEMORYPRESSURE_WARN)   ? 1
                                                                     : 0;
        const int before = BbStats::system_memory_pressure.exchange(level);
        if (before != level) {
            static const char* const names[3] = {"normal", "warning", "critical"};
            std::printf("System memory pressure: %s%s\n", names[level],
                        level ? " (the GPU slows while the system pages; close other apps to keep "
                                "frames short)"
                              : "");
            std::fflush(stdout);
        }
    });
    dispatch_resume(source);
}

// fifa-hybrid: BB_SHADER_SEED_DIR=<dir> names a read-only shader cache shipped with the app
// (<dir>/<serial>/: the SPIR-V cache, the MSL files and the Metal binary archive of a previous
// run). It is copied once into the user cache directory, when that holds no pipeline archive
// yet, so a fresh install compiles only what the seed does not hold. Files already present are
// kept. BB_SHADER_SEED=0 skips the copy. An archive made on another Mac or OS build is rejected
// by Metal at load and replaced (PsoCache); the MSL files still save the SPIR-V to MSL step.
extern std::unique_ptr<AmdGpu::Liverpool> liverpool; // gnmdriver.cpp (global scope)
namespace Libraries::VideoOut {
void DumpStallState(); // video_out.cpp
}
// 0.5.1-b: stall dump. When the game has not flipped for BB_STALL_DUMP_S seconds (default 8,
// 0 = off) after flipping before, prints what the command processor and the renderer wait for
// ([STALL] lines, once per 30 s while it lasts). Diagnostic only: it changes nothing.
static void WatchStalls() {
    const char* env = std::getenv("BB_STALL_DUMP_S");
    const int seconds = env ? std::atoi(env) : 8;
    if (seconds <= 0) {
        return;
    }
    std::thread([seconds] {
        Common::SetCurrentThreadName("bb:StallWatch");
        u64 dumped_at = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            const u64 now = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch()).count());
            const u64 flip = BbStats::last_flip_ns.load(std::memory_order_relaxed);
            if (flip == 0 || now - flip < u64(seconds) * 1000000000ull) {
                dumped_at = 0;
                continue;
            }
            if (dumped_at && now - dumped_at < 30ull * 1000000000ull) {
                continue;
            }
            dumped_at = now;
            const u64 submit = BbStats::last_submit_ns.load(std::memory_order_relaxed);
            std::fprintf(stderr, "[STALL] no flip for %.1f s (last graphics submission %.1f s ago)\n",
                         (now - flip) / 1e9, submit ? (now - submit) / 1e9 : -1.0);
            if (auto* lp = liverpool.get()) {
                lp->DumpStall();
            }
            Libraries::VideoOut::DumpStallState();
        }
    }).detach();
}

static void SeedShaderCache() {
    const char* seed = std::getenv("BB_SHADER_SEED_DIR");
    if (!seed || !seed[0]) {
        return;
    }
    if (const char* e = std::getenv("BB_SHADER_SEED"); e && e[0] == '0') {
        return;
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string serial{Common::ElfInfo::Instance().GameSerial()};
    const fs::path src = fs::path{seed} / serial;
    if (!fs::is_directory(src, ec)) {
        std::printf("Shader cache: no seed at %s\n", src.c_str());
        return;
    }
    const fs::path dst = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / serial;
    if (fs::exists(dst / "metal" / "pipelines.binaryarchive", ec)) {
        return; // a cache of its own: nothing to seed
    }
    const auto t0 = std::chrono::steady_clock::now();
    fs::create_directories(dst, ec);
    u64 files = 0, bytes = 0;
    for (const auto& entry : fs::recursive_directory_iterator(src, ec)) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const fs::path rel = fs::relative(entry.path(), src, ec);
        const fs::path to = dst / rel;
        fs::create_directories(to.parent_path(), ec);
        if (fs::exists(to, ec)) {
            continue;
        }
        if (fs::copy_file(entry.path(), to, fs::copy_options::skip_existing, ec)) {
            ++files;
            bytes += entry.file_size(ec);
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("Shader cache: seeded %llu files (%.1f MB) into %s from %s in %.0f ms\n",
                (unsigned long long)files, bytes / 1e6, dst.c_str(), src.c_str(), ms);
}


extern "C" int bbgpu_init(const BbGpuConfig* config) {
    BbSettings::Load();
#ifdef BB_PGO_GENERATE
    StartProfileWriter();
#endif
    g_sdk_version = config->sdk_version;
    if (config->user_dir) setenv("BB_GPU_USER_DIR", config->user_dir, 0);
    Core::Emulator::FillElfInfo(*config);
    SeedShaderCache();
    WatchMemoryPressure();
    WatchStalls();
    OpenWindow(config);
    Core::Loader::SymbolsResolver resolver;
    // GnmDriver creates the presenter that the VideoOut present thread uses.
    Libraries::GnmDriver::RegisterLib(&resolver);
    Libraries::VideoOut::RegisterLib(&resolver);
    return 0;
}

extern "C" int bbgpu_host_start(const BbGpuConfig* config, void* shared) {
    BbSplit::InitHost(static_cast<BbGpuShared*>(shared));
    const int result = bbgpu_init(config);
    if (result == 0) {
        auto* s = static_cast<BbGpuShared*>(shared);
        s->label_address = Libraries::VideoOut::BbSplitLabelAddress();
        s->hdr_supported = presenter && presenter->IsHDRSupported() ? 1 : 0;
        BbSplit::ForwardInterrupts();
        // Keyboard for the game's pad emulation (input proper moves here in M5).
        std::thread([s] {
            Common::SetCurrentThreadName("bb:KeyBridge");
            for (;;) {
                int count = 0;
                const bool* keys = SDL_GetKeyboardState(&count);
                if (keys) {
                    std::memcpy(s->keys, keys, std::min<size_t>(size_t(count), sizeof s->keys));
                    __atomic_store_n(&s->keys_valid, 1u, __ATOMIC_RELEASE);
                }
                __atomic_store_n(&s->overlay_captures, BbOverlay::CapturesInput() ? 1u : 0u,
                                 __ATOMIC_RELAXED);
                // Text entry (IME dialog): publish the text and its state while it is open.
                if (__atomic_load_n(&s->text_active, __ATOMIC_ACQUIRE) && g_window) {
                    std::string text;
                    const int state = g_window->PollTextInput(text);
                    const size_t n = std::min(text.size(), sizeof s->text - 1);
                    std::memcpy(s->text, text.data(), n);
                    s->text[n] = 0;
                    __atomic_store_n(&s->text_state, static_cast<u32>(state), __ATOMIC_RELEASE);
                    __atomic_fetch_add(&s->text_seq, 1u, __ATOMIC_RELEASE);
                    if (state != 0) {
                        __atomic_store_n(&s->text_active, 0u, __ATOMIC_RELEASE);
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(4));
            }
        }).detach();
    }
    return result;
}

extern "C" void bbgpu_split_post_restart(void) {
    const BbGpuRecord record{BBGPU_EV_RESTART, sizeof(BbGpuRecord)};
    BbSplit::Post(&record);
}

extern "C" int bbgpu_host_fault(uint64_t address, int write) {
    auto* rasterizer = Core::Memory::Instance()->GetRasterizer();
    if (!rasterizer) {
        return 0;
    }
    if (write) {
        BbStats::Timer timer{BbStats::t_write_faults};
        return rasterizer->OnWriteFault(address, false) ? 1 : 0;
    }
    BbStats::read_faults.fetch_add(1, std::memory_order_relaxed);
    BbStats::Timer timer{BbStats::t_read_faults};
    static const bool stats = std::getenv("BB_FAULT_STATS") != nullptr;
    if (!stats) {
        return rasterizer->ReadMemory(address, 8, false) ? 1 : 0;
    }
    // fifa-hybrid diagnostics: [RFAULT] read-fault count and host time per 2 s.
    static std::atomic<u64> n{0}, us{0};
    static std::atomic<s64> window_start{0};
    const auto t0 = std::chrono::steady_clock::now();
    const int handled = rasterizer->ReadMemory(address, 8, false) ? 1 : 0;
    n.fetch_add(1, std::memory_order_relaxed);
    us.fetch_add(u64(std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - t0).count()),
                 std::memory_order_relaxed);
    const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
    s64 ws = window_start.load(std::memory_order_relaxed);
    if (ws == 0) {
        window_start.compare_exchange_strong(ws, now);
    } else if (now - ws >= 2000000000 && window_start.compare_exchange_strong(ws, now)) {
        const u64 c = n.exchange(0);
        std::printf("[RFAULT] n=%llu host=%.1fms\n", (unsigned long long)c, us.exchange(0) / 1e3);
        std::fflush(stdout);
    }
    return handled;
}

extern "C" int bbgpu_host_dispatch(const void* record) {
    const auto* r = static_cast<const BbGpuRecord*>(record);
    BbArenaRuns::Refresh(); // the view events before this record (ring order)
    if (r->type == BBGPU_REC_TEXT_BEGIN) {
        auto* t = static_cast<const BbGpuTextBegin*>(record);
        BbGpuShared* s = BbSplit::Shared();
        int opened = 0;
        if (g_window) {
            const std::string initial(t->initial, strnlen(t->initial, sizeof t->initial));
            const std::string prompt(t->prompt, strnlen(t->prompt, sizeof t->prompt));
            g_window->BeginTextInput(initial, prompt.empty() ? "Text" : prompt, (t->flags & 1u) != 0);
            __atomic_store_n(&s->text_state, 0u, __ATOMIC_RELAXED);
            __atomic_store_n(&s->text_active, 1u, __ATOMIC_RELEASE);
            opened = 1;
        }
        bbgpu_reply_send(s, t->slot, opened);
        return 1;
    }
    if (r->type == BBGPU_REC_INVALIDATE) {
        const auto* range = static_cast<const BbGpuRange*>(record);
        Core::Memory::Instance()->InvalidateMemory(range->address, range->size);
        return 1;
    }
    if (r->type == BBGPU_REC_CPU_WRITE) {
        const auto* w = static_cast<const BbGpuCpuWrite*>(record);
        if (auto* rasterizer = Core::Memory::Instance()->GetRasterizer()) {
            rasterizer->OnCpuWrite(w->address, w->size, w->fault);
        }
        return 1;
    }
    return Libraries::GnmDriver::BbSplitHost(r) || Libraries::VideoOut::BbSplitHost(r) ? 1 : 0;
}

extern "C" void bbgpu_split_attach(void* shared, int host_pid) {
    auto* s = static_cast<BbGpuShared*>(shared);
    s->host_pid = host_pid;
    s->game_pid = getpid();
    BbSplit::Attach(s);
}

extern "C" const bool* bbgpu_split_keyboard(void) {
    if (!BbSplit::Remote() || !__atomic_load_n(&BbSplit::Shared()->keys_valid, __ATOMIC_ACQUIRE)) {
        return nullptr;
    }
    static_assert(sizeof(bool) == 1);
    return reinterpret_cast<const bool*>(BbSplit::Shared()->keys);
}

extern "C" uint32_t bbgpu_bench_frame_draws(void) {
    return BbSplit::Remote() ? __atomic_load_n(&BbSplit::Shared()->bench_frame_draws, __ATOMIC_RELAXED)
                             : 0;
}

extern "C" uint64_t bbgpu_bench_host_flips(void) {
    return BbSplit::Remote() ? __atomic_load_n(&BbSplit::Shared()->host_flips, __ATOMIC_RELAXED) : 0;
}

extern "C" void bbgpu_bench_set_phase(uint32_t phase, const char* error) {
    if (!BbSplit::Remote()) {
        return;
    }
    BbGpuShared* s = BbSplit::Shared();
    if (error) {
        std::snprintf(s->bench_error, sizeof(s->bench_error), "%s", error);
    }
    __atomic_store_n(&s->bench_phase, phase, __ATOMIC_RELEASE);
}

extern "C" void bbgpu_split_push(const void* record) {
    BbSplit::Push(record);
}

extern "C" int bbgpu_init_remote(const BbGpuConfig* config, void* shared_block) {
    auto* s = static_cast<BbGpuShared*>(shared_block);
    BbSettings::Load();
    g_sdk_version = config->sdk_version;
    Core::Emulator::FillElfInfo(*config);
    BbSplit::InitRemote(s);
    std::snprintf(s->title, sizeof s->title, "%s", config->title ? config->title : "Bloodborne");
    std::snprintf(s->serial, sizeof s->serial, "%s", config->serial ? config->serial : "UNKNOWN");
    std::snprintf(s->user_dir, sizeof s->user_dir, "%s", config->user_dir ? config->user_dir : "user");
    s->sdk_version = config->sdk_version;
    s->psf_attributes = config->psf_attributes;
    s->width = config->width;
    s->height = config->height;
    s->tsc_hz = runtime_tsc_frequency();
    s->process_start_ns = 0; // the host's process clock: same CLOCK_MONOTONIC origin as ours
    {
        timespec now{};
        clock_gettime(CLOCK_MONOTONIC, &now);
        s->process_start_ns = u64(now.tv_sec) * 1'000'000'000ull + u64(now.tv_nsec) -
                              runtime_process_time_counter();
    }
    BbGpuStart start{{BBGPU_REC_START, sizeof(BbGpuStart)}, 0, 0};
    const s64 result = BbSplit::Request(&start, &start.slot);
    if (result != 0) {
        std::printf("GPU split: the GPU host could not start the renderer (%lld)\n",
                    static_cast<long long>(result));
        return -1;
    }
    Core::Loader::SymbolsResolver resolver;
    Libraries::GnmDriver::RegisterLib(&resolver);
    Libraries::VideoOut::RegisterLib(&resolver);
    BbSplit::StartEventDispatcher();
    std::printf("GPU split: rendering in the GPU host (labels at %#llx, HDR %s)\n",
                static_cast<unsigned long long>(s->label_address), s->hdr_supported ? "yes" : "no");
    return 0;
}

#ifdef __APPLE__
static s32 g_window_width, g_window_height;
#endif

// bbgpu-host (and bb-probe) end through _exit, which runs no destructors: the pipeline archive
// is written here first. Every caller waits for the same flush (the first starts it on a thread
// of its own), at most `timeout_ms`; a write cut short by the _exit that follows leaves the old
// file intact.
extern "C" int bbgpu_flush_caches(unsigned timeout_ms) {
    struct State {
        std::mutex mutex;
        std::condition_variable cv;
        bool done{};
        bool ok{true};
    };
    static State* const state = new State; // never destroyed: the flush thread may outlive exit
    static std::once_flag once;
    const auto start = std::chrono::steady_clock::now();
    std::call_once(once, [] {
        std::thread([] {
            Common::SetCurrentThreadName("bb:ExitFlush");
            const bool ok = presenter ? presenter->FlushPipelineArchive() : true;
            {
                std::scoped_lock lock{state->mutex};
                state->done = true;
                state->ok = ok;
            }
            state->cv.notify_all();
        }).detach();
    });
    std::unique_lock lock{state->mutex};
    const bool done = state->cv.wait_for(lock, std::chrono::milliseconds{timeout_ms},
                                         [] { return state->done; });
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (!done) {
        std::printf("Exit: the pipeline archive write did not finish within %u ms; exiting "
                    "without it\n",
                    timeout_ms);
    } else {
        std::printf("Exit: pipeline archive flushed in %.0f ms%s\n", ms,
                    state->ok ? "" : " (the write failed)");
    }
    std::fflush(stdout);
    return done ? (state->ok ? 0 : -1) : 1;
}

static void OpenWindow(const BbGpuConfig* config) {
    const std::string title = config->title ? config->title : "Bloodborne";
    const s32 width = config->width, height = config->height;
    if (config->metal_layer) {
        // fifa-hybrid: the host application (shadPS4) owns the window and its Metal layer.
        auto* window = new Frontend::WindowSDL(width, height, config->metal_layer,
                                               config->layer_scale > 0 ? config->layer_scale : 1.0f);
        std::scoped_lock lock{g_window_mutex};
        g_window = window;
        g_window_ready = true;
        return;
    }
#ifdef __APPLE__
    g_window_width = width;
    g_window_height = height;
#endif
#ifndef __APPLE__
    auto window_loop = [title, width, height] {
        Common::SetCurrentThreadName("bb:window");
        auto* window = new Frontend::WindowSDL(width, height, title.c_str());
        {
            std::scoped_lock lock{g_window_mutex};
            g_window = window;
            g_window_ready = true;
        }
        g_window_cv.notify_all();
        while (window->PollEvents()) {
            SDL_Delay(2);
        }
        LOG_INFO(Frontend, "Window closed by user");
        bbgpu_flush_caches(BBGPU_EXIT_FLUSH_MS);
        std::_Exit(0);
    };
#endif
#ifdef __APPLE__
    // Cocoa only works on the main thread: bb-probe runs the game on another thread and runs
    // the main run loop on the main thread, so the window goes to the main queue. Its events
    // are polled from a 2 ms run loop timer there, not a loop that never returns: AppKit finishes some
    // work (the full-screen transition) in blocks of its own on the main queue, which a
    // never-ending block starves (the window stuck off screen at its old size).
    dispatch_async_f(dispatch_get_main_queue(), new std::string(title), [](void* ctx) {
        const std::unique_ptr<std::string> title{static_cast<std::string*>(ctx)};
        auto* window = new Frontend::WindowSDL(g_window_width, g_window_height, title->c_str());
        {
            std::scoped_lock lock{g_window_mutex};
            g_window = window;
            g_window_ready = true;
        }
        g_window_cv.notify_all();
        // A run loop timer (unlike a dispatch timer) is a source of the main run loop: it keeps
        // CFRunLoopRun() running, and in the common modes it also fires during live resizes.
        CFRunLoopTimerContext context{0, window, nullptr, nullptr, nullptr};
        CFRunLoopTimerRef timer = CFRunLoopTimerCreate(
            kCFAllocatorDefault, CFAbsoluteTimeGetCurrent(), 0.002, 0, 0,
            [](CFRunLoopTimerRef, void* ctx) {
                if (!static_cast<Frontend::WindowSDL*>(ctx)->PollEvents()) {
                    LOG_INFO(Frontend, "Window closed by user");
                    bbgpu_flush_caches(BBGPU_EXIT_FLUSH_MS);
                    std::_Exit(0);
                }
            },
            &context);
        CFRunLoopAddTimer(CFRunLoopGetMain(), timer, kCFRunLoopCommonModes);
    });
#else
    g_window_thread = std::thread(std::move(window_loop));
    g_window_thread.detach();
#endif
    {
        std::unique_lock lock{g_window_mutex};
        g_window_cv.wait(lock, [] { return g_window_ready; });
    }
}

// fifa-hybrid: the host runtime (shadPS4) drives the renderer's memory tracking directly, with
// the same four calls it makes on its own rasterizer, instead of the map/unmap/invalidate hooks.
// The run table (BbArenaRuns, the "runs" arena layout) is fed and refreshed here too: there is no
// ring thread in this mode, so the view event and the refresh happen on the mapping thread.
namespace {
std::mutex g_fifa_memory_mutex;
}
extern "C" void bbgpu_memory_register(uint64_t address, uint64_t size) {
    if (auto* r = Core::Memory::Instance()->GetRasterizer()) r->RegisterMemory(address, size);
}
extern "C" void bbgpu_memory_map(uint64_t address, uint64_t size) {
    std::scoped_lock lock{g_fifa_memory_mutex};
    BbArenaRuns::OnMap(address, size);
    if (auto* r = Core::Memory::Instance()->GetRasterizer()) r->MapMemory(address, size);
    BbArenaRuns::Refresh();
}
extern "C" void bbgpu_memory_unmap(uint64_t address, uint64_t size) {
    std::scoped_lock lock{g_fifa_memory_mutex};
    BbArenaRuns::OnUnmap(address, size);
    if (auto* r = Core::Memory::Instance()->GetRasterizer()) r->UnmapMemory(address, size);
    BbArenaRuns::Refresh();
}
extern "C" void bbgpu_memory_invalidate(uint64_t address, uint64_t size) {
    if (auto* r = Core::Memory::Instance()->GetRasterizer()) r->InvalidateMemory(address, size);
}
extern "C" void bbgpu_memory_protect(uint64_t address, uint64_t size) {
    std::scoped_lock lock{g_fifa_memory_mutex};
    BbArenaRuns::OnProtect(address, size);
    BbArenaRuns::Refresh();
}

namespace Libraries::Kernel { void StartKernelService(); }
extern "C" void bbgpu_register_kernel(void) {
    Libraries::Kernel::StartKernelService();
    Core::Loader::SymbolsResolver resolver;
    Libraries::Kernel::RegisterEventQueue(&resolver);
    Libraries::AvPlayer::RegisterLib(&resolver);
}

extern "C" uintptr_t bbgpu_resolve(const char* scoped_nid) {
    const char* hash = std::strchr(scoped_nid, '#');
    const size_t length = hash ? size_t(hash - scoped_nid) : std::strlen(scoped_nid);
    for (const auto& symbol : g_symbols) {
        if (symbol.nid.size() == length && !std::memcmp(symbol.nid.data(), scoped_nid, length)) {
            return uintptr_t(symbol.address);
        }
    }
    return 0;
}

extern "C" int bbgpu_handle_fault(void* ucontext, void* address) {
    if (BbSplit::Remote()) {
        // M4: GPU page tracking lives in the host; it handles the fault and unprotects.
        return BbSplit::RelayFault(reinterpret_cast<u64>(address), Common::IsWriteError(ucontext))
                   ? 1
                   : 0;
    }
    return Core::Signals::Instance()->DispatchAccessViolation(ucontext, address) ? 1 : 0;
}

extern "C" unsigned bbgpu_symbol_count(void) {
    return unsigned(g_symbols.size());
}

extern "C" uint64_t bbgpu_get_flip_count(void) {
    return BbStats::flips.load(std::memory_order_relaxed);
}

extern "C" uint64_t bbgpu_get_submit_count(void) {
    if (BbSplit::Remote()) {
        return __atomic_load_n(&BbSplit::Shared()->host_submits, __ATOMIC_RELAXED);
    }
    return BbStats::submissions.load(std::memory_order_relaxed);
}

namespace Libraries::VideoOut {
void GetFrametimeStats(double* avg_fps, double* p95_ms, double* p99_ms);
}

extern "C" void bbgpu_get_frametime_percentiles(double* avg_fps, double* p95_ms, double* p99_ms) {
    Libraries::VideoOut::GetFrametimeStats(avg_fps, p95_ms, p99_ms);
}

extern "C" void bbgpu_dump_host_threads_hang(void* file_handle) {
    FILE* f = static_cast<FILE*>(file_handle);
    if (!f) return;
    std::fprintf(f, "=== HOST GPU / PRESENTER THREADS ===\n");
    std::fprintf(f, "GPU Stats:\n");
    std::fprintf(f, "  Submissions: %llu, Flips: %llu, Draws: %llu, Dispatches: %llu\n",
                 (unsigned long long)BbStats::submissions.load(std::memory_order_relaxed),
                 (unsigned long long)BbStats::flips.load(std::memory_order_relaxed),
                 (unsigned long long)BbStats::draws.load(std::memory_order_relaxed),
                 (unsigned long long)BbStats::dispatches.load(std::memory_order_relaxed));
    std::fprintf(f, "  GPU Idle Time: %.2f ms\n",
                 (double)BbStats::gpu_idle_ns.load(std::memory_order_relaxed) / 1000000.0);
    std::fprintf(f, "  Scheduler Sync Wait: %.2f ms\n",
                 (double)BbStats::sync_recording_ns.load(std::memory_order_relaxed) / 1000000.0);
    std::fprintf(f, "  Host Copies Wait: %.2f ms\n",
                 (double)BbStats::host_copies_wait_ns.load(std::memory_order_relaxed) / 1000000.0);
    std::fprintf(f, "  GPU Tick Wait: %.2f ms\n",
                 (double)BbStats::tick_wait_ns.load(std::memory_order_relaxed) / 1000000.0);
    std::fprintf(f, "  Host Wait (fences/queue): %.2f ms\n",
                 (double)BbStats::t_host_wait.load(std::memory_order_relaxed) / 1000000.0);
    std::fprintf(f, "  Texture Creation Time: %.2f ms, Staging Time: %.2f ms\n",
                 (double)BbStats::t_image_create.load(std::memory_order_relaxed) / 1000000.0,
                 (double)BbStats::t_staging.load(std::memory_order_relaxed) / 1000000.0);
    std::fprintf(f, "  Page Protection Revoke Calls: %llu (%llu pages)\n",
                 (unsigned long long)BbStats::protect_revoke_calls.load(std::memory_order_relaxed),
                 (unsigned long long)BbStats::protect_revoke_pages.load(std::memory_order_relaxed));
    std::fprintf(f, "====================================\n\n");
    std::fflush(f);
}

// Stuck-GPU diagnostics: the Metal device's tick counters (the old breadcrumb buffer markers
// need VK_AMD_buffer_marker; a Metal stall shows as a gap between the tick being waited on and
// the last tick known done).
extern "C" void bbgpu_dump_breadcrumbs(void* file_handle) {
    if (!file_handle) return;
    FILE* f = static_cast<FILE*>(file_handle);
    std::fprintf(f, "\n=== GPU STATE ===\n");
    if (const auto* p = presenter.get()) {
        const Metal::Device& device = p->GetDevice();
        std::fprintf(f,
                     "Metal device %s: next tick %llu, last tick done %llu, command buffer "
                     "errors %llu\n",
                     device.Name().c_str(), (unsigned long long)device.CurrentTick(),
                     (unsigned long long)device.KnownGpuTick(),
                     (unsigned long long)device.ErrorCount());
    } else {
        std::fprintf(f, "no presenter (the renderer is not running in this process)\n");
    }
    std::fprintf(f, "=================\n\n");
    std::fflush(f);
}

// Kernel service thread: runs boost::asio timers for equeue timer events.
namespace Libraries::Kernel {
boost::asio::io_context io_context;
static std::mutex m_asio_req;
static std::condition_variable_any cv_asio_req;
static std::atomic<u32> asio_requests;
static std::jthread service_thread;

void KernelSignalRequest() {
    std::unique_lock lock{m_asio_req};
    ++asio_requests;
    cv_asio_req.notify_one();
}

static void KernelServiceThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("bb:kernel_service");
    while (!stoken.stop_requested()) {
        {
            std::unique_lock lock{m_asio_req};
            cv_asio_req.wait(lock, stoken, [] { return asio_requests != 0; });
        }
        if (stoken.stop_requested()) break;
        io_context.run();
        io_context.restart();
        asio_requests = 0;
    }
}

void StartKernelService() {
    service_thread = std::jthread{KernelServiceThread};
}

int* PS4_SYSV_ABI __Error() {
    return runtime_errno();
}
} // namespace Libraries::Kernel

// RenderDoc capture hooks: not used by the port.
namespace VideoCore {
void LoadRenderDoc() {}
void StartCapture() {}
void EndCapture() {}
void TriggerCapture() {}
void SetOutputDir(const std::filesystem::path&, const std::string&) {}
bool IsRenderDocLoaded() { return false; }
void RequestScreenshot(ScreenshotRequest) {}
u32 ConsumeGameOnlyScreenshotRequests() { return 0; }
u32 ConsumeWithOverlaysScreenshotRequests() { return 0; }
ScreenshotRequests ConsumeScreenshotRequests() { return {}; }
} // namespace VideoCore

extern "C" int bbgpu_overlay_captures_input(void) {
    if (BbSplit::Remote()) {
        return __atomic_load_n(&BbSplit::Shared()->overlay_captures, __ATOMIC_RELAXED) ? 1 : 0;
    }
    return BbOverlay::CapturesInput() ? 1 : 0;
}

extern "C" int bbgpu_mouse_look_enabled(void) {
    return BbSettings::Get().mouse_look ? 1 : 0;
}

extern "C" int bbgpu_text_input_begin(const char* initial, const char* prompt, int password) {
    if (BbSplit::Remote()) {
        BbGpuTextBegin record{};
        record.r = {BBGPU_REC_TEXT_BEGIN, sizeof(BbGpuTextBegin)};
        record.flags = password ? 1u : 0u;
        std::snprintf(record.initial, sizeof record.initial, "%s", initial ? initial : "");
        std::snprintf(record.prompt, sizeof record.prompt, "%s", prompt ? prompt : "Text");
        return BbSplit::Request(&record, &record.slot) != 0 ? 1 : 0;
    }
    if (!g_window) return 0;
    g_window->BeginTextInput(initial ? initial : "", prompt ? prompt : "Text", password != 0);
    return 1;
}

extern "C" int bbgpu_text_input_poll(char* out, uint64_t size) {
    if (BbSplit::Remote()) {
        BbGpuShared* s = BbSplit::Shared();
        const u32 state = __atomic_load_n(&s->text_state, __ATOMIC_ACQUIRE);
        if (size) {
            const size_t n = std::min<size_t>(strnlen(s->text, sizeof s->text), size - 1);
            std::memcpy(out, s->text, n);
            out[n] = 0;
        }
        return static_cast<int>(state);
    }
    if (!g_window) return 2;
    std::string text;
    const int state = g_window->PollTextInput(text);
    if (size) {
        const size_t n = std::min<size_t>(text.size(), size - 1);
        std::memcpy(out, text.data(), n);
        out[n] = 0;
    }
    return state;
}
