// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <unordered_map>

#include "bbport_arena_runs.h"
#include "bbport_copy.h"

#include <deque>
#include <mutex>
#include <thread>
#include <span>
#include <vector>
#include <boost/container/small_vector.hpp>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_copy.h"
#include "video_core/buffer_cache/chunk_table.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Metal {
class Device;
}

namespace Vulkan {
class Runtime;
class Scheduler;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

// fifa-hybrid diagnostics: a guest range armed by the shader probe (BB_PROBE_WATCH=<buffer index>);
// the buffer cache logs the uploads, write-backs, GPU-written binds, DMA copies and CPU faults
// that touch it ([WATCH] lines).
inline std::atomic<u64> g_watch_lo{0}, g_watch_hi{0};
inline bool WatchOverlaps(u64 addr, u64 size) {
    const u64 lo = g_watch_lo.load(std::memory_order_relaxed), hi = g_watch_hi.load(std::memory_order_relaxed);
    return hi != 0 && addr < hi && addr + size > lo;
}


// fifa-hybrid diagnostics: guest address watched by BB_BUFWATCH (env) or BB_RESOLVE_WATCH.
extern std::atomic<u64> g_bb_bufwatch;


class TextureCache;
class MemoryTracker;
class PageManager;

class BufferCache {
public:
    /// fifa-hybrid diagnostics: CPU writes that first needed a GPU readback (BB_FAULT_STATS).
    std::atomic<u64> write_flushes{0};
    std::atomic<u64> cpu_write_wins{0}; ///< written pages whose pending GPU write-back was dropped
    std::mutex hot_read_mutex;
    std::unordered_map<VAddr, u32> read_fault_counts; ///< per 4 KiB page
    std::vector<VAddr> hot_read_pages;
    /// fifa-hybrid: the game wrote [addr, addr+size) after lifting the protection itself.
    void OnCpuWrite(VAddr addr, u64 size, VAddr fault);
    /// fifa-hybrid: a CPU read fault on a GPU-modified page; after BB_HOT_READ faults the page
    /// becomes a hot read page (readable, written back after every submission).
    void NoteReadFault(VAddr addr);
    /// Downloads the GPU-modified words of the hot read pages (Rasterizer::OnSubmit).
    void FlushHotReadPages();

private:
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    /// Granularity of the page table (what MoltenVK's sparse alignment gave: 16 KiB, shift 14).
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;
    static constexpr u64 MacArenaBase = 0x7000000000; ///< runtime_memory.c USER_MIN (macOS)
    u64 mac_arena_end = 0; ///< end of the arena (smaller where Metal buffers are limited)
    /// Highest arena offset requested so far, in 256 MiB steps (logged when it grows: how big
    /// the arena needs to be; it is committed in full, ~1 GB of memory per GiB).
    std::atomic<u64> arena_high_steps{0};

public:
    /// Read-only bindings up to this size are copied into a stream buffer (bbport: public for
    /// the draw pipeline's constant ring).
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

    explicit BufferCache(Metal::Device& device, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    /// bbport: null unless BB_DIRECT_MEMORY_ACCESS=1 (no shader reads it otherwise).
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// bbport: the buffers holding guest memory (chunks), for the ordering of address-reached
    /// reads.
    [[nodiscard]] std::span<const ChunkTable::Chunk> GetChunks() const noexcept {
        return chunks.Chunks();
    }

    /// Retrieves the fault buffer.
    /// bbport: null unless BB_DIRECT_MEMORY_ACCESS=1, like the page table.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager ? fault_manager->GetFaultBuffer() : nullptr;
    }

    /// Retrieves the stream buffer.
    StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns the page table granularity (log2 of the block size).
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// bbport: a guest write fault at `device_addr` was handled; unprotect its neighbourhood.
    void ExtendWriteFault(VAddr device_addr);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool assume_locks = false);

    /// Finds a buffer for the specified region.
    /// bbport: `note_use` = false leaves marking the range used in this submission (upload
    /// hoisting) to the caller, which may first want to hoist a copy of it. `caller`: who asks,
    /// for the GPU runs telemetry (bbport_arena_runs.h).
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(
        VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer = false,
        bool note_use = true, BbArenaRuns::Caller caller = BbArenaRuns::Caller::Bind);

    /// Metal's indexed draw path truncates offsets above 4 GiB on M2 (observed).
    /// Rebase those ranges into fence-managed staging; preserve GPU-written indices.
    /// bbport: only for an arena without CPU contents; otherwise the encoder draws through a
    /// window of the arena (Metal::LowOffset) and the arena range is returned.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainIndexBuffer(VAddr device_addr, u32 size);

    /// bbport: indirect draw and dispatch arguments for Metal's indirectBufferOffset, 32-bit like
    /// the index offset (the same copies as ObtainIndexBuffer past 4 GiB). Only for arguments
    /// Metal reads at that offset: records behind a GPU count are read by a kernel at any offset.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainDrawArgumentBuffer(VAddr device_addr,
                                                                         u32 size);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Processes the fault buffer.
    void ProcessFaultBuffer();

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

private:
    /// bbport: the buffer holding [device_addr, device_addr + size) (a chunk); stops the game when
    /// none does.
    const Buffer* GetChunk(VAddr device_addr, u64 size);

    /// bbport: the chunk holding [device_addr, device_addr + size), or null (GetChunk stops the
    /// game instead). For paths that can decline: read-backs, size-0 requests.
    const Buffer* FindChunk(VAddr device_addr, u64 size) const;

    /// bbport: creates a buffer holding guest memory [start, start + bytes) and `guard` bytes past
    /// it, sets it up for the GPU (resident, upload hoisting, ordering, declared on encoders) and
    /// publishes it as a chunk of [start, start + bytes).
    const Buffer* SetUpChunk(VAddr start, u64 bytes, const std::string& name, u64 guard = 0);

    /// bbport: span and runs layouts: creates the chunks the game's GPU runs (BbArenaRuns) still
    /// lack. On the GPU command thread (a queued command: the draw pipe is drained), or at start.
    void MaterializeRuns();
    static void OnRunsChanged(void* self);

    /// bbport: span and runs layouts: a request no chunk holds (none seen in play; counted and
    /// logged). Read: a copy of the guest bytes. Written: a scratch range (the PS4's GPU cannot
    /// write memory it has not mapped for it either). Starting in a chunk and ending in its guard:
    /// the chunk (the guard takes the rest).
    std::pair<const Buffer*, u64> ObtainMiss(VAddr device_addr, u32 size, bool is_written,
                                             bool is_texel_buffer, bool note_use);

    /// bbport: guest bytes copied into staging (texture uploads, misses).
    std::pair<const Buffer*, u64> StagingCopy(VAddr device_addr, u32 size);

    /// bbport: span and runs layouts: every 10 s, Metal errors or bytes written into a guard make
    /// the next launch use the previous layout (the fallback file in the user folder).
    void MonitorLayout(std::stop_token stop);
    void WriteLayoutFallback(const std::string& reason);

    /// bbport: a range for Metal's 32-bit index and indirect offsets: the arena's when it ends
    /// within the limit, else a copy (`caller`: Index or Indirect, for the counters).
    std::pair<const Buffer*, u64> LowOffsetRange(VAddr device_addr, u32 size,
                                                 BbArenaRuns::Caller caller);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size);

public:
    /// fifa-hybrid: writes the GPU-modified bytes of the pages holding [addr, addr+size) back
    /// to guest memory asynchronously (no GPU drain) and unmarks them, so a guest polling a
    /// tiny shader-written counter reads the previous value now and the new one a tick later.
    void DownloadSmallAsync(VAddr addr, u64 size);

private:

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    /// bbport: batched small copies on the copy threads (BbCopy::QueueCopy).
    static void RunGuestCopy(const BbCopy::Item& item);
    void SmallGuestCopy(const BbCopy::Item& item);

    const Buffer* UploadCopies(const Buffer* arena, std::span<BufferCopy> copies,
                               size_t total_size_bytes);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    Metal::Device& device;
    Vulkan::Scheduler& scheduler;
    // bbport: texel buffer ranges known not to alias an image, per image registry generation.
    struct ImageMiss {
        VAddr address = 0;
        u32 size = 0;
        u64 generation = ~0ULL;
    };
    std::array<ImageMiss, 256> image_miss_cache{};
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    RangeSet gpu_modified_ranges;
    /// fifa-hybrid: the part of gpu_modified_ranges that is not a steady rewrite of a tiny
    /// buffer (see NoteGpuWrite): no asynchronous write-back, the CPU's read waits for the GPU.
    RangeSet oneshot_ranges;
    struct GpuWriteHistory {
        u64 last_ns = 0;
        u32 count = 0;
    };
    std::unordered_map<VAddr, GpuWriteHistory> gpu_write_history;
    void NoteGpuWrite(VAddr addr, u64 size);
public:
    /// 0.5.1-g: the GPU side wrote guest memory itself (see TextureCache::NoteHostWrite): the
    /// arena copy of those pages is stale, so they are marked CPU-modified as a fault would.
    void NoteHostWrite(VAddr addr, u64 size);
    /// Debug: bytes of [addr, addr + size) on clean pages (neither CPU- nor GPU-modified) where
    /// the arena differs from guest memory; ~0 without a chunk. `clean` gets the bytes compared.
    u64 DebugArenaDiff(VAddr addr, u64 size, u64* clean, u64* first_diff);
private:
    void DropHotRead(VAddr page, const char* why);
    void DropHotReadIn(VAddr addr, u64 size, const char* why);

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;
    /// bbport: bound for size-0 requests no chunk covers (they stopped the game in GetChunk);
    /// nothing reads it.
    Buffer write_sink;

    /// bbport: the chunks' buffers (a deque: their addresses stay put) and their lookup table.
    std::deque<Buffer> arenas;
    ChunkTable chunks;

    /// bbport: how guest memory is laid out in GPU buffers (BB_ARENA_LAYOUT).
    enum class Layout : u8 { Legacy, Span, Runs };
    Layout layout = Layout::Legacy;
    u64 legacy_bytes = 0; ///< the legacy arena's length: the cap on all chunks
    u64 guard_bytes = 0;  ///< past each chunk (span, runs)
    std::atomic<u64> chunk_bytes{0};
    std::mutex materialize_mutex;
    std::atomic<u64> misses{0};
    std::jthread layout_monitor;

    /// Blocks whose page table entries are filled (the arena itself is fully backed).
    IntervalList<Interval> resident_ranges;

    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};
};

} // namespace VideoCore
