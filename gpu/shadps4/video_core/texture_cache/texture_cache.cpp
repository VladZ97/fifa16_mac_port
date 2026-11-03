#include <cmath>
#include <unordered_set>
#include <cstdio>
// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <map>
#include <set>
#include <string_view>
#include <xxhash.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>

#include "bbport_toggles.h"
#include <cstdlib>
#include <boost/container/small_vector.hpp>
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/hash.h"
#include "common/scope_exit.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

namespace VideoCore {

std::set<VAddr> g_bb_dump_volume_addrs; // fifa-hybrid: volumes to dump (BB_TONE_LOG fills it)

static constexpr u64 PageShift = 12;
static constexpr u64 NumFramesBeforeRemoval = 32;

TextureCache::TextureCache(Metal::Device& device_, Vulkan::Scheduler& scheduler_,
                           Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                           BufferCache& buffer_cache_, PageManager& tracker_)
    : device{device_}, scheduler{scheduler_}, runtime{runtime_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, tracker{tracker_},
      tile_manager{device, scheduler, runtime},
      readback_linear_images{EmulatorSettings.IsReadbackLinearImagesEnabled()} {

    u32 max_samplers = device.GetLimits().max_samplers;
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // Set up garbage collection parameters.
    // Metal reports the allocated size and a recommended working set (budget and local memory).
    u64 budget = device.GetLimits().recommended_working_set;
    const u64 local_mem = device.GetLimits().recommended_working_set;
    const bool is_low_vram = (local_mem > 0 && local_mem <= 4096ULL * 1024 * 1024) ||
                             (budget > 0 && budget <= 4096ULL * 1024 * 1024);
    const u64 base_mem = is_low_vram && local_mem > 0 ? local_mem : budget;
    if (base_mem > 0) {
        trigger_gc_memory = base_mem / 100 * (is_low_vram ? 70 : 70);
        pressure_gc_memory = base_mem / 100 * (is_low_vram ? 90 : 85);
        critical_gc_memory = base_mem / 100 * (is_low_vram ? 95 : 95);
    } else {
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        trigger_gc_memory = 0;
    }
}

TextureCache::~TextureCache() = default;

void TextureCache::ProcessDownloadImages() {
    if (static const char* pe = std::getenv("BB_PROBE_ADDR"); pe) {
        static int pn = 0;
        if (pn++ % 200 == 0) {
            const u64 pa = std::strtoull(pe, nullptr, 16);
            mach_vm_address_t a = pa; mach_vm_size_t sz = 0; vm_region_basic_info_data_64_t info{};
            mach_msg_type_number_t cnt = VM_REGION_BASIC_INFO_COUNT_64; mach_port_t obj = 0;
            mach_vm_region(mach_task_self(), &a, &sz, VM_REGION_BASIC_INFO_64,
                           (vm_region_info_t)&info, &cnt, &obj);
            std::unique_lock lk{download_images_mutex};
            const auto it = ondemand_page_refs.find(pa & ~0xfffULL);
            std::fprintf(stderr, "[PROBE] %#llx region=%#llx prot=%d refs=%d pending=%zu\n",
                         (unsigned long long)pa, (unsigned long long)a, info.protection,
                         it == ondemand_page_refs.end() ? -1 : int(it->second), ondemand_images.size());
        }
    }
    // fifa-hybrid: the write-backs of GPU-written images land as deferred priority operations
    // (one staging download each, no GPU drain per image). A synchronous download here cost a
    // full scheduler.Finish() per image per submission: FIFA composites dozens of textures on
    // the GPU each frame around the kit screen and took minutes to get through its menus.
    // BB_READBACK_SYNC=1 restores the synchronous write-back.
    static const bool sync = [] {
        const char* env = std::getenv("BB_READBACK_SYNC");
        return env && env[0] == '1';
    }();
    std::unique_lock lk{download_images_mutex};
    for (const ImageId image_id : download_images) {
        DownloadImageMemory(image_id, sync);
    }
    download_images.clear();
}

// fifa-hybrid: GPU-written *tiled* images are read back too (upstream b4e7ae7 downloaded only
// linear ones, as a linear copy). FIFA composites kit/skin textures and bakes small environment
// maps on the GPU into tiled textures that the CPU then reads or partly rewrites; without the
// readback it sees stale memory (black players, black stadium). BB_READBACK_TILED=<max pixels>
// (default 256x256; 0 = off) bounds the per-write cost; frame-sized and depth/HDR targets are
// never written back.
static u64 ReadbackTiledMaxPixels() {
    static const u64 value = [] {
        const char* env = std::getenv("BB_READBACK_TILED");
        if (!env || !*env) {
            return u64{256 * 256};
        }
        return std::strtoull(env, nullptr, 10);
    }();
    return value;
}

bool TextureCache::ShouldReadbackImage(const Image& image) const {
    if (!image.info.props.is_tiled || image.info.size.width <= 8) {
        return true;
    }
    const u32 w = image.info.size.width, h = image.info.size.height;
    if (u64{w} * h > ReadbackTiledMaxPixels()) {
        return false;
    }
    static const bool all_formats = [] {
        const char* env = std::getenv("BB_READBACK_ALLFMT");
        return env && env[0] == '1';
    }();
    switch (all_formats ? vk::Format::eUndefined : image.info.pixel_format) {
    case vk::Format::eB10G11R11UfloatPack32:
    case vk::Format::eR16G16B16A16Sfloat:
    case vk::Format::eR32Sfloat:
    case vk::Format::eD32Sfloat:
    case vk::Format::eD32SfloatS8Uint:
    case vk::Format::eD16Unorm:
    case vk::Format::eD24UnormS8Uint:
        return false;
    default:
        break;
    }
    if (w == 1920 || w == 1280 || w == 960 || h == 1080 || h == 720 || h == 540) {
        return false;
    }
    return true;
}

// fifa-hybrid: on-demand readbacks (ported from the FIFA side build of shadPS4). Eager
// downloads after every GPU write (BB_READBACK_EAGER=1) cost a staging copy per image per
// submission and, synchronously, a GPU drain each; FIFA writes dozens of small composites a
// frame. Instead the image's pages get a read watcher: the first CPU read faults, the image is
// downloaded then (synchronously, so the read sees it), and the watcher is removed.
void TextureCache::QueueReadback(ImageId image_id, Image& image) {
    static const bool eager = [] {
        const char* env = std::getenv("BB_READBACK_EAGER");
        return env && env[0] == '1';
    }();
    // debug: BB_READBACK_EAGER_SIZES="WxH,WxH" downloads only those sizes eagerly.
    static const std::string eager_sizes = [] {
        const char* env = std::getenv("BB_READBACK_EAGER_SIZES");
        return env ? "," + std::string(env) + "," : std::string{};
    }();
    const bool eager_size =
        !eager_sizes.empty() &&
        eager_sizes.find(fmt::format(",{}x{},", image.info.size.width, image.info.size.height)) !=
            std::string::npos;
    std::unique_lock lk{download_images_mutex};
    if (eager || eager_size) {
        download_images.emplace(image_id);
        return;
    }
    // fifa-hybrid: large images (the kit composites the loader reads back while the warm-up
    // drill runs) are also downloaded asynchronously after this submission, so the later CPU
    // fault usually finds the bytes in place and skips the GPU round trip and drain that cost
    // 10-60 ms each. BB_ONDEMAND_PREFETCH_KB=<n> (default 128, 0 off) sets the size threshold.
    static const u64 prefetch_bytes = [] {
        const char* env = std::getenv("BB_ONDEMAND_PREFETCH_KB");
        return (env ? u64(std::atoll(env)) : 128ull) * 1024ull;
    }();
    if (prefetch_bytes != 0 && image.info.guest_size >= prefetch_bytes) {
        download_images.emplace(image_id);
        if (static const bool fstats = std::getenv("BB_FAULT_STATS") != nullptr; fstats) {
            static int lines = 0;
            if (lines++ < 300) {
                std::fprintf(stderr, "[ODQ] prefetch img %#llx %ux%u size %#llx serial=%llu bc=%d\n",
                             (unsigned long long)image.info.guest_address, image.info.size.width,
                             image.info.size.height, (unsigned long long)image.info.guest_size,
                             (unsigned long long)image.write_serial, int(image.info.props.is_block));
            }
        }
    }
    // Overlapping images (an atlas and a tile rendered into it) must be written back in GPU
    // write order, so the latest write wins; re-queues of a pending image move it last.
    ondemand_seq[image_id.index] = ondemand_seq_next++;
    if (!ondemand_images.insert(image_id).second) {
        return; // already protected and pending
    }
    static const bool vb_log = std::getenv("BB_VB_LOG") != nullptr;
    static int qn = 0;
    if (vb_log && qn++ < 3000) {
        std::fprintf(stderr, "[RBQ] q img=%u addr=%#llx size=%#llx %ux%u\n", image_id.index,
                     (unsigned long long)image.info.guest_address,
                     (unsigned long long)image.info.guest_size, image.info.size.width,
                     image.info.size.height);
    }
    const bool droppable = DroppableOnWrite(image);
    ondemand_ranges[image_id.index] = {image.info.guest_address, image.info.guest_size, droppable};
    ProtectForCpuRead(image.info.guest_address, image.info.guest_size, true, droppable);
}

bool TextureCache::DroppableOnWrite(const Image& image) {
    if (image.block_shadow || image.shadow_of) {
        return false;
    }
    // Only screen-derived render targets (1920x1080, 960x540, ...): texture composites (kit and
    // face builds, 256x256 RGBA8 and the like) are power-of-two and their GPU result must reach
    // guest memory before the CPU touches it (the goalkeeper's shirt broke otherwise).
    const u32 w = image.info.size.width, h = image.info.size.height;
    const bool power_of_two = (w & (w - 1)) == 0 && (h & (h - 1)) == 0;
    if (power_of_two || w < 256 || h < 128) {
        return false;
    }
    const std::string name = vk::to_string(image.info.pixel_format);
    return name.find("Bc") == std::string::npos && name.find("Uint") == std::string::npos &&
           name.find("Sint") == std::string::npos;
}

// Refcounted per page (two images may share a page, and an image's range may change between
// the protect and the unprotect), so the page manager's own count never underflows.
void TextureCache::ProtectForCpuRead(VAddr addr, size_t size, bool protect, bool droppable) {
    if (size == 0) {
        return;
    }
    // Droppable images count as image read watchers (the page manager flags pages watched only
    // by them, the game writes those itself); the others count as plain read watchers.
    auto& refs = droppable ? ondemand_page_refs : ondemand_page_refs_plain;
    const VAddr first = PageManager::GetPageAddr(addr);
    const VAddr last = PageManager::GetNextPageAddr(addr + size - 1);
    for (VAddr page = first; page < last; page += PageManager::PM_PAGE_SIZE_PUBLIC) {
        if (protect) {
            if (refs[page]++ == 0) {
                if (droppable) {
                    tracker.UpdateReadWatchers<true, true>(page, PageManager::PM_PAGE_SIZE_PUBLIC);
                } else {
                    tracker.UpdateReadWatchers<true, false>(page, PageManager::PM_PAGE_SIZE_PUBLIC);
                }
            }
        } else {
            auto it = refs.find(page);
            if (it == refs.end()) {
                continue;
            }
            if (--it->second == 0) {
                refs.erase(it);
                if (droppable) {
                    tracker.UpdateReadWatchers<false, true>(page, PageManager::PM_PAGE_SIZE_PUBLIC);
                } else {
                    tracker.UpdateReadWatchers<false, false>(page, PageManager::PM_PAGE_SIZE_PUBLIC);
                }
            }
        }
    }
}

void TextureCache::FinishCpuReadWork(CpuReadWork& work) {
    if (!work.copies.empty()) {
        scheduler.Wait(work.tick);
        for (const auto& copy : work.copies) {
            Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(copy.addr), copy.download.mapped,
                                                      copy.size);
            runtime.GetStagingPool().FreeDeferred(copy.download);
        }
    }
    std::unique_lock lk{download_images_mutex};
    for (const auto& [a, sz, droppable] : work.unprotect) {
        ProtectForCpuRead(a, sz, false, droppable);
    }
}

void TextureCache::DownloadImagesForCpuRead(VAddr addr, size_t size, bool for_write,
                                            bool have_lock, CpuReadWork* work) {
    boost::container::small_vector<std::pair<ImageId, bool>, 4> ids;
    {
        std::unique_lock cache_lock{mutex, std::defer_lock};
        if (!have_lock) {
            cache_lock.lock();
        }
        std::unique_lock lk{download_images_mutex};
        if (ondemand_images.empty()) {
            return;
        }
        // Protection is per page: a fault on a page shared with an image resolves the image
        // even when the accessed bytes lie outside it.
        static const bool vb_log = std::getenv("BB_VB_LOG") != nullptr;
        static int fn = 0;
        if (vb_log && fn++ < 3000) {
            std::fprintf(stderr, "[RBQ] fault addr=%#llx w=%d pending=%zu\n",
                         (unsigned long long)addr, for_write, ondemand_images.size());
        }
        const auto pages_start = PageManager::GetPageAddr(addr);
        const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);
        const auto collect = [&](VAddr start, u64 len) {
            ForEachImageInRegion(start, len, [&](ImageId image_id, Image& image) {
                if (ondemand_images.contains(image_id) &&
                    std::find_if(ids.begin(), ids.end(), [&](const auto& e) {
                        return e.first == image_id;
                    }) == ids.end()) {
                    ids.emplace_back(image_id, True(image.flags & ImageFlagBits::GpuModified));
                }
            });
        };
        collect(pages_start, pages_end - pages_start);
        // Pending images overlapping a selected one are written back too (transitively), or a
        // later download of the bigger image would overwrite the smaller one's newer pixels.
        static const bool no_overlap = std::getenv("BB_RB_NOOVERLAP") != nullptr;
        for (size_t i = 0; !no_overlap && i < ids.size(); ++i) {
            const auto& im = slot_images[ids[i].first];
            collect(im.info.guest_address, im.info.guest_size);
        }
        std::sort(ids.begin(), ids.end(), [&](const auto& a, const auto& b) {
            return ondemand_seq[a.first.index] < ondemand_seq[b.first.index];
        });
        for (const auto& [id, _] : ids) {
            ondemand_images.erase(id);
            ondemand_seq.erase(id.index);
        }
    }
    static const bool log_images = std::getenv("BB_LOG_IMAGES") != nullptr;
    for (const auto& [id, modified] : ids) {
        if (log_images) {
            const auto& im = slot_images[id];
            LOG_INFO(Render_Vulkan, "ONDEMAND {} addr={:#x} {}x{} fmt={} modified={}",
                     for_write ? "write" : "read", im.info.guest_address, im.info.size.width,
                     im.info.size.height, vk::to_string(im.info.pixel_format), modified);
        }
        if (static const bool fstats = std::getenv("BB_FAULT_STATS") != nullptr; fstats) {
            static int lines = 0;
            const Image& im = slot_images[id];
            if (lines++ < 400) {
                std::fprintf(stderr, "[ODR] %s img %#llx %ux%u size %#llx mod=%d landed=%llu serial=%llu bc=%d shadow=%d\n",
                             for_write ? "write" : "read", (unsigned long long)im.info.guest_address,
                             im.info.size.width, im.info.size.height, (unsigned long long)im.info.guest_size,
                             int(modified), (unsigned long long)(im.prefetch_landed ? im.prefetch_landed->load() : 0),
                             (unsigned long long)im.write_serial, int(im.info.props.is_block), int(bool(im.block_shadow)));
            }
        }
        if (modified && !slot_images[id].PrefetchLanded()) {
            DownloadImageMemory(id, true, work);
        }
    }
    if (work && !work->copies.empty()) {
        scheduler.Flush();
        work->tick = scheduler.CurrentTick() - 1;
    }
    for (const auto& [id, _] : ids) {
        // Lift the read protection of the range protected at queue time, so a page the
        // download did not cover cannot fault forever.
        std::unique_lock lk{download_images_mutex};
        if (auto it = ondemand_ranges.find(id.index); it != ondemand_ranges.end()) {
            if (work) {
                work->unprotect.emplace_back(it->second.first, it->second.second, it->second.droppable);
            } else {
                ProtectForCpuRead(it->second.first, it->second.second, false, it->second.droppable);
            }
            ondemand_ranges.erase(it);
        }
    }
}

// fifa-hybrid: BB_ONDEMAND_WRITE_DROP=1 policy. A CPU write into an image awaiting an on-demand
// readback drops the readback instead of downloading first (a GPU round trip per fault, 1.5-3 ms
// on the GPU thread). The CPU's bytes win; the rest of the image in guest memory stays stale
// until the GPU writes it again, which is what a render target or a recycled texture sees anyway.
void TextureCache::ForgetOndemandImagesIn(VAddr addr, u64 size, bool droppable_only) {
    boost::container::small_vector<ImageId, 4> ids;
    {
        std::unique_lock lk{download_images_mutex};
        for (const auto& [index, range] : ondemand_ranges) {
            if (range.first < addr + size && range.first + range.second > addr &&
                (range.droppable || !droppable_only)) {
                ids.push_back(ImageId{index});
            }
        }
    }
    for (const ImageId id : ids) {
        ForgetOndemandImage(id);
    }
}

// fifa-hybrid diagnostics (BB_FAULT_STATS): names, once per image, the on-demand images a CPU
// write fault lands in (each such write costs a GPU round trip).
static std::mutex g_odw_counts_mutex;
static std::unordered_map<VAddr, std::pair<u64, std::string>> g_odw_counts; // per image: writes, label

void TextureCache::PrintOndemandWriteCounts() {
    std::scoped_lock lk{g_odw_counts_mutex};
    std::vector<std::pair<u64, std::string>> rows;
    for (auto& [addr, v] : g_odw_counts) {
        if (v.first) {
            rows.emplace_back(v.first, v.second);
        }
        v.first = 0;
    }
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::string line = "[ODWC]";
    for (size_t i = 0; i < rows.size() && i < 6; ++i) {
        line += fmt::format(" {}x{}", rows[i].second, rows[i].first);
    }
    std::printf("%s\n", line.c_str());
}

void TextureCache::LogOndemandWrite(VAddr addr, u64 size) {
    static std::unordered_set<VAddr> seen;
    static u32 lines = 0;
    std::unique_lock lk{download_images_mutex};
    for (const auto& [index, range] : ondemand_ranges) {
        // Page-granular overlap (as HasOndemandIn sees it); "P" marks writes that only share a
        // page with the image (outside its bytes).
        const VAddr page_lo = PageManager::GetPageAddr(range.first);
        const VAddr page_hi = PageManager::GetNextPageAddr(range.first + range.second - 1);
        if (page_lo >= addr + size || page_hi <= addr) {
            continue;
        }
        const bool exact = range.first < addr + size && range.first + range.second > addr;
        const Image& image = slot_images[ImageId{index}];
        {
            std::scoped_lock ck{g_odw_counts_mutex};
            auto& e = g_odw_counts[image.info.guest_address ^ (exact ? 0 : 1)];
            ++e.first;
            if (e.second.empty()) {
                e.second = fmt::format("{}x{}f{}{}{}@{:x}", image.info.size.width,
                                       image.info.size.height, u32(image.info.pixel_format),
                                       range.droppable ? "d" : "D", exact ? "" : "P",
                                       image.info.guest_address);
            }
        }
        if (!exact) {
            continue;
        }
        if (lines < 80 && seen.insert(image.info.guest_address).second) {
            ++lines;
            std::fprintf(stderr, "[ODW] write %#llx in img %u addr=%#llx size=%u %ux%ux%u fmt=%u\n",
                         (unsigned long long)addr, index,
                         (unsigned long long)image.info.guest_address, image.info.guest_size,
                         image.info.size.width, image.info.size.height, image.info.size.depth,
                         u32(image.info.pixel_format));
        }
    }
}

bool TextureCache::HasOndemandOverlapping(VAddr addr, u64 size) {
    std::unique_lock lk{download_images_mutex};
    if (ondemand_images.empty() || size == 0) {
        return false;
    }
    for (const ImageId id : ondemand_images) {
        if (!slot_images.is_allocated(id)) {
            continue;
        }
        const Image& image = slot_images[id];
        if (image.info.guest_address < addr + size &&
            addr < image.info.guest_address + image.info.guest_size) {
            return true;
        }
    }
    return false;
}

bool TextureCache::HasOndemandIn(VAddr addr, u64 size) {
    std::unique_lock lk{download_images_mutex};
    if ((ondemand_page_refs.empty() && ondemand_page_refs_plain.empty()) || size == 0) {
        return false;
    }
    const VAddr first = PageManager::GetPageAddr(addr);
    const VAddr last = PageManager::GetNextPageAddr(addr + size - 1);
    for (VAddr page = first; page < last; page += PageManager::PM_PAGE_SIZE_PUBLIC) {
        if (ondemand_page_refs.contains(page) || ondemand_page_refs_plain.contains(page)) {
            return true;
        }
    }
    return false;
}

void TextureCache::DropOndemandCovered(VAddr addr, u64 size) {
    static const bool off = std::getenv("BB_NO_DMA_DROP") != nullptr;
    static const bool log = std::getenv("BB_DMA_LOG") != nullptr;
    if (off || !HasOndemandIn(addr, size)) {
        return;
    }
    boost::container::small_vector<ImageId, 4> covered;
    {
        std::scoped_lock lock{mutex};
        std::unique_lock lk{download_images_mutex};
        ForEachImageInRegion(addr, size, [&](ImageId image_id, Image& image) {
            if (!ondemand_images.contains(image_id)) {
                return;
            }
            const bool inside = image.info.guest_address >= addr &&
                                image.info.guest_address + image.info.guest_size <= addr + size;
            if (inside) {
                covered.push_back(image_id);
            }
            static std::atomic<int> n{0};
            if (log && n++ < 300) {
                std::fprintf(stderr, "[DMADROP] dst %#llx size %#llx img %u %#llx size %#llx %ux%u %s\n",
                             (unsigned long long)addr, (unsigned long long)size, image_id.index,
                             (unsigned long long)image.info.guest_address,
                             (unsigned long long)image.info.guest_size, image.info.size.width,
                             image.info.size.height, inside ? "dropped" : "partial");
            }
        });
    }
    for (const ImageId id : covered) {
        ForgetOndemandImage(id);
    }
}

void TextureCache::ForgetOndemandImage(ImageId image_id) {
    std::unique_lock lk{download_images_mutex};
    static const bool vb_log = std::getenv("BB_VB_LOG") != nullptr;
    if (vb_log && ondemand_images.contains(image_id)) {
        std::fprintf(stderr, "[RBQ] forget img=%u\n", image_id.index);
    }
    if (ondemand_images.erase(image_id)) {
        ondemand_seq.erase(image_id.index);
        if (auto it = ondemand_ranges.find(image_id.index); it != ondemand_ranges.end()) {
            ProtectForCpuRead(it->second.first, it->second.second, false, it->second.droppable);
            ondemand_ranges.erase(it);
        }
    }
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync, CpuReadWork* work) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    SyncBlockShadow(image); // the newest blocks may sit in the one-texel-per-block shadow
    // fifa-hybrid: every mip through the tile manager (tiled images are re-tiled on the GPU).
    const auto image_addr = image.info.guest_address;
    const auto image_mips = image.info.resources.levels;
    u32 copy_size = 0;
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image_mips; ++mip) {
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> mip, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[mip];
        const u32 extent_width = mip_pitch ? std::min<u32>(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min<u32>(mip_height, height) : height;
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
        copy_size += mip_size;
    }
    if (([] { static const bool v = std::getenv("BB_VB_LOG") != nullptr; return v; }())) {
        std::fprintf(stderr, "[RBQ] dl addr=%#llx size=%#llx %ux%u fmt=%d copy=%#x sync=%d\n",
                     (unsigned long long)image_addr, (unsigned long long)image.info.guest_size,
                     image.info.size.width, image.info.size.height, int(image.info.pixel_format),
                     copy_size, int(sync));
    }
    if (buffer_copies.empty() || copy_size == 0 || copy_size > image.info.guest_size) {
        return;
    }
    const auto download =
        runtime.GetStagingPool().Request(copy_size, MemoryType::HostCached, 16, !sync || work);
    tile_manager.TileImage(image, buffer_copies, download.buffer, download.offset);
    if (work) {
        work->copies.push_back({image_addr, copy_size, download});
    } else if (sync) {
        scheduler.Finish();
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(image_addr), download.mapped,
                                                  copy_size);
    } else {
        if (!image.prefetch_landed) {
            image.prefetch_landed = std::make_shared<std::atomic<u64>>(0);
        }
        scheduler.DeferPriorityOperation([this, image_addr, download, copy_size,
                                          landed = image.prefetch_landed,
                                          serial = image.write_serial] {
            Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(image_addr),
                                                      download.mapped, copy_size);
            runtime.GetStagingPool().FreeDeferred(download);
            landed->store(serial, std::memory_order_release);
        });
    }
}

bool TextureCache::ResolvePrefetchedOndemand(VAddr addr, size_t size) {
    boost::container::small_vector<ImageId, 4> ids;
    std::unique_lock cache_lock{mutex};
    std::unique_lock lk{download_images_mutex};
    if (ondemand_images.empty() || size == 0) {
        return false;
    }
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);
    const auto collect = [&](VAddr start, u64 len) {
        ForEachImageInRegion(start, len, [&](ImageId image_id, Image&) {
            if (ondemand_images.contains(image_id) &&
                std::find(ids.begin(), ids.end(), image_id) == ids.end()) {
                ids.push_back(image_id);
            }
        });
    };
    collect(pages_start, pages_end - pages_start);
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto& im = slot_images[ids[i]];
        collect(im.info.guest_address, im.info.guest_size);
    }
    if (ids.empty()) {
        return false;
    }
    for (const ImageId id : ids) {
        const Image& image = slot_images[id];
        if (True(image.flags & ImageFlagBits::GpuModified) && !image.PrefetchLanded()) {
            return false;
        }
    }
    static std::atomic<u64> resolved{0};
    static const bool stats = std::getenv("BB_FAULT_STATS") != nullptr;
    for (const ImageId id : ids) {
        ondemand_images.erase(id);
        ondemand_seq.erase(id.index);
        if (auto it = ondemand_ranges.find(id.index); it != ondemand_ranges.end()) {
            ProtectForCpuRead(it->second.first, it->second.second, false, it->second.droppable);
            ondemand_ranges.erase(it);
        }
        if (stats && (resolved.fetch_add(1) % 50) == 0) {
            const Image& image = slot_images[id];
            std::fprintf(stderr, "[PREFETCH] resolved %llu faults without GPU work (last %#llx %ux%u)\n",
                         (unsigned long long)resolved.load(), (unsigned long long)image.info.guest_address,
                         image.info.size.width, image.info.size.height);
        }
    }
    return true;
}

void TextureCache::DebugDumpVolume(ImageId image_id) {
    static const char* dir = std::getenv("BB_DUMP_VOLUMES");
    static int count = 0;
    Image& image = slot_images[image_id];
    // fifa-hybrid: BB_DUMP_VOLUME_ADDR=<hex> also dumps that volume (GPU copy) on its first
    // three binds, GPU-modified or not (the second grading LUT pair is generated on the GPU).
    static const VAddr want = [] {
        const char* env = std::getenv("BB_DUMP_VOLUME_ADDR");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    static int want_count = 0;
    static std::map<VAddr, int> role_counts; // addresses the tonemapper hook asked for
    // BB_DUMP_VOLUME_TILE=<n>: also the first three volumes of that tile mode (the graded LUTs
    // of a match are thick-tiled, mode 19, at addresses that change per run).
    static const int want_tile = [] {
        const char* env = std::getenv("BB_DUMP_VOLUME_TILE");
        return env ? std::atoi(env) : -1;
    }();
    const bool wanted = (want != 0 && image.info.guest_address == want && want_count < 3) ||
                        (want_tile >= 0 && int(image.info.tile_mode) == want_tile && want_count < 3) ||
                        (g_bb_dump_volume_addrs.contains(image.info.guest_address) &&
                         role_counts[image.info.guest_address]++ < 2);
    if (!dir || (!wanted && (count >= 8 || False(image.flags & ImageFlagBits::GpuModified)))) {
        return;
    }
    // fifa-hybrid: BB_DUMP_AFTER=<s> delays volume dumps too (the grading LUTs of a rain match are
    // generated minutes after the first binds at the title screens).
    {
        static const double after_s = [] {
            const char* env = std::getenv("BB_DUMP_AFTER");
            return env ? std::atof(env) : 0.0;
        }();
        static const auto t0 = std::chrono::steady_clock::now();
        if (after_s > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < after_s) {
            return;
        }
    }
    if (wanted) {
        ++want_count;
    } else {
        ++count;
    }
    const std::string path = fmt::format("{}/gpu_{:x}_{}x{}x{}{}.bin", dir, image.info.guest_address,
                                         image.info.size.width, image.info.size.height,
                                         image.info.size.depth, wanted ? fmt::format("_b{}", want_count) : "");
    // The guest copy before the download (what the upload read), tile mode in the name.
    if (FILE* f = std::fopen(fmt::format("{}.guest_t{}", path, u32(image.info.tile_mode)).c_str(), "wb")) {
        std::fwrite(std::bit_cast<const u8*>(image.info.guest_address), 1, image.info.guest_size, f);
        std::fclose(f);
    }
    { // The GPU copy in linear layout (what the shader samples), independent of the guest tiling.
        const u32 w = image.info.size.width, h = image.info.size.height, d = image.info.size.depth;
        const u64 size = u64(w) * h * d * (image.info.num_bits / 8);
        std::array<vk::BufferImageCopy, 1> copies{vk::BufferImageCopy{
            .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageExtent = {w, h, d},
        }};
        const auto download = runtime.GetStagingPool().Request(size, MemoryType::HostCached, 16, false);
        copies[0].bufferOffset = download.offset;
        runtime.DownloadImage(&image, download.buffer, copies);
        scheduler.Finish();
        if (FILE* f = std::fopen((path + ".linear").c_str(), "wb")) {
            std::fwrite(download.mapped, 1, size, f);
            std::fclose(f);
        }
    }
    DownloadImageMemory(image_id, true);
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(std::bit_cast<const u8*>(image.info.guest_address), 1, image.info.guest_size, f);
        std::fclose(f);
    }
    std::fprintf(stderr, "VOLDUMP %s\n", path.c_str());
}

std::set<VAddr> g_bb_dump_extra;
void TextureCache::DepthProbe(VAddr addr) {
    std::vector<ImageId> ids;
    ForEachImageInRegion(addr, 4096, [&](ImageId id, Image& image) {
        if (image.info.guest_address == addr) ids.push_back(id);
    });
    for (const ImageId id : ids) {
        Image& image = slot_images[id];
        if (!image.backing) continue;
        const u32 w = image.info.size.width, h = image.info.size.height;
        const u64 size = u64(w) * h * 4;
        const auto dl = runtime.GetStagingPool().Request(size, MemoryType::HostCached, 16, false);
        scheduler.EndRendering();
        const bool depth = image.has_depth;
        scheduler.Record([src = image.GetImage(), w, h, dst = dl.buffer->Ref(dl.offset), size, depth](Metal::Encoder& e) {
            if (depth) e.CopyTextureToBuffer(src, 0, 0, {}, {w, h, 1}, dst, w * 4, u32(size), Metal::BlitAspect::Depth);
            else e.CopyTextureToBuffer(src, 0, 0, {}, {w, h, 1}, dst, w * 4, u32(size));
        });
        scheduler.Finish();
        const float* v = reinterpret_cast<const float*>(dl.mapped);
        const size_t n = size_t(w) * h, stride = std::max<size_t>(1, n / 65536);
        double sum = 0; float mn = 1e30f, mx = -1e30f; size_t zeros = 0, cnt = 0, ones = 0;
        for (size_t i = 0; i < n; i += stride) { const float x = v[i]; sum += x; mn = std::min(mn, x); mx = std::max(mx, x); ++cnt; zeros += x == 0.0f; ones += x >= 1.0f; }
        std::fprintf(stderr, "[DPROBE] %#llx %ux%u %s depth=%d serial %llu flags %#x: mean %.4f min %.4f max %.4f zero %.1f%% one %.1f%%\n",
                     (unsigned long long)addr, w, h, vk::to_string(image.info.pixel_format).c_str(), int(depth),
                     (unsigned long long)image.write_serial, u32(image.flags), cnt ? sum / cnt : 0.0, mn, mx,
                     cnt ? 100.0 * zeros / cnt : 0.0, cnt ? 100.0 * ones / cnt : 0.0);
    }
    if (ids.empty()) std::fprintf(stderr, "[DPROBE] %#llx: no image\n", (unsigned long long)addr);
}

void TextureCache::TwinWatch() {
    static const bool on = std::getenv("BB_TWIN_WATCH") != nullptr;
    if (!on || twin_watch_ids.empty()) {
        return;
    }
    static int round = 0;
    ++round;
    auto stats = [](const float* v, size_t n, size_t stride) {
        double sum = 0; float mn = 1e30f, mx = -1e30f; size_t zeros = 0, cnt = 0, ones = 0;
        for (size_t i = 0; i < n; i += stride) {
            const float x = v[i];
            sum += x; mn = std::min(mn, x); mx = std::max(mx, x); ++cnt;
            zeros += x == 0.0f; ones += x >= 1.0f;
        }
        return fmt::format("mean {:.4f} min {:.4f} max {:.4f} zero {:.1f}% one {:.1f}%", cnt ? sum / cnt : 0.0, mn, mx,
                           cnt ? 100.0 * zeros / cnt : 0.0, cnt ? 100.0 * ones / cnt : 0.0);
    };
    for (size_t k = 0; k < twin_watch_ids.size();) {
        Image* image = TryGetImage(twin_watch_ids[k].first, twin_watch_ids[k].second);
        if (!image || !image->depth_twin || !image->backing) {
            std::fprintf(stderr, "[TWINWATCH] r%d entry %zu gone (image %p)\n", round, k, (void*)image);
            twin_watch_ids.erase(twin_watch_ids.begin() + k);
            continue;
        }
        const u32 w = image->info.size.width, h = image->info.size.height;
        const u64 size = u64(w) * h * 4;
        std::array<vk::BufferImageCopy, 1> copies{vk::BufferImageCopy{
            .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageExtent = {w, h, 1},
        }};
        const auto dl = runtime.GetStagingPool().Request(size, MemoryType::HostCached, 16, false);
        copies[0].bufferOffset = dl.offset;
        const u64 serial_before = image->write_serial, synced_before = image->depth_twin_synced;
        runtime.DownloadImage(image, dl.buffer, copies);
        const auto tdl = runtime.GetStagingPool().Request(size, MemoryType::HostCached, 16, false);
        scheduler.Record([src = image->depth_twin.Get(), w, h, dst = tdl.buffer->Ref(tdl.offset), size](Metal::Encoder& e) {
            e.CopyTextureToBuffer(src, 0, 0, {}, {w, h, 1}, dst, w * 4, u32(size), Metal::BlitAspect::Depth);
        });
        scheduler.Finish();
        const size_t n = size_t(w) * h, stride = std::max<size_t>(1, n / 65536);
        std::fprintf(stderr, "[TWINWATCH] r%d %#llx %ux%u serial %llu synced %llu flags %#x | src %s | twin %s\n", round,
                     (unsigned long long)image->info.guest_address, w, h, (unsigned long long)serial_before,
                     (unsigned long long)synced_before, u32(image->flags),
                     stats(reinterpret_cast<const float*>(dl.mapped), n, stride).c_str(),
                     stats(reinterpret_cast<const float*>(tdl.mapped), n, stride).c_str());
        ++k;
    }
    std::fflush(stderr);
}

void TextureCache::DebugDumpTexture(ImageId image_id, bool shadowed) {
    // debug: BB_DUMP_TEXTURES=<dir> writes mip 0 of sampled 2D textures as the GPU holds them.
    static const char* textures_dir = std::getenv("BB_DUMP_TEXTURES");
    static const char* shadowed_dir = std::getenv("BB_DUMP_SHADOWED");
    static std::set<VAddr> seen, seen_shadowed;
    const char* dir = shadowed ? shadowed_dir : textures_dir;
    Image& image = slot_images[image_id];
    // BB_DUMP_AFTER=<s>: only non-block (render target) images, from <s> seconds after the first call
    static const char* after = std::getenv("BB_DUMP_AFTER");
    static const auto t0 = std::chrono::steady_clock::now();
    // BB_DUMP_ADDRS=<dir>:<hex>,<hex>: dump these guest addresses (any kind, up to 3 times each).
    static const char* addrs_env = std::getenv("BB_DUMP_ADDRS");
    // fifa-hybrid: BB_DUMP_R32=<dir>: like BB_DUMP_ADDRS for every R32 image up to 1024 wide.
    static const char* r32_env = std::getenv("BB_DUMP_R32");
    static const u32 r32_max = [] {
        const char* e = std::getenv("BB_DUMP_R32_MAX");
        return e ? u32(std::atoi(e)) : 1024u;
    }();
    if (r32_env && image.info.pixel_format == vk::Format::eR32Sfloat && image.info.size.width <= r32_max &&
        image.info.size.width >= 64) {
        static std::string r32_dir = [] {
            std::string d(r32_env); return d + ":0"; }();
        if (!addrs_env) {
            addrs_env = r32_dir.c_str();
        }
        if (g_bb_dump_extra.insert(image.info.guest_address).second) {
            std::fprintf(stderr, "[R32DUMP] candidate %#llx %ux%u t=%.1f\n",
                         (unsigned long long)image.info.guest_address, image.info.size.width,
                         image.info.size.height,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    if (addrs_env) {
        static std::string adir;
        static std::map<VAddr, int> want;
        static bool parsed = false;
        if (!parsed) {
            parsed = true;
            std::string e(addrs_env);
            const auto c = e.find(':');
            adir = e.substr(0, c);
            for (size_t p = c + 1; p < e.size();) {
                size_t q = e.find(',', p);
                if (q == std::string::npos) q = e.size();
                want[std::stoull(e.substr(p, q - p), nullptr, 16)] = 0;
                p = q + 1;
            }
        }
        auto it = want.find(image.info.guest_address);
        if (it == want.end() && g_bb_dump_extra.contains(image.info.guest_address)) {
            it = want.emplace(image.info.guest_address, 0).first;
        }
        if (it == want.end() || it->second >= 3) {
            return;
        }
        // fifa-hybrid: BB_DUMP_AFTER=<s> delays the address dumps too (the first binds are the
        // title screens, where a scene target still holds its clear value).
        if (after && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() <
                         std::atof(after)) {
            static int waits = 0;
            if (++waits % 2000 == 1) {
                std::fprintf(stderr, "[DUMPADDR] waiting for t>=%s (now %.1f) for %#llx\n", after,
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                             (unsigned long long)image.info.guest_address);
            }
            return;
        }
        shadowed = false;
        dir = adir.c_str();
        const int n = it->second++;
        const u32 w = image.info.size.width, h = image.info.size.height;
        const u32 bs = image.info.props.is_block ? 4 : 1;
        const u64 size = u64((w + bs - 1) / bs) * ((h + bs - 1) / bs) * (image.info.num_bits / 8);
        std::array<vk::BufferImageCopy, 1> copies{vk::BufferImageCopy{
            .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
            .imageExtent = {w, h, 1},
        }};
        const auto download = runtime.GetStagingPool().Request(size, MemoryType::HostCached, 16, false);
        copies[0].bufferOffset = download.offset;
        runtime.DownloadImage(&image, download.buffer, copies);
        scheduler.Finish();
        const std::string path = fmt::format("{}/gpu_{:x}_n{}_{}_{}x{}_t{}.bin", dir, image.info.guest_address, n,
                                             vk::to_string(image.info.pixel_format), w, h, u32(image.info.tile_mode));
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(download.mapped, 1, size, f);
            std::fclose(f);
        }
        if (FILE* f = std::fopen((path + ".guest").c_str(), "wb")) {
            std::fwrite(std::bit_cast<const u8*>(image.info.guest_address), 1, image.info.guest_size, f);
            std::fclose(f);
        }
        // fifa-hybrid: the D32 twin (Image::depth_twin) as the compare samplers see it.
        if (image.depth_twin && !image.info.props.is_block) {
            const u64 tsize = u64(w) * h * 4;
            const auto tdl = runtime.GetStagingPool().Request(tsize, MemoryType::HostCached, 16, false);
            scheduler.Record([src = image.depth_twin.Get(), w, h, dst = tdl.buffer->Ref(tdl.offset), tsize](Metal::Encoder& e) {
                e.CopyTextureToBuffer(src, 0, 0, {}, {w, h, 1}, dst, w * 4, u32(tsize), Metal::BlitAspect::Depth);
            });
            scheduler.Finish();
            if (FILE* f = std::fopen((path + ".twin").c_str(), "wb")) {
                std::fwrite(tdl.mapped, 1, tsize, f);
                std::fclose(f);
            }
        }
        std::fprintf(stderr, "[DUMPADDR] %s flags %#x gsize %u\n", path.c_str(), u32(image.flags), u32(image.info.guest_size));
        return;
    }
    if (!dir || image.info.props.is_volume || image.info.size.width < 256) {
        return;
    }
    static std::map<VAddr, int> shadow_syncs;
    const int sync_no = shadowed ? shadow_syncs[image.info.guest_address]++ : 0;
    if (shadowed) {
        if (seen_shadowed.size() >= 48 || sync_no >= 3) {
            return;
        }
        seen_shadowed.insert(image.info.guest_address);
    } else if (after) {
        static const bool after_block = std::getenv("BB_DUMP_AFTER_BLOCK") != nullptr;
        static const bool after_all = std::getenv("BB_DUMP_AFTER_ALL") != nullptr;
        if ((!after_all && image.info.props.is_block != after_block) || image.info.props.is_depth ||
            image.info.size.width < (after_all ? 256u : 512u) ||
            std::chrono::steady_clock::now() - t0 <
                                             std::chrono::seconds(std::atoi(after))) {
            return;
        }
    } else if (!image.info.props.is_block) {
        return;
    } else if (std::getenv("BB_DUMP_GPUBC") &&
               (False(image.flags & ImageFlagBits::GpuModified) || image.info.num_bits != u32(std::max(64, std::atoi(std::getenv("BB_DUMP_GPUBC")))))) {
        return;
    }
    static const size_t dump_max = std::getenv("BB_DUMP_MAX") ? std::atoi(std::getenv("BB_DUMP_MAX")) : 64;
    if (!shadowed && (seen.size() >= dump_max || !seen.insert(image.info.guest_address).second)) {
        return;
    }
    for (u32 layer = 0; layer < image.info.resources.layers; ++layer)
    for (u32 mip = 0; mip < (after && !([] { static const bool v = std::getenv("BB_DUMP_MIPS") != nullptr; return v; }()) ? 1u : image.info.resources.levels); ++mip) {
        const u32 w = std::max(image.info.size.width >> mip, 1u);
        const u32 h = std::max(image.info.size.height >> mip, 1u);
        std::array<vk::BufferImageCopy, 1> copies{vk::BufferImageCopy{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource{.aspectMask = image.info.props.is_depth
                                                ? vk::ImageAspectFlagBits::eDepth
                                                : vk::ImageAspectFlagBits::eColor,
                              .mipLevel = mip, .baseArrayLayer = layer, .layerCount = 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {w, h, 1},
        }};
        const u32 bs = image.info.props.is_block ? 4 : 1;
        const u64 size = u64((w + bs - 1) / bs) * ((h + bs - 1) / bs) * (image.info.num_bits / 8);
        const auto download = runtime.GetStagingPool().Request(size, MemoryType::HostCached, 16, false);
        copies[0].bufferOffset = download.offset;
        runtime.DownloadImage(&image, download.buffer, copies);
        scheduler.Finish();
        const std::string path = fmt::format("{}/tex_{:x}_s{}_l{}_m{}_{}_{}x{}_t{}.bin", dir,
                                             image.info.guest_address, sync_no, layer, mip,
                                             vk::to_string(image.info.pixel_format), w, h,
                                             u32(image.info.tile_mode));
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fwrite(download.mapped, 1, size, f);
            std::fclose(f);
        }
    }
    u64 guest_nz = 0;
    for (const u8* g = std::bit_cast<const u8*>(image.info.guest_address),
                  *ge = g + image.info.guest_size;
         g < ge; ++g) {
        guest_nz += *g != 0;
    }
    std::fprintf(stderr, "TEXDUMP %x levels=%u flags=%x guest_nz=%llu/%llu\n",
                 u32(image.info.guest_address), image.info.resources.levels, u32(image.flags),
                 (unsigned long long)guest_nz, (unsigned long long)image.info.guest_size);
}

static inline u64 MaybeDirtyHash(const Image& image) {
    return XXH3_64bits(std::bit_cast<const u8*>(image.info.guest_address), image.info.guest_size);
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // Initialize hash
        image.hash = MaybeDirtyHash(image);
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    std::scoped_lock lock{mutex};
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);
    ForEachImageInRegion(pages_start, pages_end - pages_start, [&](ImageId image_id, Image& image) {
        // fifa-hybrid: a CPU write into an image awaiting an on-demand readback ends the
        // readback and lifts the read protection, otherwise the write faults forever.
        if (([] { static const bool v = std::getenv("BB_VB_LOG") != nullptr; return v; }())) {
            std::fprintf(stderr, "[RBQ] inval img=%u addr=%#llx+%#zx\n", image_id.index,
                         (unsigned long long)addr, size);
        }
        ForgetOndemandImage(image_id);
        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
        }
    });
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size, u64 writer) {
    std::scoped_lock lock{mutex};
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only consider images that match base address.
        // TODO: Maybe also consider subresources
        if (image.info.guest_address != address) {
            return;
        }
        // bbport: BB_LOG_UPLOAD_MB=<n> also names, once, each writer that dirties an image of
        // at least n MB (diagnostics).
        static const double log_upload_mb = [] {
            const char* env = std::getenv("BB_LOG_UPLOAD_MB");
            return env ? std::atof(env) : 0.0;
        }();
        if (log_upload_mb > 0 && image.info.guest_size >= log_upload_mb * 1e6) {
            static std::set<std::pair<u64, VAddr>> logged;
            if (logged.emplace(writer, address).second) {
                LOG_WARNING(Render_Vulkan,
                            "GPU write dirties image {:#x} size {:#x} ({}x{} format {}): {} {:#x}, "
                            "write size {:#x}, image flags {:#x}",
                            address, image.info.guest_size, image.info.size.width,
                            image.info.size.height, static_cast<u32>(image.info.pixel_format),
                            writer ? "shader" : "DMA copy", writer, max_size,
                            static_cast<u32>(image.flags));
            }
        }
        if (static const bool img_formats = std::getenv("BB_IMG_FORMATS") != nullptr;
            img_formats && image.info.props.is_volume) {
            static std::set<std::pair<u64, VAddr>> vlogged;
            if (vlogged.emplace(writer, address).second) {
                std::fprintf(stderr, "VOLWRITE %#llx by %s %#llx size %#llx\n",
                             (unsigned long long)address, writer ? "shader" : "DMA",
                             (unsigned long long)writer, (unsigned long long)max_size);
            }
        }
        // fifa-hybrid diagnostics: BB_REFRESH_LOG=1 names the GPU writer of every R32 image
        // (shadow maps filled through buffer writes or DMA).
        if (static const bool rlog = std::getenv("BB_REFRESH_LOG") != nullptr;
            rlog && image.info.pixel_format == vk::Format::eR32Sfloat) {
            static std::set<std::pair<u64, VAddr>> seen;
            if (seen.emplace(writer, address).second && seen.size() < 200) {
                std::fprintf(stderr, "[GPUWRITE] image %#llx %ux%u R32 written by %s %#llx size %#llx\n",
                             (unsigned long long)address, image.info.size.width, image.info.size.height,
                             writer ? "shader" : "DMA", (unsigned long long)writer, (unsigned long long)max_size);
            }
        }
        // Ensure image is reuploaded when accessed again.
        image.flags |= ImageFlagBits::GpuDirty;
    });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    std::scoped_lock lk{mutex};

    ImageIds deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

static bool DepthTwinEnabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("BB_DEPTH_TWIN");
        return !(e && e[0] == '0');
    }();
    return enabled;
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = cache_image.info.resources < requested_info.resources;

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        // fifa-hybrid: an R32 colour image compare-sampled as depth keeps a D32 twin instead
        // (Image::depth_twin; BB_DEPTH_TWIN=0 restores the recreation).
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth &&
                    !(DepthTwinEnabled() && cache_image.DepthTwinEligible() &&
                      !requested_info.props.has_stencil && bpp_match);
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        new_info.resources = std::max(requested_info.resources, cache_image.info.resources);
        new_info.UpdateSize();
        const auto new_image_id = slot_images.insert(device, runtime, slot_image_views, new_info);
        // bbport: the replacement keeps the lazy usages the old image got.
        slot_images[new_image_id].ApplyNewUsage(slot_images[cache_image_id].CurrentUsage());
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;
        runtime.CopyColorAndDepth(&cache_image, &new_image);

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NumFramesBeforeRemoval;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            static const bool bc_log = std::getenv("BB_BCVIEW_LOG") != nullptr;
            static int bc_logged = 0;
            if (bc_log && (image_info.props.is_block || cache_image.info.props.is_block) &&
                bc_logged < 200) {
                ++bc_logged;
                std::fprintf(stderr,
                             "[BCOVL] addr=%llx req fmt=%u %ux%u lv%u ly%u gs=%llx blk=%d | "
                             "cache fmt=%u %ux%u lv%u ly%u gs=%llx blk=%d gpu=%d\n",
                             (unsigned long long)image_info.guest_address,
                             (unsigned)image_info.pixel_format, image_info.size.width,
                             image_info.size.height, image_info.resources.levels,
                             image_info.resources.layers,
                             (unsigned long long)image_info.guest_size,
                             (int)image_info.props.is_block,
                             (unsigned)cache_image.info.pixel_format, cache_image.info.size.width,
                             cache_image.info.size.height, cache_image.info.resources.levels,
                             cache_image.info.resources.layers,
                             (unsigned long long)cache_image.info.guest_size,
                             (int)cache_image.info.props.is_block,
                             (int)True(cache_image.flags & ImageFlagBits::GpuModified));
            }
            // GPU texture compression wrote the blocks through a uint image before the BC
            // texture was first used: the new BC image takes them over.
            if (image_info.props.is_block && !cache_image.info.props.is_block &&
                lhs_block_size == rhs_block_size &&
                True(cache_image.flags & ImageFlagBits::GpuModified) &&
                cache_image.info.size.width == image_info.size.width / 4 &&
                cache_image.info.size.height == image_info.size.height / 4 &&
                cache_image.info.resources.layers == image_info.resources.layers &&
                !([] { static const bool v = std::getenv("BB_NO_BLOCK_SEED") != nullptr; return v; }())) {
                block_seed = cache_image_id;
                return {merged_image_id, -1, -1};
            }
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        const bool pow2_padding_only =
            image_info.props.is_pow2 != cache_image.info.props.is_pow2 &&
            image_info.tile_mode == cache_image.info.tile_mode &&
            image_info.size == cache_image.info.size &&
            image_info.pitch == cache_image.info.pitch && image_info.resources.levels == 1 &&
            cache_image.info.resources.levels == 1 && image_info.resources.layers == 1 &&
            cache_image.info.resources.layers == 1;

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size || pow2_padding_only) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // fifa-hybrid: same address reused with a different pow2 padding / mip count (kit
        // composites: 1024x1024 6 mips, then 1 mip padded). Treat like an address reuse.
        if (!([] { static const bool v = std::getenv("BB_OVERLAP_STRICT") != nullptr; return v; }())) {
            LOG_WARNING(Render_Vulkan,
                        "image overlap at {:#x}: replacing {}x{} lv{} gs {:#x} with lv{} gs {:#x}",
                        image_info.guest_address, cache_image.info.size.width,
                        cache_image.info.size.height, cache_image.info.resources.levels,
                        cache_image.info.guest_size, image_info.resources.levels,
                        image_info.guest_size);
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_ERROR(Render_Vulkan,
                  "Unresolvable image overlap with equal memory address:\n"
                  "=== OLD IMAGE (cached) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  Last accessed:  tick {}\n"
                  "  Safe to delete: {}\n"
                  "  isPow2:         {}\n"
                  "  Alt tile:       {}\n"
                  "\n"
                  "=== NEW IMAGE (requested) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  isPow2:         {}\n"
                  "  Alt tile:       {}\n"
                  "\n"
                  "=== COMPARISON ===\n"
                  "  Same format:           {}\n"
                  "  Same type:             {}\n"
                  "  Same tile mode:        {}\n"
                  "  Same block size:       {}\n"
                  "  Same BlockDim:         {}\n"
                  "  Same pitch:            {}\n"
                  "  Same pow2:             {}\n"
                  "  Same alt tile:         {}\n"
                  "  Old resources <= new:  {} (old: {}, new: {})\n"
                  "  Old size <= new size:  {}\n"
                  "  Expected size (calc):  {} bytes\n"
                  "  Size ratio (new/expected): {:.2f}x\n"
                  "  Size ratio (new/old):  {:.2f}x\n"
                  "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  Merged image ID:       {}\n"
                  "  Binding type:          {}\n"
                  "  Current tick:          {}\n"
                  "  Age (ticks since last access): {}",

                  // Old image details
                  cache_image.info.guest_address, cache_image.info.guest_size,
                  vk::to_string(cache_image.info.pixel_format),
                  static_cast<int>(cache_image.info.type), cache_image.info.size.width,
                  cache_image.info.size.height, cache_image.info.size.depth, cache_image.info.pitch,
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  cache_image.info.num_samples, static_cast<u32>(cache_image.info.tile_mode),
                  cache_image.info.num_bits, +cache_image.info.props.is_block,
                  cache_image.info.guest_size, cache_image.tick_accessed_last, safe_to_delete,
                  bool(cache_image.info.props.is_pow2), cache_image.info.alt_tile,

                  // New image details
                  image_info.guest_address, image_info.guest_size,
                  vk::to_string(image_info.pixel_format), static_cast<int>(image_info.type),
                  image_info.size.width, image_info.size.height, image_info.size.depth,
                  image_info.pitch, image_info.resources.levels, image_info.resources.layers,
                  image_info.num_samples, static_cast<u32>(image_info.tile_mode),
                  image_info.num_bits, image_info.props.is_block, image_info.guest_size,
                  bool(image_info.props.is_pow2), image_info.alt_tile,

                  // Comparison
                  (image_info.pixel_format == cache_image.info.pixel_format),
                  (image_info.type == cache_image.info.type),
                  (image_info.tile_mode == cache_image.info.tile_mode),
                  (image_info.num_bits == cache_image.info.num_bits),
                  (image_info.BlockDim() == cache_image.info.BlockDim()),
                  (image_info.pitch == cache_image.info.pitch),
                  (image_info.props.is_pow2 == cache_image.info.props.is_pow2),
                  (image_info.alt_tile == cache_image.info.alt_tile),
                  (cache_image.info.resources <= image_info.resources),
                  cache_image.info.resources.levels, image_info.resources.levels,
                  (cache_image.info.guest_size <= image_info.guest_size), expected_size,

                  // Size ratios
                  static_cast<double>(image_info.guest_size) / expected_size,
                  static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

                  // Difference between actual and expected sizes with percentages
                  static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

                  static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

                  merged_image_id.index, static_cast<int>(binding), scheduler.CurrentTick(),
                  scheduler.CurrentTick() - cache_image.tick_accessed_last);

        UNREACHABLE_MSG("Encountered unresolvable image overlap with equal memory address.");
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        if (auto mip = image_info.MipOf(cache_image.info); mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        if (auto mip = cache_image.info.MipOf(image_info); mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    runtime.CopyMip(&cache_image, &merged_image, mip, slice);
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}


void TextureCache::DebugProbeImage(ImageId id, const char* tag) {
    Image& img = slot_images[id];
    const auto uf = [](u32 v, u32 mbits) {
        const u32 m = v & ((1u << mbits) - 1), e = (v >> mbits) & 31;
        if (e == 31) return -1.0f;
        return e == 0 ? std::ldexp(float(m) / (1u << mbits), -14)
                      : std::ldexp(1.0f + float(m) / (1u << mbits), int(e) - 15);
    };
    const u32 layers = std::min(img.info.resources.layers, 6u);
    std::string line = fmt::format("[PROBELUM] {} id {} addr {:#x}", tag, id.index, img.info.guest_address);
    for (u32 m = 0; m < img.info.resources.levels; ++m) {
        const u32 w = std::max(img.info.size.width >> m, 1u), h = std::max(img.info.size.height >> m, 1u);
        const u64 size = u64(w) * h * 4 * layers;
        std::array<vk::BufferImageCopy, 1> c{vk::BufferImageCopy{
            .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = m,
                              .baseArrayLayer = 0, .layerCount = layers},
            .imageExtent = {w, h, 1}}};
        const auto dl = runtime.GetStagingPool().Request(size, MemoryType::HostCached, 16, false);
        c[0].bufferOffset = dl.offset;
        runtime.DownloadImage(&img, dl.buffer, c);
        scheduler.Finish();
        const u32* b = reinterpret_cast<const u32*>(dl.mapped);
        double sum[3] = {};
        u64 bad = 0;
        const u64 n = u64(w) * h * layers;
        for (u64 i = 0; i < n; ++i) {
            const float r = uf(b[i] & 0x7ff, 6), g = uf((b[i] >> 11) & 0x7ff, 6), bl = uf(b[i] >> 22, 5);
            if (r < 0 || g < 0 || bl < 0) { ++bad; continue; }
            sum[0] += r; sum[1] += g; sum[2] += bl;
        }
        line += fmt::format(" | m{} {:.3f} {:.3f} {:.3f} bad {}", m, sum[0] / n, sum[1] / n, sum[2] / n, bad);
    }
    std::fprintf(stderr, "%s\n", line.c_str());
    // fifa-hybrid: the separate images the cache holds inside this image's range (the per-level
    // render targets of the chain): their level 0 means, to compare with the parent's levels.
    boost::container::small_vector<ImageId, 8> subs;
    ForEachImageInRegion(img.info.guest_address, img.info.guest_size, [&](ImageId sub_id, Image& sub) {
        if (sub_id != id && sub.info.size.width < img.info.size.width &&
            sub.info.pixel_format == img.info.pixel_format) {
            subs.push_back(sub_id);
        }
    });
    for (const ImageId sub_id : subs) {
        Image& sub = slot_images[sub_id];
        const u32 w = sub.info.size.width, h = sub.info.size.height;
        const u64 size = u64(w) * h * 4;
        std::array<vk::BufferImageCopy, 1> c{vk::BufferImageCopy{
            .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0,
                              .baseArrayLayer = 0, .layerCount = 1},
            .imageExtent = {w, h, 1}}};
        const auto dl = runtime.GetStagingPool().Request(size, MemoryType::HostCached, 16, false);
        c[0].bufferOffset = dl.offset;
        runtime.DownloadImage(&sub, dl.buffer, c);
        scheduler.Finish();
        const u32* b = reinterpret_cast<const u32*>(dl.mapped);
        double sum[3] = {};
        const u64 n = u64(w) * h;
        for (u64 i = 0; i < n; ++i) {
            sum[0] += uf(b[i] & 0x7ff, 6); sum[1] += uf((b[i] >> 11) & 0x7ff, 6); sum[2] += uf(b[i] >> 22, 5);
        }
        std::fprintf(stderr, "[PROBELUM]   sub id %u addr %#llx %ux%u levels %u: %.3f %.3f %.3f\n",
                     sub_id.index, (unsigned long long)sub.info.guest_address, w, h,
                     sub.info.resources.levels, sum[0] / n, sum[1] / n, sum[2] / n);
    }
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    const auto new_image_id = slot_images.insert(device, runtime, slot_image_views, info);
    // bbport: the larger image keeps the lazy usages the old one got (the game grows mip
    // chains one level at a time).
    slot_images[new_image_id].ApplyNewUsage(slot_images[image_id].CurrentUsage());
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    static const bool img_formats = std::getenv("BB_IMG_FORMATS") != nullptr;
    if (img_formats) {
        static std::atomic<int> logged{0};
        if (logged.fetch_add(1) < 64) {
            const auto& a = src_image.info;
            std::fprintf(stderr,
                         "IMGEXPAND %#llx %s type=%u %ux%ux%u layers=%u mips=%u tile=%u flags=%#x -> "
                         "%s type=%u %ux%ux%u layers=%u mips=%u tile=%u\n",
                         (unsigned long long)a.guest_address, vk::to_string(a.pixel_format).c_str(),
                         u32(a.type), a.size.width, a.size.height, a.size.depth, a.resources.layers,
                         a.resources.levels, u32(a.tile_mode), u32(src_image.flags),
                         vk::to_string(info.pixel_format).c_str(), u32(info.type), info.size.width,
                         info.size.height, info.size.depth, info.resources.layers,
                         info.resources.levels, u32(info.tile_mode));
        }
    }

    RefreshImage(new_image);
    static int probed = 0;
    const bool probe = ([] { static const bool v = std::getenv("BB_BCVIEW_LOG") != nullptr; return v; }()) && info.props.is_block &&
                       !src_image.info.props.is_block && info.num_bits == 128 && probed < 8;
    const auto halves = [&](Image& img) {
        const u32 bs = img.info.props.is_block ? 4 : 1;
        const u32 w = img.info.size.width, h = img.info.size.height;
        const u64 size = u64(w / bs) * (h / bs) * 16;
        std::array<vk::BufferImageCopy, 1> c{vk::BufferImageCopy{
            .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0,
                              .baseArrayLayer = 0, .layerCount = 1},
            .imageExtent = {w, h, 1}}};
        const auto dl = runtime.GetStagingPool().Request(size, MemoryType::HostCached, 16, false);
        c[0].bufferOffset = dl.offset;
        runtime.DownloadImage(&img, dl.buffer, c);
        scheduler.Finish();
        u64 n[4] = {};
        const u8* b = static_cast<const u8*>(dl.mapped);
        for (u64 i = 0; i < size; ++i) {
            n[(i & 15) / 4] += b[i] != 0;
        }
        std::fprintf(stderr, "[BCPROBE] %s %#llx x=%llu y=%llu z=%llu w=%llu of %llu\n",
                     vk::to_string(img.info.pixel_format).c_str(),
                     (unsigned long long)img.info.guest_address, (unsigned long long)n[0],
                     (unsigned long long)n[1], (unsigned long long)n[2], (unsigned long long)n[3],
                     (unsigned long long)size / 4);
    };
    if (probe) {
        halves(src_image);
    }
    runtime.CopyImage(&src_image, &new_image);
    if (probe) {
        ++probed;
        halves(new_image);
    }

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    FreeImage(image_id);
    TrackImage(new_image_id);
    return new_image_id;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    const auto& info = desc.info;
    ASSERT(info.guest_address != 0);

    std::scoped_lock lock{mutex};

    u64 key_hash = info.guest_address ^ info.guest_size << 17 ^ u64(info.pixel_format) << 40 ^
                   u64(info.size.width) << 24 ^ info.size.height ^ u64(desc.type) << 58 ^
                   u64(exact_fmt) << 63;
    key_hash = (key_hash ^ key_hash >> 33) * 0xFF51AFD7ED558CCDull;
    key_hash = (key_hash ^ key_hash >> 33) * 0xC4CEB9FE1A85EC53ull;
    key_hash ^= key_hash >> 33;
    auto& cached = find_image_cache[key_hash % find_image_cache.size()];
    // An exact match stays valid while that image is registered with the same description;
    // resolved overlaps (views into other images) only until any image registration changes.
    const auto still_exact = [&] {
        if (cached.view_mip >= 0 || cached.view_slice >= 0 || !cached.image_id) {
            return false;
        }
        const Image& image = slot_images[cached.image_id];
        return True(image.flags & ImageFlagBits::Registered) &&
               image.info.guest_address == info.guest_address &&
               image.info.guest_size == info.guest_size && image.info.size == info.size &&
               image.info.pixel_format == info.pixel_format;
    };
    if ((cached.generation == registry_generation.load(std::memory_order_relaxed) || still_exact()) &&
        cached.address == info.guest_address &&
        cached.size == info.guest_size && cached.extent == info.size &&
        cached.format == info.pixel_format && cached.type == info.type &&
        cached.exact_fmt == exact_fmt && cached.binding == desc.type &&
        cached.levels == info.resources.levels && cached.layers == info.resources.layers &&
        !BbToggle::Disabled(BbToggle::FindImageCache)) {
        Image& image = slot_images[cached.image_id];
        image.tick_accessed_last = scheduler.CurrentTick();
        TouchImage(image);
        if (cached.view_mip > 0) {
            desc.view_info.range.base.level = cached.view_mip;
        }
        if (cached.view_slice > 0) {
            desc.view_info.range.base.layer = cached.view_slice;
        }
        return cached.image_id;
    }

    ImageIds image_ids;
    ForEachImageInRegion(info.guest_address, info.guest_size,
                         [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });

    ImageId image_id{};

    // Check for a perfect match first
    for (const auto& cache_id : image_ids) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    block_seed = {};
    if (!image_id) {
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (image_resolved.info.resources < info.resources) {
            // The image was clearly picked up wrong.
            FreeImage(image_id);
            image_id = {};
            LOG_WARNING(Render_Vulkan, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        static const bool bcfind = std::getenv("BB_BCVIEW_LOG") != nullptr;
        static int bcfind_n = 0;
        if (bcfind && info.props.is_block && bcfind_n < 300) {
            ++bcfind_n;
            std::fprintf(stderr, "[BCFIND] new %s %ux%u lv%u at %#llx gs=%llx, %zu overlaps\n",
                         vk::to_string(info.pixel_format).c_str(), info.size.width,
                         info.size.height, info.resources.levels,
                         (unsigned long long)info.guest_address,
                         (unsigned long long)info.guest_size, image_ids.size());
            for (const auto& id : image_ids) {
                const auto& o = slot_images[id].info;
                std::fprintf(stderr, "[BCFIND]   ov %s %ux%u lv%u at %#llx gs=%llx gpu=%d\n",
                             vk::to_string(o.pixel_format).c_str(), o.size.width, o.size.height,
                             o.resources.levels, (unsigned long long)o.guest_address,
                             (unsigned long long)o.guest_size,
                             (int)True(slot_images[id].flags & ImageFlagBits::GpuModified));
            }
        }
        image_id = slot_images.insert(device, runtime, slot_image_views, info);
        // bbport: the creating binding's lazy usages from the start (a texture first bound as
        // storage, or in another pixel format), instead of an upgrade copy at its first view.
        auto& created = slot_images[image_id];
        created.ApplyNewUsage(created.NeededUsage(desc.view_info));
        RegisterImage(image_id);
        if (block_seed && created.info.props.is_block) {
            Image& seed = slot_images[block_seed];
            static int probed = 0;
            const bool probe = ([] { static const bool v = std::getenv("BB_BCVIEW_LOG") != nullptr; return v; }()) && probed < 8;
            const auto halves = [&](Image& img) {
                const u32 bs = img.info.props.is_block ? 4 : 1;
                const u32 w = img.info.size.width, h = img.info.size.height;
                const u64 size = u64(w / bs) * (h / bs) * 16;
                std::array<vk::BufferImageCopy, 1> c{vk::BufferImageCopy{
                    .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                      .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                    .imageExtent = {w, h, 1}}};
                const auto dl = runtime.GetStagingPool().Request(size, MemoryType::HostCached, 16, false);
                c[0].bufferOffset = dl.offset;
                runtime.DownloadImage(&img, dl.buffer, c);
                scheduler.Finish();
                u64 lo = 0, hi = 0;
                const u8* b = static_cast<const u8*>(dl.mapped);
                for (u64 i = 0; i < size; ++i) {
                    ((i & 15) < 8 ? lo : hi) += b[i] != 0;
                }
                return std::pair{lo, hi};
            };
            if (probe) {
                auto [lo, hi] = halves(seed);
                std::fprintf(stderr, "[BCPROBE] seed %#llx lo=%llu hi=%llu\n",
                             (unsigned long long)seed.info.guest_address,
                             (unsigned long long)lo, (unsigned long long)hi);
            }
            runtime.CopyBlockBytes(&seed, &slot_images[image_id]);
            if (probe) {
                ++probed;
                auto [lo, hi] = halves(slot_images[image_id]);
                std::fprintf(stderr, "[BCPROBE] bc lo=%llu hi=%llu\n", (unsigned long long)lo,
                             (unsigned long long)hi);
            }
            static int seeded = 0;
            if (seeded++ < 40) {
                std::fprintf(stderr, "[BCSEED] %s %ux%u lv%u <- %s %ux%u lv%u at %#llx\n",
                             vk::to_string(info.pixel_format).c_str(), info.size.width,
                             info.size.height, info.resources.levels,
                             vk::to_string(seed.info.pixel_format).c_str(), seed.info.size.width,
                             seed.info.size.height, seed.info.resources.levels,
                             (unsigned long long)info.guest_address);
            }
            FreeImage(block_seed);
        }
        block_seed = {};
        // fifa-hybrid diagnostic: BB_IMG_FORMATS=1 prints each new (format, tiling) combination once.
        static const bool img_formats = std::getenv("BB_IMG_FORMATS") != nullptr;
        if (img_formats) {
            static std::mutex m;
            static std::set<std::tuple<u32, u32, u32, u32>> seen;
            std::scoped_lock lk{m};
            const auto key = std::make_tuple(u32(info.pixel_format), u32(info.tile_mode),
                                             u32(info.array_mode), u32(info.props.is_tiled));
            const bool lut_like = (info.size.width == 32 && (info.size.height == 32 || info.size.height == 1024)) ||
                                  (info.size.width == 1024 && info.size.height == 32);
            if (seen.insert(key).second || lut_like) {
                std::fprintf(stderr,
                             "IMGFMT %s type=%u tile=%u array=%u tiled=%d bits=%u %ux%ux%u vol=%d mips=%u layers=%u "
                             "addr=%#llx pitch=%u\n",
                             vk::to_string(info.pixel_format).c_str(), u32(info.type), u32(info.tile_mode),
                             u32(info.array_mode), int(info.props.is_tiled), info.num_bits,
                             info.size.width, info.size.height, info.size.depth,
                             int(info.props.is_volume), info.resources.levels,
                             info.resources.layers, (unsigned long long)info.guest_address,
                             info.pitch);
            }
        }
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    cached = FindImageCacheEntry{
        .address = info.guest_address,
        .size = info.guest_size,
        .extent = info.size,
        .format = info.pixel_format,
        .type = info.type,
        .exact_fmt = exact_fmt,
        .binding = desc.type,
        .levels = info.resources.levels,
        .layers = info.resources.layers,
        .generation = registry_generation.load(std::memory_order_relaxed),
        .image_id = image_id,
        .view_mip = view_mip,
        .view_slice = view_slice,
    };
    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    ImageIds image_ids;
    ForEachImageInRegion(address, size, [&](ImageId image_id, Image& image) {
        if (image.info.guest_address != address) {
            return;
        }
        if (ensure_valid && !image.SafeToDownload()) {
            return;
        }
        image_ids.push_back(image_id);
    });
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

ImageId TextureCache::ResolveBlockShadow(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    const auto& req = desc.info;
    if (image.info.props.is_block && !req.props.is_block && ([] { static const bool v = std::getenv("BB_BCVIEW_LOG") != nullptr; return v; }())) {
        static std::unordered_set<u64> seen;
        if (seen.insert(image.info.guest_address ^ (u64(req.resources.levels) << 56)).second) {
            std::fprintf(stderr, "[BCVIEW] bc %s %ux%u lv%u gs%#llx tile%d at %#llx <- req %s %ux%u lv%u base%u gs%#llx tile%d %s\n",
                         vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
                         image.info.size.height, image.info.resources.levels,
                         (unsigned long long)image.info.guest_size, int(image.info.tile_mode),
                         (unsigned long long)image.info.guest_address,
                         vk::to_string(req.pixel_format).c_str(), req.size.width, req.size.height,
                         req.resources.levels, desc.view_info.range.base.level,
                         (unsigned long long)req.guest_size, int(req.tile_mode),
                         desc.type == BindingType::Storage ? "storage" : "sampled");
        }
    }
    if (!image.info.props.is_block || req.props.is_block ||
        req.resources.layers != image.info.resources.layers ||
        req.resources.levels > image.info.resources.levels ||
        (([] { static const bool v = std::getenv("BB_SHADOW_SINGLE_MIP") != nullptr; return v; }()) && (req.resources.levels != 1 || image.info.resources.levels != 1)) ||
        req.guest_size > image.info.guest_size ||
        req.size.width != image.info.size.width / 4 ||
        req.size.height != image.info.size.height / 4 || ([] { static const bool v = std::getenv("BB_NO_BLOCK_SHADOW") != nullptr; return v; }())) {
        if (image.info.props.is_block && !req.props.is_block && ([] { static const bool v = std::getenv("BB_BCVIEW_LOG") != nullptr; return v; }())) {
            static int n = 0;
            if (n++ < 40) {
                std::fprintf(stderr, "[BCSKIP] %#llx ly %u/%u lv %u/%u gs %#llx/%#llx %ux%u/%ux%u\n",
                             (unsigned long long)image.info.guest_address, req.resources.layers,
                             image.info.resources.layers, req.resources.levels,
                             image.info.resources.levels, (unsigned long long)req.guest_size,
                             (unsigned long long)image.info.guest_size, req.size.width,
                             req.size.height, image.info.size.width, image.info.size.height);
            }
        }
        return image_id;
    }
    UpdateImage(image_id);
    if (image.block_shadow) {
        Image& shadow = slot_images[image.block_shadow];
        if (shadow.info.pixel_format != req.pixel_format ||
            shadow.info.resources.levels < req.resources.levels) {
            // The game grows the chain a level per dispatch: keep what the old shadow got.
            SyncBlockShadow(image);
            DeleteImage(image.block_shadow);
            image.block_shadow = {};
        }
    }
    if (!image.block_shadow) {
        ImageInfo info = req;
        info.guest_address = 0;
        const ImageId shadow_id = slot_images.insert(device, runtime, slot_image_views, info);
        Image& shadow = slot_images[shadow_id];
        shadow.shadow_of = image_id;
        shadow.flags &= ~ImageFlagBits::Dirty;
        slot_images[image_id].block_shadow = shadow_id;
        slot_images[image_id].shadow_newer = false;
        slot_images[image_id].shadow_current = false;
        ++registry_generation; // cached descriptor sets bound the image itself
        const auto& bc_info = slot_images[image_id].info;
        LOG_INFO(Render_Vulkan, "block shadow {} {}x{} for {} {}x{} at {:#x}",
                 vk::to_string(info.pixel_format), info.size.width, info.size.height,
                 vk::to_string(bc_info.pixel_format), bc_info.size.width, bc_info.size.height,
                 bc_info.guest_address);
    }
    Image& bc = slot_images[image_id];
    Image& shadow = slot_images[bc.block_shadow];
    if (!bc.shadow_newer && !bc.shadow_current) {
        if (([] { static const bool v = std::getenv("BB_BCVIEW_LOG") != nullptr; return v; }())) {
            static int n = 0;
            if (n++ < 200) {
                std::fprintf(stderr, "[BCCOPY] to-shadow %#llx %s\n",
                             (unsigned long long)bc.info.guest_address,
                             desc.type == BindingType::Storage ? "storage" : "sampled");
            }
        }
        runtime.CopyBlockBytes(&bc, &shadow);
        bc.shadow_current = true;
    }
    if (desc.type == BindingType::Storage) {
        bc.shadow_newer = true;
        bc.flags |= ImageFlagBits::GpuModified;
        // The CPU reads GPU-compressed textures back (kits): watch the BC image's memory.
        if (readback_linear_images && !([] { static const bool v = std::getenv("BB_NO_SHADOW_READBACK") != nullptr; return v; }())) {
            QueueReadback(image_id, bc);
        }
    }
    return bc.block_shadow;
}

void TextureCache::SyncBlockShadow(Image& image) {
    if (image.block_shadow && image.shadow_newer) {
        image.shadow_newer = false;
        image.shadow_current = true;
        if (([] { static const bool v = std::getenv("BB_BCVIEW_LOG") != nullptr; return v; }())) {
            static int n = 0;
            if (n++ < 200) {
                std::fprintf(stderr, "[BCCOPY] from-shadow %#llx\n",
                             (unsigned long long)image.info.guest_address);
            }
        }
        if (([] { static const bool v = std::getenv("BB_HALFSTAT") != nullptr; return v; }()) && slot_images[image.block_shadow].info.num_bits == 128) {
            static int hs = 0;
            auto stat = [&](Image& im, const char* tag) {
                const u32 w = im.info.size.width, h = im.info.size.height;
                const u32 bs = im.info.props.is_block ? 4 : 1;
                const u64 size = u64((w + bs - 1) / bs) * ((h + bs - 1) / bs) * 16;
                std::array<vk::BufferImageCopy, 1> c{vk::BufferImageCopy{
                    .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                                      .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
                    .imageExtent = {w, h, 1}}};
                const auto dl = runtime.GetStagingPool().Request(size, MemoryType::HostCached, 16, false);
                c[0].bufferOffset = dl.offset;
                runtime.DownloadImage(&im, dl.buffer, c);
                scheduler.Finish();
                u64 nz[4] = {};
                const u32* d = reinterpret_cast<const u32*>(dl.mapped);
                for (u64 i = 0; i < size / 4; ++i) nz[i & 3] += d[i] != 0;
                std::fprintf(stderr, "[HALFSTAT] %s %#llx %s %ux%u nz x=%llu y=%llu z=%llu w=%llu of %llu\n",
                             tag, (unsigned long long)image.info.guest_address,
                             vk::to_string(im.info.pixel_format).c_str(), w, h,
                             (unsigned long long)nz[0], (unsigned long long)nz[1],
                             (unsigned long long)nz[2], (unsigned long long)nz[3],
                             (unsigned long long)(size / 16));
            };
            std::fprintf(stderr, "[HALFSTAT] sync %#llx levels=%u\n", (unsigned long long)image.info.guest_address, image.info.resources.levels);
            if (hs++ < 40) {
                stat(slot_images[image.block_shadow], "uint");
                runtime.CopyBlockBytes(&slot_images[image.block_shadow], &image);
                stat(image, "bc");
            }
        }
        runtime.CopyBlockBytes(&slot_images[image.block_shadow], &image);
        // debug: BB_DUMP_SHADOWED=<dir> dumps GPU-compressed images after their first sync.
        if (([] { static const bool v = std::getenv("BB_DUMP_SHADOWED") != nullptr; return v; }())) {
            DebugDumpTexture(slot_images[image.block_shadow].shadow_of, true);
        }
    }
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc, ViewMemo* memo,
                                     bool refresh) {
    // fifa-hybrid: a depth view of a colour image reads its D32 twin, synced here.
    if (Image& twin_image = slot_images[image_id];
        desc.info.props.is_depth && !twin_image.has_depth && DepthTwinEnabled() &&
        twin_image.DepthTwinEligible()) {
        runtime.SyncDepthTwin(&twin_image);
        if (static const bool watch = std::getenv("BB_TWIN_WATCH") != nullptr; watch && twin_image.depth_twin) {
            const std::pair<ImageId, u64> key{image_id, twin_image.image_uid};
            if (std::ranges::find(twin_watch_ids, key) == twin_watch_ids.end()) {
                twin_watch_ids.push_back(key);
                std::fprintf(stderr, "[TWINWATCH] registered %#llx %ux%u serial %llu flags %#x\n",
                             (unsigned long long)twin_image.info.guest_address, twin_image.info.size.width,
                             twin_image.info.size.height, (unsigned long long)twin_image.write_serial,
                             u32(twin_image.flags));
            }
        }
    }

    Image& image = slot_images[image_id];
    if (image.shadow_of) {
        // A shadow has no guest memory of its own: synced by ResolveBlockShadow.
        return image.FindView(desc.view_info);
    }
    SyncBlockShadow(image);
    if (desc.type == BindingType::Storage) {
        if (([] { static const bool v = std::getenv("BB_BCVIEW_LOG") != nullptr; return v; }()) &&
            (image.info.pixel_format == vk::Format::eR32G32B32A32Uint ||
             image.info.pixel_format == vk::Format::eR32G32Uint)) {
            static std::set<std::tuple<u64, u32, u32>> seen;
            if (seen.insert({image.info.guest_address, u32(desc.info.pixel_format),
                             desc.view_info.range.base.level}).second && seen.size() < 80) {
                std::fprintf(stderr, "[UINTW] %s %ux%u lv%u at %#llx view %s %ux%u mip%u\n",
                             vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
                             image.info.size.height, image.info.resources.levels,
                             (unsigned long long)image.info.guest_address,
                             vk::to_string(desc.info.pixel_format).c_str(), desc.info.size.width,
                             desc.info.size.height, desc.view_info.range.base.level);
            }
        }
        if (image.info.props.is_block && ([] { static const bool v = std::getenv("BB_BCVIEW_LOG") != nullptr; return v; }())) {
            static std::unordered_set<u64> seen;
            if (seen.insert(image.info.guest_address).second) {
                std::fprintf(stderr, "[BCGPU] storage %s %ux%u lv%u at %#llx view %s\n",
                             vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
                             image.info.size.height, image.info.resources.levels,
                             (unsigned long long)image.info.guest_address,
                             vk::to_string(desc.info.pixel_format).c_str());
            }
        }
        image.flags |= ImageFlagBits::GpuModified;
    }
    if (refresh) {
        UpdateImage(image_id);
    }
    if (desc.type == BindingType::Storage && readback_linear_images &&
        image.info.guest_address != 0 && ShouldReadbackImage(image)) {
        QueueReadback(image_id, image); // after the refresh, see FindRenderTarget
    }
    if (memo && !BbToggle::Disabled(BbToggle::TextureViewMemo)) {
        if (memo->image_id == image_id && memo->backing == image.backing && memo->view_id) {
            return slot_image_views[memo->view_id];
        }
        ImageView& view = image.FindView(desc.view_info);
        const auto& ids = image.backing->image_view_ids;
        const u32 last = image.backing->last_view;
        memo->image_id = image_id;
        memo->backing = image.backing;
        memo->view_id = last < ids.size() && &slot_image_views[ids[last]] == &view ? ids[last]
                                                                                    : ids.back();
        return view;
    }
    return image.FindView(desc.view_info);
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    if (image.info.props.is_block && ([] { static const bool v = std::getenv("BB_BCVIEW_LOG") != nullptr; return v; }())) {
        static std::unordered_set<u64> seen;
        if (seen.insert(image.info.guest_address).second) {
            std::fprintf(stderr, "[BCGPU] rt %s %ux%u at %#llx\n",
                         vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
                         image.info.size.height, (unsigned long long)image.info.guest_address);
        }
    }
    image.flags |= ImageFlagBits::GpuModified;
    image.usage.render_target = 1u;
    UpdateImage(image_id);
    // Queued after the refresh: RefreshImage writes back pending on-demand readbacks in the
    // image's range, which would otherwise download this image before the draw and drop the
    // watch, so the CPU later read the pre-draw contents.
    if (readback_linear_images && ShouldReadbackImage(image)) {
        QueueReadback(image_id, image);
    }

    // Register meta data for this color buffer
    if (desc.info.meta_info.cmask_addr) {
        surface_metas.emplace(desc.info.meta_info.cmask_addr,
                              MetaDataInfo{.type = MetaType::CMask});
        image.info.meta_info.cmask_addr = desc.info.meta_info.cmask_addr;
    }

    if (desc.info.meta_info.fmask_addr) {
        surface_metas.emplace(desc.info.meta_info.fmask_addr,
                              MetaDataInfo{.type = MetaType::FMask});
        image.info.meta_info.fmask_addr = desc.info.meta_info.fmask_addr;
    }

    return image.FindView(desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc, bool gl_depth) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    image.usage.depth_target = 1u;
    if (static const bool remap_log = std::getenv("BB_REMAP_LOG") != nullptr;
        remap_log && image.gl_depth != gl_depth) {
        std::fprintf(stderr, "[REMAP] depth target %#llx %ux%u gl_depth %d -> %d\n",
                     (unsigned long long)image.info.guest_address, image.info.size.width,
                     image.info.size.height, int(image.gl_depth), int(gl_depth));
    }
    // Sticky: a shadow map is cleared by a draw under the default clip space and then rendered
    // under the GL-style one (the flag flipped every frame). The remap maps the clear values to
    // the right extremes too (1 -> 1, 0 -> -1), so once GL-style the image stays flagged.
    if (gl_depth) {
        image.gl_depth = true;
    }
    UpdateImage(image_id);

    // Register meta data for this depth buffer
    if (desc.info.meta_info.htile_addr) {
        surface_metas.emplace(desc.info.meta_info.htile_addr,
                              MetaDataInfo{.type = MetaType::HTile,
                                           .clear_mask = image.info.meta_info.htile_clear_mask});
        image.info.meta_info.htile_addr = desc.info.meta_info.htile_addr;
    }

    // If there is a stencil attachment, link depth and stencil.
    if (desc.info.stencil_addr != 0) {
        ImageId stencil_id{};
        ForEachImageInRegion(desc.info.stencil_addr, desc.info.stencil_size,
                             [&](ImageId image_id, Image& image) {
                                 if (image.info.guest_address == desc.info.stencil_addr) {
                                     stencil_id = image_id;
                                 }
                             });
        if (!stencil_id) {
            ImageInfo info{};
            info.guest_address = desc.info.stencil_addr;
            info.guest_size = desc.info.stencil_size;
            info.size = desc.info.size;
            stencil_id = slot_images.insert(device, runtime, slot_image_views, info);
            RegisterImage(stencil_id);
        }
        Image& stencil_image = slot_images[stencil_id];
        TouchImage(stencil_image);
        stencil_image.AssociateDepth(image_id, image.image_uid);
    }

    return image.FindView(desc.view_info, false);
}

void TextureCache::RefreshImage(Image& image) {
    if (False(image.flags & ImageFlagBits::Dirty)) {
        return;
    }
    if (static const bool slog = std::getenv("BB_STENCIL_LOG") != nullptr; slog && image.info.props.is_depth) {
        std::fprintf(stderr, "[SREFRESH] depth image %#llx %ux%u refreshed (dirty)\n",
                     (unsigned long long)image.info.guest_address, image.info.size.width, image.info.size.height);
    }
    // fifa-hybrid diagnostics: BB_REFRESH_LOG=1 also prints every refresh of a small R32 image
    // (the CPU-filled shadow maps) with its flags.
    if (static const bool rlog2 = std::getenv("BB_REFRESH_LOG") != nullptr;
        rlog2 && image.info.pixel_format == vk::Format::eR32Sfloat && image.info.size.width <= 1024) {
        static const auto t0 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[R32UP] t=%.1f image %#llx %ux%u flags %#x refresh\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                     (unsigned long long)image.info.guest_address, image.info.size.width,
                     image.info.size.height, u32(image.flags));
    }
    // fifa-hybrid diagnostics: BB_REFRESH_LOG=1 prints every upload over a GPU-written image (guest
    // memory replacing GPU results: suspicious for render targets such as shadow maps).
    if (static const bool rlog = std::getenv("BB_REFRESH_LOG") != nullptr;
        rlog && (True(image.flags & ImageFlagBits::GpuModified) ||
                 (True(image.flags & ImageFlagBits::GpuDirty) &&
                  image.info.pixel_format == vk::Format::eR32Sfloat))) {
        static const auto t0 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "[REFRESH] t=%.1f gpu-written image %#llx %ux%u %s dirty-> upload (samples %u)\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                     (unsigned long long)image.info.guest_address, image.info.size.width,
                     image.info.size.height, vk::to_string(image.info.pixel_format).c_str(),
                     image.info.num_samples);
    }
    if (image.block_shadow && ([] { static const bool v = std::getenv("BB_BCVIEW_LOG") != nullptr; return v; }())) {
        std::fprintf(stderr, "[BCREFRESH] addr=%#llx %ux%u newer=%d cur=%d flags=%#x\n",
                     (unsigned long long)image.info.guest_address, image.info.size.width,
                     image.info.size.height, (int)image.shadow_newer, (int)image.shadow_current,
                     (unsigned)image.flags);
    }
    // New guest blocks replace the image's: the shadow copies them on its next bind.
    image.shadow_current = false;
    image.shadow_newer = false;
    if (image.info.num_samples > 1) {
        // fifa-hybrid: multisample images can't be uploaded, and a new Metal texture holds
        // whatever was in that memory before (Vulkan drivers tend to hand out zeroed memory),
        // which may show as colour noise / NaNs where the game blends into it. Start
        // never-written colour ones from zero (BB_NO_MSAA_ZERO=1 disables).
        static const bool no_zero = std::getenv("BB_NO_MSAA_ZERO") != nullptr;
        if (!no_zero && False(image.flags & ImageFlagBits::GpuModified) &&
            !image.info.props.is_depth) {
            vk::ClearValue zero{};
            zero.color = vk::ClearColorValue{std::array<float, 4>{0.f, 0.f, 0.f, 0.f}};
            runtime.ClearImage(&image, {.base = {}, .extent = image.info.resources}, zero);
        }
        return;
    }
    BbStats::Timer timer{BbStats::t_refresh};

    // fifa-hybrid: the upload reads guest memory on the GPU side (no CPU fault), so GPU-written
    // images awaiting an on-demand readback in this range are written back first.
    if (HasOndemandIn(image.info.guest_address, image.info.guest_size)) {
        static int rn = 0;
        if (([] { static const bool v = std::getenv("BB_VB_LOG") != nullptr; return v; }()) && rn++ < 2000) {
            std::fprintf(stderr, "[RBQ] refresh-resolve addr=%#llx size=%#llx %ux%u\n",
                         (unsigned long long)image.info.guest_address,
                         (unsigned long long)image.info.guest_size, image.info.size.width,
                         image.info.size.height);
        }
        if (!([] { static const bool v = std::getenv("BB_RB_NOREFRESHRESOLVE") != nullptr; return v; }())) {
            DownloadImagesForCpuRead(image.info.guest_address, image.info.guest_size, false, true);
        }
    }

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        const u64 hash = MaybeDirtyHash(image);
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);

    BbStats::image_upload_bytes.fetch_add(image.info.guest_size, std::memory_order_relaxed);
    boost::container::small_vector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];

        // Protect GPU modified resources from accidental CPU reuploads.
        // bbport: an image only ever filled from GPU buffer writes (the 1080p R32F one the game
        // writes with a compute shader every frame: 9 MB) skips the CPU hash. The hash is only
        // compared once the image is GPU-modified, and here it would read memory whose GPU write
        // may still be pending; 0 makes that later comparison upload, as a changed hash would.
        const u8* mip_addr = std::bit_cast<u8*>(image.info.guest_address) + mip_offset;
        const u64 mip_hash =
            is_gpu_dirty && !is_gpu_modified ? 0 : XXH3_64bits(mip_addr, mip_size);
        if (is_gpu_modified && !is_gpu_dirty && image.mip_hashes[m] == mip_hash) {
            continue;
        }
        image.mip_hashes[m] = mip_hash;

        const u32 extent_width = mip_pitch ? std::min(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }

    scheduler.EndRendering();

    // bbport: BB_LOG_UPLOAD_MB=<n> logs every image refresh of at least n MB (diagnostics).
    static const double log_upload_mb = [] {
        const char* env = std::getenv("BB_LOG_UPLOAD_MB");
        return env ? std::atof(env) : 0.0;
    }();
    if (log_upload_mb > 0 && image.info.guest_size >= log_upload_mb * 1e6) {
        LOG_WARNING(Render_Vulkan,
                    "Image refresh {:#x} size {:#x}: {}x{} format {} layers {} mips {} ({} "
                    "copied), tiled {}, cpu dirty {}, maybe cpu dirty {}, gpu modified {}, gpu "
                    "dirty {}, region gpu modified {}",
                    image.info.guest_address, image.info.guest_size, image.info.size.width,
                    image.info.size.height, static_cast<u32>(image.info.pixel_format), num_layers,
                    num_mips, image_copies.size(), bool(image.info.props.is_tiled),
                    True(image.flags & ImageFlagBits::CpuDirty),
                    True(image.flags & ImageFlagBits::MaybeCpuDirty), is_gpu_modified,
                    is_gpu_dirty,
                    buffer_cache.IsRegionGpuModified(image.info.guest_address,
                                                     image.info.guest_size));
    }

    // fifa-hybrid diagnostic: BB_DUMP_VOLUMES=<dir> writes the raw guest bytes of each volume
    // image refresh (first 8) as <dir>/vol_<addr>_<w>x<h>x<d>_t<tile>.bin.
    static const char* dump_volumes = std::getenv("BB_DUMP_VOLUMES");
    if (dump_volumes && image.info.props.is_volume) {
        static std::atomic<int> dumped{0};
        if (dumped.fetch_add(1) < 8) {
            const auto path = fmt::format("{}/vol_{:x}_{}x{}x{}_t{}.bin", dump_volumes,
                                          image.info.guest_address, image.info.size.width,
                                          image.info.size.height, image.info.size.depth,
                                          u32(image.info.tile_mode));
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(std::bit_cast<u8*>(image.info.guest_address), 1,
                            image.info.guest_size, f);
                std::fclose(f);
            }
        }
    }

    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    const auto [buffer, offset] = tile_manager.DetileImage(in_buffer, in_offset, image.info);
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    runtime.UploadImage(&image, buffer, image_copies);
}

void* TextureCache::GetSampler(const AmdGpu::Sampler& sampler,
                                     AmdGpu::BorderColorBuffer border_color_base,
                                     const bool is_depth, float extra_lod_bias) {
    // Compare and plain uses of one S# need separate samplers; so do extra LOD biases.
    const u64 hash = HashCombine(HashCombine(XXH3_64bits(&sampler, sizeof(sampler)), is_depth),
                                 u64(std::bit_cast<u32>(extra_lod_bias)));

    std::scoped_lock lock{samplers_mutex};
    const auto [it, new_sampler] = samplers.try_emplace(hash, device, sampler, border_color_base,
                                                        is_depth, extra_lod_bias);
    if (new_sampler) {
        samplers.at(hash).lru_id = sampler_lru_cache.Insert(hash, gc_tick);
    } else {
        sampler_lru_cache.Touch(it->second.lru_id, gc_tick);
    }

    return it->second.Handle();
}

void TextureCache::RegisterImage(ImageId image_id) {
    BbStats::images_registered.fetch_add(1, std::memory_order_relaxed);
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    ++registry_generation;
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    image.lru_id = lru_cache.Insert(image_id, gc_tick);
    image.lru_touched_tick = gc_tick;
    ForEachPage(image.info.guest_address, image.info.guest_size,
                [this, image_id](u64 page) { page_table[page].push_back(image_id); });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    image.flags &= ~ImageFlagBits::Registered;
    ++registry_generation;
    lru_cache.Free(image.lru_id);
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == nullptr) {
            UNREACHABLE_MSG("Unregistering unregistered page=0x{:x}", page << PageShift);
            return;
        }
        auto& image_ids = *page_it;
        const auto vector_it = std::ranges::find(image_ids, image_id);
        if (vector_it == image_ids.end()) {
            ASSERT_MSG(false, "Unregistering unregistered image in page=0x{:x}", page << PageShift);
            return;
        }
        image_ids.erase(vector_it);
    });
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    if (!image.IsTracked()) {
        // Re-track the whole image
        image.track_addr = image_begin;
        image.track_addr_end = image_end;
        tracker.UpdatePageWatchers<1>(image_begin, image.info.guest_size);
    } else {
        if (image_begin < image.track_addr) {
            TrackImageHead(image_id);
        }
        if (image.track_addr_end < image_end) {
            TrackImageTail(image_id);
        }
    }
}

void TextureCache::TrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    if (image_begin == image.track_addr) {
        return;
    }
    ASSERT(image.track_addr != 0 && image_begin < image.track_addr);
    const auto size = image.track_addr - image_begin;
    image.track_addr = image_begin;
    tracker.UpdatePageWatchers<1>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_end == image.track_addr_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0 && image.track_addr_end < image_end);
    const auto addr = image.track_addr_end;
    const auto size = image_end - image.track_addr_end;
    image.track_addr_end = image_end;
    tracker.UpdatePageWatchers<1>(addr, size);
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!image.IsTracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    image.track_addr = 0;
    image.track_addr_end = 0;
    if (size != 0) {
        tracker.UpdatePageWatchers<false>(addr, size);
    }
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (!image.IsTracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = tracker.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers<false>(image_begin, size);
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (!image.IsTracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = tracker.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers<false>(addr, size);
}

void TextureCache::GarbageCollectImages() {
    { // Metal always reports usage (Device::AllocatedSize)
        // Without the resident buffers: the guest arena (6 GiB, private) counts in full from its
        // creation, which kept a 16 GB Mac above the trigger and evicted textures for nothing.
        // The budget below leaves them out instead.
        const u64 allocated = device.AllocatedSize();
        const u64 resident = device.ResidentBytes();
        total_used_memory = allocated > resident ? allocated - resident : 0;
        BbStats::gc_allocated_bytes.store(allocated, std::memory_order_relaxed);
        BbStats::gc_resident_bytes.store(resident, std::memory_order_relaxed);
        // bbport: on integrated GPUs (Steam Deck) the usage covers system-memory heaps holding
        // much more than images (buffers backing guest memory), and the startup budget left
        // ~1 GB after its 8 GB system reserve: usage stayed above the critical mark, so the
        // collector evicted images used two or three frames ago on every submission and wrote
        // GPU-written ones back. Compare with the driver's current budget instead.
        // BB_GC_BUDGET_MB=N: this rule with a fixed budget on any GPU (tests on a desktop).
        static const u64 forced_budget = [] {
            const char* env = std::getenv("BB_GC_BUDGET_MB");
            return env ? std::strtoull(env, nullptr, 10) << 20 : 0;
        }();
        u64 budget = forced_budget ? forced_budget : device.GetLimits().recommended_working_set;
        const u64 local_mem = device.GetLimits().recommended_working_set;
        if (budget != 0 || local_mem != 0) {
            const bool is_low_vram = (local_mem > 0 && local_mem <= 4096ULL * 1024 * 1024) ||
                                     (budget > 0 && budget <= 4096ULL * 1024 * 1024);
            u64 base_mem = !forced_budget && is_low_vram && local_mem > 0 ? local_mem : budget;
            // bbport: the resident buffers (the guest arena above all) count toward Metal's
            // working set too, so images get what they leave. Measured against the whole set,
            // images could grow to ~8 GB beside an 8 GiB arena on a 16 GB Mac and the GPU
            // process sat at Metal's limit (11.53 of 11.55 GB, issue #19). BB_GC_BUDGET_MB
            // stands for the working set (BB_GC_BUDGET_MB=11015: a 16 GB Mac's).
            // The guest memory buffers count as the previous layout made them (the arena and the
            // direct memory access tables): memory a leaner layout frees goes to the rest of the
            // Mac, not to images. Other resident buffers (object motion) count as they are.
            // BB_GC_RESIDENT_REF=live: the resident bytes as they are (the marks then rise by
            // 70-88% of what the layout frees).
            static const bool live_reference = [] {
                const char* e = std::getenv("BB_GC_RESIDENT_REF");
                return e && std::string_view{e} == "live";
            }();
            const u64 layout = BbStats::layout_resident_bytes.load(std::memory_order_relaxed);
            const u64 legacy = BbStats::legacy_resident_bytes.load(std::memory_order_relaxed);
            const u64 charged =
                live_reference || resident < layout ? resident : resident - layout + legacy;
            static bool marks_logged = false;
            base_mem = base_mem > charged + (2ULL << 30) ? base_mem - charged : 2ULL << 30;
            static const u64 trig_pct = [is_low_vram] {
                const char* e = std::getenv("BB_GC_TRIGGER");
                return e ? std::strtoull(e, nullptr, 10) : (is_low_vram ? 70 : 70);
            }();
            static const u64 pres_pct = [is_low_vram] {
                const char* e = std::getenv("BB_GC_PRESSURE");
                return e ? std::strtoull(e, nullptr, 10) : (is_low_vram ? 90 : 80);
            }();
            static const u64 crit_pct = [is_low_vram] {
                const char* e = std::getenv("BB_GC_CRITICAL");
                return e ? std::strtoull(e, nullptr, 10) : (is_low_vram ? 95 : 88);
            }();
            trigger_gc_memory = base_mem / 100 * trig_pct;
            pressure_gc_memory = base_mem / 100 * pres_pct;
            critical_gc_memory = base_mem / 100 * crit_pct;
            if (!marks_logged && layout != 0) {
                marks_logged = true;
                std::printf("Texture budget: resident buffers charged as %.0f MiB (%.0f resident, "
                            "%s); marks %.0f / %.0f / %.0f MiB\n",
                            charged / 1048576.0, resident / 1048576.0,
                            live_reference ? "BB_GC_RESIDENT_REF=live"
                                           : "the guest memory buffers as the previous layout",
                            trigger_gc_memory / 1048576.0, pressure_gc_memory / 1048576.0,
                            critical_gc_memory / 1048576.0);
            }
        }
    }
    BbStats::gc_used_bytes.store(total_used_memory, std::memory_order_relaxed);
    BbStats::gc_trigger_bytes.store(trigger_gc_memory, std::memory_order_relaxed);
    // bbport: gc_tick (one per guest submission, hundreds a second) at each of the last 64
    // seconds, for ages in seconds. The usage compared is all of our VRAM, not only images: on a
    // discrete GPU it stays over the trigger, where an age of 16 ticks (~3 frames) evicted every
    // texture briefly off screen and uploaded it again. Below the pressure mark an image goes
    // once unused for BB_GC_IDLE_SECONDS (20): the textures of an area left behind.
    const u64 second = u64(std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count());
    if (second != gc_second) {
        gc_second = second;
        BbStats::coarse_second.store(u32(second), std::memory_order_relaxed);
        gc_tick_at_second[second % gc_tick_at_second.size()] = gc_tick;
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    static const u64 idle_seconds = [] {
        const char* env = std::getenv("BB_GC_IDLE_SECONDS");
        return std::clamp<u64>(env ? std::strtoull(env, nullptr, 10) : 20, 1, 63);
    }();
    // The tick at the start of that second (an older one where no submission came then).
    const u64 idle_tick =
        gc_tick_at_second[(second - idle_seconds) % gc_tick_at_second.size()];
    const u64 pressure_seconds = std::clamp<u64>(idle_seconds / 4, 3ULL, 5ULL);
    const u64 raw_press_tick =
        gc_tick_at_second[(second - pressure_seconds) % gc_tick_at_second.size()];
    const u64 pressure_tick = raw_press_tick ? raw_press_tick : (gc_tick > 300 ? gc_tick - 300 : 0);
    const u64 raw_crit_tick =
        gc_tick_at_second[(second - 1) % gc_tick_at_second.size()];
    const u64 critical_tick = raw_crit_tick ? raw_crit_tick : (gc_tick > 120 ? gc_tick - 120 : 0);

    std::scoped_lock lock{mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 below_tick = 0;
    size_t num_deletions = 0;
    u32 visited = 0;
    u32 sync_downloads_this_pass = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggresive = allow_aggressive && total_used_memory >= critical_gc_memory;
        if (aggresive) {
            below_tick = critical_tick;
            num_deletions = 16;
        } else if (pressured) {
            below_tick = pressure_tick;
            num_deletions = 6;
        } else {
            below_tick = idle_tick;
            num_deletions = 4;
        }
        visited = 0;
    };
    const auto clean_up = [&](ImageId image_id) {
        if (num_deletions == 0 || ++visited > 256) {
            return true;
        }
        auto& image = slot_images[image_id];
        const bool download = image.SafeToDownload();
        const bool tiled = image.info.IsTiled();
        // bbport: images that cannot go now (GPU-written: their contents exist only here, and
        // tiled ones cannot be written back at all) neither use up the deletions nor stay first
        // in line. The oldest ten being such render targets stopped the collector for good, and
        // VRAM grew with every area visited up to the pressure mark (~10 GB on 16 GB).
        if ((tiled && download) || (download && !pressured)) {
            lru_cache.Touch(image.lru_id, gc_tick);
            BbStats::gc_kept_images.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (download) {
            if (sync_downloads_this_pass >= 1) {
                // Rate-limit synchronous GPU flushes to at most 1 per submission pass
                // to prevent multiple consecutive pipeline stalls and frame hitches.
                lru_cache.Touch(image.lru_id, gc_tick);
                return false;
            }
            // bbport: synchronously, while the image still protects its pages. A deferred
            // write-back landed after FreeImage had unprotected them, over whatever the game
            // had mapped there since (0x28ce9b5).
            DownloadImageMemory(image_id, true);
            ++gc_downloads;
            ++sync_downloads_this_pass;
        }
        --num_deletions;
        ++gc_evictions;
        BbStats::gc_freed_images.fetch_add(1, std::memory_order_relaxed);
        FreeImage(image_id);
        if (total_used_memory < critical_gc_memory) {
            if (aggresive) {
                num_deletions >>= 2;
                aggresive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    lru_cache.ForEachItemBelow(below_tick, clean_up);

    if (total_used_memory >= critical_gc_memory) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        lru_cache.ForEachItemBelow(below_tick, clean_up);
    }
    // bbport: evictions under memory pressure, at most every 5 s (BB_FRAME_STATS or not).
    if (pressured || gc_downloads != 0) {
        const auto now = std::chrono::steady_clock::now();
        if (now - gc_report_time >= std::chrono::seconds(5)) {
            std::printf("Texture cache: memory pressure, %llu of %llu MiB (critical %llu): "
                        "%llu images evicted, %llu written back since the last report\n",
                        (unsigned long long)(total_used_memory >> 20),
                        (unsigned long long)(pressure_gc_memory >> 20),
                        (unsigned long long)(critical_gc_memory >> 20),
                        (unsigned long long)gc_evictions, (unsigned long long)gc_downloads);
            gc_report_time = now;
            gc_evictions = gc_downloads = 0;
        }
    }
}

void TextureCache::GarbageCollectSamplers() {
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }
    // Command buffers do not retain what they use: evicted samplers live until the GPU passed
    // the current tick (deferred below, outside samplers_mutex).
    std::vector<Sampler> retired;
    SCOPE_EXIT {
        if (!retired.empty()) {
            scheduler.DeferOperation([retired = std::move(retired)] {});
        }
    };
    std::scoped_lock lock{samplers_mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggresive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](u64 hash) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        Sampler& sampler = samplers.at(hash);
        const size_t lru_id = sampler.lru_id;
        retired.push_back(std::move(sampler));
        samplers.erase(hash);
        sampler_lru_cache.Free(lru_id);
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_samplers >= critical_gc_samplers) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };

    GarbageCollectImages();
    GarbageCollectSamplers();
}

void TextureCache::TouchImage(const Image& image) {
    if (image.lru_touched_tick == gc_tick) {
        return;
    }
    image.lru_touched_tick = gc_tick;
    lru_cache.Touch(image.lru_id, gc_tick);
}

void TextureCache::DeleteImage(ImageId image_id) {
    // fifa-hybrid: an image deleted while awaiting an on-demand readback (garbage collection,
    // overlap resolution) writes its GPU contents back first; the CPU (FIFA's crowd colours)
    // reads them after the image is gone, when no read watcher is left to fault.
    bool pending;
    {
        std::unique_lock lk{download_images_mutex};
        pending = ondemand_images.contains(image_id);
    }
    if (pending) {
        DownloadImageMemory(image_id, false);
    }
    ForgetOndemandImage(image_id);
    Image& image = slot_images[image_id];
    if (image.block_shadow) {
        const ImageId shadow_id = image.block_shadow;
        image.block_shadow = {};
        DeleteImage(shadow_id);
    }
    ASSERT_MSG(!image.IsTracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        surface_metas.erase(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        surface_metas.erase(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        surface_metas.erase(meta_info.htile_addr);
    }

    {
        std::unique_lock lk{download_images_mutex};
        if (download_images.contains(image_id)) {
            download_images.erase(image_id);
        }
    }

    // Reclaim image and any image views it references.
    scheduler.DeferOperation([this, image_id] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.erase(image_view_id);
            }
        }
        slot_images.erase(image_id);
    });
}

} // namespace VideoCore
