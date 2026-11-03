// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <dlfcn.h>
#include <array>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include "bbport_toggles.h"
#include "common/alignment.h"
#include "common/div_ceil.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_metal/mtl_encoder.h"
#include "video_core/renderer_metal/mtl_texture.h"
#include "video_core/renderer_metal/vk_to_mtl.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "video_core/renderer_vulkan/vk_scene_resolution.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"

#include <vulkan/vulkan_format_traits.hpp>

namespace Vulkan {

static vk::ImageType ConvertImageType(AmdGpu::ImageType type) noexcept {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
        return vk::ImageType::e1D;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DArray:
        return vk::ImageType::e2D;
    case AmdGpu::ImageType::Color3D:
        return vk::ImageType::e3D;
    default:
        UNREACHABLE_MSG("Unexpected image type {}", u32(type));
    }
}

static std::pair<u32, u32> SanitizeCopyLayers(const VideoCore::ImageInfo& src_info,
                                              const VideoCore::ImageInfo& dst_info,
                                              const u32 depth) {
    const auto vk_src_type = ConvertImageType(src_info.type);
    const auto vk_dst_type = ConvertImageType(dst_info.type);

    u32 src_layers = src_info.resources.layers;
    u32 dst_layers = dst_info.resources.layers;

    // 3D images can only use 1 layer.
    if (vk_src_type == vk::ImageType::e3D && src_layers != 1) {
        src_layers = 1;
    }
    if (vk_dst_type == vk::ImageType::e3D && dst_layers != 1) {
        dst_layers = 1;
    }

    // If the image type is equal, layer count must match. Take the minimum of both.
    if (vk_src_type == vk_dst_type) {
        if (src_layers != dst_layers) {
            src_layers = dst_layers = std::min(src_layers, dst_layers);
        }
    } else {
        // For 2D <-> 3D copies, 2D layer count must equal 3D depth.
        if (vk_src_type == vk::ImageType::e2D && vk_dst_type == vk::ImageType::e3D &&
            src_layers != depth) {
            src_layers = depth;
        }
        if (vk_src_type == vk::ImageType::e3D && vk_dst_type == vk::ImageType::e2D &&
            dst_layers != depth) {
            dst_layers = depth;
        }
    }

    return std::make_pair(src_layers, dst_layers);
}

static u32 BufferImageCopySize(const vk::BufferImageCopy& copy, const vk::Format pixel_format) {
    const u32 row_length = copy.bufferRowLength ? copy.bufferRowLength : copy.imageExtent.width;
    const u32 height = copy.bufferImageHeight ? copy.bufferImageHeight : copy.imageExtent.height;

    const auto block = vk::blockExtent(pixel_format);
    const u32 block_size = vk::blockSize(pixel_format);
    const u32 row_pitch = (row_length / block[0]) * block_size;
    const u32 slice_pitch = (height / block[1]) * row_pitch;

    const u32 width_in_blocks = (copy.imageExtent.width + block[0] - 1) / block[0];
    const u32 height_in_blocks = (copy.imageExtent.height + block[1] - 1) / block[1];
    const u32 num_slices = copy.imageExtent.depth * copy.imageSubresource.layerCount;

    return (num_slices - 1) * slice_pitch + (height_in_blocks - 1) * row_pitch +
           width_in_blocks * block_size;
}

Runtime::Runtime(Metal::Device& device_, Scheduler& scheduler_)
    : ordering{!device_.HazardTracking()}, device{device_}, scheduler{scheduler_},
      staging_pool{device_, scheduler_} {
    // Device-local staging (texture copies through buffers, tiling) is ordered like the arena.
    staging_pool.SetDeviceLocalHook([this](const VideoCore::Buffer* buffer, u64 size, bool alive) {
        if (alive) {
            RegisterOrdered(buffer, size);
        } else {
            UnregisterOrdered(buffer);
        }
    });
    blit_helper = std::make_unique<VideoCore::BlitHelper>(device, scheduler);
}

namespace {
/// The rest of the current command waits for its own copies so far: the uploads of a staging
/// round trip read what its downloads wrote (staging is ordered by the Runtime; tracked
/// buffers: Metal orders the encoders).
void WaitForOwnCopies(Metal::Encoder& e) {
    Metal::OrderRequest own;
    own.copy.Add(e.OrderCommand());
    e.OrderWait(own);
}
} // namespace

// 1 KiB pages; the tag is the low 16 bits of the submission tick. A tag that wraps around can only
// make a page look used (no hoist), never unused.
static constexpr u64 HoistPageBits = 10;

void Runtime::RegisterHoistTarget(const VideoCore::Buffer* buffer, u64 size) {
    const u64 pages = (size + (1ull << HoistPageBits) - 1) >> HoistPageBits;
    TagArray tick{static_cast<u16*>(std::calloc(pages, sizeof(u16)))};
    if (tick) {
        hoist_targets[buffer] = {std::move(tick), pages};
    } else {
        LOG_ERROR(Render, "No memory for the upload hoisting tags of {} bytes: no hoisting", size);
    }
    RegisterOrdered(buffer, size);
}

void Runtime::RegisterOrdered(const VideoCore::Buffer* buffer, u64 size) {
    if (ordering) {
        barriers.Register(buffer, size);
        device.OrderByCaller(buffer->Handle());
    }
}

void Runtime::UnregisterOrdered(const VideoCore::Buffer* buffer) {
    if (ordering) {
        device.OrderByCaller(buffer->Handle(), false);
        barriers.Unregister(buffer);
    }
}

void Runtime::BeginStagingRoundTrip(const VideoCore::Buffer* staging, u64 offset, u64 size) {
    // A staging range written by texture downloads, then read by uploads in the same recorded
    // command (with a barrier between them): after the last users of the range.
    FlushBarriers(IsBufferAccessed(staging, offset, size, true));
}

void Runtime::EndStagingRoundTrip(const VideoCore::Buffer* staging, u64 offset, u64 size) {
    AccessBuffer(staging, offset, size, true);
    AccessBuffer(staging, offset, size, false);
}

Metal::OrderRequest Runtime::IsBufferAccessed(const VideoCore::Buffer* handle, u64 offset,
                                              u64 size, bool write) const {
    return ordering ? barriers.Conflicts(handle, offset, size, write) : Metal::OrderRequest{};
}

u64 Runtime::BeginCommand() {
    // Commands numbered while the scheduler's tick was `command_tick` are in that submission.
    const u64 tick = scheduler.CurrentTick();
    if (tick != command_tick) {
        if (barriers.Command()) {
            submitted_commands.emplace_back(command_tick, barriers.Command());
        }
        command_tick = tick;
    }
    const u64 known = scheduler.KnownGpuTick();
    while (!submitted_commands.empty() && submitted_commands.front().first <= known) {
        barriers.Retire(submitted_commands.front().second);
        submitted_commands.pop_front();
    }
    const u64 command = barriers.BeginCommand();
    scheduler.SetOrderCommand(command);
    return command;
}

void Runtime::FlushBarriers(const Metal::OrderRequest& need) {
    if (!ordering) {
        return;
    }
    BeginCommand();
    if (need.Empty()) {
        return;
    }
    ((need.Scopes() & (Metal::ScopeRenderEnd | Metal::ScopeRenderVertex))
         ? BbStats::arena_barriers
         : BbStats::arena_other_barriers)
        .fetch_add(1, std::memory_order_relaxed);
    scheduler.Record([need](Metal::Encoder& e) { e.OrderWait(need); });
}

bool Runtime::UsedInFlight(const VideoCore::Buffer* buffer,
                           std::span<const VideoCore::BufferCopy> copies, bool source) const {
    const auto it = hoist_targets.find(buffer);
    if (it == hoist_targets.end()) {
        return false;
    }
    // Submissions committed but not known complete: ticks (known, current). A tag is the low
    // 16 bits of its tick, so one 65536 submissions old can look in flight (a wait too many).
    const u64 current = scheduler.CurrentTick(), known = scheduler.KnownGpuTick();
    const u64 in_flight = current > known + 1 ? current - known - 1 : 0;
    if (in_flight == 0) {
        return false;
    }
    const auto in_flight_tag = [&](u16 tag) {
        const u16 age = static_cast<u16>(static_cast<u16>(current) - tag);
        return age != 0 && age <= in_flight;
    };
    if (in_flight_tag(std::atomic_ref<u16>(it->second.whole_tick).load(std::memory_order_relaxed))) {
        return true;
    }
    for (const auto& copy : copies) {
        if (copy.size == 0) {
            continue;
        }
        const u64 offset = source ? copy.src_offset : copy.dst_offset;
        const u64 last = std::min((offset + copy.size - 1) >> HoistPageBits, it->second.pages - 1);
        for (u64 page = offset >> HoistPageBits; page <= last; ++page) {
            if (in_flight_tag(
                    std::atomic_ref<u16>(it->second.tick[page]).load(std::memory_order_relaxed))) {
                return true;
            }
        }
    }
    return false;
}

void Runtime::NoteSubmissionUse(const VideoCore::Buffer* buffer, u64 offset, u64 size) {
    const auto it = hoist_targets.find(buffer);
    if (it == hoist_targets.end() || size == 0) {
        return;
    }
    const u16 tick = static_cast<u16>(scheduler.CurrentTick());
    if (offset == 0 && size >= (it->second.pages << HoistPageBits)) {
        // The whole buffer (direct memory access): one tag, not a walk over every page.
        std::atomic_ref<u16>(it->second.whole_tick).store(tick, std::memory_order_relaxed);
        return;
    }
    const u64 last = std::min((offset + size - 1) >> HoistPageBits, it->second.pages - 1);
    for (u64 page = offset >> HoistPageBits; page <= last; ++page) {
        std::atomic_ref<u16>(it->second.tick[page]).store(tick, std::memory_order_relaxed);
    }
}

bool Runtime::HoistCopy(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                        std::span<const VideoCore::BufferCopy> copies, bool shader_copy) {
    if (!scheduler.IsRendering() || BbToggle::Disabled(BbToggle::UploadHoisting) ||
        (shader_copy && BbToggle::Disabled(BbToggle::CopyShaderHoisting))) {
        return false; // outside a pass an inline copy costs no pass break
    }
    const u16 tick = static_cast<u16>(scheduler.CurrentTick());
    const auto unused = [&](const VideoCore::Buffer* buffer, bool source) {
        const auto it = hoist_targets.find(buffer);
        if (it == hoist_targets.end() ||
            std::atomic_ref<u16>(it->second.whole_tick).load(std::memory_order_relaxed) == tick) {
            return false;
        }
        for (const auto& copy : copies) {
            if (copy.size == 0) {
                continue;
            }
            const u64 offset = source ? copy.src_offset : copy.dst_offset;
            const u64 last = (offset + copy.size - 1) >> HoistPageBits;
            if (last >= it->second.pages) {
                return false;
            }
            for (u64 page = offset >> HoistPageBits; page <= last; ++page) {
                if (std::atomic_ref<u16>(it->second.tick[page]).load(std::memory_order_relaxed) ==
                    tick) {
                    return false; // used earlier in this submission: the copy must stay in order
                }
            }
        }
        return true;
    };
    // A source outside the registered buffers is CPU-written staging (the stream buffer).
    if (!unused(dst, false) || (shader_copy && hoist_targets.contains(src) && !unused(src, true))) {
        return false;
    }
    // The upload runs before this submission's commands. A destination (or a shader copy's
    // source) still used by a submission in flight stays in order (it would have to wait for
    // all work before it). Later commands of this submission that touch its ranges get a
    // barrier on the uploads (marked here, Metal::ScopeAhead).
    if (ordering &&
        (UsedInFlight(dst, copies, false) || (shader_copy && UsedInFlight(src, copies, true)))) {
        BbStats::hoists_in_flight.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    scheduler.RecordHoistedCopy(src->Ref(), dst->Ref(), copies, shader_copy);
    if (ordering) {
        // A number of this submission's (the uploads run before it, and retire with it). No
        // command is between its FlushBarriers and its marks here: hoisting is decided while
        // a command's resources are bound, before its FlushBarriers.
        BeginCommand();
        for (const auto& copy : copies) {
            barriers.Mark(dst, copy.dst_offset, copy.size, true, Metal::EncoderAhead);
            barriers.Mark(src, copy.src_offset, copy.size, false, Metal::EncoderAhead);
        }
    }
    if (!shader_copy) {
        BbStats::hoisted_copies.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

void Runtime::TickFrame() {
    staging_pool.TickFrame();
}

void Runtime::CopyBuffer(const VideoCore::Buffer& src, const VideoCore::Buffer& dst,
                         std::span<const VideoCore::BufferCopy> copies,
                         std::source_location where) {
    scheduler.EndRendering(where);
    Metal::OrderRequest need;
    for (const auto& copy : copies) {
        need |= IsBufferAccessed(&src, copy.src_offset, copy.size) |
                IsBufferAccessed(&dst, copy.dst_offset, copy.size, true);
    }
    FlushBarriers(need);
    scheduler.Record([src_ref = src.Ref(), dst_ref = dst.Ref(),
                      regions = scheduler.RecordData(copies)](Metal::Encoder& e) {
        for (const auto& copy : regions) {
            e.CopyBuffer({src_ref.buffer, src_ref.offset + copy.src_offset},
                         {dst_ref.buffer, dst_ref.offset + copy.dst_offset}, copy.size);
        }
    });
    for (const auto& copy : copies) {
        AccessBuffer(&src, copy.src_offset, copy.size);
        AccessBuffer(&dst, copy.dst_offset, copy.size, true);
    }
}

void Runtime::FillBuffer(const VideoCore::Buffer& dst, u64 offset, u64 size, u32 value) {
    scheduler.EndRendering();
    FlushBarriers(IsBufferAccessed(&dst, offset, size, true));
    scheduler.Record([ref = dst.Ref(offset), size, value](Metal::Encoder& e) {
        e.FillBuffer(ref, size, value);
    });
    AccessBuffer(&dst, offset, size, true);
}

std::atomic<u64> g_bb_draw_seq{0};
static const auto g_bb_seq_t0 = std::chrono::steady_clock::now();
bool BbSeqWindow() {
    static const char* at = std::getenv("BB_SEQ_AT");
    if (!at) return false;
    static const double t_at = std::atof(at);
    static const auto t0 = g_bb_seq_t0; // library load (the first call may be late in the match)
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return t >= t_at && t < t_at + 1.0;
}

void Runtime::AccessImage(VideoCore::Image* image, bool write) {
    BeforeImageAccess();
    if (write) {
        ++image->write_serial;
        ++image->diag_other_writes; // (RT binds and uploads count themselves and subtract)
        if (image->info.size.width == 2048 && image->info.pixel_format == vk::Format::eR32Sfloat && BbSeqWindow()) {
            Dl_info info{};
            void* ret = __builtin_return_address(0);
            dladdr(ret, &info);
            std::fprintf(stderr, "[SEQ] %llu access-write %#llx from %s+%#llx\n", (unsigned long long)g_bb_draw_seq.load(),
                         (unsigned long long)image->info.guest_address, info.dli_sname ? info.dli_sname : "?",
                         (unsigned long long)(reinterpret_cast<uintptr_t>(ret) - reinterpret_cast<uintptr_t>(info.dli_saddr)));
        }
    }
    if (scene_targets) {
        scene_targets->NativeAccess(*image, write);
    }
}

Runtime::TransferMark::TransferMark(Runtime& runtime, const char* what,
                                    const VideoCore::Image& image) {
    profiler = GpuProfiler::Get();
    if (!profiler || !profiler->Records(&runtime.scheduler)) {
        profiler = nullptr;
        return;
    }
    resume = profiler->Current();
    const auto& info = image.info;
    const u64 key = (u64(reinterpret_cast<uintptr_t>(what)) << 20) ^
                    (u64(info.pixel_format) << 40) ^ (u64(info.size.width) << 16) ^
                    info.size.height ^ 0x7A5Full;
    profiler->Mark(key, [&] {
        return fmt::format("{} {} {}x{}", what, vk::to_string(info.pixel_format),
                           info.size.width, info.size.height);
    });
}

Runtime::TransferMark::~TransferMark() {
    if (profiler && resume) {
        profiler->Resume(resume);
    }
}

namespace {

using Metal::BlitAspect;

/// Logs `message` once per Vulkan format.
void LogOncePerFormat(vk::Format format, const char* message) {
    static std::mutex mutex;
    static std::vector<std::pair<vk::Format, const char*>> logged;
    std::scoped_lock lock{mutex};
    if (std::ranges::find(logged, std::pair{format, message}) == logged.end()) {
        logged.emplace_back(format, message);
        LOG_WARNING(Render_Vulkan, "{} ({})", message, vk::to_string(format));
    }
}

/// False (logged once) for images whose format Metal has no texture format for (RGB32 and the
/// others an RGBA8 texture stands in for, see image.cpp): their copies are skipped.
bool HasMetalFormat(const VideoCore::Image& image) {
    if (Metal::VkToMtl::PixelFormat(image.info.pixel_format) != 0) {
        return true;
    }
    LogOncePerFormat(image.info.pixel_format,
                     "image format has no Metal texture format: copies of it are skipped");
    return false;
}

bool IsCombinedDepthStencil(u32 pixel_format) {
    return Metal::VkToMtl::IsDepthPixelFormat(pixel_format) &&
           Metal::VkToMtl::HasStencilPixelFormat(pixel_format);
}

/// The aspect a buffer copy of `mask` moves. Metal copies one aspect of a combined depth-stencil
/// texture at a time (depth when both are asked for); other formats have one.
BlitAspect AspectOf(u32 pixel_format, vk::ImageAspectFlags mask) {
    if (!IsCombinedDepthStencil(pixel_format)) {
        return BlitAspect::All;
    }
    if (!(mask & vk::ImageAspectFlagBits::eDepth) && (mask & vk::ImageAspectFlagBits::eStencil)) {
        return BlitAspect::Stencil;
    }
    return BlitAspect::Depth;
}

/// How `aspect` of a texture of `pixel_format` lies in a buffer: bytes per block and block size
/// in texels. Depth of combined formats is 32-bit, stencil 8-bit. bytes == 0: not copyable.
struct BlockLayout {
    u32 bytes{}, width{1}, height{1};
};
BlockLayout LayoutOf(u32 pixel_format, BlitAspect aspect) {
    switch (aspect) {
    case BlitAspect::Depth:
        return {4};
    case BlitAspect::Stencil:
        return {1};
    case BlitAspect::All:
        break;
    }
    return {Metal::BytesPerBlock(pixel_format), Metal::BlockWidth(pixel_format),
            Metal::BlockHeight(pixel_format)};
}

u32 MipSize(u32 size, u32 level) {
    return std::max(size >> level, 1u);
}

u32 DepthOf(const Metal::TextureDesc& desc, u32 level) {
    return desc.type == Metal::TextureType::Tex3D ? MipSize(desc.depth, level) : 1u;
}

/// The clear colour for a colour attachment of the image's Metal texture: the value member its
/// numeric format reads, moved to the channels the Metal format stores them in (BGRA-like twins).
std::array<double, 4> ClearColorOf(vk::Format format, const vk::ClearColorValue& value) {
    const std::string_view numeric = vk::componentNumericFormat(format, 0);
    std::array<double, 4> color{};
    for (u32 i = 0; i < 4; ++i) {
        color[i] = numeric == "UINT"   ? double(value.uint32[i])
                   : numeric == "SINT" ? double(value.int32[i])
                                       : double(value.float32[i]);
    }
    const auto swizzle = Metal::VkToMtl::FormatSwizzle(format);
    const u8 channels[4] = {swizzle.r, swizzle.g, swizzle.b, swizzle.a};
    std::array<double, 4> stored = color;
    for (u32 i = 0; i < 4; ++i) {
        if (channels[i] >= Metal::VkToMtl::SwizzleRed) {
            stored[channels[i] - Metal::VkToMtl::SwizzleRed] = color[i];
        }
    }
    return stored;
}

constexpr std::array<u8, 4> IdentitySwizzle{
    Metal::VkToMtl::IdentitySwizzle.r, Metal::VkToMtl::IdentitySwizzle.g,
    Metal::VkToMtl::IdentitySwizzle.b, Metal::VkToMtl::IdentitySwizzle.a};

/// One buffer <-> texture copy (offset relative to the buffer reference).
struct BufferTextureCopy {
    u64 offset;
    u32 bytes_per_row, bytes_per_image;
    u32 level, slice;
    Metal::Origin3 origin;
    Metal::Size3 size;
    BlitAspect aspect;
};

/// A texture-to-texture copy as recorded: both sizes (they differ in texels when a staging
/// buffer reinterprets blocks), the staging layout and offset.
struct TextureCopy {
    u32 src_level, src_slice, src_z;
    u32 dst_level, dst_slice, dst_z;
    Metal::Size3 src_size, dst_size;
    u32 bytes_per_row, bytes_per_image;
    u64 offset;
};

} // namespace

void Runtime::CopyBufferImage(VideoCore::Image& image, const VideoCore::Buffer& buffer,
                              std::span<const vk::BufferImageCopy> copies, bool upload) {
    if (!HasMetalFormat(image)) {
        return;
    }
    const Metal::TextureDesc& desc = image.backing->desc;
    if (desc.samples > 1) {
        // Metal has no buffer copies of multisample textures.
        LogOncePerFormat(image.info.pixel_format,
                         "CopyBufferImage: buffer copy of a multisampled image skipped");
        return;
    }
    boost::container::small_vector<BufferTextureCopy, 16> blits;
    for (const auto& copy : copies) {
        const auto& sub = copy.imageSubresource;
        const BlitAspect aspect = AspectOf(desc.pixel_format, sub.aspectMask);
        const BlockLayout block = LayoutOf(desc.pixel_format, aspect);
        if (block.bytes == 0) {
            LogOncePerFormat(image.info.pixel_format,
                             "image format has no buffer copy layout: copy skipped");
            continue;
        }
        const u32 row_length = copy.bufferRowLength ? copy.bufferRowLength : copy.imageExtent.width;
        const u32 image_height =
            copy.bufferImageHeight ? copy.bufferImageHeight : copy.imageExtent.height;
        const u32 bytes_per_row = Common::DivCeil(row_length, block.width) * block.bytes;
        const u32 bytes_per_image = bytes_per_row * Common::DivCeil(image_height, block.height);
        for (u32 layer = 0; layer < sub.layerCount; ++layer) {
            blits.push_back({
                .offset = copy.bufferOffset + u64(layer) * bytes_per_image * copy.imageExtent.depth,
                .bytes_per_row = bytes_per_row,
                .bytes_per_image = bytes_per_image,
                .level = sub.mipLevel,
                .slice = sub.baseArrayLayer + layer,
                .origin = {u32(copy.imageOffset.x), u32(copy.imageOffset.y),
                           u32(copy.imageOffset.z)},
                .size = {copy.imageExtent.width, copy.imageExtent.height, copy.imageExtent.depth},
                .aspect = aspect,
            });
        }
    }
    if (blits.empty()) {
        return;
    }
    scheduler.Record([texture = image.GetImage(), ref = buffer.Ref(), upload,
                      blits = scheduler.RecordData(std::span<const BufferTextureCopy>{blits})](
                         Metal::Encoder& e) {
        for (const auto& b : blits) {
            const Metal::BufferRef at{ref.buffer, ref.offset + b.offset};
            if (upload) {
                e.CopyBufferToTexture(at, b.bytes_per_row, b.bytes_per_image, texture, b.level,
                                      b.slice, b.origin, b.size, b.aspect);
            } else {
                e.CopyTextureToBuffer(texture, b.level, b.slice, b.origin, b.size, at,
                                      b.bytes_per_row, b.bytes_per_image, b.aspect);
            }
        }
    });
}

void Runtime::UploadImage(VideoCore::Image* dst, const VideoCore::Buffer* src,
                          std::span<const vk::BufferImageCopy> upload_copies) {
    ++dst->write_serial; // fifa-hybrid: the depth twin re-syncs
    ++dst->diag_uploads;

    SetBackingSamples(dst, dst->info.num_samples, false);
    scheduler.EndRendering();
    const TransferMark mark{*this, "image upload", *dst};
    AccessImage(dst, true);

    Metal::OrderRequest need;
    for (const auto& copy : upload_copies) {
        const auto copy_size = BufferImageCopySize(copy, dst->info.pixel_format);
        need |= IsBufferAccessed(src, copy.bufferOffset, copy_size);
    }
    FlushBarriers(need);
    CopyBufferImage(*dst, *src, upload_copies, true);

    for (const auto& copy : upload_copies) {
        const auto copy_size = BufferImageCopySize(copy, dst->info.pixel_format);
        AccessBuffer(src, copy.bufferOffset, copy_size);
    }

    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::DownloadImage(VideoCore::Image* src, const VideoCore::Buffer* dst,
                            std::span<const vk::BufferImageCopy> download_copies) {
    SetBackingSamples(src, src->info.num_samples);
    scheduler.EndRendering();
    const TransferMark mark{*this, "image download", *src};
    AccessImage(src, false);

    Metal::OrderRequest need;
    for (const auto& copy : download_copies) {
        const auto copy_size = BufferImageCopySize(copy, src->info.pixel_format);
        need |= IsBufferAccessed(dst, copy.bufferOffset, copy_size, true);
    }
    FlushBarriers(need);
    CopyBufferImage(*src, *dst, download_copies, false);

    for (const auto& copy : download_copies) {
        const auto copy_size = BufferImageCopySize(copy, src->info.pixel_format);
        AccessBuffer(dst, copy.bufferOffset, copy_size, true);
    }
}

void Runtime::CopyRegions(VideoCore::Image* src, VideoCore::Image* dst,
                          std::span<const TextureRegion> regions) {
    if (regions.empty() || !HasMetalFormat(*src) || !HasMetalFormat(*dst)) {
        return;
    }
    // bbport: a copy between pixel formats views one side as the other: an image without its
    // lazy PixelFormatView usage gets it (as before lazy usages) rather than a staging bounce.
    for (VideoCore::Image* side : {src, dst}) {
        const VideoCore::Image* other = side == src ? dst : src;
        const auto& own = side->backing->desc;
        const u32 wanted = other->backing->desc.pixel_format;
        if (own.pixel_format != wanted && !own.format_views && side->allowed_format_views &&
            Metal::SafeViewPixelFormat(own.pixel_format, wanted, false) != wanted &&
            Metal::SafeViewPixelFormat(own.pixel_format, wanted, true) == wanted) {
            UpgradeUsage(side, VideoCore::Image::UsageFormatViews);
            break;
        }
    }
    if (dst->info.props.is_block && std::getenv("BB_BCVIEW_LOG")) {
        static int n = 0;
        if (n++ < 60) {
            std::fprintf(stderr, "[BCGPU] copy %s %ux%u lv%u at %#llx <- %s %ux%u lv%u at %#llx\n",
                         vk::to_string(dst->info.pixel_format).c_str(), dst->info.size.width,
                         dst->info.size.height, dst->info.resources.levels,
                         (unsigned long long)dst->info.guest_address,
                         vk::to_string(src->info.pixel_format).c_str(), src->info.size.width,
                         src->info.size.height, src->info.resources.levels,
                         (unsigned long long)src->info.guest_address);
        }
    }
    const Metal::TextureDesc& sd = src->backing->desc;
    const Metal::TextureDesc& dd = dst->backing->desc;
    void* src_texture = src->GetImage();
    void* dst_texture = dst->GetImage();
    Metal::Object view;
    bool bounce = false;
    if (sd.pixel_format != dd.pixel_format) {
        // A blit needs one pixel format: view either side as the other when Metal can, else
        // reinterpret blocks of the same size (BC <-> uncompressed, depth <-> colour) through a
        // staging buffer.
        if (Metal::SafeViewPixelFormat(sd.pixel_format, dd.pixel_format, sd.format_views) ==
            dd.pixel_format) {
            view = Metal::CreateTextureView(src_texture, dd.pixel_format, sd.type, 0, sd.levels, 0,
                                            sd.layers, IdentitySwizzle);
            src_texture = view.Get();
        } else if (Metal::SafeViewPixelFormat(dd.pixel_format, sd.pixel_format,
                                              dd.format_views) == sd.pixel_format) {
            view = Metal::CreateTextureView(dst_texture, sd.pixel_format, dd.type, 0, dd.levels, 0,
                                            dd.layers, IdentitySwizzle);
            dst_texture = view.Get();
        } else {
            bounce = Metal::BytesPerBlock(sd.pixel_format) != 0 &&
                     Metal::BytesPerBlock(sd.pixel_format) == Metal::BytesPerBlock(dd.pixel_format);
        }
        if (!bounce && !view) {
            LogOncePerFormat(dst->info.pixel_format,
                             "image copy between incompatible formats skipped (destination)");
            return;
        }
    }
    if (bounce ? (sd.samples > 1 || dd.samples > 1) : sd.samples != dd.samples) {
        LogOncePerFormat(dst->info.pixel_format,
                         "image copy between different sample counts skipped (destination)");
        return;
    }

    // Clamp each region to both subresources, in whole blocks of each side.
    const BlockLayout sb = LayoutOf(sd.pixel_format, BlitAspect::All);
    const BlockLayout db = LayoutOf(dd.pixel_format, BlitAspect::All);
    boost::container::small_vector<TextureCopy, 16> copies;
    u64 staging_size = 0;
    for (const auto& r : regions) {
        if (r.src_level >= sd.levels || r.dst_level >= dd.levels || r.src_slice >= sd.layers ||
            r.dst_slice >= dd.layers || r.src_z >= DepthOf(sd, r.src_level) ||
            r.dst_z >= DepthOf(dd, r.dst_level)) {
            continue;
        }
        const u32 sw = MipSize(sd.width, r.src_level), sh = MipSize(sd.height, r.src_level);
        const u32 dw = MipSize(dd.width, r.dst_level), dh = MipSize(dd.height, r.dst_level);
        const u32 blocks_x = std::min(Common::DivCeil(std::min(r.width, sw), sb.width),
                                      Common::DivCeil(dw, db.width));
        const u32 blocks_y = std::min(Common::DivCeil(std::min(r.height, sh), sb.height),
                                      Common::DivCeil(dh, db.height));
        const u32 depth = std::min({r.depth, DepthOf(sd, r.src_level) - r.src_z,
                                    DepthOf(dd, r.dst_level) - r.dst_z});
        TextureCopy copy{
            .src_level = r.src_level,
            .src_slice = r.src_slice,
            .src_z = r.src_z,
            .dst_level = r.dst_level,
            .dst_slice = r.dst_slice,
            .dst_z = r.dst_z,
            .src_size = {std::min(blocks_x * sb.width, sw), std::min(blocks_y * sb.height, sh),
                         depth},
            .dst_size = {std::min(blocks_x * db.width, dw), std::min(blocks_y * db.height, dh),
                         depth},
            .bytes_per_row = blocks_x * sb.bytes,
            .bytes_per_image = blocks_x * sb.bytes * blocks_y,
            .offset = staging_size,
        };
        if (!bounce) {
            // Same block size on both sides: one size that fits both.
            copy.src_size.width = std::min(copy.src_size.width, copy.dst_size.width);
            copy.src_size.height = std::min(copy.src_size.height, copy.dst_size.height);
        }
        staging_size += Common::AlignUp(u64(copy.bytes_per_image) * depth, 16);
        copies.push_back(copy);
    }
    if (copies.empty()) {
        return;
    }

    Metal::BufferRef staging{};
    StagingBufferRef staging_ref{};
    if (bounce) {
        staging_ref = staging_pool.Request(staging_size, VideoCore::MemoryType::DeviceLocal, 16);
        staging = staging_ref.buffer->Ref(staging_ref.offset);
        BeginStagingRoundTrip(staging_ref.buffer, staging_ref.offset, staging_size);
    }
    scheduler.Record([src_texture, dst_texture, view = std::move(view), staging, bounce,
                      copies = scheduler.RecordData(std::span<const TextureCopy>{copies})](
                         Metal::Encoder& e) mutable {
        if (view) {
            e.KeepAlive(std::move(view));
        }
        if (!bounce) {
            for (const auto& c : copies) {
                e.CopyTexture(src_texture, c.src_level, c.src_slice, {0, 0, c.src_z},
                              dst_texture, c.dst_level, c.dst_slice, {0, 0, c.dst_z}, c.src_size);
            }
            return;
        }
        for (const auto& c : copies) {
            e.CopyTextureToBuffer(src_texture, c.src_level, c.src_slice, {0, 0, c.src_z},
                                  c.src_size, {staging.buffer, staging.offset + c.offset},
                                  c.bytes_per_row, c.bytes_per_image);
        }
        // The uploads read what the downloads wrote: another encoder, after them (staging is
        // ordered by this Runtime; tracked buffers: Metal orders the encoders).
        e.EndEncoding();
        WaitForOwnCopies(e);
        for (const auto& c : copies) {
            e.CopyBufferToTexture({staging.buffer, staging.offset + c.offset}, c.bytes_per_row,
                                  c.bytes_per_image, dst_texture, c.dst_level, c.dst_slice,
                                  {0, 0, c.dst_z}, c.dst_size);
        }
    });
    if (bounce) {
        EndStagingRoundTrip(staging_ref.buffer, staging_ref.offset, staging_size);
    }
}

void Runtime::CopyImage(VideoCore::Image* src, VideoCore::Image* dst) {
    const u32 num_mips = std::min(src->info.resources.levels, dst->info.resources.levels);

    // Format mismatch warning (safe but useful)
    if (src->info.pixel_format != dst->info.pixel_format) {
        LOG_DEBUG(Render_Vulkan,
                  "Copy between different formats: src={}, dst={}. "
                  "Result may be undefined.",
                  vk::to_string(src->info.pixel_format), vk::to_string(dst->info.pixel_format));
    }

    const u32 base_width = src->info.size.width;
    const u32 base_height = src->info.size.height;
    const u32 base_depth =
        dst->info.type == AmdGpu::ImageType::Color3D ? dst->info.size.depth : src->info.size.depth;

    // Match sample count before copying
    SetBackingSamples(dst, dst->info.num_samples, false);
    SetBackingSamples(src, src->info.num_samples);

    const bool src_is_3d = ConvertImageType(src->info.type) == vk::ImageType::e3D;
    const bool dst_is_3d = ConvertImageType(dst->info.type) == vk::ImageType::e3D;

    // 2D <-> 3D: array layer i is depth plane i.
    boost::container::small_vector<TextureRegion, 16> regions;
    for (u32 mip = 0; mip < num_mips; ++mip) {
        const u32 mip_w = std::max(base_width >> mip, 1u);
        const u32 mip_h = std::max(base_height >> mip, 1u);
        const u32 mip_d = std::max(base_depth >> mip, 1u);

        const auto [src_layers, dst_layers] = SanitizeCopyLayers(src->info, dst->info, mip_d);

        if (src_is_3d && dst_is_3d) {
            regions.push_back({mip, 0, 0, mip, 0, 0, mip_w, mip_h, mip_d});
        } else if (!src_is_3d && dst_is_3d) {
            for (u32 layer = 0; layer < src_layers; ++layer) {
                regions.push_back({mip, layer, 0, mip, 0, layer, mip_w, mip_h, 1});
            }
        } else if (src_is_3d) {
            for (u32 layer = 0; layer < dst_layers; ++layer) {
                regions.push_back({mip, 0, layer, mip, layer, 0, mip_w, mip_h, 1});
            }
        } else {
            for (u32 layer = 0; layer < std::min(src_layers, dst_layers); ++layer) {
                regions.push_back({mip, layer, 0, mip, layer, 0, mip_w, mip_h, 1});
            }
        }
    }

    scheduler.EndRendering();
    AccessImage(src, false);
    AccessImage(dst, true);
    CopyRegions(src, dst, regions);

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::CopyImageWithBuffer(VideoCore::Image* src, VideoCore::Image* dst,
                                  const VideoCore::Buffer* buffer, u64 offset) {
    const u32 num_mips = std::min(src->info.resources.levels, dst->info.resources.levels);
    const u32 num_layers = std::min(src->info.resources.layers, dst->info.resources.layers);
    ASSERT(src->info.resources.layers == dst->info.resources.layers && num_mips == 1);

    SetBackingSamples(dst, dst->info.num_samples, false);
    SetBackingSamples(src, src->info.num_samples);

    scheduler.EndRendering();
    AccessImage(src, false);
    AccessImage(dst, true);

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
    if (!HasMetalFormat(*src) || !HasMetalFormat(*dst)) {
        return;
    }
    // Each side lays the texels out with its own format (Vulkan's tight buffer copies); the depth
    // aspect of a combined depth-stencil format (the stencil is not copied).
    const Metal::TextureDesc& sd = src->backing->desc;
    const Metal::TextureDesc& dd = dst->backing->desc;
    const BlitAspect src_aspect =
        IsCombinedDepthStencil(sd.pixel_format) ? BlitAspect::Depth : BlitAspect::All;
    const BlitAspect dst_aspect =
        IsCombinedDepthStencil(dd.pixel_format) ? BlitAspect::Depth : BlitAspect::All;
    const BlockLayout sb = LayoutOf(sd.pixel_format, src_aspect);
    const BlockLayout db = LayoutOf(dd.pixel_format, dst_aspect);
    if (sb.bytes == 0 || db.bytes == 0 || sd.samples > 1 || dd.samples > 1) {
        LogOncePerFormat(dst->info.pixel_format,
                         "image copy through a buffer not possible: skipped (destination)");
        return;
    }
    Metal::Size3 size{std::min(sd.width, dd.width), std::min(sd.height, dd.height),
                      std::min(sd.depth, dd.depth)};
    Metal::Size3 src_size = size;
    Metal::Size3 dst_size = size;
    if (sb.bytes == db.bytes && (sb.width != db.width || sb.height != db.height)) {
        // fifa-hybrid: a block-compressed image and an uncompressed image with one texel per
        // block (BC3 <-> RGBA32Uint): the same bytes, copied block for block.
        const u32 bw = std::min(Common::DivCeil(sd.width, sb.width),
                                Common::DivCeil(dd.width, db.width));
        const u32 bh = std::min(Common::DivCeil(sd.height, sb.height),
                                Common::DivCeil(dd.height, db.height));
        src_size = {std::min(bw * sb.width, sd.width), std::min(bh * sb.height, sd.height),
                    size.depth};
        dst_size = {std::min(bw * db.width, dd.width), std::min(bh * db.height, dd.height),
                    size.depth};
        size = src_size;
    }
    const u32 src_row = Common::DivCeil(src_size.width, sb.width) * sb.bytes;
    const u32 src_image = src_row * Common::DivCeil(src_size.height, sb.height);
    const u32 dst_row = Common::DivCeil(dst_size.width, db.width) * db.bytes;
    const u32 dst_image = dst_row * Common::DivCeil(dst_size.height, db.height);
    const u32 layers = std::min({num_layers, sd.layers, dd.layers});

    const u64 staging_size = u64(std::max(src_image, dst_image)) * size.depth * layers;
    BeginStagingRoundTrip(buffer, offset, staging_size);
    scheduler.Record([src_texture = src->GetImage(), dst_texture = dst->GetImage(),
                      ref = buffer->Ref(offset), size, src_size, dst_size, layers, src_row,
                      src_image, dst_row, dst_image, src_aspect, dst_aspect](Metal::Encoder& e) {
        for (u32 layer = 0; layer < layers; ++layer) {
            e.CopyTextureToBuffer(src_texture, 0, layer, {}, src_size,
                                  {ref.buffer, ref.offset + u64(layer) * src_image * size.depth},
                                  src_row, src_image, src_aspect);
        }
        // The uploads read what the downloads wrote: another encoder, after them (staging is
        // ordered by this Runtime; tracked buffers: Metal orders the encoders).
        e.EndEncoding();
        WaitForOwnCopies(e);
        for (u32 layer = 0; layer < layers; ++layer) {
            e.CopyBufferToTexture({ref.buffer, ref.offset + u64(layer) * dst_image * size.depth},
                                  dst_row, dst_image, dst_texture, 0, layer, {}, dst_size,
                                  dst_aspect);
        }
    });
    EndStagingRoundTrip(buffer, offset, staging_size);
}

void Runtime::CopyMip(VideoCore::Image* src, VideoCore::Image* dst, u32 mip, u32 slice) {
    const auto dst_dim = dst->info.props.is_block ? 2 : 0;
    const auto mip_block_w = std::max(dst->info.size.width >> (mip + dst_dim), 1u);
    const auto mip_block_h = std::max(dst->info.size.height >> (mip + dst_dim), 1u);
    const auto mip_block_p = std::max(dst->info.mips_layout[mip].pitch >> dst_dim, 1u);

    const auto src_dim = src->info.props.is_block ? 2 : 0;
    ASSERT(mip_block_w == (src->info.size.width >> src_dim));
    ASSERT(mip_block_h == (src->info.size.height >> src_dim));
    ASSERT(mip_block_p == (src->info.pitch >> src_dim));

    SetBackingSamples(dst, dst->info.num_samples);
    SetBackingSamples(src, src->info.num_samples);

    scheduler.EndRendering();
    AccessImage(src, false);
    AccessImage(dst, true);

    // Every layer of the source, as the Vulkan copy does (cube faces rendered one mip at a time).
    const auto [src_layers, dst_layers] =
        SanitizeCopyLayers(src->info, dst->info, src->info.size.depth);
    const u32 layers = std::min(src_layers, dst_layers > slice ? dst_layers - slice : 1u);
    boost::container::small_vector<TextureRegion, 8> regions;
    for (u32 layer = 0; layer < layers; ++layer) {
        regions.push_back({0, layer, 0, mip, slice + layer, 0, src->info.size.width,
                           src->info.size.height, src->info.size.depth});
    }
    CopyRegions(src, dst, regions);

    dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::CopyBlockBytes(VideoCore::Image* src, VideoCore::Image* dst) {
    const u32 num_mips = std::min(src->info.resources.levels, dst->info.resources.levels);
    if (num_mips > 1 || src->info.resources.layers > 1) {
        // Mip chains (GPU-compressed kits) and arrays (player textures): each level and layer
        // block for block, through CopyRegions' staging bounce.
        if (!HasMetalFormat(*src) || !HasMetalFormat(*dst)) {
            return;
        }
        SetBackingSamples(dst, dst->info.num_samples, false);
        SetBackingSamples(src, src->info.num_samples);
        scheduler.EndRendering();
        AccessImage(src, false);
        AccessImage(dst, true);
        const u32 layers = std::min(src->info.resources.layers, dst->info.resources.layers);
        boost::container::small_vector<TextureRegion, 16> regions;
        for (u32 mip = 0; mip < num_mips; ++mip) {
            const u32 w = std::max(src->info.size.width >> mip, 1u);
            const u32 h = std::max(src->info.size.height >> mip, 1u);
            for (u32 layer = 0; layer < layers; ++layer) {
                regions.push_back({mip, layer, 0, mip, layer, 0, w, h, 1});
            }
        }
        CopyRegions(src, dst, regions);
        dst->flags |= (src->flags & VideoCore::ImageFlagBits::GpuModified);
        dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
        return;
    }
    const u64 size = std::max<u64>(std::max(src->info.guest_size, dst->info.guest_size), 64_KB);
    const auto copy_ref = staging_pool.Request(size, VideoCore::MemoryType::DeviceLocal);
    CopyImageWithBuffer(src, dst, copy_ref.buffer, copy_ref.offset);
}

void Runtime::SyncDepthTwin(VideoCore::Image* image) {
    if (!image->DepthTwinEligible()) {
        return;
    }
    if (image->depth_twin && image->depth_twin_synced == image->write_serial) {
        return;
    }
    const Metal::TextureDesc& sd = image->backing->desc;
    if (!image->depth_twin) {
        Metal::TextureDesc d = sd;
        d.pixel_format = Metal::VkToMtl::PixelFormat(vk::Format::eD32Sfloat);
        d.render_target = false;
        d.shader_write = false;
        d.shader_atomic = false;
        d.format_views = false;
        image->depth_twin = Metal::CreateTexture(device, d, "depth twin", nullptr);
        if (!image->depth_twin) {
            LOG_ERROR(Render_Vulkan, "depth twin of {:#x} {}x{} not created", image->info.guest_address,
                      sd.width, sd.height);
            return;
        }
        LOG_INFO(Render_Vulkan, "depth twin for R32 image {:#x} {}x{} levels {}",
                 image->info.guest_address, sd.width, sd.height, sd.levels);
        if (static const bool watch = std::getenv("BB_TWIN_WATCH") != nullptr; watch) {
            std::fprintf(stderr, "[TWINWATCH] twin created for %#llx %ux%u serial %llu flags %#x\n",
                         (unsigned long long)image->info.guest_address, sd.width, sd.height,
                         (unsigned long long)image->write_serial, u32(image->flags));
        }
    }
    scheduler.EndRendering();
    AccessImage(image, false);
    if (static const bool tlog = std::getenv("BB_TWIN_LOG") != nullptr; tlog) {
        static int n = 0;
        if (n++ % 100 == 0) {
            std::fprintf(stderr, "[TWIN] sync #%d image %#llx %ux%u serial %llu\n", n,
                         (unsigned long long)image->info.guest_address, sd.width, sd.height,
                         (unsigned long long)image->write_serial);
        }
    }
    for (u32 level = 0; level < sd.levels; ++level) {
        const u32 w = std::max(1u, sd.width >> level), h = std::max(1u, sd.height >> level);
        for (u32 layer = 0; layer < sd.layers; ++layer) {
            const u64 size = u64(w) * h * 4;
            const auto staging = staging_pool.Request(size, VideoCore::MemoryType::DeviceLocal, 256);
            scheduler.Record([src = image->GetImage(), dst = image->depth_twin.Get(), ref = staging.buffer->Ref(staging.offset),
                              w, h, level, layer, size](Metal::Encoder& e) {
                e.CopyTextureToBuffer(src, level, layer, {}, {w, h, 1}, ref, w * 4, u32(size));
                e.CopyBufferToTexture(ref, w * 4, u32(size), dst, level, layer, {}, {w, h, 1});
            });
        }
    }
    image->depth_twin_synced = image->write_serial;
    if (sd.width == 2048 && BbSeqWindow()) {
        std::fprintf(stderr, "[SEQ] %llu sync %#llx serial %llu\n", (unsigned long long)g_bb_draw_seq.load(),
                     (unsigned long long)image->info.guest_address, (unsigned long long)image->write_serial);
    }
    // fifa-hybrid diagnostics (BB_TWIN_WATCH=1): every 2 s per image, right after the sync:
    // value statistics of the source and the twin, and what moved write_serial since the last probe.
    if (static const bool watch = std::getenv("BB_TWIN_WATCH") != nullptr; watch) {
        const auto now = std::chrono::steady_clock::now();
        if (now - image->diag_last_probe >= std::chrono::seconds(2)) {
            image->diag_last_probe = now;
            const u32 w = sd.width, h = sd.height;
            const u64 size = u64(w) * h * 4;
            const auto dl = staging_pool.Request(size, VideoCore::MemoryType::HostCached, 16, false);
            const auto tdl = staging_pool.Request(size, VideoCore::MemoryType::HostCached, 16, false);
            scheduler.Record([src = image->GetImage(), twin = image->depth_twin.Get(), w, h, size,
                              d1 = dl.buffer->Ref(dl.offset), d2 = tdl.buffer->Ref(tdl.offset)](Metal::Encoder& e) {
                e.CopyTextureToBuffer(src, 0, 0, {}, {w, h, 1}, d1, w * 4, u32(size));
                e.CopyTextureToBuffer(twin, 0, 0, {}, {w, h, 1}, d2, w * 4, u32(size), Metal::BlitAspect::Depth);
            });
            scheduler.Finish();
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
            const size_t n = size_t(w) * h, stride = std::max<size_t>(1, n / 65536);
            std::fprintf(stderr, "[TWINWATCH] sync %#llx %ux%u serial %llu flags %#x since last: rt %u up %u other %u | src %s | twin %s\n",
                         (unsigned long long)image->info.guest_address, w, h, (unsigned long long)image->write_serial,
                         u32(image->flags), image->diag_rt_writes, image->diag_uploads,
                         image->diag_other_writes - image->diag_rt_writes - image->diag_uploads, stats(reinterpret_cast<const float*>(dl.mapped), n, stride).c_str(),
                         stats(reinterpret_cast<const float*>(tdl.mapped), n, stride).c_str());
            image->diag_rt_writes = image->diag_uploads = image->diag_other_writes = 0;
        }
    }
}

void Runtime::CopyColorAndDepth(VideoCore::Image* src, VideoCore::Image* dst) {
    if (src->info.num_samples == 1 && dst->info.num_samples == 1) {
        if (src->info.props.is_depth == dst->info.props.is_depth) {
            CopyImage(src, dst);
        } else {
            // Metal has no colour <-> depth blit: through the intermediate copy buffer.
            static constexpr size_t COPY_BUFFER_SIZE = 128_MB;
            const auto copy_ref =
                staging_pool.Request(COPY_BUFFER_SIZE, VideoCore::MemoryType::DeviceLocal);
            CopyImageWithBuffer(src, dst, copy_ref.buffer, copy_ref.offset);
        }
    } else if (src->info.num_samples == 1 && dst->info.num_samples > 1 &&
               dst->info.props.is_depth) {
        // Perform a rendering pass to transfer the channels of source as samples in dest.
        scheduler.EndRendering();
        AccessImage(src, false);
        AccessImage(dst, true);
        blit_helper->ReinterpretColorAsMsDepth(
            dst->info.size.width, dst->info.size.height, dst->info.num_samples,
            src->info.pixel_format, dst->info.pixel_format, src->GetImage(), dst->GetImage());
    } else {
        LOG_WARNING(Render_Vulkan, "Unimplemented depth overlap copy");
    }
}

void Runtime::CopyDepthStencil(VideoCore::Image* src, VideoCore::Image* dst,
                               const VideoCore::SubresourceRange& sub_range) {
    scheduler.EndRendering();
    AccessImage(src, false);
    AccessImage(dst, true);
    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;

    // Metal blits no depth-stencil texture to another one of another format and copies one
    // aspect of a combined format at a time: each aspect both images have goes through a buffer.
    const Metal::TextureDesc& sd = src->backing->desc;
    const Metal::TextureDesc& dd = dst->backing->desc;
    struct Aspect {
        BlitAspect src, dst;
        u32 bytes;
    };
    boost::container::static_vector<Aspect, 2> aspects;
    const auto add = [&](BlitAspect combined) {
        const BlitAspect sa = IsCombinedDepthStencil(sd.pixel_format) ? combined : BlitAspect::All;
        const BlitAspect da = IsCombinedDepthStencil(dd.pixel_format) ? combined : BlitAspect::All;
        const u32 bytes = LayoutOf(sd.pixel_format, sa).bytes;
        if (bytes == 0 || bytes != LayoutOf(dd.pixel_format, da).bytes) {
            LogOncePerFormat(dst->info.pixel_format,
                             "depth-stencil copy between incompatible formats skipped");
            return;
        }
        aspects.push_back({sa, da, bytes});
    };
    if (src->has_depth && dst->has_depth) {
        add(BlitAspect::Depth);
    }
    if (src->has_stencil && dst->has_stencil) {
        add(BlitAspect::Stencil);
    }
    if (sd.samples > 1 || dd.samples > 1) {
        LogOncePerFormat(dst->info.pixel_format, "multisampled depth-stencil copy skipped");
        return;
    }
    const u32 first = sub_range.base.layer;
    const u32 end = std::min({first + sub_range.extent.layers, sd.layers, dd.layers});
    if (aspects.empty() || first >= end) {
        return;
    }
    const Metal::Size3 size{std::min({dst->info.size.width, sd.width, dd.width}),
                            std::min({dst->info.size.height, sd.height, dd.height}), 1};
    const u64 pixels = u64(size.width) * size.height;
    u64 staging_size = 0;
    for (const auto& aspect : aspects) {
        staging_size += Common::AlignUp(pixels * aspect.bytes, 16) * (end - first);
    }
    const auto ref = staging_pool.Request(staging_size, VideoCore::MemoryType::DeviceLocal, 16);
    BeginStagingRoundTrip(ref.buffer, ref.offset, staging_size);
    scheduler.Record([src_texture = src->GetImage(), dst_texture = dst->GetImage(),
                      staging = ref.buffer->Ref(ref.offset), aspects, first, end, size,
                      pixels](Metal::Encoder& e) {
        const auto each = [&](auto&& copy) {
            u64 offset = staging.offset;
            for (const auto& aspect : aspects) {
                const u64 stride = Common::AlignUp(pixels * aspect.bytes, 16);
                for (u32 layer = first; layer < end; ++layer, offset += stride) {
                    copy(aspect, layer, Metal::BufferRef{staging.buffer, offset});
                }
            }
        };
        each([&](const Aspect& aspect, u32 layer, Metal::BufferRef at) {
            e.CopyTextureToBuffer(src_texture, 0, layer, {}, size, at, size.width * aspect.bytes,
                                  u32(pixels * aspect.bytes), aspect.src);
        });
        // The uploads read what the downloads wrote: another encoder, after them (staging is
        // ordered by this Runtime; tracked buffers: Metal orders the encoders).
        e.EndEncoding();
        WaitForOwnCopies(e);
        each([&](const Aspect& aspect, u32 layer, Metal::BufferRef at) {
            e.CopyBufferToTexture(at, size.width * aspect.bytes, u32(pixels * aspect.bytes),
                                  dst_texture, 0, layer, {}, size, aspect.dst);
        });
    });
    EndStagingRoundTrip(ref.buffer, ref.offset, staging_size);
}

void Runtime::ResolveImage(VideoCore::Image* src, VideoCore::Image* dst,
                           const VideoCore::SubresourceRange& src_range,
                           const VideoCore::SubresourceRange& dst_range) {
    SetBackingSamples(dst, 1, false);
    if (std::getenv("BB_MSAA_LOG")) { // MSAALOG
        static int n = 0;
        if (n++ < 400)
            std::fprintf(stderr, "[MSAALOG] resolve src=%#llx %ux%u s=%u -> dst=%#llx fmt=%d\n",
                         (unsigned long long)src->info.guest_address, src->info.size.width,
                         src->info.size.height, src->backing->desc.samples,
                         (unsigned long long)dst->info.guest_address, (int)dst->info.pixel_format);
    }
    scheduler.EndRendering();
    AccessImage(src, false);
    AccessImage(dst, true);
    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;

    const auto [src_layers, dst_layers] = SanitizeCopyLayers(src->info, dst->info, 1);
    const u32 layers = std::min(src_layers, dst_layers);
    const Metal::TextureDesc& sd = src->backing->desc;
    if (sd.samples == 1) {
        boost::container::small_vector<TextureRegion, 8> regions;
        for (u32 layer = 0; layer < layers; ++layer) {
            regions.push_back({0, src_range.base.layer + layer, 0, 0, dst_range.base.layer + layer,
                               0, src->info.size.width, src->info.size.height, 1});
        }
        CopyRegions(src, dst, regions);
        return;
    }
    if (!HasMetalFormat(*src) || !HasMetalFormat(*dst)) {
        return;
    }
    if (Metal::IsIntegerPixelFormat(sd.pixel_format)) {
        // Metal cannot multisample-resolve integer formats.
        LogOncePerFormat(src->info.pixel_format,
                         "ResolveImage: resolve of an integer format skipped (source)");
        return;
    }
    // A pass that loads the multisampled texture and resolves it into the destination (Metal's
    // resolve needs the same pixel format: view the destination as the source's when they differ).
    const Metal::TextureDesc& dd = dst->backing->desc;
    void* dst_texture = dst->GetImage();
    Metal::Object view;
    if (dd.pixel_format != sd.pixel_format) {
        if (Metal::SafeViewPixelFormat(dd.pixel_format, sd.pixel_format, dd.format_views) !=
            sd.pixel_format) {
            LogOncePerFormat(dst->info.pixel_format,
                             "resolve between incompatible formats skipped (destination)");
            return;
        }
        view = Metal::CreateTextureView(dst_texture, sd.pixel_format, dd.type, 0, dd.levels, 0,
                                        dd.layers, IdentitySwizzle);
        dst_texture = view.Get();
    }
    const u32 src_base = src_range.base.layer;
    const u32 dst_base = dst_range.base.layer;
    const u32 count = std::min({layers, sd.layers - std::min(src_base, sd.layers),
                                dd.layers - std::min(dst_base, dd.layers)});
    scheduler.Record([src_texture = src->GetImage(), dst_texture, view = std::move(view), src_base,
                      dst_base, count](Metal::Encoder& e) mutable {
        if (view) {
            e.KeepAlive(std::move(view));
        }
        for (u32 layer = 0; layer < count; ++layer) {
            Metal::RenderPassDesc pass{};
            pass.colors[0].texture = src_texture;
            pass.colors[0].slice = src_base + layer;
            pass.colors[0].resolve_texture = dst_texture;
            pass.colors[0].resolve_slice = dst_base + layer;
            e.BeginRenderPass(pass);
            e.EndRenderPass();
        }
    });
}

void Runtime::ClearImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range,
                         const vk::ClearValue& clear_value) {
    scheduler.EndRendering();
    AccessImage(dst, true);
    dst->flags |= VideoCore::ImageFlagBits::GpuModified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;

    if (dst->has_depth || dst->has_stencil) {
        // As in Vulkan (vkCmdClearColorImage), only colour images are cleared here.
        LogOncePerFormat(dst->info.pixel_format, "colour clear of a depth image skipped");
        return;
    }
    // A clear-only render pass per level and layer (per depth plane of 3D textures).
    const Metal::TextureDesc& desc = dst->backing->desc;
    if (!desc.render_target) {
        LogOncePerFormat(dst->info.pixel_format, "clear of a block-compressed image skipped");
        return;
    }
    struct Target {
        u32 level, slice, plane;
    };
    boost::container::small_vector<Target, 16> targets;
    const u32 level_end = std::min(range.base.level + range.extent.levels, desc.levels);
    const u32 layer_end = std::min(range.base.layer + range.extent.layers, desc.layers);
    for (u32 level = range.base.level; level < level_end; ++level) {
        if (desc.type == Metal::TextureType::Tex3D) {
            for (u32 plane = 0; plane < DepthOf(desc, level); ++plane) {
                targets.push_back({level, 0, plane});
            }
            continue;
        }
        for (u32 layer = range.base.layer; layer < layer_end; ++layer) {
            targets.push_back({level, layer, 0});
        }
    }
    if (targets.empty()) {
        return;
    }
    scheduler.Record([texture = dst->GetImage(),
                      color = ClearColorOf(dst->info.pixel_format, clear_value.color),
                      targets = scheduler.RecordData(std::span<const Target>{targets})](
                         Metal::Encoder& e) {
        for (const auto& t : targets) {
            Metal::RenderPassDesc pass{};
            pass.colors[0].texture = texture;
            pass.colors[0].level = t.level;
            pass.colors[0].slice = t.slice;
            pass.colors[0].depth_plane = t.plane;
            pass.colors[0].clear = true;
            pass.colors[0].clear_color = color;
            e.BeginRenderPass(pass);
            e.EndRenderPass();
        }
    });
}

void Runtime::SetBackingSamples(VideoCore::Image* image, u32 num_samples, bool copy_backing) {
    BeforeImageAccess();
    auto& backing = image->backing;
    auto& backing_images = image->backing_images;
    const auto& info = image->info;
    if (!backing || backing->num_samples == num_samples) {
        return;
    }
    ASSERT_MSG(!image->info.props.is_depth, "Swapping samples is only valid for color images");
    VideoCore::Image::BackingImage* new_backing;
    auto it = std::ranges::find(backing_images, num_samples,
                                &VideoCore::Image::BackingImage::num_samples);
    if (it == backing_images.end()) {
        // A new texture from the image's description with the new sample count.
        new_backing = &image->CreateBacking(num_samples);
    } else {
        new_backing = std::addressof(*it);
    }

    if (std::getenv("BB_MSAA_LOG")) { // MSAALOG
        static int n = 0;
        if (n++ < 400)
            std::fprintf(stderr, "[MSAALOG] swap %#llx %ux%u %u->%u copy=%d guest=%u fmt=%d\n",
                         (unsigned long long)info.guest_address, info.size.width,
                         info.size.height, backing->num_samples, num_samples, (int)copy_backing,
                         info.num_samples, (int)info.pixel_format);
    }
    if (copy_backing) {
        scheduler.EndRendering();
        ASSERT(image->info.resources.levels == 1 && image->info.resources.layers == 1);
        AccessImage(image, false);

        // Copy between ms and non ms backing images
        blit_helper->CopyBetweenMsImages(info.size.width, info.size.height,
                                         new_backing->num_samples, info.pixel_format,
                                         backing->num_samples > 1, backing->texture.Get(),
                                         new_backing->texture.Get());
    }

    backing = new_backing;
}

void Runtime::UpgradeUsage(VideoCore::Image* image, u32 needed) {
    BeforeImageAccess();
    (void)needed;
    auto* const old_backing = image->backing;
    // Both usages at once: an image written by shaders is often read in another pixel format
    // too (the game's streamed textures: written as RGBA8 uint, sampled as unorm or sRGB), and
    // each upgrade replaces the texture.
    image->texture_desc.shader_write = image->allowed_shader_write;
    image->texture_desc.format_views = image->allowed_format_views;
    auto& fresh = image->CreateBacking(old_backing->num_samples);
    scheduler.EndRendering();
    AccessImage(image, false);
    const Metal::TextureDesc& d = fresh.desc;
    // The old backing's texture and views live until the copy's command buffer completes
    // (commands recorded before it may still use them); the backing is never found again.
    void* const src = old_backing->texture.Get();
    std::vector<Metal::Object> retired;
    retired.push_back(std::move(old_backing->texture));
    for (const auto view_id : old_backing->image_view_ids) {
        retired.push_back(std::move((*image->slot_image_views)[view_id].view));
        image->slot_image_views->erase(view_id);
    }
    old_backing->image_view_ids.clear();
    old_backing->image_view_infos.clear();
    old_backing->last_view = 0;
    old_backing->num_samples = 0;
    scheduler.Record([src, dst = fresh.texture.Get(), d, retired = std::move(retired)](
                         Metal::Encoder& e) mutable {
        const bool is_3d = d.type == Metal::TextureType::Tex3D;
        for (u32 level = 0; level < d.levels; ++level) {
            const Metal::Size3 size{std::max(d.width >> level, 1u), std::max(d.height >> level, 1u),
                                    is_3d ? std::max(d.depth >> level, 1u) : 1u};
            for (u32 slice = 0; slice < d.layers; ++slice) {
                e.CopyTexture(src, level, slice, {}, dst, level, slice, {}, size);
            }
        }
        for (auto& object : retired) {
            e.KeepAlive(std::move(object));
        }
    });
    image->backing = &fresh;
    BbStats::texture_upgrades.fetch_add(1, std::memory_order_relaxed);
    BbStats::texture_upgrade_bytes.fetch_add(fresh.allocated_size, std::memory_order_relaxed);
}

void Runtime::AccessBuffer(const VideoCore::Buffer* handle, u64 offset, u64 size, bool write,
                           u8 encoders, Metal::ReadStage stage) {
    NoteSubmissionUse(handle, offset, size);
    MarkAccess(handle, offset, size, write, encoders, stage);
}

} // namespace Vulkan
