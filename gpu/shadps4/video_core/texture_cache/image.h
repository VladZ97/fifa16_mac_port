// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/enum.h"
#include "common/incremental_id.h"
#include "common/types.h"
#include "video_core/renderer_metal/mtl_object.h"
#include "video_core/renderer_metal/mtl_texture.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"

#include <atomic>
#include <deque>
#include <memory>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

namespace Vulkan {
class Runtime;
} // namespace Vulkan

namespace Metal {
class Device;
} // namespace Metal

namespace VideoCore {

enum ImageFlagBits : u32 {
    Empty = 0,
    MaybeCpuDirty = 1 << 0, ///< The page this image is in was touched before the image address
    CpuDirty = 1 << 1,      ///< Contents have been modified from the CPU
    GpuDirty = 1 << 2, ///< Contents have been modified from the GPU (valid data in buffer cache)
    Dirty = MaybeCpuDirty | CpuDirty | GpuDirty,
    GpuModified = 1 << 3, ///< Contents have been modified from the GPU
    Registered = 1 << 6,  ///< True when the image is registered
    Picked = 1 << 7,      ///< Temporary flag to mark the image as picked
};
DECLARE_ENUM_FLAG_OPERATORS(ImageFlagBits)

/// bbport: lazy texture usages (Image::allowed_shader_write); BB_LAZY_TEXTURE_USAGE=0 turns them off.
bool LazyTextureUsage();

struct Image {
    explicit Image(Metal::Device& device, Vulkan::Runtime& runtime,
                   Common::SlotVector<ImageView>& slot_image_views, const ImageInfo& info);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&&) = default;
    Image& operator=(Image&&) = default;

    bool Overlaps(VAddr overlap_cpu_addr, size_t overlap_size) const noexcept {
        const VAddr overlap_end = overlap_cpu_addr + overlap_size;
        const auto image_addr = info.guest_address;
        const auto image_end = info.guest_address + info.guest_size;
        return image_addr < overlap_end && overlap_cpu_addr < image_end;
    }

    /// The current backing's id<MTLTexture> (unretained).
    void* GetImage() const {
        return backing->texture.Get();
    }

    u64 GetHostImageSize() const {
        return backing->allocated_size;
    }

    bool IsTracked() {
        return track_addr != 0 && track_addr_end != 0;
    }

    bool SafeToDownload() const {
        return True(flags & ImageFlagBits::GpuModified) && False(flags & (ImageFlagBits::Dirty));
    }

    void AssociateDepth(ImageId depth_image_id, u64 depth_image_uid) {
        depth_id = depth_image_id;
        depth_uid = depth_image_uid;
    }

    void DisassociateDepth() {
        depth_id = {};
        depth_uid = {};
    }

    ImageView& FindView(const ImageViewInfo& view_info, bool ensure_guest_samples = true);

    struct BackingImage;
    /// Adds a backing texture with `num_samples` samples (clamped to what the device supports),
    /// made from the image's texture description; the image's current backing is unchanged.
    BackingImage& CreateBacking(u32 num_samples);

public:
    // bbport: fields every binding of the image reads, together ahead of the large ImageInfo
    // (they were spread over several cache lines: one miss each per texture per draw).
    ImageFlagBits flags = ImageFlagBits::Dirty;
    struct {
        u32 is_bound : 1;
        u32 is_target : 1;
        u32 needs_rebind : 1;
        u32 force_general : 1;
    } binding{};
    struct {
        u32 texture : 1;
        u32 storage : 1;
        u32 render_target : 1;
        u32 depth_target : 1;
        u32 vo_surface : 1;
    } usage{};
    bool scene_proxy = false; ///< bbport: SceneTargets holds a reduced-size proxy of it
    bool gl_depth = false;    ///< fifa-hybrid: depth target of a [-1, 1] viewport pass (PushData::depth_remap)
    VAddr track_addr = 0;
    VAddr track_addr_end = 0;
    VAddr guest_begin = 0; ///< info.guest_address
    VAddr guest_end = 0;   ///< info.guest_address + info.guest_size
    BackingImage* backing{};
    /// bbport: gc tick of the last LRU touch; skips the LRU list (a cache miss) when current.
    mutable u64 lru_touched_tick = ~0ULL;
    u64 tick_accessed_last{};
    ImageId depth_id{};
    /// fifa-hybrid: Metal has no uncompressed views of block-compressed textures. A one-texel-per-
    /// block view (BC3 as RGBA32Uint, GPU texture compression) uses this separate image instead.
    ImageId block_shadow{};
    ImageId shadow_of{};       ///< set on a shadow: its block-compressed image
    bool shadow_newer = false; ///< the shadow was written after the last copy into the image
    bool shadow_current = false; ///< the shadow holds the image's blocks (no copy needed)
    u64 depth_uid{};
    u64 image_uid{};

    ImageInfo info;
    Vulkan::Runtime* runtime;
    Common::SlotVector<ImageView>* slot_image_views;
    Metal::Device* device;
    bool has_depth = false;
    bool has_stencil = false;
    /// fifa-hybrid: the backing's stencil plane was never written (a fresh texture holds
    /// undefined data, 0xff on Apple GPUs). PS4 memory starts zeroed and FIFA's UI masks (write
    /// bit 0, test EQUAL 1, write 0) assume it: the first pass that binds the stencil clears it to 0.
    bool stencil_undefined = false;
    /// fifa-hybrid: a colour R32 image that shaders compare-sample as depth (FIFA's shadow maps:
    /// R32 colour targets read with image_sample_c). Metal's sample_compare needs a depth pixel
    /// format, so the image keeps a D32 twin texture synced from the colour backing whenever it
    /// was written since (write_serial counts GPU writes and uploads). Replaces recreating the
    /// image as D32 and back every frame through 16 MB copies (undefined content at times).
    Metal::Object depth_twin;
    u64 depth_twin_synced = ~u64{0};
    u64 write_serial = 1; // read from the fault thread too (a stale read only costs a download)
    /// fifa-hybrid: on-demand readback prefetch. Large GPU-written images awaiting a CPU read
    /// fault are also downloaded asynchronously right after their write; `prefetch_landed` holds
    /// the write_serial whose bytes reached guest memory (0 none). A fault that finds it equal
    /// to write_serial needs no GPU round trip (TextureCache::ResolvePrefetchedOndemand).
    std::shared_ptr<std::atomic<u64>> prefetch_landed;
    [[nodiscard]] bool PrefetchLanded() const {
        return prefetch_landed && prefetch_landed->load(std::memory_order_acquire) != 0 &&
               prefetch_landed->load(std::memory_order_acquire) == write_serial;
    }
    [[nodiscard]] bool DepthTwinEligible() const {
        return !has_depth && info.pixel_format == vk::Format::eR32Sfloat && info.num_samples == 1 &&
               !info.props.is_block && !info.props.is_volume;
    }
    /// Sample counts the device supports for textures: `mask & N` for N = 1, 2, 4, 8 (the bits of
    /// vk::SampleCountFlags).
    u32 supported_samples_mask = 1;
    /// The texture description backings are made from (samples are per backing).
    Metal::TextureDesc texture_desc{};
    /// bbport: the ShaderWrite and PixelFormatView usages the image may need. Metal compresses
    /// private textures losslessly only without both, so single-sample backings are made without
    /// them and get both, once, when a storage binding or a view in another pixel format needs
    /// one (NeededUsage, Runtime::UpgradeUsage). BB_LAZY_TEXTURE_USAGE=0: always both (as before).
    bool allowed_shader_write = false;
    bool allowed_format_views = false;
    static constexpr u32 UsageShaderWrite = 1;
    static constexpr u32 UsageFormatViews = 2;
    /// The usages (Usage*) a view `view_info` of the current backing needs and it lacks.
    [[nodiscard]] u32 NeededUsage(const ImageViewInfo& view_info) const;
    /// The usages (Usage*) the current backing has.
    [[nodiscard]] u32 CurrentUsage() const;
    /// A new image (no contents, no views yet) that needs usages (Usage*): the binding that
    /// creates it, or the image it replaces (a larger mip chain), needs or had them. Its backing
    /// is made again with both (no copy, no upgrade later).
    void ApplyNewUsage(u32 needed);

    struct BackingImage {
        Metal::Object texture; ///< id<MTLTexture>
        Metal::TextureDesc desc;
        u64 allocated_size{};
        boost::container::small_vector<ImageViewInfo, 4> image_view_infos;
        boost::container::small_vector<ImageViewId, 4> image_view_ids;
        u32 last_view = 0; ///< bbport: index of the view FindView returned last
        u32 num_samples;
    };
    std::deque<BackingImage> backing_images;
    boost::container::static_vector<u64, 16> mip_hashes{};
    u64 lru_id{};
    u64 hash{};

private:
    static Common::IncrementalIdProvider<u64> global_image_uid;
};

} // namespace VideoCore
