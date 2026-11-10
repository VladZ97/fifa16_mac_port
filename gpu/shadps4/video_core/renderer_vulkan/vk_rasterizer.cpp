#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <unordered_set>
// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
// SPDX-License-Identifier: GPL-2.0-or-later

#include <xxhash.h>
#include "video_core/renderer_vulkan/ui_composition.h"
#include "bbport_toggles.h"
#include "bbport_cpu.h"
#include "video_core/renderer_vulkan/vk_frame_capture.h"
#include "common/debug.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "bbport_metalfx.h"
#include "ffx_metal/ffx_metal.h"
#include "bbport_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_depth_attachment.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/blit_helper.h" // 0.5.1-c: copy dispatch pass
#include "video_core/renderer_metal/mtl_texture.h"
#include <vulkan/vulkan_format_traits.hpp>
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "bbport_threads.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {
extern std::set<VAddr> g_bb_dump_volume_addrs;
}

namespace Vulkan {
static int PassLabels();
static u64 g_probe_key = 0; // DEBUG(bbport): compute key of the pipeline being bound

// fifa-hybrid: a pass whose clip space is [-w, w] stores window depth in [-1, 1] on the PS4
// (viewport zoffset 0, zscale 1); Metal clamps the viewport to [0, 1] and the vertex stage
// stores (z + 1) / 2 instead, so shaders reading such a depth target get the inverse mapping
// (PushData::depth_remap). BB_DEPTH_REMAP=0 disables.
static bool DepthRemapEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_DEPTH_REMAP"); // opt-in: it blurred close-ups (DOF
        return env && env[0] == '1';                       // reads the remapped depth) and did not
                                                           // change the brightness
    }();
    return enabled;
}

static bool GlDepthPass(const AmdGpu::Regs& regs) {
    return DepthRemapEnabled() &&
           regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
}

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(Metal::Device& device_, Scheduler& scheduler_, Runtime& runtime_,
                       AmdGpu::Liverpool* liverpool_)
    : device{device_}, scheduler{scheduler_}, runtime{runtime_}, page_manager{this},
      buffer_cache{device, scheduler, runtime, liverpool_, texture_cache, page_manager},
      texture_cache{device, scheduler, runtime, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{device, scheduler, liverpool, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    // Metal has no null descriptors. Shader writes through a null binding land here and are not
    // discarded, so the contents are only mostly zero (docs/MACOS_BACKLOG.md).
    null_buffer.emplace(device, VideoCore::MemoryType::HostUncached, 0, NullBufferSize,
                        "NullBuffer");
    ASSERT_MSG(!null_buffer->mapped_data.empty(), "Null buffer is not host visible");
    std::memset(null_buffer->mapped_data.data(), 0, NullBufferSize);
    // Before the rasterizer is bound: Liverpool enqueues buffers only once it sees it.
    draw_prep = std::make_unique<DrawPreparation>(pipeline_cache);
    scene_targets = std::make_unique<SceneTargets>(device, scheduler, runtime, texture_cache);
    // bbport: the upscaler that runs, before anything reads it (object motion's on/off here,
    // the upscaler's resources): unavailable providers fall back -- MetalFX only where our device
    // has the temporal scaler, FSR 3.1 on Apple7 and later (the FidelityFX Metal back end) -- and
    // the fallback is never saved.
    BbSettings::ConfigureUpscalerSupport(BbMetalFx::Supported(device), FfxMetal::Supported(device));
    // Object motion first: it fixes the buffer addresses the motion shader variants embed.
    object_motion = std::make_unique<ObjectMotion>(device, scheduler);
    camera_motion = std::make_unique<CameraMotion>(device, scheduler, texture_cache, runtime);
    camera_motion->SetObjectMotion(object_motion.get());
    upscaler = std::make_unique<TemporalUpscaler>(device, scheduler, texture_cache, runtime,
                                                  *camera_motion, *scene_targets);
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);

    GpuProfiler::Init(device, scheduler);
    if (DrawPipeWanted()) {
        constant_ring = std::make_unique<ConstantRing>(device, scheduler);
        draw_pipe = std::make_unique<DrawPipe>(&RunDrawPacket, this);
    }
    scheduler.SetPassEndHook(
        [](void* self) {
            auto* r = static_cast<Rasterizer*>(self);
            r->FlushDeferredDispatches(); // first: TryDeferDispatch keeps clear of pending copies
            FlushDeferredCopies(*r);
        },
        this);
    // bbport: the frame capture's event trace also lists the SRT walkers' guest reads.
    if (([] { static const bool v = std::getenv("BB_CAPTURE_TRIGGER") != nullptr; return v; }())) {
        Shader::srt_read_hook = [](u64 address, u32 size) {
            if (FrameCapture::Tracing()) {
                FrameCapture::Trace(fmt::format("W {:#x} {}", address, size));
            }
        };
    }
    // bbport: this thread joins the texture binding helper before it changes image state.
    runtime.SetImageAccessHook(&JoinBindHelper, this);
}

Rasterizer::~Rasterizer() = default;

inline const AmdGpu::Regs& Rasterizer::Regs() const {
    return stage_regs_used.load(std::memory_order_relaxed) && stage_regs ? *stage_regs
                                                                          : liverpool->regs;
}

inline AmdGpu::CbDbExtent Rasterizer::CbExtent(u32 index) const {
    return stage_regs_used.load(std::memory_order_relaxed) && stage_regs
               ? pipe_cb_extent[index]
               : liverpool->last_cb_extent[index];
}

inline const AmdGpu::ComputeProgram& Rasterizer::CsRegs() const {
    return stage_regs_used.load(std::memory_order_relaxed) && stage_cs ? *stage_cs
                                                                        : liverpool->GetCsRegs();
}

inline AmdGpu::CbDbExtent Rasterizer::DbExtent() const {
    return stage_regs_used.load(std::memory_order_relaxed) && stage_regs ? pipe_db_extent
                                                                          : liverpool->last_db_extent;
}

namespace {
/// magic_enum::enum_contains, from a table: it scans every enumerator (per texture per draw).
bool IsKnownFormat(AmdGpu::DataFormat data_fmt, AmdGpu::NumberFormat num_fmt) {
    static const auto tables = [] {
        std::pair<std::array<bool, 256>, std::array<bool, 256>> t{};
        for (const auto value : magic_enum::enum_values<AmdGpu::DataFormat>()) {
            if (static_cast<u32>(value) < 256) {
                t.first[static_cast<u32>(value)] = true;
            }
        }
        for (const auto value : magic_enum::enum_values<AmdGpu::NumberFormat>()) {
            if (static_cast<u32>(value) < 256) {
                t.second[static_cast<u32>(value)] = true;
            }
        }
        return t;
    }();
    const u32 d = static_cast<u32>(data_fmt), n = static_cast<u32>(num_fmt);
    return d < 256 && n < 256 && tables.first[d] && tables.second[n];
}

/// Recording thread: the Metal pipelines of a graphics pipeline, waiting for the creation pool
/// in shader_compile=wait mode. Failed (or no render pipeline): none, so the draw helper skips
/// the draw and counts it once (skipped_no_pipeline).
Metal::Draw::Runs::Graphics DrawPipelines(const GraphicsPipeline& pipeline) {
    pipeline.WaitReady();
    if (pipeline.Failed() || !pipeline.Render()) {
        return {};
    }
    // Guest shaders reach only the arena (ordered by the Runtime's barriers: a plain access for
    // the Encoder) and tracked buffers.
    return {pipeline.Render(), pipeline.Tessellation(), pipeline.Geometry(), Metal::Access::Read};
}

/// Recording thread: the same for a compute pipeline (an empty view: skipped and counted).
Metal::Draw::ComputeView ComputeView(const ComputePipeline& pipeline) {
    pipeline.WaitReady();
    if (pipeline.Failed() || !pipeline.Compute()) {
        return {};
    }
    Metal::Draw::ComputeView view = Metal::Draw::View(*pipeline.Compute());
    view.buffers = Metal::Access::Read; // as DrawPipelines

    return view;
}

/// Producer side: the draw may leave patch draws waiting in the draw runs (a tessellated draw;
/// known from the stages before the pipeline is built).
bool IsTessellated(const GraphicsPipeline& pipeline) {
    return pipeline.GetStages()[u32(Shader::SwStage::TessellationControl)] != nullptr;
}
} // namespace

bool Rasterizer::FilterDraw() {
    const auto& regs = Regs();
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

template <typename... Parts>
VideoCore::ImageId Rasterizer::FindTargetMemoized(VideoCore::TextureCache::ImageDesc& desc,
                                                  LastTarget& last, auto&& make_desc,
                                                  const Parts&... parts) {
    static_assert((sizeof(Parts) + ...) <= 256, "target memo key too large");
    std::array<u8, 256> key{};
    u32 key_size = 0;
    const auto append = [&](const auto& part) {
        std::memcpy(key.data() + key_size, &part, sizeof(part));
        key_size += sizeof(part);
    };
    (append(parts), ...);
    const u64 generation = texture_cache.RegistryGeneration();
    const bool memo_enabled = !BbToggle::Disabled(BbToggle::TextureBindingMemo);
    if (memo_enabled && last.generation == generation && last.key_size == key_size &&
        std::memcmp(last.key.data(), key.data(), key_size) == 0) {
        texture_cache.MarkFound(last.image_id);
        return last.image_id;
    }
    const auto remember = [&](VideoCore::ImageId image_id) {
        last.key = key;
        last.key_size = key_size;
        last.generation = generation;
        last.image_id = image_id;
    };
    u64 hash = 0xCBF29CE484222325ull;
    for (u32 i = 0; i < key_size; i += 8) {
        u64 word;
        std::memcpy(&word, key.data() + i, 8);
        hash = (hash ^ word) * 0x100000001B3ull;
        hash ^= hash >> 29;
    }
    auto& memo = target_memo[hash % target_memo.size()];
    if (memo.generation == generation && memo.key_size == key_size &&
        std::memcmp(memo.key.data(), key.data(), key_size) == 0 && memo_enabled) {
        desc = memo.desc;
        texture_cache.MarkFound(memo.image_id);
        remember(memo.image_id);
        return memo.image_id;
    }
    make_desc();
    const VideoCore::ImageId image_id = texture_cache.FindImage(desc);
    if (generation == texture_cache.RegistryGeneration()) {
        memo.key = key;
        memo.key_size = key_size;
        memo.generation = generation;
        memo.image_id = image_id;
        memo.desc = desc;
        remember(image_id);
    } else {
        last.generation = ~0ULL;
    }
    return image_id;
}

}
namespace VideoCore { extern std::set<VAddr> g_bb_dump_extra; }
namespace Vulkan {
void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = Regs();
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (([] { static const bool v = std::getenv("BB_BCVIEW_LOG") != nullptr; return v; }()) && col_buf && col_buf.Address() >= 0x250000000ull &&
            col_buf.Address() < 0x259000000ull) {
            static std::set<u64> seen_rt;
            const auto* ps_i = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
            const u64 ps_h = ps_i ? ps_i->pgm_hash : 0;
            u32 bcr; std::memcpy(&bcr, &regs.blend_control[cb], 4);
            const u64 k = col_buf.Address() ^ (u64(col_buf.GetDataFmt()) << 40) ^ (u64(target_mask) << 48) ^ (ps_h * 31) ^ (u64(bcr) << 8);
            if (seen_rt.insert(k).second && seen_rt.size() < 80) {
                std::fprintf(stderr, "[STAGERT] ps %016llx blend %#x bypass %d cb%d at %#llx dfmt %u nfmt %u pitch %u h %u mask %#x mrt %#x skip %d export %#x tile %u\n",
                             (unsigned long long)ps_h, bcr, int(col_buf.info.blend_bypass),
                             cb, (unsigned long long)col_buf.Address(), u32(col_buf.GetDataFmt()),
                             u32(col_buf.GetNumberFmt()), col_buf.Pitch(), col_buf.Height(), target_mask,
                             key.mrt_mask, int(skip_cb_binding), u32(regs.color_export_format.raw), u32(col_buf.GetTileMode()));
                if (ps_i) for (const auto& r : ps_i->images) { const auto rt = r.GetSharp(*ps_i);
                    if (u32(rt.GetDataFmt()) >= 35 && u32(rt.GetDataFmt()) <= 41) VideoCore::g_bb_dump_extra.insert(rt.Address());
                    std::fprintf(stderr, "[STAGEPS]   tex %#llx f%u n%u %ux%u t%u sel%u%u%u%u mips%u\n", (unsigned long long)rt.Address(), u32(rt.GetDataFmt()), u32(rt.GetNumberFmt()), rt.width + 1, rt.height + 1, u32(rt.tiling_index), u32(rt.dst_sel_x), u32(rt.dst_sel_y), u32(rt.dst_sel_z), u32(rt.dst_sel_w), u32(rt.last_level - rt.base_level + 1)); }
            }
        }
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = CbExtent(cb);
        const u32 tag = 1;
        image_id = bound_images.emplace_back(FindTargetMemoized(
            desc, last_targets[cb], [&] { std::construct_at(&desc, col_buf, hint); }, tag,
            col_buf, hint));
        bound_images_written.push_back(1);
        bound_images_prev_write.push_back(0);
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    // bbport: the pipeline key uses the same rule for its depth/stencil formats.
    if (BoundDepthAttachment(regs).Bound()) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = DbExtent();
        auto& [image_id, desc] = db_desc;
        const u32 tag = 2;
        image_id = bound_images.emplace_back(FindTargetMemoized(
            desc, last_targets[AmdGpu::NUM_COLOR_BUFFERS],
            [&] {
                std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                                  htile_address, hint);
            },
            tag, regs.depth_buffer, regs.depth_view, regs.depth_control, htile_address, hint));
        bound_images_written.push_back(1);
        bound_images_prev_write.push_back(0);
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.first = {};
    }
    // bbport: the G-buffer pass (5+ color targets) holds the scene depth, and its constants the
    // main camera (shadow passes bind the same layout with the light's camera).
    const bool scene_gbuffer = std::popcount(key.mrt_mask) >= 5 && db_desc.first;
    if (scene_gbuffer && !upscaler->Enabled() && BbSettings::Get().ultrawide_width) {
        const auto& size = texture_cache.GetImage(db_desc.first).info.size;
        auto& settings = BbSettings::Get();
        if (settings.active_render_width != int(size.width) ||
            settings.active_render_height != int(size.height)) {
            settings.active_render_width = size.width;
            settings.active_render_height = size.height;
            std::printf("Native ultrawide: observed scene depth %ux%u (no host scaling)\n",
                        size.width, size.height);
        }
    }
    gbuffer_draw = camera_motion->Enabled() && scene_gbuffer;
    if (gbuffer_draw) {
        camera_motion->OnGBufferPass(db_desc.first);
    }
    if (upscaler->Enabled() && cb_descs[0].first) {
        upscaler->OnColorTarget(cb_descs[0].first);
    }
    // bbport: scene color: a full-size RGBA16F target drawn with the scene depth.
    if (upscaler->Enabled() && db_desc.first && db_desc.first == camera_motion->Depth() &&
        cb_descs[0].first && std::popcount(key.mrt_mask) <= 2) {
        const auto& color = texture_cache.GetImage(cb_descs[0].first);
        const auto& depth = texture_cache.GetImage(db_desc.first);
        if (color.info.pixel_format == vk::Format::eR16G16B16A16Sfloat &&
            color.info.size.width == depth.info.size.width &&
            color.info.size.height == depth.info.size.height) {
            upscaler->OnSceneColor(cb_descs[0].first);
            // Blended geometry without depth writes (not full-screen passes): transparents and
            // effects start. Blended layers writing depth (decals, wet/blood ground) are
            // surfaces and stay in the history.
            if (regs.blend_control[0].enable && !regs.color_buffers[0].info.blend_bypass &&
                !regs.depth_control.depth_write_enable &&
                (regs.num_indices > 6 || regs.num_instances.NumInstances() > 1)) {
                upscaler->OnBlendedSceneDraw();
            } else if (regs.num_indices <= 6 && regs.num_instances.NumInstances() <= 1) {
                // A full-screen pass over the scene (the fog composite): the transparents are in.
                upscaler->OnSceneComposite();
            }
        }
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::Info& info,
    const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.UserData()[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.UserData()[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = Regs().color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, CbExtent(0));
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    runtime.ClearImage(&image, desc.view_info.range, clear_value);
    ScopeMarkerEnd();
}

// bbport: two-stage draw pipeline (vk_draw_pipe.h). Packet layout, 8-byte aligned parts:
// DrawPacket, u16 block indices, u32 block words (RegDirty::BlockWords each), then per stage a
// PacketStage with its user data and flattened user data, then (verification) the full file.
namespace {
enum class PacketKind : u32 { Draw, Task, Dispatch, Special, Indirect, IndirectDispatch };
struct TaskPacket {
    PacketKind kind;
    u32 size;
    Rasterizer::OrderedTask task;
};
struct DrawPacket {
    PacketKind kind;
    const Pipeline* pipeline;
    const PreparedDraw* prepared;
    u32 index_offset;
    u8 is_indexed;
    u8 regs_reset;
    u16 num_blocks;
    u32 num_stages;
    u32 verify;
    std::array<AmdGpu::CbDbExtent, AmdGpu::NUM_COLOR_BUFFERS> cb_extent;
    AmdGpu::CbDbExtent db_extent;
    u64 ring_end; ///< constant ring position after this packet's copies
};
struct PacketStage {
    const Shader::Info* info;
    VAddr pgm_base;
    u32 ud_size;
    u32 flat_size;
    u32 num_ring;
    u32 pad;
};
constexpr u32 AlignPacket(u32 size) {
    return (size + 7) & ~7u;
}
u32 VerifyInterval() {
    static const u32 interval = [] {
        const char* env = std::getenv("BB_PIPE_VERIFY");
        return env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 0u;
    }();
    return interval;
}
} // namespace

bool Rasterizer::DrawPipeWanted() {
    const char* env = std::getenv("BB_DRAW_PIPE");
    if (env && env[0]) {
        return env[0] == '1';
    }
    // Stage B spins while draws flow. Measured ahead with 16 threads (+19%) and with 4 cores /
    // 8 threads (taskset, Steam Deck-like: +18%).
    return BbThreads::Available() >= 8;
}

bool Rasterizer::UseDrawPipe() const {
    return draw_pipe && !host_markers_enabled && !BbToggle::Disabled(BbToggle::DrawPipeline);
}

bool Rasterizer::OnStageA() const {
    return std::this_thread::get_id() == liverpool->GetGpuCommandProcessorThread();
}

namespace {
/// Stage A: cycles waited per drain site (BB_FRAME_STATS), keyed by function name and line.
std::unordered_map<const char*, std::unordered_map<u32, u64>> drain_sites;
} // namespace

void Rasterizer::DrainDrawPipe(u32 reason, u32 line, const char* function) {
    if (!draw_pipe || !OnStageA()) {
        return;
    }
    static const bool stats = std::getenv("BB_FRAME_STATS") != nullptr;
    if (!stats) {
        draw_pipe->Drain(reason);
        return;
    }
    const u64 before = draw_pipe->drain_cycles;
    draw_pipe->Drain(reason);
    if (const u64 waited = draw_pipe->drain_cycles - before) {
        drain_sites[function][line] += waited;
    }
}

bool Rasterizer::IsGpuSideThread() const {
    return OnStageA() || DrawPipe::OnStageB();
}

bool Rasterizer::IsGpuSideThreadId(u32 tid) const {
    // The userfaultfd thread handles the fault while the faulting thread waits: without locks
    // only if that thread owns the caches. The GPU command thread does not while the draw
    // recording thread may be running, so its faults take the locked path of guest threads.
    if (draw_pipe) {
        return tid == draw_pipe->StageBThreadId();
    }
    return tid == liverpool->GetGpuCommandProcessorThreadId();
}

namespace {
std::mutex vb_w_mutex;
std::vector<std::tuple<VAddr, u64, char>> vb_w_ranges;
} // namespace

void VbNoteWrite(VAddr address, u64 size, char tag) {
    static const bool vb_log_w = std::getenv("BB_VB_LOG") != nullptr;
    if (vb_log_w && size != 0) {
        std::scoped_lock lk{vb_w_mutex};
        if (vb_w_ranges.size() < 400000) {
            vb_w_ranges.emplace_back(address, size, tag);
        }
    }
}

void Rasterizer::NotePendingGpuWrite(VAddr address, u64 size) {
    static const bool vb_log_w = std::getenv("BB_VB_LOG") != nullptr;
    if (vb_log_w) {
        std::scoped_lock lk{vb_w_mutex};
        if (vb_w_ranges.size() < 400000) {
            vb_w_ranges.emplace_back(address, size, 'P');
        }
    }
    if (size != 0 && size <= 64) {
        // fifa-hybrid: tiny shader-written buffers (occlusion counters, flags) that the guest
        // polls are written back after the submission (FlushTinyWritten).
        std::scoped_lock lk{tiny_written_mutex};
        tiny_written.emplace_back(address, size);
    }
    if (!draw_pipe || !size) {
        return;
    }
    // Draws write the same buffers over and over: an entry for the range only moves on.
    const u64 position = draw_pipe->Head();
    const VAddr end = address + size;
    for (auto& write : pending_writes) {
        if (address <= write.end && write.begin <= end) {
            write.begin = std::min(write.begin, address);
            write.end = std::max(write.end, end);
            write.position = position;
            pending_min = std::min(pending_min, address);
            pending_max = std::max(pending_max, end);
            return;
        }
    }
    pending_writes.push_back({address, end, position});
    pending_min = std::min(pending_min, address);
    pending_max = std::max(pending_max, end);
}

bool Rasterizer::PendingWriteOverlaps(VAddr address, u64 size) {
    if (pending_writes.empty() || address >= pending_max || address + size <= pending_min) {
        return false;
    }
    // Now and then drop entries whose writes the recording thread has done: the bounds shrink.
    if ((++pending_checks & 63) == 0) {
        std::erase_if(pending_writes, [&](const PendingWrite& write) {
            return draw_pipe->Reached(write.position);
        });
        pending_min = ~VAddr{0};
        pending_max = 0;
        for (const auto& write : pending_writes) {
            pending_min = std::min(pending_min, write.begin);
            pending_max = std::max(pending_max, write.end);
        }
    }
    const VAddr end = address + size;
    for (const auto& write : pending_writes) {
        if (address < write.end && write.begin < end && !draw_pipe->Reached(write.position)) {
            return true;
        }
    }
    return false;
}

void Rasterizer::CollectRingBindings(const Shader::Info& stage, const PreparedDraw* prepared,
                           boost::container::static_vector<RingBinding, Shader::NUM_BUFFERS>& out) {
    const PreparedStage* prepared_stage = nullptr;
    if (prepared && !BbToggle::Disabled(BbToggle::PreparedResources)) {
        for (u32 i = 0; i < prepared->num_stages; ++i) {
            const auto& candidate = prepared->stages[i];
            if (&candidate.program->info == &stage &&
                candidate.num_buffers == stage.buffers.size()) {
                prepared_stage = &candidate;
                break;
            }
        }
    }
    const auto& limits = device.GetLimits();
    const u64 alignment = std::max<u64>(limits.storage_alignment, limits.uniform_alignment);
    const auto copy = [&](u32 index, const void* source, VAddr address, u64 size) {
        const auto offset = constant_ring->Allocate(size, alignment);
        if (!offset) {
            return;
        }
        u8* dst = constant_ring->Data(*offset);
        if (source) {
            std::memcpy(dst, source, size);
        } else {
            memory->CopySparseMemory(address, dst, size);
            if (FrameCapture::Tracing()) {
                FrameCapture::Trace(fmt::format("R {} {:08x} {:#x} {}", u32(stage.sw_stage),
                                                u32(stage.pgm_hash), address, size));
            }
        }
        out.push_back({index, static_cast<u32>(size), *offset, address});
    };
    for (u32 index = 0; index < stage.buffers.size(); ++index) {
        const auto& desc = stage.buffers[index];
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::Flatbuf &&
                !stage.flattened_ud_buf.empty()) {
                copy(index, stage.flattened_ud_buf.data(), 0,
                     stage.flattened_ud_buf.size() * sizeof(u32));
            }
            continue;
        }
        const auto vsharp = prepared_stage ? prepared_stage->buffer_sharps[index]
                                           : desc.GetSharp(stage);
        const VAddr address = vsharp.base_address;
        if (address == 0 || vsharp.GetSize() == 0) {
            continue;
        }
        const u64 size = memory->ClampRangeSize(address, vsharp.GetSize());
        if (desc.is_written) {
            NotePendingGpuWrite(address, size);
            continue;
        }
        // The stream path of BufferCache::ObtainBuffer, taken here: small, read-only, not
        // written by the GPU (now or by work still queued for the recording thread).
        if (size == 0 || size > VideoCore::BufferCache::STREAM_THRESHOLD ||
            buffer_cache.IsRegionGpuModified(address, size) || PendingWriteOverlaps(address, size)) {
            continue;
        }
        // fifa-hybrid: memory a GPU-written image still owns (on-demand readback pending) is
        // stale in RAM; leave it to ObtainBuffer, which downloads the image first.
        if (!([] { static const bool v = std::getenv("BB_RB_RINGOLD") != nullptr; return v; }()) && texture_cache.HasOndemandIn(address, size)) {
            static int rn = 0;
            if (([] { static const bool v = std::getenv("BB_VB_LOG") != nullptr; return v; }()) && rn++ < 2000) {
                std::fprintf(stderr, "[RBQ] ring-skip addr=%#llx size=%#llx\n",
                             (unsigned long long)address, (unsigned long long)size);
            }
            continue;
        }
        copy(index, nullptr, address, size);
    }
}

void Rasterizer::PostDraw(const Pipeline* pipeline, const PreparedDraw* used_prepared,
                          bool is_indexed, u32 index_offset, const AmdGpu::ComputeProgram* cs,
                          const IndirectDraw* indirect) {
    auto& dirty = liverpool->pipe_dirty;
    if (!pipe_synced) {
        // Stage B has not run yet: its register copy starts as this one.
        pipe_regs = liverpool->regs;
        dirty.Clear();
        pipe_synced = true;
    }
    std::array<u16, AmdGpu::RegDirty::NumBlocks> blocks;
    u32 num_blocks = 0;
#if defined(__GLIBCXX__)
    for (size_t block = dirty.blocks._Find_first(); block < dirty.blocks.size();
         block = dirty.blocks._Find_next(block)) {
        blocks[num_blocks++] = static_cast<u16>(block);
    }
#else
    for (size_t block = 0; block < dirty.blocks.size(); ++block) {
        if (dirty.blocks.test(block)) {
            blocks[num_blocks++] = static_cast<u16>(block);
        }
    }
#endif
    const auto stages =
        pipeline ? pipeline->GetStages() : std::span<const Shader::Info* const>{};
    // Constants: copied here into the ring, the recording thread only binds them.
    thread_local std::array<boost::container::static_vector<RingBinding, Shader::NUM_BUFFERS>,
                            Shader::MaxStageTypes>
        rings;
    u32 num_stages = 0;
    u32 size = sizeof(DrawPacket) + AlignPacket(num_blocks * sizeof(u16)) +
               num_blocks * AmdGpu::RegDirty::BlockWords * sizeof(u32);
    for (const auto* stage : stages) {
        if (stage) {
            auto& ring = rings[num_stages];
            ring.clear();
            if (constant_ring && !BbToggle::Disabled(BbToggle::ConstantRing)) {
                CollectRingBindings(*stage, used_prepared, ring);
            }
            ++num_stages;
            size += sizeof(PacketStage) +
                    AlignPacket(u32(stage->user_data.size() + stage->flattened_ud_buf.size()) *
                                sizeof(u32)) +
                    u32(ring.size() * sizeof(RingBinding));
        }
    }
    if (cs) {
        size += AlignPacket(sizeof(AmdGpu::ComputeProgram));
    }
    if (indirect) {
        size += AlignPacket(sizeof(IndirectDraw));
    }
    const u32 interval = VerifyInterval();
    const bool verify = interval && (draw_pipe->packets % interval) == 0;
    if (verify) {
        size += sizeof(AmdGpu::Regs);
    }

    u8* out = draw_pipe->Begin(size);
    u8* const start = out;
    auto& packet = *reinterpret_cast<DrawPacket*>(out);
    packet.kind = cs         ? (indirect ? PacketKind::IndirectDispatch : PacketKind::Dispatch)
                  : indirect ? PacketKind::Indirect
                  : pipeline ? PacketKind::Draw
                             : PacketKind::Special;
    packet.pipeline = pipeline;
    packet.prepared = used_prepared;
    packet.index_offset = index_offset;
    packet.is_indexed = is_indexed;
    packet.regs_reset = dirty.reset;
    packet.num_blocks = static_cast<u16>(num_blocks);
    packet.num_stages = num_stages;
    packet.verify = verify;
    packet.cb_extent = liverpool->last_cb_extent;
    packet.db_extent = liverpool->last_db_extent;
    packet.ring_end = constant_ring ? constant_ring->Position() : 0;
    out += sizeof(DrawPacket);
    std::memcpy(out, blocks.data(), num_blocks * sizeof(u16));
    out += AlignPacket(num_blocks * sizeof(u16));
    for (u32 i = 0; i < num_blocks; ++i) {
        std::memcpy(out,
                    liverpool->regs.reg_array.data() + blocks[i] * AmdGpu::RegDirty::BlockWords,
                    AmdGpu::RegDirty::BlockWords * sizeof(u32));
        out += AmdGpu::RegDirty::BlockWords * sizeof(u32);
    }
    u32 stage_index = 0;
    for (const auto* stage : stages) {
        if (!stage) {
            continue;
        }
        const auto& ring = rings[stage_index++];
        auto& header = *reinterpret_cast<PacketStage*>(out);
        header = {stage,
                  stage->pgm_base,
                  static_cast<u32>(stage->user_data.size()),
                  static_cast<u32>(stage->flattened_ud_buf.size()),
                  static_cast<u32>(ring.size()),
                  0};
        out += sizeof(PacketStage);
        std::memcpy(out, ring.data(), ring.size() * sizeof(RingBinding));
        out += ring.size() * sizeof(RingBinding);
        std::memcpy(out, stage->user_data.data(), stage->user_data.size_bytes());
        std::memcpy(out + stage->user_data.size_bytes(), stage->flattened_ud_buf.data(),
                    stage->flattened_ud_buf.size() * sizeof(u32));
        out += AlignPacket(u32(stage->user_data.size() + stage->flattened_ud_buf.size()) *
                           sizeof(u32));
    }
    if (cs) {
        std::memcpy(out, cs, sizeof(AmdGpu::ComputeProgram));
        out += AlignPacket(sizeof(AmdGpu::ComputeProgram));
    }
    if (indirect) {
        std::memcpy(out, indirect, sizeof(IndirectDraw));
        out += AlignPacket(sizeof(IndirectDraw));
    }
    if (verify) {
        std::memcpy(out, &liverpool->regs, sizeof(AmdGpu::Regs));
        out += sizeof(AmdGpu::Regs);
    }
    dirty.Clear();
    draw_pipe->Commit(static_cast<u32>(out - start));
    PrintPipeStats();
}

void Rasterizer::RetireSubmission() {
    if (!draw_pipe || !OnStageA()) {
        return;
    }
    while (!pipe_keepalive.empty() && draw_pipe->Reached(pipe_keepalive.front().first)) {
        pipe_keepalive.pop_front();
    }
    if (!draw_pipe->Idle()) {
        pipe_keepalive.emplace_back(draw_pipe->Head(), draw_prep->KeepCurrent());
    }
}

void Rasterizer::DumpStall() {
    std::fprintf(stderr, "[STALL] draw pipe: %s, head %llu; deferred copies pending %d, deferred dispatches %zu\n",
                 draw_pipe ? (draw_pipe->Idle() ? "idle" : "BUSY") : "off",
                 (unsigned long long)DrawPipeHead(), int(DeferredCopiesPending()), deferred_dispatches.size());
    scheduler.DumpStall();
}

bool Rasterizer::RunInOrder(OrderedTask task, const void* data, u32 size, u64 toggle) {
    if (!UseDrawPipe() || BbToggle::Disabled(toggle)) {
        DrainDrawPipe(DrawPipe::ReasonRasterizer);
        task(*this, static_cast<const u8*>(data));
        return false;
    }
    const u32 total = sizeof(TaskPacket) + AlignPacket(size);
    u8* out = draw_pipe->Begin(total);
    *reinterpret_cast<TaskPacket*>(out) = {PacketKind::Task, size, task};
    std::memcpy(out + sizeof(TaskPacket), data, size);
    draw_pipe->Commit(total);
    return true;
}

void Rasterizer::RunDrawPacket(void* context, const u8* data, u32 size) {
    auto& self = *static_cast<Rasterizer*>(context);
    if (*reinterpret_cast<const PacketKind*>(data) == PacketKind::Task) {
        const auto& task = *reinterpret_cast<const TaskPacket*>(data);
        task.task(self, data + sizeof(TaskPacket));
        return;
    }
    const auto& packet = *reinterpret_cast<const DrawPacket*>(data);
    const u8* in = data + sizeof(DrawPacket);
    const auto* blocks = reinterpret_cast<const u16*>(in);
    in += AlignPacket(packet.num_blocks * sizeof(u16));
    auto& regs = self.pipe_regs;
    if (packet.regs_reset) {
        regs.SetDefaults();
    }
    for (u32 i = 0; i < packet.num_blocks; ++i) {
        std::memcpy(regs.reg_array.data() + blocks[i] * AmdGpu::RegDirty::BlockWords, in,
                    AmdGpu::RegDirty::BlockWords * sizeof(u32));
        in += AmdGpu::RegDirty::BlockWords * sizeof(u32);
    }
    ASSERT(packet.num_stages <= Shader::Info::MaxUdSnapshots);
    Shader::Info::snapshots_in_use.store(true, std::memory_order_relaxed); // 0.5.1-c
    for (u32 i = 0; i < packet.num_stages; ++i) {
        const auto& stage = *reinterpret_cast<const PacketStage*>(in);
        in += sizeof(PacketStage);
        self.ring_stages[i] = {stage.info, reinterpret_cast<const RingBinding*>(in),
                               stage.num_ring};
        in += stage.num_ring * sizeof(RingBinding);
        const auto* words = reinterpret_cast<const u32*>(in);
        Shader::Info::ud_snapshots[i] = {stage.info, words, stage.ud_size, words + stage.ud_size,
                                         stage.flat_size, stage.pgm_base};
        in += AlignPacket((stage.ud_size + stage.flat_size) * sizeof(u32));
    }
    Shader::Info::num_ud_snapshots = packet.num_stages;
    self.num_ring_stages = packet.num_stages;
    const AmdGpu::ComputeProgram* cs = nullptr;
    if (packet.kind == PacketKind::Dispatch || packet.kind == PacketKind::IndirectDispatch) {
        cs = reinterpret_cast<const AmdGpu::ComputeProgram*>(in);
        in += AlignPacket(sizeof(AmdGpu::ComputeProgram));
    }
    const IndirectDraw* indirect = nullptr;
    if (packet.kind == PacketKind::Indirect || packet.kind == PacketKind::IndirectDispatch) {
        indirect = reinterpret_cast<const IndirectDraw*>(in);
        in += AlignPacket(sizeof(IndirectDraw));
    }
    if (packet.verify) {
        // The resource tables the GPU thread walked must read the same now: else a write
        // still queued here (WriteData, DMA) changed them after that thread read them.
        static u32 flat_reports = 0;
        for (u32 i = 0; i < packet.num_stages && flat_reports < 32; ++i) {
            const auto& snapshot = Shader::Info::ud_snapshots[i];
            const auto& srt = snapshot.info->srt_info;
            // Stages the pipeline selection did not refresh (no user data) keep their tables.
            if (!srt.walker_func || snapshot.flat_size == 0 || snapshot.user_data_size == 0) {
                continue;
            }
            std::vector<u32> flat(snapshot.flat_size);
            std::memcpy(flat.data(), snapshot.user_data,
                        std::min(snapshot.user_data_size, snapshot.flat_size) * sizeof(u32));
            // Pointers in the tables may be stale by now: a fault only skips the check.
            BbRecoverBuf recover;
            if (BB_RECOVER_SET(recover)) {
                runtime_fault_recover = nullptr;
                continue;
            }
            runtime_fault_recover = &recover;
            Shader::RunSrtWalker(srt.walker_func, snapshot.user_data, flat.data());
            runtime_fault_recover = nullptr;
            if (std::memcmp(flat.data(), snapshot.flat, snapshot.flat_size * sizeof(u32)) != 0) {
                u32 first = 0, count = 0;
                for (u32 w = 0; w < snapshot.flat_size; ++w) {
                    if (flat[w] != snapshot.flat[w]) {
                        first = count++ ? first : w;
                    }
                }
                std::printf("Draw pipe: resource tables of shader %016llx changed after the GPU "
                            "thread read them (%u of %u words, first %u: %#x -> %#x, user data "
                            "%u words)\n",
                            static_cast<unsigned long long>(snapshot.info->pgm_hash), count,
                            snapshot.flat_size, first, snapshot.flat[first], flat[first],
                            snapshot.user_data_size);
                ++flat_reports;
            }
        }
        const auto* live = reinterpret_cast<const AmdGpu::Regs*>(in);
        in += sizeof(AmdGpu::Regs);
        static u32 reports = 0;
        for (u32 block = 0; block < AmdGpu::RegDirty::NumBlocks && reports < 32; ++block) {
            const u32 at = block * AmdGpu::RegDirty::BlockWords;
            if (std::memcmp(regs.reg_array.data() + at, live->reg_array.data() + at,
                            AmdGpu::RegDirty::BlockWords * sizeof(u32)) != 0) {
                for (u32 w = at; w < at + AmdGpu::RegDirty::BlockWords; ++w) {
                    if (regs.reg_array[w] != live->reg_array[w]) {
                        std::printf("Draw pipe: register copy differs at word %#x (%#x vs %#x)\n",
                                    w, regs.reg_array[w], live->reg_array[w]);
                        ++reports;
                        break;
                    }
                }
            }
        }
    }
    ASSERT(static_cast<u32>(in - data) == size);
    stage_regs_used.store(true, std::memory_order_relaxed); // 0.5.1-c
    stage_regs = &regs;
    self.pipe_cb_extent = packet.cb_extent;
    self.pipe_db_extent = packet.db_extent;
    if (packet.kind == PacketKind::Special) {
        // A draw FilterDraw handles (fast clear elimination, resolve, depth copy, skip).
        FrameCapture::Poll();
        self.scheduler.PopPendingOperations();
        self.FilterDraw();
    } else if (cs) {
        stage_cs = cs;
        const auto* pipeline = static_cast<const ComputePipeline*>(packet.pipeline);
        if (indirect) {
            self.DispatchIndirectRecord(pipeline, indirect->args, indirect->stride);
        } else {
            self.DispatchRecord(pipeline);
        }
        stage_cs = nullptr;
    } else if (indirect) {
        self.DrawIndirectRecord(static_cast<const GraphicsPipeline*>(packet.pipeline),
                                packet.is_indexed, *indirect);
    } else {
        self.DrawRecord(static_cast<const GraphicsPipeline*>(packet.pipeline), packet.prepared,
                        packet.is_indexed, packet.index_offset);
    }
    Shader::Info::num_ud_snapshots = 0;
    self.num_ring_stages = 0;
    if (self.constant_ring) {
        self.constant_ring->Stamp(packet.ring_end);
    }
}

void Rasterizer::PrintPipeStats() {
    static const bool stats = std::getenv("BB_FRAME_STATS") != nullptr;
    if (!stats || (draw_pipe->packets & 1023) != 0) {
        return;
    }
    static auto window = std::chrono::steady_clock::now();
    static u64 last_packets = 0, last_drains = 0, last_drain_cycles = 0, last_busy = 0;
    static u64 last_tsc = bbport::CpuCycles();
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - window).count();
    if (seconds < 5.0) {
        return;
    }
    const u64 tsc = bbport::CpuCycles();
    const double cycles = double(tsc - last_tsc);
    const u64 busy = draw_pipe->busy_cycles.load(std::memory_order_relaxed);
    std::printf("Draw pipe: %.0f draws/s pipelined, %.0f drains/s that waited, stage A waited "
                "%.1f%%, stage B busy %.1f%%\n",
                (draw_pipe->packets - last_packets) / seconds,
                (draw_pipe->drains - last_drains) / seconds,
                100.0 * (draw_pipe->drain_cycles - last_drain_cycles) / cycles,
                100.0 * (busy - last_busy) / cycles);
    std::printf("  drains by reason/s:");
    for (u32 r = 0; r < DrawPipe::NumReasons; ++r) {
        if (const u64 n = draw_pipe->drains_by_reason[r]) {
            static constexpr const char* names[] = {"ConstRam", "commands", "compute",
                                                    "submission end", "draw", "rasterizer"};
            const double share = 100.0 * draw_pipe->cycles_by_reason[r] / cycles;
            draw_pipe->cycles_by_reason[r] = 0;
            if (r < 256) {
                std::printf(" %s=%.0f (%.1f%%)",
                            magic_enum::enum_name(AmdGpu::PM4ItOpcode(r)).data(), n / seconds,
                            share);
            } else {
                std::printf(" %s=%.0f (%.1f%%)", names[r - 256], n / seconds, share);
            }
        }
    }
    std::printf("\n  stage A waits over 0.5%%:");
    for (const auto& [function, lines] : drain_sites) {
        for (const auto& [line, waited] : lines) {
            if (waited * 200 > cycles) {
                std::printf(" %s:%u %.1f%%", function, line, 100.0 * waited / cycles);
            }
        }
    }
    std::printf("\n");
    drain_sites.clear();
    draw_pipe->drains_by_reason = {};
    window = now;
    last_tsc = tsc;
    last_packets = draw_pipe->packets;
    last_drains = draw_pipe->drains;
    last_drain_cycles = draw_pipe->drain_cycles;
    last_busy = busy;
}

bool Rasterizer::FilterDrawPasses() const {
    // FilterDraw's checks without its side effects: false when it would skip the draw or run
    // a pass of its own (fast clear elimination, resolve, depth/stencil copy).
    const auto& regs = Regs();
    using Mode = AmdGpu::ColorControl::OperationMode;
    const auto mode = regs.color_control.mode;
    if (mode == Mode::EliminateFastClear || mode == Mode::FmaskDecompress ||
        mode == Mode::Resolve || regs.primitive_type == AmdGpu::PrimitiveType::None) {
        return false;
    }
    const bool depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const bool stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    return !(mode == Mode::Disable && (depth_copy || stencil_copy));
}

// fifa-hybrid diagnostics (BB_DRAWLOG_TRIGGER): the open draw log and its draw counter.
static FILE* g_drawlog = nullptr;
struct DrawlogStream {
    u32 vs;
    u32 stream;
    VAddr addr;
    u64 size;
};
static std::map<std::pair<VAddr, u64>, DrawlogStream> g_drawlog_streams;
// The draw-log window is open: read faults and image downloads are listed on stderr.
std::atomic<bool> g_bb_diag_window{false};
static u64 g_drawlog_draw = 0;

void Rasterizer::Draw(bool is_indexed, u32 index_offset, const PreparedDraw* prepared) {
    RENDERER_TRACE;
    BbStats::draws.fetch_add(1, std::memory_order_relaxed);
    // BB_DROP_TESS=1/2/3 (measurement only, wrong image): drop tessellated draws (1), rect and
    // quad lists (2) or both (3), to bound what MoltenVK's tessellation pass splits cost.
    static const int drop_tess = [] {
        const char* env = std::getenv("BB_DROP_TESS");
        return env ? std::atoi(env) : 0;
    }();
    if (drop_tess) {
        const auto& r = liverpool->regs;
        const bool rect = r.primitive_type == AmdGpu::PrimitiveType::RectList ||
                          r.primitive_type == AmdGpu::PrimitiveType::QuadList;
        if (((drop_tess & 1) && r.stage_enable.hs_en) ||
            ((drop_tess & 2) && rect && !r.stage_enable.hs_en)) {
            return;
        }
    }
    if (BbStats::enabled) {
        // Draws MoltenVK runs through its tessellation emulation (a pass split each).
        const auto& r = liverpool->regs;
        if (r.stage_enable.hs_en) {
            BbStats::tess_draws.fetch_add(1, std::memory_order_relaxed);
        } else if (r.primitive_type == AmdGpu::PrimitiveType::RectList ||
                   r.primitive_type == AmdGpu::PrimitiveType::QuadList) {
            BbStats::rect_draws.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // BB_DRAWLOG_TRIGGER=<file>: when that file appears (its content is the output path) every draw of the
    // next 70 ms is written there: a D line per draw, then per stage its constant buffers (B) and images (I).
    // BB_AUDIT=<seconds, default 20>: every so often, tracked clean images are compared with guest memory.
    if (static const int audit_s = [] {
            const char* e = std::getenv("BB_AUDIT");
            return e ? std::max(std::atoi(e), 1) == 1 ? 20 : std::atoi(e) : 0;
        }();
        audit_s) {
        static u32 skip = 0;
        if ((++skip & 0xff) == 0) {
            const auto now = std::chrono::steady_clock::now();
            static auto next_audit = now + std::chrono::seconds(audit_s);
            if (now >= next_audit) {
                next_audit = now + std::chrono::seconds(audit_s);
                texture_cache.AuditImages();
            }
        }
    }
    if (static const char* dlt = std::getenv("BB_DRAWLOG_TRIGGER"); dlt) {
        const auto now = std::chrono::steady_clock::now();
        static auto next_poll = now;
        static auto stop_at = now;
        static auto diag_stop = now;
        if (g_bb_diag_window.load(std::memory_order_relaxed) && now >= diag_stop) {
            g_bb_diag_window.store(false, std::memory_order_relaxed);
            std::fprintf(stderr, "[DIAGWIN] end\n");
        }
        if (g_drawlog && now >= stop_at) {
            // BB_VDIFF=1: the vertex streams listed in the window, arena against guest memory.
            for (const auto& [key, v] : g_drawlog_streams) {
                u64 clean = 0, first = 0;
                const u64 diff = buffer_cache.DebugArenaDiff(v.addr, v.size, &clean, &first);
                std::fprintf(g_drawlog, "A %08x s%u %#llx size %llu: arena differs in %lld of %llu clean bytes, first %#llx\n",
                             v.vs, v.stream, (unsigned long long)v.addr, (unsigned long long)v.size,
                             (long long)diff, (unsigned long long)clean, (unsigned long long)first);
            }
            g_drawlog_streams.clear();
            std::fclose(g_drawlog);
            g_drawlog = nullptr;
        }
        if (!g_drawlog && now >= next_poll) {
            next_poll = now + std::chrono::milliseconds(40);
            if (FILE* t = std::fopen(dlt, "r")) {
                char out[1024] = {};
                if (std::fgets(out, sizeof(out), t)) {
                    out[std::strcspn(out, "\r\n")] = 0;
                    g_drawlog = std::fopen(out, "w");
                    stop_at = now + std::chrono::milliseconds(70);
                    // BB_DIAG_MS=<ms>: length of the read-fault/download/write-back listing window.
                    static const int diag_ms = [] {
                        const char* env = std::getenv("BB_DIAG_MS");
                        return env ? std::atoi(env) : 500;
                    }();
                    diag_stop = now + std::chrono::milliseconds(diag_ms);
                    g_bb_diag_window.store(true, std::memory_order_relaxed);
                    std::fprintf(stderr, "[DIAGWIN] begin %s\n", out);
                }
                std::fclose(t);
                std::remove(dlt);
            }
        }
        if (g_drawlog) {
            const auto& r = liverpool->regs;
            ++g_drawlog_draw;
            std::fprintf(g_drawlog, "D %llu idx %u prim %u cb0 %#llx %ux%u z %#llx\n",
                         (unsigned long long)g_drawlog_draw, r.num_indices, u32(r.primitive_type),
                         (unsigned long long)r.color_buffers[0].Address(), r.color_buffers[0].Pitch(),
                         r.color_buffers[0].Height(), (unsigned long long)r.depth_buffer.DepthAddress());
        }
    }

    // BB_DB_ENTRY=<pitch>: every draw the game submits with a depth buffer of that pitch, counted per kind
    // before any filter (to tell draws that never arrive from draws dropped on the way).
    if (static const u32 dbe = [] { const char* e = std::getenv("BB_DB_ENTRY"); return e ? u32(std::atoi(e)) : 0u; }();
        dbe) {
        const auto& r = liverpool->regs;
        if ((r.depth_buffer.DepthAddress() || r.depth_buffer.DepthWriteAddress()) && r.depth_buffer.Pitch() == dbe) {
            const std::string key = fmt::format(
                "z {:#x} zw {:#x} {}x{} zfmt {} cmode {} prim {} idx {} den {} dwr {} dfunc {} fzd {} fzv {} cb0 {:#x} ps {:#x}",
                u64(r.depth_buffer.DepthAddress()), u64(r.depth_buffer.DepthWriteAddress()), r.depth_buffer.Pitch(),
                r.depth_buffer.Height(), u32(r.depth_buffer.z_info.format), u32(r.color_control.mode),
                u32(r.primitive_type), r.num_indices, int(r.depth_control.depth_enable),
                int(r.depth_control.depth_write_enable), int(r.depth_control.depth_func),
                int(r.depth_render_override.force_z_dirty), int(r.depth_render_override.force_z_valid),
                u64(r.color_buffers[0].Address()), u64(r.ps_program.Address<uintptr_t>()));
            static std::map<std::string, u64> counts;
            const u64 c = ++counts[key];
            if (counts.size() < 300 && (c == 1 || c % 1000 == 0)) {
                std::fprintf(stderr, "[DBE] %s count %llu\n", key.c_str(), (unsigned long long)c);
            }
        }
    }

    // bbport: with the draw pipeline this thread only selects the pipeline and hands the draw
    // to the recording thread (DrawRecord there); draws FilterDraw handles itself run here.
    const bool pipelined = UseDrawPipe();
    if (pipelined && !FilterDrawPasses() &&
        !BbToggle::Disabled(BbToggle::PipelinedMemoryWrites)) {
        PostDraw(nullptr, nullptr, false, 0);
        return;
    }
    if (!pipelined || !FilterDrawPasses()) {
        DrainDrawPipe(DrawPipe::ReasonDraw);
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
        if (!FilterDraw()) {
            return;
        }
    }
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline({}, prepared);
    const PreparedDraw* used_prepared = pipeline_cache.UsedPrepared();
    if (prepared) {
        draw_prep->Count(pipeline != nullptr &&
                         prepared->state.load(std::memory_order_acquire) == PreparedDraw::Ready &&
                         prepared->reg_checksum == liverpool->gfx_reg_checksum);
    }
    if (!pipeline) {
        return;
    }
    if (pipelined && FilterDrawPasses()) {
        PostDraw(pipeline, used_prepared, is_indexed, index_offset);
        return;
    }
    DrawRecord(pipeline, used_prepared, is_indexed, index_offset);
}

// fifa-hybrid diagnostics: BB_SHADOW_FORCE=<bits> overrides state for draws of ps f56f5d1c (FIFA's
// shadow-map pass): 1 depth test off, 2 cull none, 4 rasterizer discard off.
static thread_local bool g_bb_shadow_draw = false;
static u32 BbShadowForce() {
    static const u32 bits = [] { const char* e = std::getenv("BB_SHADOW_FORCE"); return e ? u32(std::atoi(e)) : 0u; }();
    return g_bb_shadow_draw ? bits : 0u;
}

void Rasterizer::DrawRecord(const GraphicsPipeline* pipeline, const PreparedDraw* used_prepared,
                            bool is_indexed, u32 index_offset) {
    ++g_bb_draw_seq; // fifa-hybrid diagnostics
    {
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        g_bb_shadow_draw = ps && u32(ps->pgm_hash) == 0xf56f5d1cu;
    }

    if (DrawPipe::OnStageB()) {
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
    }
    const auto& regs = Regs();
    if (BbSeqWindow()) {
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        if (ps && u32(ps->pgm_hash) == 0xf56f5d1cu) {
            const auto& vp = regs.viewports[0];
            std::fprintf(stderr, "[SEQ] %llu draw-shadow idx %u indexed %d mode %u cb0 %#llx passes %d prim %u cullF %u cullB %u ff %u ucp %#x zclipN %u zclipF %u clipdis %u vpclamp %u vp %g %g %g %g z %g %g scis %d,%d-%d,%d\n",
                         (unsigned long long)g_bb_draw_seq.load(), regs.num_indices, int(is_indexed),
                         u32(regs.color_control.mode), (unsigned long long)regs.color_buffers[0].Address(),
                         int(FilterDrawPasses()), u32(regs.primitive_type), u32(regs.polygon_control.cull_front),
                         u32(regs.polygon_control.cull_back), u32(regs.polygon_control.front_face),
                         u32(regs.clipper_control.user_clip_plane_enable), u32(regs.clipper_control.zclip_near_disable),
                         u32(regs.clipper_control.zclip_far_disable), u32(regs.clipper_control.clip_disable),
                         u32(regs.depth_render_override.disable_viewport_clamp), vp.xscale, vp.xoffset, vp.yscale, vp.yoffset,
                         vp.zscale, vp.zoffset, int(regs.screen_scissor.top_left_x), int(regs.screen_scissor.top_left_y),
                         int(regs.screen_scissor.bottom_right_x), int(regs.screen_scissor.bottom_right_y));
            std::fprintf(stderr, "[SEQ]   cw-on-zfail %u cw-off-zpass %u rop3 %#x blend0 %#x cb0 dfmt %u nfmt %u mrtmask %#x zexport %u rastkill %u vtxkill %u\n",
                         u32(regs.depth_control.enable_color_writes_on_depth_fail),
                         u32(regs.depth_control.disable_color_writes_on_depth_pass), u32(regs.color_control.rop3),
                         std::bit_cast<u32>(regs.blend_control[0]), u32(regs.color_buffers[0].GetDataFmt()),
                         u32(regs.color_buffers[0].GetNumberFmt()), u32(regs.color_target_mask.raw),
                         u32(regs.depth_shader_control.z_export_enable), u32(regs.clipper_control.dx_rasterization_kill),
                         u32(regs.clipper_control.vtx_kill_or));
        }
    }
    // bbport: the pass copying a finished frame to a display buffer; the previous draw's
    // target is that frame.
    // Object motion needs the frame boundary too (its history halves alternate per frame).
    if ((camera_motion->Enabled() || object_motion->Enabled()) && regs.color_buffers[0] &&
        FrameCapture::IsDisplayBuffer(regs.color_buffers[0].Address())) {
        scene_started = false;
        camera_motion->OnDisplayPass(cb_descs[0].first);
        if (upscaler->OnFrameStart()) {
            object_motion->InvalidateHistory();
            camera_motion->InvalidateHistory();
        }
        camera_motion->SetJitter(upscaler->Jitter());
        object_motion->OnFrameStart();
        NoteFrameStart();
        static const char* scene_debug = std::getenv("BB_SCENE_DEBUG");
        scene_debug_frame = scene_debug && std::remove(scene_debug) == 0;
        scene_targets->debug = scene_debug_frame;
    }
    // bbport debug: BB_SKIP_PS="hash,hash" drops the draws using these pixel shaders.
    static const std::vector<u64> skip_ps = [] {
        std::vector<u64> out;
        if (const char* s = std::getenv("BB_SKIP_PS")) {
            for (char* e = nullptr; *s; s = *e ? e + 1 : e) {
                out.push_back(std::strtoull(s, &e, 16));
            }
        }
        return out;
    }();
    // DEBUG(bbport): BB_LIST_PS=1 logs each pixel shader the first time it is drawn (temporary).
    static const bool list_ps = std::getenv("BB_LIST_PS") != nullptr;
    if (list_ps) {
        static std::unordered_set<u64> seen;
        static const auto t0 = std::chrono::steady_clock::now();
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        const auto* vs = pipeline->GetStages()[u32(Shader::SwStage::Vertex)];
        if (ps && seen.insert(ps->pgm_hash).second) {
            std::fprintf(stderr, "[LISTPS] t=%.1f ps %llx vs %llx rt0 %#llx fmt %u mrt %x\n",
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                             .count(),
                         (unsigned long long)ps->pgm_hash,
                         (unsigned long long)(vs ? vs->pgm_hash : 0),
                         (unsigned long long)(regs.color_buffers[0] ? regs.color_buffers[0].Address()
                                                                    : 0),
                         u32(regs.color_buffers[0].GetDataFmt()),
                         pipeline->GetGraphicsKey().mrt_mask);
        }
    }
    if (!skip_ps.empty()) {
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        if (ps && std::find(skip_ps.begin(), skip_ps.end(), ps->pgm_hash) != skip_ps.end()) {
            return;
        }
    }
    // fifa-hybrid diagnostics: BB_RESOLVE_LOG=1 prints the EQAA resolve's constants (ps 92176db0:
    // buffer 0 dwords 0-3 = colour samples, coverage samples, fmask bits, fmask mask; buffer 1
    // dword 4 = weight) once a second.
    if (static const bool resolve_log = std::getenv("BB_RESOLVE_LOG") != nullptr; resolve_log) {
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        if (ps && (ps->pgm_hash & 0xffffffffull) == 0x92176db0ull && ps->buffers.size() >= 2) {
            static auto last = std::chrono::steady_clock::now() - std::chrono::seconds(2);
            const auto now = std::chrono::steady_clock::now();
            if (now - last >= std::chrono::milliseconds(1000)) {
                last = now;
                const auto s0 = ps->buffers[0].GetSharp(*ps), s1 = ps->buffers[1].GetSharp(*ps);
                if (s0.base_address && s1.base_address && memory->IsValidGpuMapping(s0.base_address, 0) &&
                    memory->IsValidGpuMapping(s1.base_address, 0)) {
                    const auto* d0 = reinterpret_cast<const u32*>(s0.base_address);
                    const auto* d1 = reinterpret_cast<const u32*>(s1.base_address);
                    float w;
                    std::memcpy(&w, &d1[4], sizeof w);
                    std::fprintf(stderr, "[RESOLVE] b0[0..3]=%u %u %u %#x b1[4]=%g (aa=%u cb_samples=%u) b1=%#llx\n",
                                 d0[0], d0[1], d0[2], d0[3], w, u32(regs.aa_config.NumSamples()),
                                 u32(regs.color_buffers[0].NumSamples()), (unsigned long long)s1.base_address);
                    // BB_RESOLVE_WATCH=1: watch b1[4] (who writes it, write-back values).
                    if (static const bool rw = std::getenv("BB_RESOLVE_WATCH") != nullptr; rw) {
                        VideoCore::g_bb_bufwatch.store(u64(s1.base_address) + 16, std::memory_order_relaxed);
                    }
                }
            }
        }
    }
    // fifa-hybrid diagnostics: BB_TONE_LOG=1 prints the tonemapper's constants once a second
    // (pixel shader c37c824e7df, buffer 1 dwords: 29 exposure, 30 curve mode).
    static const bool tone_log = std::getenv("BB_TONE_LOG") != nullptr;
    if (tone_log) {
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        // The tonemapper: five buffers, eight images of which images 3-6 are the 3D grading LUTs.
        const auto is_tonemapper = [&] {
            if (!ps || ps->buffers.size() != 5 || ps->images.size() != 8) {
                return false;
            }
            for (u32 i = 3; i <= 6; ++i) {
                if (ps->images[i].GetSharp(*ps).GetType() != AmdGpu::ImageType::Color3D) {
                    return false;
                }
            }
            return true;
        };
        if (is_tonemapper()) {
            static auto last = std::chrono::steady_clock::now() - std::chrono::seconds(2);
            const auto now = std::chrono::steady_clock::now();
            if (now - last >= std::chrono::milliseconds(1000)) {
                last = now;
                const auto sharp = ps->buffers[1].GetSharp(*ps);
                const VAddr address = sharp.base_address;
                if (address != 0 && memory->IsValidGpuMapping(address, 0)) {
                    const auto* d = reinterpret_cast<const u32*>(address);
                    const auto f = [&](u32 i) {
                        float v;
                        std::memcpy(&v, &d[i], sizeof v);
                        return v;
                    };
                    std::fprintf(stderr,
                                 "[TONE] ps %llx buf=%#llx size=%u d9=%g d11=%g d12=%g d28=%g "
                                 "exp(d29)=%g mode(d30)=%u d31=%g d32=%g\n",
                                 (unsigned long long)ps->pgm_hash, (unsigned long long)address,
                                 sharp.GetSize(), f(9), f(11), f(12), f(28), f(29), d[30], f(31),
                                 f(32));
                    for (u32 i = 0; i < ps->images.size(); ++i) {
                        const auto tsharp = ps->images[i].GetSharp(*ps);
                        const VideoCore::ImageInfo info{tsharp, ps->images[i]};
                        if (tsharp.GetType() == AmdGpu::ImageType::Color3D) {
                            VideoCore::g_bb_dump_volume_addrs.insert(tsharp.Address());
                        }
                        std::fprintf(stderr,
                                     "[TONE]   img%u addr=%#llx %ux%ux%u type=%u data_fmt=%u "
                                     "num_fmt=%u -> %s mips=%u tile=%u dst_sel=%u,%u,%u,%u\n",
                                     i, (unsigned long long)tsharp.Address(), tsharp.width,
                                     tsharp.height, tsharp.depth, u32(tsharp.GetType()),
                                     u32(tsharp.GetDataFmt()), u32(tsharp.GetNumberFmt()),
                                     vk::to_string(info.pixel_format).c_str(),
                                     info.resources.levels, u32(tsharp.tiling_index),
                                     u32(tsharp.dst_sel_x), u32(tsharp.dst_sel_y),
                                     u32(tsharp.dst_sel_z), u32(tsharp.dst_sel_w));
                    }
                }
            }
        }
    }
    // fifa-hybrid A/B (temporary): BB_SKIP_SQUARE_DEPTH=1 drops depth-only draws into square
    // targets (the shadow maps), so the scene renders against shadow maps left at their clear value.
    static const bool skip_square_depth = std::getenv("BB_SKIP_SQUARE_DEPTH") != nullptr;
    if (skip_square_depth && pipeline->GetGraphicsKey().mrt_mask == 0 &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.Pitch() == regs.depth_buffer.Height()) {
        return;
    }
    bind_prepared = used_prepared;
    motion_draw = pipeline->GetGraphicsKey().motion_vectors;
    motion_geometry = 0;

    SetDrawEncoders(*pipeline);
    PrepareRenderState(pipeline);
    // bbport: one color target, or none with a depth target (Scaleform's stencil masks).
    if (const u32 mrt_mask = pipeline->GetGraphicsKey().mrt_mask;
        upscaler->Enabled() && (std::popcount(mrt_mask) == 1 || (mrt_mask == 0 && db_desc.first))) {
        const auto& viewport = Regs().viewports[0];
        upscaler->OnDraw(pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash,
                         mrt_mask ? cb_descs[0].first : VideoCore::ImageId{}, db_desc.first,
                         UiComposition::NativeViewport(viewport.xscale * 2, viewport.yscale * 2));
    }
    const PreparedDraw* draw_prepared = bind_prepared;
    // bbport: vertex and index buffers are resolved while the helper binds textures (their
    // commands are recorded after BeginRendering, as before).
    draw_inputs = {pipeline, draw_prepared, index_offset, is_indexed, true, false};
    const bool bound = BindResources(pipeline);
    // fifa-hybrid diagnostics: BB_DUMP_R32=<dir> dumps the bound small R32 images through the
    // memoized binding path too (DebugDumpTexture self-limits per address).
    if (static const char* dprobe = std::getenv("BB_DEPTH_PROBE"); dprobe && bound) {
        static auto last = std::chrono::steady_clock::now();
        static const VAddr want = std::strtoull(dprobe, nullptr, 16);
        if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds(2)) {
            last = now;
            texture_cache.DepthProbe(want);
        }
    }
    if (static const bool twin_watch = std::getenv("BB_TWIN_WATCH_DRAW") != nullptr; twin_watch && bound) {
        static auto last = std::chrono::steady_clock::now();
        if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds(2)) {
            last = now;
            texture_cache.TwinWatch(); // fifa-hybrid diagnostics: [TWINWATCH] twin/source statistics
        }
    }
    if (static const bool dump_r32 = std::getenv("BB_DUMP_R32") != nullptr; dump_r32 && bound) {
        for (const auto image_id : bound_images) {
            if (image_id) {
                const auto& info = texture_cache.GetImage(image_id).info;
                if (info.pixel_format == vk::Format::eR32Sfloat && info.size.width <= 1024) {
                    static std::map<VAddr, std::chrono::steady_clock::time_point> seen;
                    const auto now = std::chrono::steady_clock::now();
                    auto [it, fresh] = seen.try_emplace(info.guest_address, now);
                    if (fresh || now - it->second > std::chrono::seconds(30)) {
                        it->second = now;
                        std::fprintf(stderr, "[R32LOOP] bound %#llx %ux%u\n",
                                     (unsigned long long)info.guest_address, info.size.width, info.size.height);
                    }
                    texture_cache.DebugDumpTexture(image_id);
                }
            }
        }
    }
    bind_prepared = nullptr; // indirect draws and dispatches bind without prepared sharps
    const bool inputs_resolved = draw_inputs.resolved;
    draw_inputs.pending = false;
    if (!bound) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    if (!inputs_resolved) {
        ResolveVertexBuffers(pipeline, draw_prepared);
        if (is_indexed) {
            ResolveIndexBuffer(index_offset);
        }
    }

    // bbport: screen-space (clip disabled) draws into the upscaler's output-size images.
    push_data.xscale *= target_scale[0];
    push_data.xoffset *= target_scale[0];
    push_data.yscale *= target_scale[1];
    push_data.yoffset *= target_scale[1];
    if (motion_draw && motion_geometry) {
        const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
        const auto [base_vertex, first_instance] =
            GetDrawOffsets(regs, vs, pipeline->GetFetchShader());
        const u32 index_size =
            regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16 ? 2u : 4u;
        const VAddr index_address = is_indexed
            ? regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size : 0;
        // Without a character-size skeleton the draw is gated on its bone palette changing
        // since last frame (model constants follow the camera, the palette does not). The
        // check comes first: most gated draws are static world pieces with long index lists.
        static const bool all_motion = [] {
            const char* value = std::getenv("BB_OBJECT_MOTION_ALL");
            return value && value[0] == '1';
        }();
        bool gated = !all_motion;
        u64 palette = 0;
        for (const auto& resource : vs.buffers) {
            if (resource.IsSpecial()) continue;
            const auto buffer = resource.GetSharp(vs);
            const u32 size = buffer.GetSize();
            if (!buffer.Valid() || buffer.GetStride() != 16) continue;
            const auto role = Motion::ClassifyBuffer(size);
            if (role == Motion::BufferRole::Skeleton) {
                gated = false;
                break;
            }
            const VAddr address = buffer.base_address;
            if (role == Motion::BufferRole::SmallSkeleton &&
                memory->IsValidGpuMapping(address, 0) &&
                memory->ClampRangeSize(address, size) == size) {
                palette = XXH3_64bits_withSeed(reinterpret_cast<const void*>(address), size,
                                               palette);
            }
        }
        push_data.motion_param = 0;
        if (!gated || object_motion->Moving({.shader = vs.pgm_hash,
                                             .geometry = motion_geometry,
                                             .indices = index_address,
                                             .index_count = regs.num_indices,
                                             .instances = regs.num_instances.NumInstances(),
                                             .first_instance = first_instance},
                                            palette)) {
            Motion::VertexRange range{base_vertex, regs.num_indices};
            u64 topology = 0;
            if (is_indexed) {
                range = {};
                const u64 bytes = u64(regs.num_indices) * index_size;
                if (index_address && memory->IsValidGpuMapping(index_address, 0) &&
                    memory->ClampRangeSize(index_address, bytes) == bytes) {
                    const bool restart = regs.enable_primitive_restart != 0;
                    const u32 count = regs.num_indices;
                    const auto scanned = object_motion->IndexRange(
                        {index_address, count, index_size, s32(base_vertex), restart}, [&] {
                            const auto* data = reinterpret_cast<const void*>(index_address);
                            const auto scan_range =
                                index_size == 2
                                    ? Motion::IndexedRange(
                                          std::span(static_cast<const u16*>(data), count),
                                          s32(base_vertex), restart)
                                    : Motion::IndexedRange(
                                          std::span(static_cast<const u32*>(data), count),
                                          s32(base_vertex), restart);
                            return Motion::IndexRangeCache::Result{scan_range,
                                                                   XXH3_64bits(data, bytes)};
                        });
                    range = scanned.range;
                    topology = scanned.topology;
                }
            }
            push_data.motion_param = object_motion->PrepareDraw({
                .shader = vs.pgm_hash,
                .geometry = motion_geometry,
                .indices = index_address,
                .topology = topology,
                .index_count = regs.num_indices,
                .instances = regs.num_instances.NumInstances(),
                .first_instance = first_instance,
                .vertices = range,
            });
        }
    }
    // bbport: jitter geometry drawn with the scene depth, not full-screen passes (a shifted
    // full-screen quad leaves an edge column unwritten).
    draw_jitter = {};
    if (upscaler->Enabled() && db_desc.first && db_desc.first == camera_motion->Depth() &&
        (regs.num_indices > 6 || regs.num_instances.NumInstances() > 1)) {
        draw_jitter = upscaler->Jitter();
    }
    const Metal::Draw::DrawState draw_state = UpdateDynamicState(pipeline, is_indexed, state);
    MidFrameFlush(state);
    MarkPass(pipeline, state);
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const u32 num_indices = regs.num_indices;
    const u32 num_instances = regs.num_instances.NumInstances();
    const u32 first_vertex = vertex_offset;
    const u32 first_instance = instance_offset;
    // Indexed: the index buffer is bound at the first index; the base vertex is added to each.
    const Metal::Draw::DrawArgs draw_args{
        .count = num_indices,
        .instances = num_instances,
        .first = is_indexed ? 0u : first_vertex,
        .vertex_offset = is_indexed ? s32(first_vertex) : 0,
        .first_instance = first_instance,
        .indexed = is_indexed,
    };
    OrderAccesses();
    // Right before the command: the copies share its chunk (push_data is final here).
    const Metal::Draw::Resources resources = RecordResources(is_indexed);
    // bbport: the pipeline may still be compiling on the creation pool; the recording thread
    // waits for it as late as possible (failed: the draw helper skips and counts it).
    scheduler.RecordDraw(
        [=, runs = &scheduler.GetDrawRuns()](Metal::Encoder& e) {
            runs->Draw(e, DrawPipelines(*pipeline), draw_state, resources, draw_args);
        },
        IsTessellated(*pipeline));
    // fifa-hybrid diagnostics: BB_DUMP_PS=<hash,...> writes colour target 0 after every draw of
    // these pixel shaders (from BB_DUMP_PS_AT seconds after the first draw, at most BB_DUMP_PS_MAX
    // (60) files) as <BB_DUMP_DIR>/dNNN_ps<hash>_<w>x<h>.raw; BB_DUMP_PS_RECT=x,y,w,h crops.
    // 4-byte formats and R8 only (raw RGBA/BGRA order as the format says).
    {
        static const std::vector<u64> dump_ps = [] {
            std::vector<u64> out;
            if (const char* s = std::getenv("BB_DUMP_PS")) {
                for (char* e = nullptr; *s; s = *e ? e + 1 : e) {
                    out.push_back(std::strtoull(s, &e, 16));
                    if (e == s) break;
                }
            }
            return out;
        }();
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        // BB_DUMP_PS=all: every draw (its state is logged; a file only with a 4-byte target).
        static const bool dump_all = [] {
            const char* s = std::getenv("BB_DUMP_PS");
            return s && std::strcmp(s, "all") == 0;
        }();
        // BB_DUMP_RT=<hex guest address>: also every draw into that colour target.
        static const u64 dump_rt = [] {
            const char* s = std::getenv("BB_DUMP_RT");
            return s ? std::strtoull(s, nullptr, 16) : 0ull;
        }();
        const bool rt_match = dump_rt != 0 && regs.color_buffers[0] &&
                              regs.color_buffers[0].Address() == dump_rt;
        if (dump_all || rt_match || (!dump_ps.empty() && ps &&
            std::find(dump_ps.begin(), dump_ps.end(), ps->pgm_hash) != dump_ps.end())) {
            static const double at = [] {
                const char* e = std::getenv("BB_DUMP_PS_AT");
                return e ? std::atof(e) : 0.0;
            }();
            static const int max_files = [] {
                const char* e = std::getenv("BB_DUMP_PS_MAX");
                return e ? std::atoi(e) : 60;
            }();
            static const std::array<u32, 4> rect = [] {
                std::array<u32, 4> r{0, 0, 0, 0};
                if (const char* e = std::getenv("BB_DUMP_PS_RECT")) {
                    std::sscanf(e, "%u,%u,%u,%u", &r[0], &r[1], &r[2], &r[3]);
                }
                return r;
            }();
            static const std::string dir = [] {
                const char* env = std::getenv("BB_DUMP_DIR");
                return std::string{env && env[0] ? env : "out/dump"};
            }();
            static const auto t0 = std::chrono::steady_clock::now();
            static int written = 0;
            const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            // BB_DUMP_PS_TRIGGER=<file>: dumps start once the file exists (checked every 64 draws).
            static const char* trigger = std::getenv("BB_DUMP_PS_TRIGGER");
            static bool triggered = trigger == nullptr;
            static u32 polls = 0;
            if (!triggered && (++polls % 64) == 0) {
                if (FILE* f = std::fopen(trigger, "rb")) {
                    std::fclose(f);
                    triggered = true;
                }
            }
            // BB_DUMP_PS_LOGONLY=1: the state lines only, no pass ends or copies.
            static const bool log_only = std::getenv("BB_DUMP_PS_LOGONLY") != nullptr;
            if (now >= at && written < max_files && triggered) {
                const auto& dc = regs.depth_control;
                const auto& sr = regs.stencil_ref_front;
                const auto& sc = regs.stencil_control;
                const auto& bc = regs.blend_control[0];
                std::fprintf(stderr,
                             "[DUMPPS] draw t=%.1f ps %llx vs %llx rt0 %#llx db %#llx st %#llx idx %u inst %u "
                             "stencil en=%u func=%u ref=%u mask=%#x wmask=%#x fail=%u zpass=%u zfail=%u "
                             "depth en=%u w=%u blend en=%u src=%u dst=%u fn=%u asrc=%u adst=%u sep=%u cmask=%#x\n",
                             now, (unsigned long long)(ps ? ps->pgm_hash : 0),
                             (unsigned long long)vs_info.pgm_hash,
                             (unsigned long long)(regs.color_buffers[0] ? regs.color_buffers[0].Address() : 0),
                             (unsigned long long)regs.depth_buffer.DepthAddress(),
                             (unsigned long long)regs.depth_buffer.StencilAddress(), num_indices, num_instances,
                             u32(dc.stencil_enable), u32(dc.stencil_ref_func), u32(sr.stencil_test_val),
                             u32(sr.stencil_mask), u32(sr.stencil_write_mask), u32(sc.stencil_fail_front),
                             u32(sc.stencil_zpass_front), u32(sc.stencil_zfail_front), u32(dc.depth_enable),
                             u32(dc.depth_write_enable), u32(bc.enable), u32(bc.color_src_factor),
                             u32(bc.color_dst_factor), u32(bc.color_func), u32(bc.alpha_src_factor),
                             u32(bc.alpha_dst_factor), u32(bc.separate_alpha_blend),
                             u32(regs.color_target_mask.raw));
                const auto& vp = regs.viewports[0];
                const auto& ss = regs.screen_scissor;
                std::fprintf(stderr, "[DUMPPS]   viewport %g %g %g %g scissor %d,%d-%d,%d opval=%u sclear=%u zclear=%u sclearval=%u bound=%u/%u\n",
                             vp.xscale, vp.xoffset, vp.yscale, vp.yoffset, int(ss.top_left_x),
                             int(ss.top_left_y), int(ss.bottom_right_x), int(ss.bottom_right_y),
                             u32(sr.stencil_op_val), u32(regs.depth_render_control.stencil_clear_enable),
                             u32(regs.depth_render_control.depth_clear_enable), u32(regs.stencil_clear),
                             u32(BoundDepthAttachment(regs).HasDepth()), u32(BoundDepthAttachment(regs).HasStencil()));
            }
            // BB_DUMP_PS_STENCIL=1: the stencil attachment after the draw too (sNNN_..., 1 byte/px).
            static const bool dump_stencil = std::getenv("BB_DUMP_PS_STENCIL") != nullptr;
            if (dump_stencil && !log_only && now >= at && written < max_files && triggered && db_desc.first) {
                auto& dimage = texture_cache.GetImage(db_desc.first);
                if (dimage.GetImage() && regs.depth_buffer.StencilValid()) {
                    const u32 x = std::min(rect[0], dimage.info.size.width - 1);
                    const u32 y = std::min(rect[1], dimage.info.size.height - 1);
                    const u32 w = rect[2] ? std::min(rect[2], dimage.info.size.width - x) : dimage.info.size.width - x;
                    const u32 h = rect[3] ? std::min(rect[3], dimage.info.size.height - y) : dimage.info.size.height - y;
                    const u64 size = u64(w) * h;
                    scheduler.EndRendering();
                    auto buffer = std::make_shared<VideoCore::Buffer>(device, VideoCore::MemoryType::HostCached, 0, size, "Stencil dump");
                    scheduler.Record([texture = dimage.GetImage(), x, y, w, h, dst = buffer->Ref(), size](Metal::Encoder& e) {
                        e.CopyTextureToBuffer(texture, 0, 0, {x, y, 0}, {w, h, 1}, dst, w, u32(size), Metal::BlitAspect::Stencil);
                    });
                    char path[512];
                    std::snprintf(path, sizeof(path), "%s/s%03d_%ux%u.raw", dir.c_str(), written, w, h);
                    scheduler.DeferPriorityOperation([buffer, size, file = std::string{path}] {
                        if (FILE* f = std::fopen(file.c_str(), "wb")) {
                            std::fwrite(buffer->mapped_data.data(), 1, size, f);
                            std::fclose(f);
                        }
                    });
                }
            }
            if (log_only && now >= at && written < max_files && triggered) {
                ++written;
            }
            if (cb_descs[0].first && triggered && !log_only) {
            auto& image = texture_cache.GetImage(cb_descs[0].first);
            const bool four_bytes = image.info.pixel_format == vk::Format::eR8G8B8A8Unorm ||
                                    image.info.pixel_format == vk::Format::eB8G8R8A8Unorm ||
                                    image.info.pixel_format == vk::Format::eR8G8B8A8Srgb ||
                                    image.info.pixel_format == vk::Format::eB8G8R8A8Srgb ||
                                    image.info.pixel_format == vk::Format::eA2B10G10R10UnormPack32 ||
                                    image.info.pixel_format == vk::Format::eB10G11R11UfloatPack32 ||
                                    image.info.pixel_format == vk::Format::eR32Sfloat;
            const bool one_byte = image.info.pixel_format == vk::Format::eR8Unorm;
            if (now >= at && written < max_files && (four_bytes || one_byte) && image.GetImage()) {
                const u32 x = std::min(rect[0], image.info.size.width - 1);
                const u32 y = std::min(rect[1], image.info.size.height - 1);
                const u32 w = rect[2] ? std::min(rect[2], image.info.size.width - x) : image.info.size.width - x;
                const u32 h = rect[3] ? std::min(rect[3], image.info.size.height - y) : image.info.size.height - y;
                const u32 row = w * (one_byte ? 1 : 4);
                const u64 size = u64(row) * h;
                scheduler.EndRendering();
                auto buffer = std::make_shared<VideoCore::Buffer>(device, VideoCore::MemoryType::HostCached, 0, size, "Draw dump");
                scheduler.Record([texture = image.GetImage(), x, y, w, h, dst = buffer->Ref(), row, size](Metal::Encoder& e) {
                    e.CopyTextureToBuffer(texture, 0, 0, {x, y, 0}, {w, h, 1}, dst, row, u32(size));
                });
                char path[512];
                std::snprintf(path, sizeof(path), "%s/d%03d_ps%08x_%ux%u_%s.raw", dir.c_str(), written,
                              u32(ps ? ps->pgm_hash : 0), w, h, vk::to_string(image.info.pixel_format).c_str());
                std::fprintf(stderr, "[DUMPPS] %d t=%.1f ps %llx vs %llx rt %#llx %ux%u idx %u inst %u -> %s\n", written, now,
                             (unsigned long long)(ps ? ps->pgm_hash : 0), (unsigned long long)vs_info.pgm_hash,
                             (unsigned long long)image.info.guest_address, image.info.size.width,
                             image.info.size.height, num_indices, num_instances, path);
                scheduler.DeferPriorityOperation([buffer, size, file = std::string{path}] {
                    if (FILE* f = std::fopen(file.c_str(), "wb")) {
                        std::fwrite(buffer->mapped_data.data(), 1, size, f);
                        std::fclose(f);
                    }
                });
                ++written;
            }
            }
        }
    }
    if (FrameCapture::Active()) {
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        FrameCapture::Draw(vs_info.pgm_hash, ps ? ps->pgm_hash : 0, num_indices, num_instances);
        if (const auto& bc = regs.blend_control[0]; bc.enable && regs.color_buffers[0]) {
            char note[128];
            std::snprintf(note, sizeof(note), "\n  blend ps %08x idx %u: src %u dst %u func %u z %u%u",
                          u32(ps ? ps->pgm_hash : 0), num_indices, u32(bc.color_src_factor),
                          u32(bc.color_dst_factor), u32(bc.color_func),
                          u32(regs.depth_control.depth_enable),
                          u32(regs.depth_control.depth_write_enable));
            FrameCapture::Note(note);
        }
    }
    DebugState.IncDrawCall();
    if (PassLabels() == 2) {
        scheduler.EndRendering();
    }

    ResetBindings();
    scheduler.KickRecording();
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) {
    RENDERER_TRACE;
    // bbport: like direct draws, handed to the recording thread after the pipeline selection
    // (else the GPU thread waited here for a whole frame of queued draws).
    const bool pipelined = UseDrawPipe() && FilterDrawPasses() &&
                           !BbToggle::Disabled(BbToggle::PipelinedIndirectDraws);
    if (!pipelined) {
        DrainDrawPipe();
        scheduler.PopPendingOperations();
        if (!FilterDraw()) {
            return;
        }
    }

    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
    };
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params);
    if (!pipeline) {
        return;
    }
    const IndirectDraw indirect = {arg_address + offset, count_address, stride, max_count};
    if (pipelined) {
        PostDraw(pipeline, nullptr, is_indexed, 0, nullptr, &indirect);
        return;
    }
    DrawIndirectRecord(pipeline, is_indexed, indirect);
}

void Rasterizer::DrawIndirectRecord(const GraphicsPipeline* pipeline, bool is_indexed,
                                    const IndirectDraw& indirect) {
    if (DrawPipe::OnStageB()) {
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
    }
    const VAddr count_address = indirect.count;
    const u32 stride = indirect.stride;
    const u32 max_count = indirect.max_count;

    // Indirect arguments may be GPU-written: leave these draws on camera fallback.
    motion_draw = false;
    motion_geometry = 0;
    // (The count clamp and the indirect geometry kernels read the records in compute encoders.)
    SetDrawEncoders(*pipeline, true);
    PrepareRenderState(pipeline);
    if (const u32 mrt_mask = pipeline->GetGraphicsKey().mrt_mask;
        upscaler->Enabled() && (std::popcount(mrt_mask) == 1 || (mrt_mask == 0 && db_desc.first))) {
        const auto& viewport = Regs().viewports[0];
        upscaler->OnDraw(pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash,
                         mrt_mask ? cb_descs[0].first : VideoCore::ImageId{}, db_desc.first,
                         UiComposition::NativeViewport(viewport.xscale * 2, viewport.yscale * 2));
    }
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    ResolveVertexBuffers(pipeline);
    if (is_indexed) {
        ResolveIndexBuffer();
    }

    // bbport: without a GPU count Metal reads the records at a 32-bit indirectBufferOffset
    // (copied when they end past 4 GiB of the arena); with one, the clamp kernel reads them at
    // any offset and draws from its own scratch buffer (Encoder::DrawIndirectCount).
    const auto [buffer, base] =
        count_address == 0
            ? buffer_cache.ObtainDrawArgumentBuffer(indirect.args, stride * max_count)
            : buffer_cache.ObtainBuffer(indirect.args, stride * max_count, false, false, true,
                                        BbArenaRuns::Caller::Indirect);
    if (count_address != 0 && base + u64(stride) * max_count > (u64{1} << 32)) {
        BbStats::indirect_counted_high.fetch_add(1, std::memory_order_relaxed);
    }
    NoteRead(buffer, base, stride * max_count);
    if (FrameCapture::Tracing()) {
        FrameCapture::Trace(fmt::format("A {:#x} {}", indirect.args, stride * max_count));
    }

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(
            count_address, 4, false, false, true, BbArenaRuns::Caller::Indirect);
        NoteRead(count_buffer, count_offset, 4);
    }

    // bbport: screen-space (clip disabled) draws into the upscaler's output-size images.
    push_data.xscale *= target_scale[0];
    push_data.xoffset *= target_scale[0];
    push_data.yscale *= target_scale[1];
    push_data.yoffset *= target_scale[1];
    draw_jitter = {};
    if (upscaler->Enabled() && db_desc.first && db_desc.first == camera_motion->Depth()) {
        draw_jitter = upscaler->Jitter();
    }
    const Metal::Draw::DrawState draw_state = UpdateDynamicState(pipeline, is_indexed, state);
    MidFrameFlush(state);
    MarkPass(pipeline, state);
    scheduler.BeginRendering(state);

    ASSERT(stride == (is_indexed ? sizeof(VkDrawIndexedIndirectCommand)
                                 : sizeof(VkDrawIndirectCommand)));
    // The records and the GPU count are arguments of the draw calls (hazard-tracked by Metal).
    const Metal::Draw::IndirectArgs indirect_args{
        .args = buffer->Ref(base),
        .count = max_count,
        .stride = stride,
        .indexed = is_indexed,
        .count_buffer =
            count_address != 0 ? count_buffer->Ref(count_offset) : Metal::BufferRef{},
    };
    OrderAccesses();
    const Metal::Draw::Resources resources = RecordResources(is_indexed);
    // Indirect tessellated draws are skipped by the helper (no run is left waiting).
    scheduler.RecordDraw(
        [=, runs = &scheduler.GetDrawRuns()](Metal::Encoder& e) {
            runs->DrawIndirect(e, DrawPipelines(*pipeline), draw_state, resources,
                               indirect_args); // see Draw
        },
        false);
    DebugState.IncDrawCall();
    if (PassLabels() == 2) {
        scheduler.EndRendering();
    }

    ResetBindings();
    scheduler.KickRecording();
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;
    BbStats::dispatches.fetch_add(1, std::memory_order_relaxed);
    // bbport: like draws, handed to the draw recording thread after the pipeline selection.
    const bool pipelined = UseDrawPipe() && !BbToggle::Disabled(BbToggle::PipelinedDispatch);
    if (!pipelined) {
        DrainDrawPipe();
    }
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    if (pipelined) {
        PostDraw(pipeline, nullptr, false, 0, &CsRegs());
        return;
    }
    DispatchRecord(pipeline);
}

void Rasterizer::DispatchRecord(const ComputePipeline* pipeline) {
    ++g_bb_draw_seq; // fifa-hybrid diagnostics
    if (BbSeqWindow()) {
        const auto* cs = pipeline->GetStages()[0];
        std::fprintf(stderr, "[SEQ] %llu dispatch %08llx queue %s\n", (unsigned long long)g_bb_draw_seq.load(),
                     (unsigned long long)(cs ? cs->pgm_hash : 0), stage_cs == &liverpool->GetCsRegs() ? "gfx" : "asc");
    }

    FrameCapture::Poll();
    gbuffer_draw = false;

    scheduler.PopPendingOperations();

    const auto& cs_program = CsRegs();

    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    // After the resource binding: its transfers (image uploads, downloads) mark themselves.
    const auto mark = [&] {
        if (auto* profiler = GpuProfiler::Get()) {
            profiler->Mark(cs.pgm_hash ^ 0xD15Aull, [&] {
                return fmt::format("dispatch cs {:016x} ({}x{}x{} groups)", cs.pgm_hash,
                                   cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
            });
        }
    };
    if (upscaler->Enabled()) {
        upscaler->OnDispatch(cs.pgm_hash);
    }
    if (ExecuteShaderHLE(cs, Regs(), cs_program, *this)) {
        return;
    }
    if (([] { static const bool v = std::getenv("BB_BCVIEW_LOG") != nullptr; return v; }())) {
        static std::set<u64> seen_w;
        const auto in_stage = [](u64 a, u64 n = 1) { return (a < 0x26a200000ull && a + n > 0x269c00000ull) || (a < 0x258e48000ull && a + n > 0x257e48000ull); };
        for (const auto& b : cs.buffers) {
            if (b.IsSpecial() || !b.is_written) continue;
            const auto v = b.GetSharp(cs);
            if (in_stage(v.base_address, v.GetSize()) && seen_w.insert(cs.pgm_hash ^ (v.base_address << 1)).second)
                std::fprintf(stderr, "[STAGEW] cs %016llx buffer at %#llx size %u stride %u dfmt %u nfmt %u fmtd %d grid %ux%ux%u\n",
                             (unsigned long long)cs.pgm_hash, (unsigned long long)v.base_address,
                             u32(v.GetSize()), u32(v.GetStride()), u32(v.GetDataFmt()), u32(v.GetNumberFmt()),
                             int(b.is_formatted), cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
        }
        for (const auto& d : cs.images) {
            const auto t = d.GetSharp(cs);
            if (d.is_written && in_stage(t.Address(), u64(t.width + 1) * (t.height + 1) * 32) &&
                seen_w.insert(cs.pgm_hash ^ (t.Address() << 1)).second)
                std::fprintf(stderr, "[STAGEW] cs %016llx image at %#llx fmt %u nfmt %u %ux%u\n",
                             (unsigned long long)cs.pgm_hash, (unsigned long long)t.Address(),
                             u32(t.GetDataFmt()), u32(t.GetNumberFmt()), t.width + 1, t.height + 1);
        }
        if (cs.pgm_hash == 0x52abaae0ull) {
            static std::atomic<u32> nc{0};
            if (nc++ < 400) {
                std::string s;
                for (const auto& r : cs.images) {
                    const auto rt = r.GetSharp(cs);
                    s += fmt::format(" [{} {:#x} f{} n{} {}x{} sel{}{}{}{} t{}]", r.is_written ? "W" : "R",
                                     rt.Address(), u32(rt.GetDataFmt()), u32(rt.GetNumberFmt()),
                                     rt.width + 1, rt.height + 1, u32(rt.dst_sel_x), u32(rt.dst_sel_y),
                                     u32(rt.dst_sel_z), u32(rt.dst_sel_w), u32(rt.tiling_index));
                }
                std::fprintf(stderr, "[CPYCS] grid %ux%u%s\n", cs_program.dim_x, cs_program.dim_y, s.c_str());
                static u64 last_kit = 0;
                static std::atomic<u32> nd{0};
                for (const auto& r : cs.images) {
                    const auto rt = r.GetSharp(cs);
                    if (r.is_written && rt.GetDataFmt() == AmdGpu::DataFormat::Format8_8_8_8 &&
                        rt.width + 1 == 2048) {
                        last_kit = rt.Address();
                    }
                    if (!r.is_written && last_kit && rt.GetDataFmt() == AmdGpu::DataFormat::Format32_32_32_32 &&
                        rt.tiling_index == 8 && memory->IsValidGpuMapping(last_kit, 2048 * 2048 * 4)) {
                        const u8* g = reinterpret_cast<const u8*>(last_kit);
                        u64 sum[4] = {}, nz[4] = {};
                        for (u32 i = 0; i < 2048 * 2048; ++i)
                            for (u32 k = 0; k < 4; ++k) { sum[k] += g[i * 4 + k]; nz[k] += g[i * 4 + k] != 0; }
                        std::fprintf(stderr, "[KITCPU] %#llx nz %llu %llu %llu %llu mean %llu %llu %llu %llu\n",
                                     (unsigned long long)last_kit, nz[0], nz[1], nz[2], nz[3], sum[0] >> 22,
                                     sum[1] >> 22, sum[2] >> 22, sum[3] >> 22);
                        if (nd++ < 3) {
                            const char* dump_dir = std::getenv("BB_DUMP_DIR"); // debug only: kit texture dumps go here when set
                            const auto p = fmt::format("{}/kitcpu{}_R8G8B8A8_2048x2048_t8.bin", dump_dir ? dump_dir : "/tmp", nd.load());
                            if (FILE* f = std::fopen(p.c_str(), "wb")) { std::fwrite(g, 1, 2048 * 2048 * 4, f); std::fclose(f); }
                        }
                        last_kit = 0;
                    }
                }
            }
        }
        for (const auto& d : cs.images) {
            const auto t = d.GetSharp(cs);
            if (d.is_written && t.GetDataFmt() == AmdGpu::DataFormat::Format32_32_32_32) {
                static std::set<u64> seen_cs;
                if (seen_cs.insert(cs.pgm_hash ^ t.Address()).second && seen_cs.size() < 12) {
                    std::fprintf(stderr, "[ENCCS] cs %016llx writes 128b at %#llx %ux%u grid %ux%ux%u nimg=%zu\n",
                                 (unsigned long long)cs.pgm_hash,
                                 (unsigned long long)t.Address(), t.width + 1, t.height + 1,
                                 cs_program.dim_x, cs_program.dim_y, cs_program.dim_z, cs.images.size());
                    for (const auto& r : cs.images) {
                        const auto rt = r.GetSharp(cs);
                        if (!r.is_written && memory->IsValidGpuMapping(rt.Address(), 16)) {
                            const u32* g = reinterpret_cast<const u32*>(rt.Address());
                            const u32 n = (rt.width + 1) * (rt.height + 1);
                            u32 c[4] = {};
                            for (u32 i = 0; i < n; ++i)
                                for (u32 k = 0; k < 4; ++k) c[k] += g[i * 4 + k] != 0;
                            for (u32 i = 0; i < 4; ++i)
                                std::fprintf(stderr, "[ENCCS]   t%u %08x %08x %08x %08x\n", i * 37,
                                             g[i * 148], g[i * 148 + 1], g[i * 148 + 2], g[i * 148 + 3]);
                            std::fprintf(stderr, "[ENCCS]   guest src nz x=%u y=%u z=%u w=%u of %u tile=%u\n",
                                         c[0], c[1], c[2], c[3], n, u32(rt.tiling_index));
                        }
                        std::fprintf(stderr, "[ENCCS]   img w=%d fmt=%u nfmt=%u %ux%u at %#llx\n", int(r.is_written),
                                     u32(rt.GetDataFmt()), u32(rt.GetNumberFmt()), rt.width + 1, rt.height + 1,
                                     (unsigned long long)rt.Address());
                    }
                }
            }
        }
    }

    SetDrawEncoders(*pipeline);
    if (!BindResources(pipeline)) {
        return;
    }

    static const bool bcseq = std::getenv("BB_BCSEQ") != nullptr;
    const auto seq_line = [&](const char* how) {
        if (bcseq && (u32(cs.pgm_hash) == 0x510f1e38u || u32(cs.pgm_hash) == 0xa7c5bd5fu ||
                      u32(cs.pgm_hash) == 0x52abaae0u)) {
            std::fprintf(stderr, "[SEQ] cs %08x %ux%ux%u %s\n", u32(cs.pgm_hash), cs_program.dim_x,
                         cs_program.dim_y, cs_program.dim_z, how);
        }
    };
    // fifa-hybrid: a small dispatch inside a render pass runs ahead of the submission
    // (BB_HOIST_CS; TryHoistDispatch) or after the pass (BB_DEFER_CS; TryDeferDispatch)
    // instead of splitting it.
    if (TryHoistDispatch(pipeline, cs_program.dim_x, cs_program.dim_y, cs_program.dim_z,
                         cs.pgm_hash)) {
        seq_line("hoisted");
        if (FrameCapture::Active()) {
            FrameCapture::Dispatch(cs.pgm_hash, cs_program.dim_x, cs_program.dim_y,
                                   cs_program.dim_z);
        }
        DebugState.IncDispatch();
        scheduler.KickRecording();
        return;
    }
    if (TryDeferDispatch(pipeline, cs_program.dim_x, cs_program.dim_y, cs_program.dim_z,
                         cs.pgm_hash)) {
        seq_line("deferred");
        if (FrameCapture::Active()) {
            FrameCapture::Dispatch(cs.pgm_hash, cs_program.dim_x, cs_program.dim_y,
                                   cs_program.dim_z);
        }
        DebugState.IncDispatch();
        ResetBindings();
        scheduler.KickRecording();
        return;
    }

    // 0.5.1-c: a full-screen copy dispatch (CopyDispatchDetectPass) draws as a fragment pass.
    if (CopyDispatchPlan plan; PlanCopyDispatchPass(cs, cs_program.dim_x, cs_program.dim_y,
                                                    cs_program.dim_z, plan)) {
        scheduler.EndRendering();
        if (auto* profiler = GpuProfiler::Get()) { // its own profile label
            profiler->Mark(cs.pgm_hash ^ 0xC0B1ull, [&] {
                return fmt::format("copy pass cs {:016x} ({}x{} groups)", cs.pgm_hash,
                                   cs_program.dim_x, cs_program.dim_y);
            });
        }
        if (FrameCapture::Active()) {
            FrameCapture::Dispatch(cs.pgm_hash, cs_program.dim_x, cs_program.dim_y,
                                   cs_program.dim_z);
        }
        seq_line("copy pass");
        OrderAccesses();
        runtime.GetBlitHelper().CopyDispatch(plan.width, plan.height, plan.dst, plan.src,
                                             plan.sampler, plan.scale, plan.remap, plan.swizzle);
        BbStats::copy_dispatch_passes.fetch_add(1, std::memory_order_relaxed);
        DebugState.IncDispatch();
        ResetBindings();
        scheduler.KickRecording();
        return;
    }

    scheduler.EndRendering();
    seq_line("dispatch");
    mark();

    const u32 dim_x = cs_program.dim_x, dim_y = cs_program.dim_y, dim_z = cs_program.dim_z;
    if (FrameCapture::Active()) {
        FrameCapture::Dispatch(cs.pgm_hash, dim_x, dim_y, dim_z);
    }
    // An empty grid is a no-op (as in Vulkan); Metal's validation rejects it.
    if (dim_x != 0 && dim_y != 0 && dim_z != 0) {
        const Metal::Size3 groups{dim_x, dim_y, dim_z};
        OrderAccesses();
        const Metal::Draw::Resources resources = RecordResources(false);
        // Dispatches always wait for their pipeline: skipping one could leave its results
        // missing for later frames (a failed one is skipped and counted).
        scheduler.RecordDraw(
            [=, runs = &scheduler.GetDrawRuns()](Metal::Encoder& e) {
                runs->Dispatch(e, ComputeView(*pipeline), resources.bound, resources.push,
                               groups);
            },
            false);
    }
    DebugState.IncDispatch();

    ResetBindings();
    scheduler.KickRecording();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;
    // bbport: handed to the recording thread like direct dispatches.
    const bool pipelined = UseDrawPipe() && !BbToggle::Disabled(BbToggle::PipelinedIndirectDraws);
    if (!pipelined) {
        DrainDrawPipe();
    }
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    if (pipelined) {
        const IndirectDraw indirect = {address + offset, 0, size, 1};
        PostDraw(pipeline, nullptr, false, 0, &CsRegs(), &indirect);
        return;
    }
    DispatchIndirectRecord(pipeline, address + offset, size);
}

void Rasterizer::DispatchIndirectRecord(const ComputePipeline* pipeline, VAddr address,
                                        u32 size) {
    scheduler.PopPendingOperations();
    const u32 offset = 0;

    SetDrawEncoders(*pipeline);
    if (!BindResources(pipeline)) {
        return;
    }

    // bbport: dispatchThreadgroupsWithIndirectBuffer's offset is 32-bit too (copied past 4 GiB).
    const auto [buffer, base] = buffer_cache.ObtainDrawArgumentBuffer(address + offset, size);
    NoteRead(buffer, base, size, Metal::ReadStage::Late);

    scheduler.EndRendering();

    // The group counts are an argument of the dispatch (hazard-tracked by Metal).
    const Metal::BufferRef args = buffer->Ref(base);
    OrderAccesses();
    const Metal::Draw::Resources resources = RecordResources(false);
    scheduler.RecordDraw(
        [=, runs = &scheduler.GetDrawRuns()](Metal::Encoder& e) {
            runs->DispatchIndirect(e, ComputeView(*pipeline), resources.bound, resources.push,
                                   args); // see DispatchRecord
        },
        false);
    DebugState.IncDispatch();

    ResetBindings();
    scheduler.KickRecording();
}

u64 Rasterizer::Flush() {
    DrainDrawPipe();
    const u64 current_tick = scheduler.CurrentTick();
    scheduler.Flush();
    return current_tick;
}

void Rasterizer::Finish() {
    DrainDrawPipe();
    scheduler.Finish();
}

void Rasterizer::FlushTinyWritten() {
    buffer_cache.FlushHotReadPages(); // fifa-hybrid
    std::vector<std::pair<VAddr, u64>> pending;
    {
        std::scoped_lock lk{tiny_written_mutex};
        pending.swap(tiny_written);
    }
    if (pending.empty()) {
        return;
    }
    std::sort(pending.begin(), pending.end());
    VAddr last_page = ~VAddr(0);
    for (const auto& [addr, size] : pending) {
        const VAddr page = addr & ~VAddr(0xFFF);
        if (page == last_page) {
            continue;
        }
        last_page = page;
        buffer_cache.DownloadSmallAsync(addr, size);
    }
}

void Rasterizer::OnSubmit() {
    DrainDrawPipe();
    FlushTinyWritten(); // fifa-hybrid
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    runtime.TickFrame();
}

const PreparedStage* Rasterizer::FindPreparedStage(const Shader::Info& stage) const {
    // Sharps a draw-preparation worker read from the same flattened user data.
    if (!bind_prepared || BbToggle::Disabled(BbToggle::PreparedResources)) {
        return nullptr;
    }
    for (u32 i = 0; i < bind_prepared->num_stages; ++i) {
        const auto& candidate = bind_prepared->stages[i];
        if (&candidate.program->info == &stage &&
            candidate.num_images == stage.images.size() &&
            candidate.num_samplers == stage.samplers.size() &&
            candidate.num_buffers == stage.buffers.size()) {
            return &candidate;
        }
    }
    return nullptr;
}

bool Rasterizer::BindHelperWanted() {
    // A spinning helper pays off only with cores to spare (the recording thread spins too).
    const char* env = std::getenv("BB_TEXTURE_HELPER");
    if (env && env[0]) {
        return env[0] == '1';
    }
    // Measured neutral on 16 threads (docs/parallel_gpu.md): opt-in.
    return false;
}

bool Rasterizer::HelperEligible(const Pipeline* pipeline) const {
    if (pipeline->IsCompute() || !bind_helper.Available() || FrameCapture::Active() ||
        BbToggle::Disabled(BbToggle::TextureBindHelper) ||
        BbToggle::Disabled(BbToggle::TextureBindingMemo) ||
        BbToggle::Disabled(BbToggle::UpdateImageFastPath)) {
        return false;
    }
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        // Mip arrays make the descriptor layout depend on the T#s.
        for (const auto& image : stage->images) {
            if (image.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
                return false;
            }
        }
        // Storage writes invalidate cached images from the buffer side.
        for (const auto& buffer : stage->buffers) {
            if (!buffer.IsSpecial() && buffer.is_written) {
                return false;
            }
        }
    }
    return true;
}

bool Rasterizer::TexturesBindableOnHelper(const Shader::Info& stage,
                                          const PreparedStage* prepared) {
    const u64 generation = texture_cache.RegistryGeneration();
    for (u32 image_index = 0; image_index < stage.images.size(); ++image_index) {
        const auto& image_desc = stage.images[image_index];
        if (image_desc.is_written) {
            return false;
        }
        const auto tsharp =
            prepared ? prepared->image_sharps[image_index] : image_desc.GetSharp(stage);
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid ||
            !memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !IsKnownFormat(data_fmt, num_fmt)) {
            continue; // null descriptor
        }
        const auto& entry =
            prepared ? CachedImageDescEntry(tsharp, image_desc, prepared->image_hashes[image_index])
                     : CachedImageDescEntry(tsharp, image_desc);
        if (entry.found_generation != generation) {
            return false;
        }
        VideoCore::ImageId image_id = entry.found_id;
        auto* image = &texture_cache.GetImage(image_id);
        if (const auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
            image_id = depth_image_id;
            image = &texture_cache.GetImage(image_id);
        }
        // bbport: a lazy usage upgrade (Runtime::UpgradeUsage) runs on the GPU thread.
        if (image->NeededUsage(entry.desc.view_info) != 0) {
            return false;
        }
        if (image->binding.needs_rebind || !texture_cache.IsUpToDate(image_id) ||
            scene_targets->Tracks(image->image_uid) ||
            (upscaler->Enabled() && upscaler->RedirectsSampled(image_id))) {
            return false;
        }
    }
    return true;
}

void Rasterizer::RunTextureTask(void* context) {
    auto& self = *static_cast<Rasterizer*>(context);
    auto& task = self.texture_task;
    for (u32 i = 0; i < task.count; ++i) {
        auto& stage = task.stages[i];
        if (!self.TexturesBindableOnHelper(*stage.info, stage.prepared)) {
            return;
        }
        auto binding = stage.binding;
        self.BindTextures(*stage.info, stage.prepared, binding, true);
        task.completed = i + 1;
    }
}

void Rasterizer::JoinBindHelper(void* context) {
    if (!BindHelper::OnHelper()) {
        static_cast<Rasterizer*>(context)->bind_helper.Join();
    }
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    pipeline_is_compute = pipeline->IsCompute();
    // fifa-hybrid: what this command's descriptors reach must not be among the deferred
    // dispatches' ranges (their results are not recorded yet): those run first (BB_DEFER_CS).
    // A compute pipeline needs no check here: deferred, it is appended after the pending ones
    // (their order holds); not deferred, DispatchRecord's EndRendering runs them first. Its
    // uploads over their ranges are caught by the caches (SynchronizeMemory, RefreshImage).
    if (!deferred_dispatches.empty() && !flushing_deferred && !pipeline_is_compute) {
        deferred_conflict_site = "bind(draw)";
        if (DeferredDispatchesConflict(pipeline)) {
            BbStats::deferred_dispatch_conflicts.fetch_add(1, std::memory_order_relaxed);
            FlushDeferredDispatches();
        }
    }
    g_probe_key = pipeline_is_compute
                      ? static_cast<const ComputePipeline*>(pipeline)->KeyValue() : 0;
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }
    // bbport: a stage that reaches memory by address needs the page table and the fault buffer,
    // which exist only with BB_DIRECT_MEMORY_ACCESS=1. Shaders compiled without it never do, and
    // a cache entry compiled with it is rejected at load (LoadShaderMeta): this is a backstop.
    // Skipped, not bound to the zero stand-in (both descriptors would be written).
    if (!buffer_cache.GetBdaPageTableBuffer()) {
        for (const auto* stage : pipeline->GetStages()) {
            if (stage && stage->uses_dma) {
                static std::atomic<bool> reported{false};
                if (!reported.exchange(true)) {
                    LOG_ERROR(Render, "Shader {:#x} reaches memory by address without "
                                      "BB_DIRECT_MEMORY_ACCESS: its draws and dispatches are skipped",
                              stage->pgm_hash);
                }
                return false;
            }
        }
    }

    bound.clear();
    image_infos.clear();
    // bbport: G-buffer passes sample material textures for a scene rendered below the output
    // size; the temporal upscaler restores the detail of the output's mip level (FSR guide:
    // log2(render / output)). Shadows, post-processing and UI keep the guest's bias.
    sampler_lod_bias = 0.0f;
    if (!pipeline->IsCompute() && !BbToggle::Disabled(BbToggle::SceneMipBias) &&
        std::popcount(static_cast<const GraphicsPipeline*>(pipeline)->GetGraphicsKey().mrt_mask &
                      0xff) >= 5) {
        sampler_lod_bias = upscaler->SceneMipBias();
        static float reported = 0.0f;
        if (sampler_lod_bias != reported) {
            reported = sampler_lod_bias;
            std::printf("Upscaler: scene texture LOD bias %.2f\n", sampler_lod_bias);
        }
    }

    bool uses_dma = false;

    static const bool stats = std::getenv("BB_FRAME_STATS") != nullptr;
    if (stats && ((helper_full + helper_partial + helper_serial) & 1023) == 0) {
        static auto window = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        if (now - window >= std::chrono::seconds(5)) {
            window = now;
            const double forks = std::max<double>(1, helper_full + helper_partial);
            std::printf("Texture helper: %llu binds in parallel, %llu partly, %llu serial; "
                        "per fork %.0f cycles helper task, %.0f cycles GPU thread wait\n",
                        static_cast<unsigned long long>(helper_full),
                        static_cast<unsigned long long>(helper_partial),
                        static_cast<unsigned long long>(helper_serial),
                        bind_helper.task_cycles.exchange(0) / forks,
                        bind_helper.wait_cycles / forks);
            bind_helper.wait_cycles = 0;
            helper_full = helper_partial = helper_serial = 0;
        }
    }

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(Regs());
    if (!HelperEligible(pipeline)) {
        ++helper_serial;
        for (const auto* stage : pipeline->GetStages()) {
            if (!stage) {
                continue;
            }
            stage->PushUd(binding, push_data);
            const PreparedStage* prepared = FindPreparedStage(*stage);
            BindBuffers(*stage, prepared, binding, push_data);
            BindTextures(*stage, prepared, binding);
            uses_dma |= stage->uses_dma;
        }
    } else {
        // bbport: textures on the helper, buffers here. Without mip arrays every stage's
        // bindings are [buffers][images][samplers], one each, so both sides know their binding
        // numbers up front, and `bound` is sized before the helper starts (neither side grows
        // it).
        struct BufferStage {
            const Shader::Info* info;
            const PreparedStage* prepared;
            Shader::Backend::Bindings binding;
        };
        std::array<BufferStage, Shader::MaxStageTypes> buffer_stages;
        auto& task = texture_task;
        task.count = 0;
        task.completed = 0;
        task.accesses.clear();
        for (const auto* stage : pipeline->GetStages()) {
            if (!stage) {
                continue;
            }
            const u32 num_buffers = static_cast<u32>(stage->buffers.size());
            const u32 num_descriptors =
                num_buffers + static_cast<u32>(stage->images.size() + stage->samplers.size());
            stage->PushUd(binding, push_data);
            const PreparedStage* prepared = FindPreparedStage(*stage);
            buffer_stages[task.count] = {stage, prepared, binding};
            auto texture_binding = binding;
            texture_binding.buffer += num_buffers;
            texture_binding.unified += num_buffers;
            task.stages[task.count] = {stage, prepared, texture_binding};
            ++task.count;
            binding.buffer += num_buffers;
            binding.unified += num_descriptors;
            uses_dma |= stage->uses_dma;
        }
        bound.resize(binding.unified);
        bind_helper.Fork(&RunTextureTask, this);
        for (u32 i = 0; i < task.count; ++i) {
            auto& stage = buffer_stages[i];
            BindBuffers(*stage.info, stage.prepared, stage.binding, push_data);
        }
        if (draw_inputs.pending && !BbToggle::Disabled(BbToggle::EarlyDrawInputs)) {
            ResolveVertexBuffers(draw_inputs.pipeline, draw_inputs.prepared);
            if (draw_inputs.is_indexed) {
                ResolveIndexBuffer(draw_inputs.index_offset);
            }
            draw_inputs.resolved = true;
        }
        bind_helper.Join();
        // The images the helper bound, accessed here (the runtime is not thread-safe).
        for (const auto& access : task.accesses) {
            runtime.AccessImage(access.image, access.write);
        }
        task.accesses.clear();
        ++(task.completed == task.count ? helper_full
                                        : task.completed ? helper_partial : helper_serial);
        // Stages the helper left (an image needing work that records commands).
        for (u32 i = task.completed; i < task.count; ++i) {
            auto& stage = task.stages[i];
            BindTextures(*stage.info, stage.prepared, stage.binding);
        }
    }

    if (uses_dma) {
        buffer_cache.SynchronizeDmaBuffers();
        fault_process_pending = true;
    }

    return true;
}

Metal::Draw::Bound& Rasterizer::BoundAt(u32 binding) {
    if (binding >= bound.size()) {
        ASSERT_MSG(!BindHelper::OnHelper(), "The texture binding helper grew the binding table");
        bound.resize(binding + 1);
    }
    return bound[binding];
}

void Rasterizer::AccessBoundImage(VideoCore::Image& image, bool write, bool on_helper) {
    if (on_helper) {
        texture_task.accesses.push_back({&image, write});
    } else {
        runtime.AccessImage(&image, write);
    }
}

Metal::Draw::Resources Rasterizer::RecordResources(bool indexed) {
    const std::span<const Metal::Draw::Bound> bound_span{bound};
    const std::span<const u8> push{reinterpret_cast<const u8*>(&push_data), sizeof(push_data)};
    const std::span<const Metal::Draw::VertexStream> streams =
        pipeline_is_compute ? std::span<const Metal::Draw::VertexStream>{}
                            : std::span<const Metal::Draw::VertexStream>{
                                  vertex_binds.streams.data(), vertex_binds.streams.size()};
    // One reservation for the three copies and the command, so that they share a chunk (each
    // RecordData call keeps room for the command after its copy).
    scheduler.ReserveRecordData(bound_span.size_bytes() + push.size_bytes() +
                                streams.size_bytes() + 3 * 64 + 2 * 1024);
    Metal::Draw::Resources resources{
        .bound = scheduler.RecordData(bound_span),
        .push = scheduler.RecordData(push),
        .streams = scheduler.RecordData(streams),
    };
    if (indexed) {
        resources.indices = index_bind.buffer;
        resources.index_type = index_bind.type;
    }
    return resources;
}

void Rasterizer::ResolveVertexBuffers(const GraphicsPipeline* pipeline,
                                      const PreparedDraw* prepared) {
    const auto& regs = Regs();
    auto& streams = vertex_binds.streams;
    streams.clear();
    VertexInputs<VertexAttribute> attributes;
    VertexInputs<AmdGpu::Buffer> guest_buffers;
    // Inputs a draw-preparation worker read from the same registers and (verified) user data.
    // The attribute formats and step rates belong to the pipeline (its key): only the streams
    // are per draw.
    const auto& fetch = pipeline->GetFetchShader();
    const PreparedVertexInputs* ready =
        prepared && prepared->vertex.valid && fetch &&
                fetch->attributes.size() == prepared->vertex.count && !FrameCapture::Active() &&
                !BbToggle::Disabled(BbToggle::PreparedResources) &&
                !BbToggle::Disabled(BbToggle::PreparedVertexInputs)
            ? &prepared->vertex
            : nullptr;
    if (ready) {
        guest_buffers.assign(ready->buffers, ready->buffers + ready->count);
    } else {
        pipeline->GetVertexInputs(attributes, guest_buffers, regs.vgt_instance_step_rate_0,
                                  regs.vgt_instance_step_rate_1);
    }

    if (guest_buffers.empty()) {
        // If there are no streams, there is nothing further to do.
        return;
    }
    if (static const u64 want = [] { const char* e = std::getenv("BB_BUF_LOG"); return e ? std::strtoull(e, nullptr, 16) : 0ull; }(); want) {
        const auto* vs = pipeline->GetStages()[u32(Shader::SwStage::Vertex)];
        static int printed = 0;
        if (vs && (vs->pgm_hash & 0xffffffffull) == want && printed < 3) {
            ++printed;
            for (u32 i = 0; i < guest_buffers.size(); ++i) {
                const auto& vb = guest_buffers[i];
                std::fprintf(stderr, "[VBUF] %08llx stream%u addr=%#llx size=%u stride=%u nrec=%u dfmt=%u nfmt=%u idx=%u ready=%d\n",
                             (unsigned long long)vs->pgm_hash, i, (unsigned long long)vb.base_address, u32(vb.GetSize()),
                             u32(vb.GetStride()), u32(vb.num_records), u32(vb.GetDataFmt()), u32(vb.GetNumberFmt()),
                             regs.num_indices, int(ready != nullptr));
                if (vb.base_address && vb.GetSize() >= 32) {
                    const u32 n = std::min<u32>(16, vb.GetSize() / 4);
                    const float* f = reinterpret_cast<const float*>(vb.base_address);
                    const u32* u = reinterpret_cast<const u32*>(vb.base_address);
                    std::string line;
                    for (u32 k = 0; k < n; ++k) line += fmt::format("{:g}/{:#x} ", f[k], u[k]);
                    std::fprintf(stderr, "[VBUF]   data: %s\n", line.c_str());
                }
            }
        }
    }
    if (motion_draw) {
        // Include every stream: identical position buffers can be paired with different
        // skinning/instance data. Index topology and base offsets are matched separately.
        motion_geometry = ready ? ready->buffers_hash
                                : XXH3_64bits(guest_buffers.data(),
                                              guest_buffers.size() * sizeof(AmdGpu::Buffer));
    }
    // Object motion research: vertex streams of G-buffer draws.
    if (FrameCapture::Active() && gbuffer_draw) {
        for (const auto& vb : guest_buffers) {
            char note[128];
            std::snprintf(note, sizeof(note), "\n  gb vb at %#llx size %u stride %u",
                          (unsigned long long)vb.base_address, u32(vb.GetSize()),
                          u32(vb.GetStride()));
            FrameCapture::Note(note);
        }
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        const VideoCore::Buffer* buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    VertexInputs<BufferRange> ranges{};
    VertexInputs<BufferRange> ranges_merged{};
    if (ready) {
        for (u32 i = 0; i < ready->num_ranges; ++i) {
            ranges_merged.emplace_back(ready->ranges[i].base, ready->ranges[i].end);
        }
    } else {
        for (const auto& buffer : guest_buffers) {
            if (buffer.base_address != 0 && buffer.GetSize() > 0) {
                ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
            }
        }
    }

    // Merge connecting ranges together
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        std::tie(range.buffer, range.offset) =
            buffer_cache.ObtainBuffer(range.base_address, size, false);
        NoteRead(range.buffer, range.offset, size);
        if (FrameCapture::Tracing()) {
            FrameCapture::Trace(fmt::format("V {:#x} {}", range.base_address, size));
        }
    }
    if (g_drawlog) {
        static const bool vdiff = std::getenv("BB_VDIFF") != nullptr;
        const auto* vs = pipeline->GetStages()[u32(Shader::SwStage::Vertex)];
        for (u32 i = 0; i < guest_buffers.size(); ++i) {
            const auto& vb = guest_buffers[i];
            const u64 size = vb.GetSize();
            if (vb.base_address == 0 || size == 0) {
                continue;
            }
            const u32* dw = reinterpret_cast<const u32*>(vb.base_address);
            std::string words;
            for (u32 k = 0; k < std::min<u64>(size / 4, 16); ++k) {
                words += fmt::format(" {:08x}", dw[k]);
            }
            std::fprintf(g_drawlog, "V %08x s%u %#llx size %llu stride %u nrec %u h %016llx cm %d gm %d:%s\n",
                         vs ? u32(vs->pgm_hash) : 0u, i, (unsigned long long)vb.base_address,
                         (unsigned long long)size, u32(vb.GetStride()), u32(vb.num_records),
                         (unsigned long long)XXH3_64bits(dw, size),
                         int(buffer_cache.IsRegionCpuModified(vb.base_address, size)),
                         int(buffer_cache.IsRegionGpuModified(vb.base_address, size)), words.c_str());
            if (vdiff && g_drawlog_streams.size() < 256) {
                g_drawlog_streams.try_emplace({vb.base_address, size},
                                              DrawlogStream{vs ? u32(vs->pgm_hash) : 0u, i, vb.base_address, size});
            }
        }
    }

    // The streams, by attribute location (fetch shader order, like guest_buffers). A V# without
    // memory leaves its location empty: the draw helper binds its null stand-in there.
    u32 stream_with_memory = 0;
    for (u32 i = 0; i < guest_buffers.size(); ++i) {
        const auto& buffer = guest_buffers[i];
        const bool has_memory = buffer.base_address != 0 && buffer.GetSize() > 0;
        const auto host_buffer_info =
            !has_memory ? ranges_merged.end()
            : ready     ? ranges_merged.begin() + ready->range_index[stream_with_memory++]
                        : std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                              return buffer.base_address >= range.base_address &&
                                     buffer.base_address < range.end_address;
                          });
        const u32 location = fetch->attributes[i].semantic;
        if (location >= streams.capacity()) {
            static std::once_flag logged;
            std::call_once(logged, [location] {
                LOG_ERROR(Render, "Vertex attribute location {} has no vertex buffer slot",
                          location);
            });
            continue;
        }
        if (streams.size() <= location) {
            streams.resize(location + 1);
        }
        if (!has_memory) {
            continue;
        }
        ASSERT(host_buffer_info != ranges_merged.cend());
        streams[location] = {
            .buffer = host_buffer_info->buffer->Ref(host_buffer_info->offset + buffer.base_address -
                                                    host_buffer_info->base_address),
            .stride = buffer.GetStride(),
        };
    }
    static const bool vb_log = std::getenv("BB_VB_LOG") != nullptr;
    static std::atomic<u32> vb_logged{};
    if (vb_log && fetch->attributes.size() == 6 && vb_logged.fetch_add(1) < 400) {
        const auto [vb_vo, vb_io] =
            GetDrawOffsets(regs, pipeline->GetStage(Shader::SwStage::Vertex), fetch);
        std::string s = fmt::format("[VBLOG] ready={} sr0={} sr1={} vo={} io={} ni={}",
                                    ready != nullptr, regs.vgt_instance_step_rate_0,
                                    regs.vgt_instance_step_rate_1, vb_vo, vb_io,
                                    regs.num_instances.NumInstances());
        for (u32 i = 0; i < guest_buffers.size(); ++i) {
            const auto& b = guest_buffers[i];
            const auto& at = fetch->attributes[i];
            const u32 loc = at.semantic;
            u32 first = 0;
            if (b.base_address) {
                std::memcpy(&first, reinterpret_cast<const void*>(b.base_address), 4);
            }
            if (i == 5 && b.base_address && b.GetStride()) {
                u32 nz = 0, cnt = b.num_records, firstnz = ~0u;
                for (u32 k = 0; k < cnt; ++k) {
                    u32 v;
                    std::memcpy(&v, reinterpret_cast<const void*>(b.base_address + k * b.GetStride()), 4);
                    if (v != 0 && firstnz == ~0u) firstnz = k;
                    nz += v != 0;
                }
                s += fmt::format(" | a5nz {}/{} first={} gpumod={}", nz, cnt, firstnz,
                                 buffer_cache.IsRegionGpuModified(b.base_address, b.GetSize()));
                std::scoped_lock lk{vb_w_mutex};
                u32 hits = 0;
                for (const auto& [wa, ws, wt] : vb_w_ranges) {
                    if (wa < b.base_address + b.GetSize() && b.base_address < wa + ws && hits++ < 6) {
                        s += fmt::format(" W{}[{:#x}+{:#x}]", wt, wa, ws);
                    }
                }
                s += fmt::format(" whits={}", hits);
                if (b.GetStride() == 16) {
                    const u8* rp = reinterpret_cast<const u8*>(b.base_address) - 12;
                    for (u32 r = 0; r < 3; ++r) {
                        s += " R[";
                        for (u32 k = 0; k < 16; ++k) s += fmt::format("{:02x}", rp[r * 16 + k]);
                        s += "]";
                    }
                    const u8* mid = rp + (cnt / 2) * 16;
                    s += " M[";
                    for (u32 k = 0; k < 16; ++k) s += fmt::format("{:02x}", mid[k]);
                    s += "]";
                }
            }
            s += fmt::format(" | a{} loc{} step{} dfmt{} nfmt{} base={:#x} stride={} n={} "
                             "w0={:#x} host={}",
                             i, loc, u32(at.GetStepRate()), u32(b.GetDataFmt()),
                             u32(b.GetNumberFmt()), u64(b.base_address), u32(b.GetStride()),
                             u32(b.num_records), first,
                             loc < streams.size() && streams[loc].buffer.buffer != nullptr);
        }
        std::fprintf(stderr, "%s\n", s.c_str());
    }
}

void Rasterizer::ResolveIndexBuffer(u32 index_offset) {
    const auto& regs = Regs();

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [buffer, offset] =
        buffer_cache.ObtainIndexBuffer(index_address, index_buffer_size);
    NoteRead(buffer, offset, index_buffer_size);
    if (FrameCapture::Tracing()) {
        FrameCapture::Trace(fmt::format("I {:#x} {}", index_address, index_buffer_size));
    }
    index_bind = {buffer->Ref(offset), is_index16 ? Metal::IndexType::U16 : Metal::IndexType::U32};
}

void Rasterizer::ResetBindings() {
    const u64 tick = scheduler.CurrentTick();
    for (auto& image_id : bound_images) {
        auto& image = texture_cache.GetImage(image_id);
        image.binding = {};
        image.last_bound_tick = tick; // an in-order command of this submission used it
    }
    for (const auto [buffer, offset, size, is_written, read_stage, encoders] : bound_buffers) {
        runtime.AccessBuffer(buffer, offset, size, is_written, encoders, read_stage);
    }
    for (const auto [buffer, offset, size, is_written, read_stage, encoders] : read_ranges) {
        runtime.AccessBuffer(buffer, offset, size, is_written, encoders, read_stage);
    }
    bound_images.clear();
    bound_images_written.clear();
    bound_images_prev_write.clear();
    bound_image_ids.clear();
    bound_image_views.clear();
    bound_samplers.clear();
    bound_images_simple = true;
    bound_buffers.clear();
    read_ranges.clear();
}

void Rasterizer::NoteRead(const VideoCore::Buffer* buffer, u64 offset, u64 size,
                          Metal::ReadStage stage) {
    read_ranges.push_back({buffer, offset, size, false, stage, draw_encoders});
}

void Rasterizer::SetDrawEncoders(const Pipeline& pipeline, bool indirect) {
    if (pipeline.IsCompute()) {
        draw_encoders = Metal::EncoderCompute;
        return;
    }
    const auto stages = pipeline.GetStages();
    const bool kernels = stages[u32(Shader::SwStage::TessellationControl)] != nullptr ||
                         stages[u32(Shader::SwStage::Geometry)] != nullptr;
    draw_encoders = Metal::EncoderRender | (kernels || indirect ? Metal::EncoderCompute : 0);
}

void Rasterizer::OrderAccesses() {
    // bbport: a command of the pass that touches the ranges of copies deferred to its end
    // (vk_shader_hle.h) has them run first: the pass ends there and reopens.
    if (scheduler.IsRendering()) {
        bool conflict = false;
        for (const auto& b : bound_buffers) {
            conflict = conflict || DeferredCopiesConflict(b.buffer, b.offset, b.size, b.is_written);
        }
        for (const auto& r : read_ranges) {
            conflict = conflict || DeferredCopiesConflict(r.buffer, r.offset, r.size, false);
        }
        if (conflict) {
            BbStats::deferred_copy_conflicts.fetch_add(1, std::memory_order_relaxed);
            scheduler.EndRendering();
            scheduler.ResumeRendering();
        }
    }
    // fifa-hybrid: the same for the dispatches deferred to the pass end (BB_DEFER_CS): the
    // ranges a draw reads by vertex or index fetch, and the attachments it writes.
    if (scheduler.IsRendering() && !deferred_dispatches.empty() && !flushing_deferred) {
        deferred_conflict_site = "order";
        bool conflict = false;
        for (const auto& b : bound_buffers) {
            conflict = conflict || DeferredDispatchTouches(b.buffer->CpuAddr() + b.offset, b.size,
                                                           b.is_written);
        }
        for (const auto& r : read_ranges) {
            conflict = conflict || DeferredDispatchTouches(r.buffer->CpuAddr() + r.offset, r.size,
                                                           false);
        }
        for (size_t i = 0; i < bound_images.size() && !conflict; ++i) {
            const bool written = i < bound_images_written.size() && bound_images_written[i];
            conflict = DeferredDispatchTouchesImage(texture_cache.GetImage(bound_images[i]), written);
        }
        if (conflict) {
            BbStats::deferred_dispatch_conflicts.fetch_add(1, std::memory_order_relaxed);
            scheduler.EndRendering();
            scheduler.ResumeRendering();
        }
    }
    // Checked now, right before the command is recorded: the uploads and copies recorded while
    // its resources were bound are among the accesses it is checked against.
    Metal::OrderRequest need;
    for (const auto& b : bound_buffers) {
        need |= runtime.IsBufferAccessed(b.buffer, b.offset, b.size, b.is_written);
    }
    for (const auto& r : read_ranges) {
        need |= runtime.IsBufferAccessed(r.buffer, r.offset, r.size);
    }
    runtime.FlushBarriers(need);
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = CsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    runtime.CopyColorAndDepth(&src_image, &dst_image);
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = CsRegs();
    const auto& info = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    runtime.ClearImage(&image1, range, clear);
    return true;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, const PreparedStage* prepared,
                             Shader::Backend::Bindings& binding, Shader::PushData& push_data) {
    const u64 alignment = device.GetLimits().storage_alignment;
    if (g_drawlog) { // BB_DRAWLOG_TRIGGER: this stage's constant buffers
        const char* kind = stage.sw_stage == Shader::SwStage::Compute ? "cs" : stage.sw_stage == Shader::SwStage::Fragment ? "ps" : "vs";
        for (u32 b = 0; b < stage.buffers.size(); ++b) {
            const auto sharp = stage.buffers[b].GetSharp(stage);
            if (!sharp.base_address) continue;
            const u32* dw = reinterpret_cast<const u32*>(sharp.base_address);
            const u32 nd = u32(std::min<u64>(sharp.GetSize() / 4, 1024));
            u64 h = 1469598103934665603ull;
            for (u32 i = 0; i < nd; ++i) h = (h ^ dw[i]) * 1099511628211ull;
            std::fprintf(g_drawlog, "B %s %08x b%u @%#llx size %llu h %016llx:", kind, u32(stage.pgm_hash), b,
                         (unsigned long long)sharp.base_address, (unsigned long long)sharp.GetSize(),
                         (unsigned long long)h);
            for (u32 i = 0; i < std::min(nd, 160u); ++i) std::fprintf(g_drawlog, " %08x", dw[i]);
            std::fputc('\n', g_drawlog);
        }
    }
    // fifa-hybrid diagnostics: BB_BUFWATCH=<hex address> names, once per stage, the buffers a
    // shader writes that cover the address (who produces the value the game polls there).
    if (const u64 watch = VideoCore::g_bb_bufwatch.load(std::memory_order_relaxed); watch != 0) {
        static std::set<u64> seen;
        for (u32 i = 0; i < stage.buffers.size(); ++i) {
            const auto sharp = stage.buffers[i].GetSharp(stage);
            const u64 base = sharp.base_address, size = sharp.GetSize();
            if (base <= watch && watch < base + size && seen.insert(stage.pgm_hash * 16 + i).second) {
                std::fprintf(stderr, "[BUFWATCH] stage %08llx (%s) buf%u addr=%#llx size=%llu written=%d\n",
                             (unsigned long long)stage.pgm_hash,
                             stage.sw_stage == Shader::SwStage::Compute ? "cs" : stage.sw_stage == Shader::SwStage::Fragment ? "ps" : "vs",
                             i, (unsigned long long)base, (unsigned long long)size,
                             int(stage.buffers[i].is_written));
            }
        }
    }
    // fifa-hybrid diagnostics: BB_BUF_LOG=<hex pgm hash> prints that stage's buffer bindings once.
    if (static const u64 want = [] {
            const char* env = std::getenv("BB_BUF_LOG");
            return env ? std::strtoull(env, nullptr, 16) : 0ull;
        }();
        want != 0 && (stage.pgm_hash & 0xffffffffull) == want) {
        static int printed = 0;
        if (printed < 3) {
            ++printed;
            for (u32 i = 0; i < stage.buffers.size(); ++i) {
                const auto sharp = stage.buffers[i].GetSharp(stage);
                std::fprintf(stderr, "[BUF] %08llx buf%u addr=%#llx size=%u written=%d stride=%u nrec=%u fmt=%u\n",
                             (unsigned long long)stage.pgm_hash, i,
                             (unsigned long long)sharp.base_address, sharp.GetSize(),
                             int(stage.buffers[i].is_written), sharp.GetStride(), u32(sharp.num_records),
                             u32(sharp.GetDataFmt()));
                if (sharp.base_address && sharp.GetSize() >= 16) {
                    const u32 n = std::min<u32>(40, sharp.GetSize() / 4);
                    const float* f = reinterpret_cast<const float*>(sharp.base_address);
                    std::string line;
                    for (u32 k = 0; k < n; ++k) line += fmt::format("{:g} ", f[k]);
                    std::fprintf(stderr, "[BUF]   data: %s\n", line.c_str());
                }
            }
        }
    }
    for (u32 buffer_index = 0; buffer_index < stage.buffers.size(); ++buffer_index) {
        const auto& desc = stage.buffers[buffer_index];
        auto& element = BoundAt(binding.unified++);
        const RingBinding* ring = num_ring_stages ? FindRingBinding(stage, buffer_index) : nullptr;
        if (ring) {
            // Copied by the GPU command thread into the constant ring (read-only).
            if (!desc.IsSpecial()) {
                if (ring->size == 864 && gbuffer_draw &&
                    memory->IsValidGpuMapping(ring->address, 0)) {
                    camera_motion->OnConstants(reinterpret_cast<const float*>(ring->address));
                }
                push_data.AddOffset(binding.buffer, 0);
            }
            const Metal::BufferRef ring_ref = constant_ring->Handle();
            element = {.buffer = {ring_ref.buffer, ring_ref.offset + ring->offset},
                       .size = ring->size};
        } else if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                element = {.buffer = gds_buf->Ref(), .size = gds_buf->SizeBytes()};
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& stream_buffer = buffer_cache.GetStreamBuffer();
                const u32 ubo_size = stage.FlatUserData().size() * sizeof(u32);
                if (FrameCapture::Active()) {
                    FrameCapture::Buffer(stage.pgm_hash, binding.buffer, 0,
                                         stage.FlatUserData().data(), ubo_size);
                }
                const u64 offset =
                    stream_buffer.Copy(stage.FlatUserData().data(), ubo_size, alignment);
                element = {.buffer = stream_buffer.Ref(offset), .size = ubo_size};
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (Regs().clipper_control.user_clip_plane_enable == 0) {
                    element = NullBufferBinding();
                } else {
                    auto& stream_buffer = buffer_cache.GetStreamBuffer();
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = Regs().clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = stream_buffer.Copy(planes.data(), ubo_size, alignment);
                    element = {.buffer = stream_buffer.Ref(offset), .size = ubo_size};
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                if (FrameCapture::Tracing()) {
                    FrameCapture::Trace(fmt::format("M {} {:08x}", u32(stage.sw_stage),
                                                    u32(stage.pgm_hash)));
                }
                // bbport: direct memory access (BB_DIRECT_MEMORY_ACCESS=1) reads any guest
                // memory by address: the whole arena and the page table, for the ordering.
                const auto stage_of_read = stage.sw_stage == Shader::SwStage::Fragment ||
                                                   stage.sw_stage == Shader::SwStage::Compute
                                               ? Metal::ReadStage::Late
                                               : Metal::ReadStage::Vertex;
                NoteRead(bda_buffer, 0, bda_buffer->SizeBytes(), stage_of_read);
                for (const auto& chunk : buffer_cache.GetChunks()) {
                    NoteRead(chunk.buffer, 0, chunk.buffer->SizeBytes(), stage_of_read);
                }
                element = {.buffer = bda_buffer->Ref(), .size = bda_buffer->SizeBytes()};
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                element = {.buffer = fault_buffer->Ref(), .size = fault_buffer->SizeBytes()};
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetStreamBuffer();
                const auto& cs_program = CsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                lds_buffer.Commit();
                element = {.buffer = lds_buffer.Ref(offset), .size = lds_size};
            } else {
                UNREACHABLE_MSG("Unexpected buffer type {}", u32(desc.buffer_type));
            }
        } else {
            const auto vsharp =
                prepared ? prepared->buffer_sharps[buffer_index] : desc.GetSharp(stage);
            if (vsharp.GetSize() == 864 && gbuffer_draw &&
                memory->IsValidGpuMapping(vsharp.base_address, 0)) {
                camera_motion->OnConstants(reinterpret_cast<const float*>(vsharp.base_address));
            }
            // Object motion research: the vertex shader buffers of G-buffer draws.
            if (FrameCapture::Active() && gbuffer_draw && stage.sw_stage == Shader::SwStage::Vertex &&
                vsharp.base_address != 0 && memory->IsValidGpuMapping(vsharp.base_address, 0)) {
                const auto* f = reinterpret_cast<const float*>(vsharp.base_address);
                char note[192];
                std::snprintf(note, sizeof(note),
                              "\n  gb vs %08x slot %u at %#llx size %u stride %u: %g %g %g %g",
                              u32(stage.pgm_hash), binding.buffer,
                              (unsigned long long)vsharp.base_address, u32(vsharp.GetSize()),
                              u32(vsharp.GetStride()), f[0], f[1], f[2], f[3]);
                FrameCapture::Note(note);
            }
            if (FrameCapture::Active() && vsharp.base_address != 0 && vsharp.GetSize() != 0 &&
                memory->IsValidGpuMapping(vsharp.base_address, 0)) {
                FrameCapture::Buffer(stage.pgm_hash, binding.buffer, vsharp.base_address,
                                     reinterpret_cast<const void*>(vsharp.base_address),
                                     memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize()));
            }
            if (vsharp.base_address == 0 || vsharp.GetSize() == 0) {
                element = NullBufferBinding();
            } else {
                const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
                if (size != vsharp.GetSize()) {
                    LOG_ERROR(Render, "Clamped size from {} to {} for stage {:#x}",
                              vsharp.GetSize(), size, stage.pgm_hash);
                }
                const auto [buffer, offset] = buffer_cache.ObtainBuffer(
                    vsharp.base_address, size, desc.is_written, desc.is_formatted);
                if (FrameCapture::Tracing()) {
                    FrameCapture::Trace(fmt::format("B {} {:08x} {:#x} {} {}",
                                                    u32(stage.sw_stage), u32(stage.pgm_hash),
                                                    vsharp.base_address, size,
                                                    desc.is_written ? 1 : 0));
                }
                const u64 offset_aligned = Common::AlignDown(offset, alignment);
                const u64 adjust = offset - offset_aligned;
                if (adjust % 4 != 0) {
                    LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                                stage.pgm_hash);
                }
                push_data.AddOffset(binding.buffer, adjust);
                // DEBUG(bbport): BB_PROBE_SHADER=<hash> logs this shader's buffers (temporary).
                static const u64 probe_hash = [] {
                    const char* e = std::getenv("BB_PROBE_SHADER");
                    return e ? std::strtoull(e, nullptr, 16) : 0ull;
                }();
                if (probe_hash != 0 && (stage.pgm_hash == probe_hash || g_probe_key == probe_hash)) {
                    static std::atomic<int> probes{0};
                    if (probes.fetch_add(1) < 660) {
                        const auto* w = reinterpret_cast<const u32*>(vsharp.base_address);
                        u32 mx = 0;
                        const u32 n = std::min<u64>(size / 64, 4096);
                        for (u32 i = 0; i < n; ++i) {
                            mx = std::max(mx, w[i * 16]);
                        }
                        std::fprintf(stderr,
                                     "[PROBE] cs %#llx b%u addr %#llx size %llu stride %u w%d "
                                     "adj %llu off %llu | w0..3 %08x %08x %08x %08x max(w[16i]) %u "
                                     "over %u\n",
                                     (unsigned long long)stage.pgm_hash, buffer_index,
                                     (unsigned long long)vsharp.base_address,
                                     (unsigned long long)size, u32(vsharp.GetStride()),
                                     int(desc.is_written), (unsigned long long)adjust,
                                     (unsigned long long)offset, w[0], w[1], w[2], w[3], mx, n);
                        // BB_PROBE_FIELD=<buffer>,<dword>,<stride dwords>[;...]: that dword over
                        // the buffer's records (min / max / mean / distinct, records over 64).
                        // BB_PROBE_DUMP=<buffer>,<records>,<stride dwords>: hex dump of the first
                        // records (6 dispatches).
                        static const auto fields = [] {
                            std::vector<std::array<u64, 3>> v;
                            if (const char* e = std::getenv("BB_PROBE_FIELD")) {
                                std::string spec{e};
                                size_t pos = 0;
                                while (pos < spec.size()) {
                                    std::array<u64, 3> f{~0ull, 0, 0};
                                    std::sscanf(spec.c_str() + pos, "%llu,%llu,%llu", (unsigned long long*)&f[0],
                                                (unsigned long long*)&f[1], (unsigned long long*)&f[2]);
                                    v.push_back(f);
                                    const size_t semi = spec.find(';', pos);
                                    pos = semi == std::string::npos ? spec.size() : semi + 1;
                                }
                            }
                            return v;
                        }();
                        for (const auto& field : fields) {
                            if (field[2] == 0 || buffer_index != field[0]) {
                                continue;
                            }
                            const u64 recs = std::min<u64>(size / (field[2] * 4), 1u << 20);
                            u64 mn = ~0ull, mxv = 0, sum = 0, over = 0;
                            std::unordered_set<u32> distinct;
                            for (u64 i = 0; i < recs; ++i) {
                                const u32 v = w[i * field[2] + field[1]];
                                mn = std::min<u64>(mn, v);
                                mxv = std::max<u64>(mxv, v);
                                sum += v;
                                over += v > 64;
                                if (distinct.size() < 4096) {
                                    distinct.insert(v);
                                }
                            }
                            std::fprintf(stderr, "[PROBE] field b%llu[%llu] over %llu records: min %llu max %llu mean %.1f distinct %zu over64 %llu\n",
                                         (unsigned long long)field[0], (unsigned long long)field[1],
                                         (unsigned long long)recs, (unsigned long long)mn,
                                         (unsigned long long)mxv, recs ? double(sum) / recs : 0.0,
                                         distinct.size(), (unsigned long long)over);
                        }
                        static const auto dumps = [] {
                            std::vector<std::array<u64, 3>> v;
                            if (const char* e = std::getenv("BB_PROBE_DUMP")) {
                                std::string spec{e};
                                size_t pos = 0;
                                while (pos < spec.size()) {
                                    std::array<u64, 3> d{~0ull, 0, 0};
                                    std::sscanf(spec.c_str() + pos, "%llu,%llu,%llu", (unsigned long long*)&d[0],
                                                (unsigned long long*)&d[1], (unsigned long long*)&d[2]);
                                    v.push_back(d);
                                    const size_t semi = spec.find(';', pos);
                                    pos = semi == std::string::npos ? spec.size() : semi + 1;
                                }
                            }
                            return v;
                        }();
                        static const u64 watch_index = [] {
                            const char* e = std::getenv("BB_PROBE_WATCH");
                            return e ? std::strtoull(e, nullptr, 10) : ~0ull;
                        }();
                        if (buffer_index == watch_index &&
                            VideoCore::g_watch_lo.load(std::memory_order_relaxed) != vsharp.base_address) {
                            VideoCore::g_watch_lo.store(vsharp.base_address, std::memory_order_relaxed);
                            VideoCore::g_watch_hi.store(vsharp.base_address + size, std::memory_order_relaxed);
                            std::fprintf(stderr, "[WATCH] armed b%u %#llx+%llu\n", buffer_index,
                                         (unsigned long long)vsharp.base_address, (unsigned long long)size);
                        }
                        for (const auto& dump : dumps) {
                            if (dump[2] == 0 || buffer_index != dump[0]) {
                                continue;
                            }
                            // Two dumps per distinct buffer address (the table moves between the
                            // menus and the match).
                            static std::map<std::pair<u64, u64>, int> seen;
                            if (seen[{dump[0], vsharp.base_address}]++ < 2) {
                                const u64 recs = std::min<u64>(dump[1], size / (dump[2] * 4));
                                for (u64 r = 0; r < recs; ++r) {
                                    std::string line = fmt::format("[PROBE] b{} @{:#x} rec {}:", dump[0], (unsigned long long)vsharp.base_address, r);
                                    for (u64 k = 0; k < std::min<u64>(dump[2], 64); ++k) {
                                        line += fmt::format(" {:08x}", w[r * dump[2] + k]);
                                    }
                                    std::fprintf(stderr, "%s\n", line.c_str());
                                }
                            }
                        }
                    }
                }
                element = {.buffer = buffer->Ref(offset_aligned), .size = size + adjust};
                bound_buffers.emplace_back(buffer, offset, size, desc.is_written,
                                           stage.sw_stage == Shader::SwStage::Fragment ||
                                                   stage.sw_stage == Shader::SwStage::Compute
                                               ? Metal::ReadStage::Late
                                               : Metal::ReadStage::Vertex,
                                           draw_encoders);
                if (desc.is_written) {
                    // Raw storage-buffer writes can also make an aliased cached image stale.
                    texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size,
                                                          stage.pgm_hash);
                }
            }
        }
        ++binding.buffer;
    }
}

Rasterizer::ImageDescCacheEntry& Rasterizer::CachedImageDescEntry(const AmdGpu::Image& sharp,
                                                                   const Shader::ImageResource& res,
                                                                   u64 hash) {
    std::array<u64, 4> key;
    std::memcpy(key.data(), &sharp, sizeof(key));
    const u32 flags = u32(res.is_written) | u32(res.is_depth) << 1 | u32(res.is_array) << 2;
    // Two-way set associative: a frame binds a few thousand distinct T#s.
    auto* set = &image_desc_cache[(hash % (image_desc_cache.size() / 2)) * 2];
    const auto matches = [&](const ImageDescCacheEntry& e) {
        return e.flags == flags && e.sharp == key;
    };
    const bool disabled = BbToggle::Disabled(BbToggle::ImageDescCache);
    ImageDescCacheEntry* slot = nullptr;
    if (!disabled && matches(set[0])) {
        slot = &set[0];
    } else if (!disabled && matches(set[1])) {
        slot = &set[1];
    } else {
        // Replace the least recently used way that no binding of this call points at.
        slot = set[0].last_use <= set[1].last_use ? &set[0] : &set[1];
        if (slot->pinned == bind_epoch) {
            slot = slot == &set[0] ? &set[1] : &set[0];
        }
        if (slot->pinned == bind_epoch) {
            slot = &image_desc_overflow.emplace_back();
        }
    }
    slot->last_use = ++desc_use_counter;
    auto& entry = *slot;
    if (entry.flags != flags || entry.sharp != key ||
        BbToggle::Disabled(BbToggle::ImageDescCache)) {
        entry.desc = VideoCore::TextureCache::ImageDesc{sharp, res};
        entry.sharp = key;
        entry.flags = flags;
        entry.found_generation = ~0ULL;
        entry.view_memo = {};
    }
    return entry;
}

void Rasterizer::BindTextures(const Shader::Info& stage, const PreparedStage* prepared,
                              Shader::Backend::Bindings& binding, bool on_helper) {
    const u32 first_image_idx = image_infos.size();
    TextureSet* set_slot = nullptr;
    u32 depth_remap_mask = 0; // fifa-hybrid: images flagged gl_depth (PushData::depth_remap)
    if (!on_helper && BindTexturesFromSet(stage, prepared, set_slot)) {
        // One element per image (no mip arrays in a memoized set).
        for (u32 i = 0; i < stage.images.size(); ++i) {
            BoundAt(binding.unified++) = image_infos[first_image_idx + i];
        }
        if (stage.sw_stage == Shader::SwStage::Fragment || stage.sw_stage == Shader::SwStage::Compute) {
            push_data.depth_remap = set_depth_remap;
        }
        BindSamplers(stage, prepared, binding);
        return;
    }
    image_bindings.clear();
    image_binding_entries.clear();
    image_desc_overflow.clear();
    image_desc_storage.clear();
    ++bind_epoch;
    // To emulate storing to explicit mip levels, bind an array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;
    // TextureSetMemo: what this call resolves, to remember when the set qualifies.
    std::array<TextureSetEntry, TextureSet::MaxImages> resolved{};
    bool set_ok = set_slot != nullptr;
    // SampleSceneProxies: bindings that may read a reduced scene proxy instead of the native
    // image (normalized sampling only; see Shader::ImageResource::needs_native).
    boost::container::small_vector<bool, 16> binding_proxy_ok;
    const bool sample_proxies = !BbToggle::Disabled(BbToggle::SampleSceneProxies) &&
                                !on_helper && scene_targets->Reduced();

    for (u32 image_index = 0; image_index < stage.images.size(); ++image_index) {
        const auto& image_desc = stage.images[image_index];
        const auto tsharp =
            prepared ? prepared->image_sharps[image_index] : image_desc.GetSharp(stage);
        // bbport: a hash lookup per texture per draw for a diagnostic only.
        static const bool warn_meta = std::getenv("BB_WARN_META_TEXTURE") != nullptr;
        if (warn_meta && texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings.emplace_back(VideoCore::ImageId{}, &image_desc_storage.emplace_back());
            image_binding_entries.push_back(nullptr);
            binding_proxy_ok.push_back(false);
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        if (!memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !IsKnownFormat(data_fmt, num_fmt)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt));
            image_bindings.emplace_back(VideoCore::ImageId{}, &image_desc_storage.emplace_back());
            image_binding_entries.push_back(nullptr);
            binding_proxy_ok.push_back(false);
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        // Same as image_desc.NumBindings(stage), without fetching the T# again.
        const u32 num_bindings =
            mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex
                ? static_cast<u32>(tsharp.last_level - tsharp.base_level + 1)
                : 1u;
        const bool proxy_candidate = sample_proxies && num_bindings == 1 &&
                                     !image_desc.needs_native && !image_desc.is_written &&
                                     mip_fallback_mode == Shader::MipStorageFallbackMode::None;

        auto& desc_entry =
            prepared ? CachedImageDescEntry(tsharp, image_desc, prepared->image_hashes[image_index])
                     : CachedImageDescEntry(tsharp, image_desc);
        // BB_SCENE_DEBUG: proxied scene targets this binding reads at the native size, and why.
        const auto debug_native = [&](const VideoCore::Image& image) {
            if (!scene_debug_frame || proxy_candidate || !image.scene_proxy) return;
            std::printf("Scene native read: %s %016llx %s %ux%u needs_native %d written %d "
                        "bindings %u mip_fallback %d\n",
                        stage.sw_stage == Shader::SwStage::Compute ? "cs" : "gfx",
                        (unsigned long long)stage.pgm_hash,
                        vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
                        image.info.size.height, int(image_desc.needs_native),
                        int(image_desc.is_written), num_bindings, int(mip_fallback_mode));
        };
        // fifa-hybrid debug: BB_IMG_BINDS=1 logs every (shader, slot, image) binding with the view
        // range; BB_DUMP_BIND=<hash8>:<slot>[,...] writes level 0 of every layer of the image bound
        // there to <BB_DUMP_DIR>/bind_*.raw once BB_DUMP_BIND_TRIGGER (a file) exists.
        const auto debug_bind = [&](VideoCore::ImageId id, const VideoCore::Image& image,
                                    const auto& d) {
            static const bool binds = std::getenv("BB_IMG_BINDS") != nullptr;
            static const char* dump = std::getenv("BB_DUMP_BIND");
            // BB_FLAG_PROBE=<hash8>:<slot>:<dword>[,...]: for binds of that shader's image slot, counts the
            // combinations of (bound image, that dword of every constant buffer big enough to hold it).
            static const auto flag_probes = [] {
                std::vector<std::array<u32, 3>> v;
                if (const char* e = std::getenv("BB_FLAG_PROBE")) {
                    std::string spec{e};
                    size_t pos = 0;
                    while (pos < spec.size()) {
                        std::array<u32, 3> f{};
                        if (std::sscanf(spec.c_str() + pos, "%x:%u:%u", &f[0], &f[1], &f[2]) == 3) v.push_back(f);
                        const size_t c = spec.find(',', pos);
                        pos = c == std::string::npos ? spec.size() : c + 1;
                    }
                }
                return v;
            }();
            if (g_drawlog) {
                std::fprintf(g_drawlog, "I %08x img%u %#llx %ux%u %s mips %u layers %u flags %#x serial %llu uid %llu view %s L%u+%u\n",
                             u32(stage.pgm_hash), image_index, (unsigned long long)image.info.guest_address,
                             image.info.size.width, image.info.size.height,
                             vk::to_string(image.info.pixel_format).c_str(), image.info.resources.levels,
                             image.info.resources.layers, u32(image.flags), (unsigned long long)image.write_serial,
                             (unsigned long long)image.image_uid, vk::to_string(d.view_info.format).c_str(),
                             d.view_info.range.base.level, d.view_info.range.extent.levels);
            }
            for (const auto& fp : flag_probes) {
                if (u32(stage.pgm_hash) != fp[0] || image_index != fp[1]) continue;
                std::string key = fmt::format("{:08x} img{} {:#x} {}x{} {}", fp[0], fp[1], image.info.guest_address,
                                              image.info.size.width, image.info.size.height,
                                              vk::to_string(image.info.pixel_format));
                bool flagged = false;
                for (u32 b = 0; b < stage.buffers.size(); ++b) {
                    const auto sharp = stage.buffers[b].GetSharp(stage);
                    if (sharp.base_address && sharp.GetSize() >= (fp[2] + 1) * 4ull) {
                        const u32* dw = reinterpret_cast<const u32*>(sharp.base_address);
                        key += fmt::format(" b{}[{}]={:#x}", b, fp[2], dw[fp[2]]);
                        if (dw[fp[2]] == 1) {
                            flagged = true;
                            key += fmt::format(" prev {:#x} {:#x}", dw[fp[2] - 2], dw[fp[2] - 1]);
                        } else if (dw[fp[2]] == 0 && sharp.GetSize() == 320) {
                            key += fmt::format(" id {:#x}", dw[fp[2] - 1]);
                        }
                    }
                }
                {
                    const auto& r = liverpool->regs;
                    key += fmt::format(" cb0 {:#x} {}x{}", u64(r.color_buffers[0].Address()),
                                       r.color_buffers[0].Pitch(), r.color_buffers[0].Height());
                }
                static const auto t0 = std::chrono::steady_clock::now();
                const double now_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                // BB_FLAG_SNAP=<dir> (with BB_SNAPSHOT_TRIGGER): a snapshot when flagged draws reach a 1080p target,
                // at most one every 12 s and 8 in all.
                if (flagged && liverpool->regs.color_buffers[0].Height() >= 1080) {
                    static const char* snap_dir = std::getenv("BB_FLAG_SNAP");
                    static const char* trig = std::getenv("BB_SNAPSHOT_TRIGGER");
                    static double last = -100.0;
                    static u32 taken = 0;
                    static u64 seen = 0;
                    if (snap_dir && trig && ++seen > 200 && now_s - last > 9.0 && taken < 14) {
                        last = now_s;
                        if (FILE* f = std::fopen(trig, "w")) {
                            std::fprintf(f, "%s/flagsnap_%u.png\n", snap_dir, taken);
                            std::fclose(f);
                        }
                        std::fprintf(stderr, "[FLAGSNAP] %u at %.1f s: %s\n", taken, now_s, key.c_str());
                        ++taken;
                    }
                }
                static std::map<std::string, u64> counts;
                const u64 c = ++counts[key];
                if (c == 1 || c % 3000 == 0) {
                    key += fmt::format(" t {:.1f}", now_s);
                }
                if (counts.size() < 1500 && (c == 1 || c % 3000 == 0)) {
                    std::fprintf(stderr, "[FLAG] %s count %llu\n", key.c_str(), (unsigned long long)c);
                }
                // The first draws of each combination: every constant buffer in full (up to 160 dwords).
                if (c == 1 || c == 500) {
                    for (u32 b = 0; b < stage.buffers.size(); ++b) {
                        const auto sharp = stage.buffers[b].GetSharp(stage);
                        if (!sharp.base_address) continue;
                        const u32 nd = u32(std::min<u64>(sharp.GetSize() / 4, 160));
                        std::string line = fmt::format("[FLAGBUF] {:08x} c{} b{} @{:#x} size {}:", fp[0], c, b,
                                                       u64(sharp.base_address), sharp.GetSize());
                        for (u32 i = 0; i < nd; ++i) {
                            line += fmt::format(" {:08x}", reinterpret_cast<const u32*>(sharp.base_address)[i]);
                        }
                        std::fprintf(stderr, "%s\n", line.c_str());
                    }
                }
            }
            // BB_MIPTRACE=1: one line per distinct way a small GPU-written image (or any image of a
            // dispatch) is bound: view format, level range, swizzle, and the colour target of the draw.
            if (static const bool mt = std::getenv("BB_MIPTRACE") != nullptr; mt &&
                image.info.size.width <= 1024 &&
                (stage.sw_stage == Shader::SwStage::Compute || image_desc.is_written ||
                 True(image.flags & VideoCore::ImageFlagBits::GpuModified))) {
                const bool cs = stage.sw_stage == Shader::SwStage::Compute;
                const auto& cb = liverpool->regs.color_buffers[0];
                const auto& m = d.view_info.mapping;
                static std::set<std::array<u64, 8>> seen;
                const std::array<u64, 8> key{stage.pgm_hash, image_index,
                    u64(image.info.size.width) << 32 | u64(image.info.pixel_format),
                    u64(d.view_info.format) << 32 | d.view_info.range.base.level,
                    u64(m.r) << 24 | u64(m.g) << 16 | u64(m.b) << 8 | u64(m.a),
                    cs ? 0 : u64(cb.info.raw & 0x1fff), cs ? 0 : u64(cb.Pitch()),
                    u64(d.view_info.min_lod) << 16 | u64(image.info.resources.levels) << 8 |
                        u64(image_desc.is_written)};
                if (seen.size() < 1500 && seen.insert(key).second) {
                    std::fprintf(stderr,
                                 "[MIPT] %s %08x img%u %#llx %s %ux%u lv%u fl %#x w%d view %s L%u+%u minlod %u map %u%u%u%u | cb0 %#llx %ux%u fmt %u nt %u swap %u\n",
                                 cs ? "cs" : "gfx", u32(stage.pgm_hash), image_index,
                                 (unsigned long long)image.info.guest_address,
                                 vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
                                 image.info.size.height, image.info.resources.levels, u32(image.flags),
                                 int(image_desc.is_written), vk::to_string(d.view_info.format).c_str(),
                                 d.view_info.range.base.level, d.view_info.range.extent.levels,
                                 d.view_info.min_lod, u32(m.r), u32(m.g), u32(m.b), u32(m.a),
                                 cs ? 0ull : (unsigned long long)cb.Address(), cs ? 0u : cb.Pitch(),
                                 cs ? 0u : cb.Height(), cs ? 0u : u32(cb.info.format),
                                 cs ? 0u : u32(cb.info.number_type), cs ? 0u : u32(cb.info.comp_swap));
                }
            }
            // BB_BCSEQ=1: every image binding of the face texture compute shaders, in order.
            if (static const bool seq = std::getenv("BB_BCSEQ") != nullptr;
                seq && stage.sw_stage == Shader::SwStage::Compute &&
                (u32(stage.pgm_hash) == 0x510f1e38u || u32(stage.pgm_hash) == 0xa7c5bd5fu ||
                 u32(stage.pgm_hash) == 0x52abaae0u)) {
                std::fprintf(stderr,
                             "[SEQ]  bind %08x img%u id %u %#llx %s %ux%u lv%u fl %#x w%d view %s L%u+%u shadow_of %d\n",
                             u32(stage.pgm_hash), image_index, id.index,
                             (unsigned long long)image.info.guest_address,
                             vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
                             image.info.size.height, image.info.resources.levels, u32(image.flags),
                             int(image_desc.is_written), vk::to_string(d.view_info.format).c_str(),
                             d.view_info.range.base.level, d.view_info.range.extent.levels,
                             int(bool(image.shadow_of)));
            }
            if (!binds && !dump) return;
            if (binds) {
                static std::set<std::tuple<u64, u32, VAddr>> seen;
                if (seen.size() < 4000 &&
                    seen.emplace(stage.pgm_hash, image_index, image.info.guest_address).second) {
                    std::fprintf(stderr,
                                 "[BIND2] %s %08llx img%u addr %#llx %s %ux%u layers %u mips %u sharp_type %u "
                                 "view type %u fmt %s base layer %u layers %u base level %u levels %u\n",
                                 stage.sw_stage == Shader::SwStage::Compute ? "cs" : "gfx",
                                 (unsigned long long)(stage.pgm_hash & 0xffffffffull), image_index,
                                 (unsigned long long)image.info.guest_address,
                                 vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
                                 image.info.size.height, image.info.resources.layers,
                                 image.info.resources.levels, u32(tsharp.GetType()),
                                 u32(d.view_info.type), vk::to_string(d.view_info.format).c_str(),
                                 d.view_info.range.base.layer, d.view_info.range.extent.layers,
                                 d.view_info.range.base.level, d.view_info.range.extent.levels);
                }
            }
            if (dump) {
                static const std::set<std::pair<u32, u32>> want = [] {
                    std::set<std::pair<u32, u32>> w;
                    const char* e = std::getenv("BB_DUMP_BIND");
                    while (e && *e) {
                        char* end = nullptr;
                        const u32 h = u32(std::strtoull(e, &end, 16));
                        u32 slot = 0;
                        if (end && *end == ':') slot = u32(std::strtoul(end + 1, &end, 10));
                        w.emplace(h, slot);
                        e = (end && *end == ',') ? end + 1 : nullptr;
                    }
                    return w;
                }();
                // BB_DUMP_ALL_TRIGGER=<file>: once the file exists, every GPU-written block image.
                if (static const char* all = std::getenv("BB_DUMP_ALL_TRIGGER"); all) {
                    static bool all_done = false;
                    static u32 all_polls = 0;
                    if (!all_done && (++all_polls % 64) == 0) {
                        if (FILE* f = std::fopen(all, "rb")) {
                            std::fclose(f);
                            all_done = true;
                            const char* env = std::getenv("BB_DUMP_DIR");
                            texture_cache.DebugDumpGpuCompressed(env && env[0] ? env : "out/dump");
                        }
                    }
                }
                if (!want.contains({u32(stage.pgm_hash), image_index})) return;
                static const char* trigger = std::getenv("BB_DUMP_BIND_TRIGGER");
                static bool triggered = trigger == nullptr;
                static u32 polls = 0;
                if (!triggered && (++polls % 16) == 0) {
                    if (FILE* f = std::fopen(trigger, "rb")) {
                        std::fclose(f);
                        triggered = true;
                    }
                }
                static std::set<std::tuple<u32, u32, VAddr>> done;
                static const size_t dump_max = [] {
                    const char* e = std::getenv("BB_DUMP_BIND_MAX");
                    return e ? size_t(std::strtoul(e, nullptr, 10)) : size_t(64);
                }();
                if (triggered && done.size() < dump_max &&
                    done.emplace(u32(stage.pgm_hash), image_index, image.info.guest_address).second) {
                    const char* env = std::getenv("BB_DUMP_DIR");
                    const std::string path = fmt::format(
                        "{}/bind_{:08x}_img{}_{:x}_{}x{}x{}_{}_f{:x}.raw", env && env[0] ? env : "out/dump",
                        u32(stage.pgm_hash), image_index, image.info.guest_address, image.info.size.width,
                        image.info.size.height, image.info.resources.layers,
                        vk::to_string(image.info.pixel_format), u32(image.flags));
                    texture_cache.DebugDumpImage(id, path.c_str());
                }
            }
        };
        for (auto i = 0; i < num_bindings; i++) {
            // bbport: a plain binding (no mip override) of the same T# resolves to the same image
            // while no image was registered or unregistered.
            if (mip_fallback_mode == Shader::MipStorageFallbackMode::None &&
                desc_entry.found_generation == texture_cache.RegistryGeneration() &&
                !BbToggle::Disabled(BbToggle::TextureBindingMemo)) {
                desc_entry.pinned = bind_epoch;
                auto& [image_id, _] =
                    image_bindings.emplace_back(desc_entry.found_id, &desc_entry.found_desc);
                image_binding_entries.push_back(&desc_entry);
                binding_proxy_ok.push_back(proxy_candidate);
                texture_cache.MarkFound(image_id);
                auto* image = &texture_cache.GetImage(image_id);
                if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                    image_id = depth_image_id;
                    image = &texture_cache.GetImage(image_id);
                }
                if (image->gl_depth && image_index < 32) {
                    depth_remap_mask |= 1u << image_index; // fifa-hybrid: PushData::depth_remap
                }
                if (static const bool depth_log = std::getenv("BB_DEPTHBIND_LOG") != nullptr; depth_log) {
                    static std::set<std::tuple<u64, u32, VAddr>> seen;
                    if (seen.emplace(stage.pgm_hash, image_index, image->info.guest_address).second &&
                        seen.size() < 400 && image->info.size.width >= 256) {
                        std::fprintf(stderr, "[DEPTHBIND] ps %08llx img%u addr %#llx %ux%u %s gl_depth=%d\n",
                                     (unsigned long long)stage.pgm_hash, image_index,
                                     (unsigned long long)image->info.guest_address, image->info.size.width,
                                     image->info.size.height, vk::to_string(image->info.pixel_format).c_str(),
                                     int(image->gl_depth));
                std::fprintf(stderr, "[DEPTHBIND]   samples=%u sharp_type=%u\n", image->info.num_samples, u32(tsharp.GetType()));
                    }
                }
                debug_native(*image);
                debug_bind(image_id, *image, desc_entry.found_desc);
                if (image->binding.is_bound) {
                    image->binding.force_general |= image_desc.is_written;
                }
                image->binding.is_bound = 1u;
                continue;
            }
            auto& desc = image_desc_storage.emplace_back(desc_entry.desc);
            auto& [image_id, _] = image_bindings.emplace_back(VideoCore::ImageId{}, &desc);
            image_binding_entries.push_back(nullptr);
            binding_proxy_ok.push_back(proxy_candidate);

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            const u64 generation = texture_cache.RegistryGeneration();
            image_id = texture_cache.FindImage(desc);
            if (mip_fallback_mode == Shader::MipStorageFallbackMode::None &&
                generation == texture_cache.RegistryGeneration() &&
                desc_entry.pinned != bind_epoch) {
                desc_entry.found_generation = generation;
                desc_entry.found_id = image_id;
                desc_entry.found_desc = desc;
                desc_entry.view_memo = {};
            }
            auto* image = &texture_cache.GetImage(image_id);
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            if (image->gl_depth && image_index < 32) {
                depth_remap_mask |= 1u << image_index; // fifa-hybrid: PushData::depth_remap
            }
            if (static const bool depth_log = std::getenv("BB_DEPTHBIND_LOG") != nullptr; depth_log) {
                static std::set<std::tuple<u64, u32, VAddr>> seen;
                if (seen.emplace(stage.pgm_hash, image_index, image->info.guest_address).second &&
                    seen.size() < 400 && image->info.size.width >= 256) {
                    std::fprintf(stderr, "[DEPTHBIND] ps %08llx img%u addr %#llx %ux%u %s gl_depth=%d\n",
                                 (unsigned long long)stage.pgm_hash, image_index,
                                 (unsigned long long)image->info.guest_address, image->info.size.width,
                                 image->info.size.height, vk::to_string(image->info.pixel_format).c_str(),
                                 int(image->gl_depth));
                std::fprintf(stderr, "[DEPTHBIND]   samples=%u sharp_type=%u\n", image->info.num_samples, u32(tsharp.GetType()));
                }
            }
            debug_native(*image);
            debug_bind(image_id, *image, desc);
            if (static const bool pl = std::getenv("BB_PROBE_LUM") != nullptr;
                pl && image->info.pixel_format == vk::Format::eB10G11R11UfloatPack32 &&
                image->info.resources.levels == 5 && image->info.size.width == 256) {
                static std::map<u32, u32> binds;
                const u32 nb = ++binds[image_id.index];
                if (nb == 1 || nb == 600 || nb == 6000) {
                    texture_cache.DebugProbeImage(image_id, nb == 1 ? "first" : "later");
                }
            }
            if (static const bool seqlog = std::getenv("BB_TWIN_WATCH") != nullptr;
                seqlog && image->info.size.width == 2048 && image->info.pixel_format == vk::Format::eR32Sfloat &&
                BbSeqWindow()) {
                std::fprintf(stderr, "[SEQ] %llu %s-%s %#llx shader %08llx queue %s\n", (unsigned long long)g_bb_draw_seq.load(),
                             stage.sw_stage == Shader::SwStage::Compute ? "cs" : "gfx",
                             image_desc.is_written ? "write" : "read", (unsigned long long)image->info.guest_address,
                             (unsigned long long)stage.pgm_hash,
                             stage.sw_stage != Shader::SwStage::Compute ? "-" : (stage_cs == &liverpool->GetCsRegs() ? "gfx" : "asc"));
            }
            if (static const bool binds = std::getenv("BB_IMG_BINDS") != nullptr; binds) {
                // debug: one line per (shader, image, written) binding
                static std::set<std::tuple<u64, VAddr, bool>> ib;
                if (ib.emplace(stage.pgm_hash, image->info.guest_address, image_desc.is_written)
                        .second) {
                    std::fprintf(stderr, "IMGBIND %#llx %s %ux%u mips=%u %s %#llx written=%d\n",
                                 (unsigned long long)image->info.guest_address,
                                 vk::to_string(image->info.pixel_format).c_str(),
                                 image->info.size.width, image->info.size.height,
                                 image->info.resources.levels,
                                 stage.sw_stage == Shader::SwStage::Compute ? "cs" : "gfx",
                                 (unsigned long long)stage.pgm_hash, int(image_desc.is_written));
                }
            }
            if (image->info.props.is_volume) {
                texture_cache.DebugDumpVolume(image_id); // fifa-hybrid: self-limited (BB_DUMP_VOLUMES)
            }
            if (image->info.props.is_volume && ([] { static const bool v = std::getenv("BB_IMG_FORMATS") != nullptr; return v; }())) {
                static std::set<std::tuple<u64, VAddr, bool>> vb;
                if (vb.emplace(stage.pgm_hash, image->info.guest_address, image_desc.is_written)
                        .second) {
                    std::fprintf(stderr, "VOLBIND %#llx %s %#llx written=%d\n",
                                 (unsigned long long)image->info.guest_address,
                                 stage.sw_stage == Shader::SwStage::Compute ? "cs" : "gfx",
                                 (unsigned long long)stage.pgm_hash, int(image_desc.is_written));
                    if (!image_desc.is_written) {
                        texture_cache.DebugDumpVolume(image_id);
                    }
                }
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
        if (num_bindings != 1) {
            bound_images_simple = false; // 0.5.1-c
        }
    }

    // fifa-hybrid: one-texel-per-block views of block-compressed images bind a shadow image.
    for (u32 binding_index = 0; binding_index < image_bindings.size(); ++binding_index) {
        auto& [image_id, desc_ptr] = image_bindings[binding_index];
        if (!image_id || !texture_cache.GetImage(image_id).info.props.is_block ||
            desc_ptr->info.props.is_block) {
            continue;
        }
        const auto shadow_id = texture_cache.ResolveBlockShadow(image_id, *desc_ptr);
        if (shadow_id != image_id) {
            image_id = shadow_id;
            image_binding_entries[binding_index] = nullptr;
            set_ok = false;
        }
    }

    // bbport: a lazy usage upgrade replaces the image's texture. Every binding of this stage gets
    // its upgrade before any binding takes a texture: a dispatch that writes one mip and reads
    // another in a second pixel format must not write the replaced texture.
    if (!on_helper) {
        for (const auto& [image_id, desc_ptr] : image_bindings) {
            if (!image_id) {
                continue;
            }
            auto& image = texture_cache.GetImage(image_id);
            if (const u32 needed = image.NeededUsage(desc_ptr->view_info)) {
                runtime.UpgradeUsage(&image, needed);
            }
        }
    }

    // Second pass to re-bind images that were updated after binding
    for (u32 binding_index = 0; binding_index < image_bindings.size(); ++binding_index) {
        auto& [image_id, desc_ptr] = image_bindings[binding_index];
        auto* memo_entry = image_binding_entries[binding_index];
        const auto& desc = *desc_ptr;
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(); // the null stand-in
            bound_image_ids.push_back({}); // 0.5.1-c: PlanCopyDispatchPass
            bound_image_views.push_back(nullptr);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                auto& rebind_desc = image_desc_storage.emplace_back(desc);
                desc_ptr = &rebind_desc;
                image_id = texture_cache.FindImage(rebind_desc);
                if (texture_cache.GetImage(image_id).info.props.is_block &&
                    !rebind_desc.info.props.is_block) {
                    image_id = texture_cache.ResolveBlockShadow(image_id, rebind_desc);
                }
                memo_entry = nullptr;
                set_ok = false;
            }

            bound_images.emplace_back(image_id);
            bound_images_written.push_back(is_storage ? 1 : 0);
            bound_images_prev_write.push_back(texture_cache.GetImage(image_id).last_write_tick);

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(
                image_id, desc, memo_entry ? &memo_entry->view_memo : nullptr, !on_helper);
            bound_image_ids.push_back(image_id); // 0.5.1-c: PlanCopyDispatchPass
            bound_image_views.push_back(image_view.Texture());
            const auto binding = image.binding;
            if (!is_storage || ([] { static const bool v = std::getenv("BB_DUMP_ADDRS") != nullptr; return v; }())) {
                texture_cache.DebugDumpTexture(image_id);
            }

            // bbport: a proxied scene image sampled with normalized coordinates reads the proxy:
            // no resample to the native size (the native image is not touched).
            if (binding_proxy_ok[binding_index] && !is_storage && !binding.force_general &&
                !binding.is_target &&
                !(upscaler->Enabled() && upscaler->RedirectsSampled(image_id))) {
                if (const auto proxy = scene_targets->SampleProxy(image, desc.view_info)) {
                    image.usage.texture = 1u;
                    image_infos.push_back({.texture = proxy->texture});
                    if (set_ok && binding_index < resolved.size()) {
                        resolved[binding_index] = {image_id, proxy->texture, image.backing,
                                                   desc.view_info.range, true};
                    }
                    ++proxy_samples;
                    continue;
                }
            }
            // The image is either bound as storage in a separate binding or bound as render
            // target in a feedback loop: read and written. Depth images are excluded because they
            // can't be bound as storage and a feedback loop doesn't make sense for them.
            const bool written =
                is_storage ||
                ((binding.force_general || binding.is_target) && !image.info.props.is_depth);
            AccessBoundImage(image, written, on_helper);
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;
            if (FrameCapture::Active()) {
                FrameCapture::Sampled(image.info, is_storage);
            }

            void* texture = image_view.Texture();
            if (upscaler->Enabled()) {
                // bbport: the display pass reads the upscaled frame (scaled presets).
                upscaler->RedirectSampled(image_id, image_view.info, texture);
            }
            image_infos.push_back({.texture = texture});
            if (set_ok && binding_index < resolved.size()) {
                resolved[binding_index] = {image_id, image_view.Texture(), image.backing,
                                           desc.view_info.range};
                set_ok = !is_storage && !binding.force_general && !binding.is_target;
            }
        }
    }
    if (set_ok && image_bindings.size() == stage.images.size()) {
        // Remember the set: the same T#s resolve the same way while no image is (un)registered.
        set_slot->generation = texture_cache.RegistryGeneration();
        set_slot->scene_generation = scene_targets->Generation();
        set_slot->count = static_cast<u32>(stage.images.size());
        set_slot->depth_remap = depth_remap_mask;
        std::copy_n(resolved.begin(), set_slot->count, set_slot->entries.begin());
    } else if (set_slot) {
        set_slot->key = 0;
    }

    // Each image is one binding; a mip array takes `array_size` consecutive elements.
    u32 image_info_idx = first_image_idx;
    for (u32 array_size : image_descriptor_array_sizes) {
        for (u32 element = 0; element < array_size; ++element) {
            BoundAt(binding.unified + element) = image_infos[image_info_idx + element];
        }
        image_info_idx += array_size;
        binding.unified += array_size;
    }

    if (stage.sw_stage == Shader::SwStage::Fragment || stage.sw_stage == Shader::SwStage::Compute) {
        push_data.depth_remap = depth_remap_mask;
        // fifa-hybrid plumbing test: BB_DEPTH_REMAP_ALL=1 remaps every sampled image (the picture
        // must break visibly; if it does not, the push constant does not reach the shaders).
        static const bool remap_all = std::getenv("BB_DEPTH_REMAP_ALL") != nullptr;
        if (remap_all) {
            push_data.depth_remap = 0xFFFFFFFFu;
        }
    }
    // fifa-hybrid diagnostics: BB_REMAP_LOG=1 prints each (shader, mask) once.
    if (static const bool remap_log = std::getenv("BB_REMAP_LOG") != nullptr; remap_log) {
        static std::set<std::pair<u64, u32>> seen;
        if (seen.emplace(stage.pgm_hash, depth_remap_mask).second && seen.size() < 300) {
            std::fprintf(stderr, "[REMAP] %s %08llx mask=%#x images=%zu set=%d\n",
                         stage.sw_stage == Shader::SwStage::Fragment ? "ps" : "stage",
                         (unsigned long long)stage.pgm_hash, depth_remap_mask, stage.images.size(),
                         int(set_slot != nullptr));
        }
    }
    BindSamplers(stage, prepared, binding);
}

// bbport: the PS4 build samples most scene textures with little or no anisotropic filtering.
// Behind the temporal upscaler the output shows the detail of the output's mip level, and
// surfaces seen at a grazing angle (stairs, floors, walls along the view) stay blurred.
// Mipmapped linear samplers of graphics draws get BB_ANISO (16; 0 = the game's) times.
void Rasterizer::ForceAnisotropy(AmdGpu::Sampler& sharp, bool is_depth) const {
    static const int forced = [this] {
        const char* value = std::getenv("BB_ANISO");
        int target = value ? std::atoi(value) : 16;
        if (target > 0) {
            const float max_aniso = device.GetLimits().max_anisotropy;
            if (max_aniso > 0.0f && target > static_cast<int>(max_aniso)) {
                target = static_cast<int>(max_aniso);
            }
        }
        return target;
    }();
    if (!is_depth && !pipeline_is_compute) {
        // The game's filters, once per kind: min filter, mip filter, ratio.
        static std::atomic<u64> seen{0};
        const u32 kind = u32(sharp.xy_min_filter.Value()) * 3 * 5 + u32(sharp.mip_filter.Value()) * 5 +
                         u32(sharp.max_aniso.Value());
        if (kind < 64 && !(seen.fetch_or(1ull << kind) & (1ull << kind))) {
            std::printf("Sampler: game uses min filter %u, mip filter %u, anisotropy %.0fx\n",
                        u32(sharp.xy_min_filter.Value()), u32(sharp.mip_filter.Value()),
                        sharp.MaxAniso());
        }
    }
    if (forced <= 1 || is_depth || pipeline_is_compute || BbToggle::Disabled(BbToggle::ForcedAniso) ||
        sharp.mip_filter == AmdGpu::MipFilter::None ||
        (sharp.xy_min_filter != AmdGpu::Filter::Bilinear &&
         sharp.xy_min_filter != AmdGpu::Filter::AnisoLinear)) {
        return;
    }
    // A sampler with a positive LOD bias is a deliberately blurred lookup. FIFA 16's pitch shader
    // (ps ed446cb3) reads its line texture through one (+2 mips) as a conservative "is a line
    // near this pixel" mask that guards its 3x3 supersampling of the line. Anisotropic filtering
    // takes the mip from the minor axis, so at the pitch's grazing angle the mask no longer
    // reached the pixels beside the line across the screen and those skipped the supersampling:
    // a stair-stepped far edge on every pitch line (2026-11-10). BB_ANISO_BIASED=1 restores it.
    static const bool keep_biased = [] {
        const char* env = std::getenv("BB_ANISO_BIASED");
        return !(env && std::atoi(env) != 0);
    }();
    if (keep_biased && sharp.LodBias() > 0.0f) {
        return;
    }
    const auto ratio = forced >= 16 ? AmdGpu::AnisoRatio::Sixteen
        : forced >= 8 ? AmdGpu::AnisoRatio::Eight
        : forced >= 4 ? AmdGpu::AnisoRatio::Four : AmdGpu::AnisoRatio::Two;
    if (AmdGpu::IsAnisoFilter(sharp.xy_min_filter) && sharp.MaxAniso() >= forced) {
        return;
    }
    sharp.xy_min_filter.Assign(AmdGpu::Filter::AnisoLinear);
    if (sharp.xy_mag_filter == AmdGpu::Filter::Bilinear) {
        sharp.xy_mag_filter.Assign(AmdGpu::Filter::AnisoLinear);
    }
    sharp.max_aniso.Assign(ratio);
}

void Rasterizer::BindSamplers(const Shader::Info& stage, const PreparedStage* prepared,
                              Shader::Backend::Bindings& binding) {
    for (u32 sampler_index = 0; sampler_index < stage.samplers.size(); ++sampler_index) {
        const auto& sampler = stage.samplers[sampler_index];
        auto ssharp =
            prepared ? prepared->sampler_sharps[sampler_index] : sampler.GetSharp(stage);
        if (FrameCapture::Tracing()) {
            // bbport(fifa): the game's own sampler state, before the forced anisotropy.
            FrameCapture::Trace(fmt::format(
                "M {} {:08x} min {} mag {} mip {} aniso {} lod {:.2f}..{:.2f} bias {:.2f}",
                u32(stage.sw_stage), stage.pgm_hash, u32(ssharp.xy_min_filter.Value()),
                u32(ssharp.xy_mag_filter.Value()), u32(ssharp.mip_filter.Value()),
                ssharp.MaxAniso(), ssharp.MinLod(), ssharp.MaxLod(), ssharp.LodBias()));
        }
        ForceAnisotropy(ssharp, sampler.is_depth);
        void* const handle = texture_cache.GetSampler(
            ssharp, Regs().ta_bc_base, sampler.is_depth, sampler.is_depth ? 0.0f : sampler_lod_bias);
        bound_samplers.push_back(handle); // 0.5.1-c
        BoundAt(binding.unified++) = {.sampler = handle};
    }
}

Rasterizer::BeginSignature Rasterizer::MakeBeginSignature(const GraphicsPipeline* pipeline) const {
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = Regs();
    BeginSignature sig;
    const u32 num_color = std::bit_width(key.mrt_mask);
    for (u32 cb = 0; cb < num_color && cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        sig.ids[cb] = cb_descs[cb].first;
        if (cb_descs[cb].first) {
            sig.views[cb] = cb_descs[cb].second.view_info;
        }
        sig.color_samples[cb] = key.color_samples[cb];
    }
    sig.ids[AmdGpu::NUM_COLOR_BUFFERS] = db_desc.first;
    if (db_desc.first) {
        sig.views[AmdGpu::NUM_COLOR_BUFFERS] = db_desc.second.view_info;
    }
    sig.mrt_mask = key.mrt_mask;
    sig.num_samples = key.num_samples;
    sig.motion = key.motion_vectors;
    sig.scene_started = scene_started;
    sig.raster_scaling = upscaler->RasterScaling();
    sig.upscaler_state = upscaler->RedirectState();
    sig.generation = texture_cache.RegistryGeneration();
    std::memcpy(&sig.depth_control, &regs.depth_control, sizeof(u32));
    sig.depth_valid = regs.depth_buffer.DepthValid();
    sig.stencil_valid = regs.depth_buffer.StencilValid();
    return sig;
}

bool Rasterizer::BindTexturesFromSet(const Shader::Info& stage, const PreparedStage* prepared,
                                     TextureSet*& slot) {
    const u32 count = static_cast<u32>(stage.images.size());
    if (!prepared || count == 0 || count > TextureSet::MaxImages ||
        BbToggle::Disabled(BbToggle::TextureSetMemo) || FrameCapture::Active()) {
        return false;
    }
    for (const auto& image : stage.images) {
        if (image.is_written || image.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
            return false;
        }
    }
    u64 key = XXH3_64bits_withSeed(prepared->image_hashes, count * sizeof(u64),
                                   reinterpret_cast<u64>(&stage));
    key |= 1; // 0 marks an empty slot
    auto& set = texture_sets[key % texture_sets.size()];
    slot = &set;
    const u64 generation = texture_cache.RegistryGeneration();
    const bool match = set.key == key && set.stage == &stage && set.count == count &&
                       set.generation == generation &&
                       set.scene_generation == scene_targets->Generation() &&
                       std::equal(set.hashes.begin(), set.hashes.begin() + count,
                                  prepared->image_hashes);
    if (!match) {
        ++texture_set_why[set.key != key ? 0 : set.generation != generation ? 1 : 3];
        ++texture_set_misses;
        set.key = key;
        set.stage = &stage;
        set.count = 0;
        set.generation = ~0ull;
        std::copy_n(prepared->image_hashes, count, set.hashes.begin());
        return false;
    }
    // Every image still has the backing its view belongs to and needs no refresh.
    for (u32 i = 0; i < count; ++i) {
        const auto& entry = set.entries[i];
        if (!entry.id) {
            continue;
        }
        auto& image = texture_cache.GetImage(entry.id);
        if (image.info.size.width == 2048 && image.info.pixel_format == vk::Format::eR32Sfloat && BbSeqWindow()) {
            std::fprintf(stderr, "[SEQ] %llu memo-bind %#llx %s shader %08llx\n", (unsigned long long)g_bb_draw_seq.load(),
                         (unsigned long long)image.info.guest_address,
                         stage.sw_stage == Shader::SwStage::Compute ? "cs" : "gfx", (unsigned long long)stage.pgm_hash);
        }
        if (image.depth_twin) {
            runtime.SyncDepthTwin(&image); // fifa-hybrid: the twin follows the colour writes
        }
        // Proxy entries need a current proxy; native entries of a proxied image go through
        // BindTextures, which may sample the proxy instead.
        const bool proxies_on = !BbToggle::Disabled(BbToggle::SampleSceneProxies);
        if (image.backing != entry.backing || image.binding.needs_rebind ||
            image.binding.is_target || !texture_cache.IsUpToDate(entry.id) ||
            (entry.proxy ? !proxies_on ||
                               !scene_targets->ProxyCurrent(image, entry.range.base.level,
                                                            entry.range.extent.levels)
                         : image.scene_proxy && proxies_on) ||
            (upscaler->Enabled() && upscaler->RedirectsSampled(entry.id))) {
            ++texture_set_why[2];
            ++texture_set_misses;
            set.generation = ~0ull;
            return false;
        }
    }
    ++texture_set_hits;
    slot = nullptr;
    set_depth_remap = 0; // fifa-hybrid: recomputed per hit, the gl_depth flag changes after memoization
    for (u32 i = 0; i < count; ++i) {
        const auto& entry = set.entries[i];
        if (!entry.id) {
            image_infos.emplace_back(); // the null stand-in
            bound_image_ids.push_back({}); // 0.5.1-c
            bound_image_views.push_back(nullptr);
            continue;
        }
        texture_cache.MarkFound(entry.id);
        auto& image = texture_cache.GetImage(entry.id);
        if (image.gl_depth && i < 32) {
            set_depth_remap |= 1u << i;
        }
        image.binding.is_bound = 1u;
        bound_images.emplace_back(entry.id);
        bound_images_written.push_back(0);
        bound_images_prev_write.push_back(image.last_write_tick);
        bound_image_ids.push_back(entry.id); // 0.5.1-c
        bound_image_views.push_back(entry.texture);
        image.usage.texture = 1u;
        if (entry.proxy) {
            scene_targets->PrepareSample(image, entry.range.base.level);
            image_infos.push_back({.texture = entry.texture});
            ++proxy_samples;
            continue;
        }
        runtime.AccessImage(&image, false);
        image_infos.push_back({.texture = entry.texture});
    }
    return true;
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    // bbport: RenderStateMemo (see BeginMemo).
    const bool memo_on = !BbToggle::Disabled(BbToggle::RenderStateMemo) && !FrameCapture::Active();
    if (BbStats::enabled && ((begin_memo_hits + begin_memo_misses) & 0x3FFFF) == 0x3FFFF) {
        std::printf("Render state memo: %llu hits, %llu misses; texture sets: %llu hits, %llu "
                    "misses\n",
                    static_cast<unsigned long long>(begin_memo_hits),
                    static_cast<unsigned long long>(begin_memo_misses),
                    static_cast<unsigned long long>(texture_set_hits),
                    static_cast<unsigned long long>(texture_set_misses));
        std::printf("  texture set misses: other key %llu, generation %llu, image check %llu, "
                    "other %llu\n",
                    (unsigned long long)texture_set_why[0], (unsigned long long)texture_set_why[1],
                    (unsigned long long)texture_set_why[2], (unsigned long long)texture_set_why[3]);
        std::printf("  scene proxies sampled directly: %llu bindings\n",
                    static_cast<unsigned long long>(proxy_samples));
        proxy_samples = 0;
        texture_set_why = {};
        begin_memo_hits = begin_memo_misses = texture_set_hits = texture_set_misses = 0;
    }
    BeginSignature signature;
    if (memo_on) {
        signature = MakeBeginSignature(pipeline);
        const auto& regs = Regs();
        bool usable = begin_memo.valid && scheduler.IsRenderingWith(begin_memo.state) &&
                      !regs.depth_render_control.depth_clear_enable &&
                      !regs.depth_render_control.stencil_clear_enable &&
                      signature == begin_memo.signature;
        for (const auto id : signature.ids) {
            if (usable && id) {
                const auto& image = texture_cache.GetImage(id);
                usable = !image.binding.is_bound && !image.binding.needs_rebind;
            }
        }
        if (usable) {
            ++begin_memo_hits;
            attachment_feedback_loop = false;
            push_data.scene_size = begin_memo.scene_size;
            target_scale = begin_memo.target_scale;
            return begin_memo.state;
        }
        ++begin_memo_misses;
    }
    begin_memo.valid = false;
    RenderState state = BeginRenderingFull(pipeline);
    if (memo_on && !attachment_feedback_loop) {
        bool clears = state.depth.clear || state.depth.clear_stencil;
        for (u32 i = 0; i < state.num_colors; ++i) {
            clears |= state.colors[i].clear;
        }
        if (!clears) {
            // Recomputed after the full path: it may have started the scene (scene_started).
            begin_memo = {true, MakeBeginSignature(pipeline), state, push_data.scene_size,
                          target_scale};
        }
    }
    return state;
}

RenderState Rasterizer::BeginRenderingFull(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    TemporalUpscaler::Target redirect{};
    bool color_redirected = false, depth_redirected = false;
    std::pair<u32, u32> guest_extent{1, 1};
    void* original_color = nullptr; ///< id<MTLTexture> of color 0 before the upscaler's redirect
    const auto& regs = Regs();
    const auto& key = pipeline->GetGraphicsKey();
    if (std::popcount(key.mrt_mask & 0x7f) >= 5 && db_desc.first &&
        scene_targets->EligibleScene(texture_cache.GetImage(db_desc.first))) scene_started = true;
    // A proxy attachment cannot represent MSAA or a feedback loop that reads the
    // same image through the guest's native descriptor during this draw.
    bool reduced = scene_started && upscaler->RasterScaling() && key.num_samples == 1;
    // BB_SCENE_DEBUG=<file>: once the file exists, why the passes of one frame keep the
    // native size (see NoteFrameStart).
    const bool debug_pass = scene_debug_frame && scene_started;
    std::string why;
    if (debug_pass) {
        why = fmt::format("pass vs {:08x} ps {:08x} mrt {:#x} samples {} raster_scaling {}:",
                          pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash,
                          key.mrt_mask ? pipeline->GetStage(Shader::SwStage::Fragment).pgm_hash
                                       : 0,
                          key.mrt_mask, u32(key.num_samples), upscaler->RasterScaling());
    }
    // Only this pass's attachments: slots past the mask's width keep earlier passes' targets,
    // often the G-buffer images the lighting passes sample (bound), which kept those native.
    const u32 num_attachments = BbToggle::Disabled(BbToggle::SceneAttachmentsOnly)
                                    ? u32(cb_descs.size())
                                    : u32(std::bit_width(key.mrt_mask));
    // All attachments of a reduced pass share one proxy size (scene or half resolution).
    std::optional<SceneResolution::Size> pass_size;
    const auto same_size = [&](const VideoCore::Image& image, u32 level) {
        const auto proxy = scene_targets->ProxySize(image, level);
        if (!pass_size) pass_size = proxy;
        return *pass_size == proxy;
    };
    for (u32 cb = 0; cb < num_attachments; ++cb) {
        const auto& [id, desc] = cb_descs[cb];
        if (id) {
            const auto& image = texture_cache.GetImage(id);
            // Mip levels of eligible chains have their own proxies (SceneTargets::Get).
            if (!scene_targets->Eligible(image) || image.binding.is_bound ||
                image.binding.needs_rebind || desc.view_info.range.base.layer ||
                !same_size(image, desc.view_info.range.base.level)) reduced = false;
            if (debug_pass) {
                why += fmt::format(" [{} {}x{} eligible {} bound {} rebind {} level {} layer {}]",
                                   vk::to_string(image.info.pixel_format), image.info.size.width,
                                   image.info.size.height, scene_targets->Eligible(image),
                                   u32(image.binding.is_bound), u32(image.binding.needs_rebind),
                                   desc.view_info.range.base.level,
                                   desc.view_info.range.base.layer);
            }
        }
    }
    if (db_desc.first && (!scene_targets->Eligible(texture_cache.GetImage(db_desc.first)) ||
                          !same_size(texture_cache.GetImage(db_desc.first), 0))) {
        reduced = false;
    }
    if (debug_pass) {
        if (db_desc.first) {
            const auto& depth = texture_cache.GetImage(db_desc.first);
            why += fmt::format(" depth [{} {}x{} eligible {}]",
                               vk::to_string(depth.info.pixel_format), depth.info.size.width,
                               depth.info.size.height, scene_targets->Eligible(depth));
        }
        std::printf("Scene pass: reduced %d %s\n", reduced, why.c_str());
    }
    push_data.scene_size = reduced ? SceneResolution::Pack(scene_targets->Size()) : 0;
    if (BbStats::enabled) {
        BbStats::reduced_draws.fetch_add(reduced, std::memory_order_relaxed);
        BbStats::scene_draws.fetch_add(scene_started, std::memory_order_relaxed);
    }
    RenderState state;
    // Metal's largest render target (MoltenVK reported the same framebuffer limit).
    constexpr u32 MaxFramebufferSize = 16384;
    state.width = MaxFramebufferSize;
    state.height = MaxFramebufferSize;
    state.num_colors = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_colors; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.colors[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            last_targets[cb].generation = ~0ULL; // FindImage may rewrite the description
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            bound_images_written.push_back(1);
            bound_images_prev_write.push_back(0);
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        runtime.SetBackingSamples(image, key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;
        if (static const bool rt_log = std::getenv("BB_RT_LOG") != nullptr; rt_log) {
            // diagnostics: each distinct (address, format, view format) colour target once
            const auto& cbuf = regs.color_buffers[cb];
            static std::set<std::tuple<u64, u32, u32, int>> seen_rt;
            const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
            if (seen_rt.emplace(image->info.guest_address, u32(cbuf.GetDataFmt()) << 8 | u32(cbuf.GetNumberFmt()),
                                ps ? u32(ps->pgm_hash) : 0u, int(cb)).second && seen_rt.size() < 3000) {
                const auto* vs = pipeline->GetStages()[u32(Shader::SwStage::Vertex)];
                std::fprintf(stderr, "[RTFMT] vs %08x cb%u ps %08x addr %#llx %ux%u dfmt %u nfmt %u image %s view %s\n",
                             vs ? u32(vs->pgm_hash) : 0u, cb, ps ? u32(ps->pgm_hash) : 0u,
                             (unsigned long long)image->info.guest_address, image->info.size.width,
                             image->info.size.height, u32(cbuf.GetDataFmt()), u32(cbuf.GetNumberFmt()),
                             vk::to_string(image->info.pixel_format).c_str(),
                             vk::to_string(image_view.info.format).c_str());
                std::fprintf(stderr, "[RTFMT]   samples=%u key_samples=%u aa=%u addr %#llx\n", image->info.num_samples, u32(key.color_samples[cb]), u32(regs.aa_config.NumSamples()), (unsigned long long)image->info.guest_address);
            }
        }

        // fifa-hybrid diagnostics: BB_TRACE_AT=<s>[,<s>...] logs every colour-target binding (one
        // line per draw with cb0) for the next BB_TRACE_DRAWS (default 2500) draws after each listed
        // second since the first draw: a per-frame pass list to diff open play against a replay.
        if (cb == 0) {
            static const std::vector<double> trace_at = [] {
                std::vector<double> out;
                if (const char* e = std::getenv("BB_TRACE_AT")) {
                    for (char* end = nullptr; *e; e = *end ? end + 1 : end) {
                        out.push_back(std::strtod(e, &end));
                        if (end == e) break;
                    }
                }
                return out;
            }();
            if (!trace_at.empty()) {
                static const int trace_draws = [] {
                    const char* e = std::getenv("BB_TRACE_DRAWS");
                    return e ? std::atoi(e) : 2500;
                }();
                static const auto t0 = std::chrono::steady_clock::now();
                static size_t next = 0;
                static int left = 0;
                const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (next < trace_at.size() && now >= trace_at[next]) {
                    std::fprintf(stderr, "[TR] ---- trace %zu at %.1fs ----\n", next, now);
                    ++next;
                    left = trace_draws;
                }
                if (left > 0) {
                    --left;
                    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
                    const auto* vs = pipeline->GetStages()[u32(Shader::SwStage::Vertex)];
                    const auto& db = regs.depth_buffer;
                    std::fprintf(stderr, "[TR] ps %08x vs %08x cb0 %#llx %ux%u %s n%u db %#llx clip %d vpz %g/%g clear %d dtest %d dwrite %d cmask %#x\n",
                                 ps ? u32(ps->pgm_hash) : 0u, vs ? u32(vs->pgm_hash) : 0u,
                                 (unsigned long long)image->info.guest_address, image->info.size.width,
                                 image->info.size.height, vk::to_string(image->info.pixel_format).c_str(),
                                 state.num_colors, (unsigned long long)(db.DepthAddress()),
                                 int(regs.clipper_control.clip_space),
                                 regs.viewport_control.zoffset_enable ? regs.viewports[0].zoffset : 0.f,
                                 regs.viewport_control.zscale_enable ? regs.viewports[0].zscale : 1.f,
                                 int(texture_cache.IsMetaCleared(regs.color_buffers[cb].CmaskAddress(), 0)),
                                 int(regs.depth_control.depth_enable), int(regs.depth_control.depth_write_enable),
                                 u32(regs.color_target_mask.raw));
                }
            }
        }
        // fifa-hybrid diagnostics: BB_BLEND_LOG=1 logs the blend state of each (ps, state) drawing
        // into the HDR scene target (B10G11R11), once.
        if (static const bool blend_log = std::getenv("BB_BLEND_LOG") != nullptr;
            blend_log && image->info.pixel_format == vk::Format::eB10G11R11UfloatPack32) {
            const auto& bc = regs.blend_control[cb];
            const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
            static std::set<std::tuple<u64, u32, u32>> seen;
            if (seen.emplace(ps ? ps->pgm_hash : 0, std::bit_cast<u32>(bc), u32(regs.color_buffers[cb].info.blend_bypass)).second && seen.size() < 300) {
                std::fprintf(stderr, "[BLEND] ps %08llx cb%u %ux%u enable=%u src=%u dst=%u func=%u asrc=%u adst=%u afunc=%u sep=%u bypass=%u\n",
                             (unsigned long long)(ps ? ps->pgm_hash : 0), cb, image->info.size.width,
                             image->info.size.height, u32(bc.enable), u32(bc.color_src_factor),
                             u32(bc.color_dst_factor), u32(bc.color_func), u32(bc.alpha_src_factor),
                             u32(bc.alpha_dst_factor), u32(bc.alpha_func), u32(bc.separate_alpha_blend),
                             u32(regs.color_buffers[cb].info.blend_bypass));
            }
        }
        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (reduced) {
            // The native image is resolved lazily when a shader/copy actually reads it.
        } else {
            if (image->binding.is_bound) {
                // Sampled by this draw too: a feedback loop.
                ASSERT_MSG(!image->binding.force_general,
                           "Having image both as storage and render target is unsupported");
                attachment_feedback_loop = true;
            }
            ++image->diag_rt_writes;
            // fifa-hybrid diagnostics (BB_TWIN_WATCH): every render-target bind of a 2048^2 R32 image.
            if (static const bool smap = std::getenv("BB_TWIN_WATCH") != nullptr;
                smap && image->info.size.width == 2048 && image->info.pixel_format == vk::Format::eR32Sfloat) {
                static int n = 0;
                if (BbSeqWindow()) {
                    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
                    std::fprintf(stderr, "[SEQ] %llu draw-rt cb%u %#llx ps %08x db %#llx\n", (unsigned long long)g_bb_draw_seq.load(),
                                 cb, (unsigned long long)image->info.guest_address, ps ? u32(ps->pgm_hash) : 0u,
                                 (unsigned long long)regs.depth_buffer.DepthAddress());
                }
                if (++n <= 300 || n % 100 == 0) {
                    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
                    const auto* vs = pipeline->GetStages()[u32(Shader::SwStage::Vertex)];
                    const bool db_meta_clear = texture_cache.IsMetaCleared(regs.depth_htile_data_base.GetAddress(), 0);
                    std::fprintf(stderr, "[SMAP] #%d cb%u %#llx ps %08x vs %08x n%u mrt %#x mask %#x clear %d db %#llx dtest %d dwrite %d dfunc %d dclear_en %d dclear %g htile_clear %d dbimg %u clip %d vpz %g/%g rastkill %u vtxkill %u\n",
                                 n, cb, (unsigned long long)image->info.guest_address, ps ? u32(ps->pgm_hash) : 0u,
                                 vs ? u32(vs->pgm_hash) : 0u, state.num_colors, u32(key.mrt_mask), u32(regs.color_target_mask.raw),
                                 int(is_clear), (unsigned long long)regs.depth_buffer.DepthAddress(),
                                 int(regs.depth_control.depth_enable), int(regs.depth_control.depth_write_enable),
                                 int(regs.depth_control.depth_func), int(regs.depth_render_control.depth_clear_enable),
                                 regs.depth_clear, int(db_meta_clear), db_desc.first.index,
                                 int(regs.clipper_control.clip_space),
                                 regs.viewport_control.zoffset_enable ? regs.viewports[0].zoffset : 0.f,
                                 regs.viewport_control.zscale_enable ? regs.viewports[0].zscale : 1.f,
                                 u32(regs.clipper_control.dx_rasterization_kill), u32(regs.clipper_control.vtx_kill_or));
                }
            }
            runtime.AccessImage(image, true);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));

        // The view selects the level and slice (level and slice 0 of the view's texture).
        auto& attachment = state.colors[cb];
        attachment = {};
        attachment.texture = image_view.Texture();
        // fifa-hybrid: a 2D view of a 3D image is the 3D texture itself (image_view.cpp), so
        // the slice the colour buffer selects must become the render pass's depth plane, or
        // every slice of a volume rendered slice by slice (FIFA's colour grading LUT) lands in
        // plane 0.
        if (image->info.type == AmdGpu::ImageType::Color3D) {
            attachment.depth_plane = slice;
        }
        attachment.clear = is_clear;
        if (is_clear) {
            attachment.clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);
        }

        if (reduced) {
            const auto target = scene_targets->Attachment(image_id, desc.view_info);
            attachment.texture = target.texture;
        }
        image->usage.render_target = 1u;
        if (cb == 0 && upscaler->Enabled()) {
            color_redirected = upscaler->RedirectColor(image_id, desc.view_info, redirect);
            if (color_redirected) {
                guest_extent = {image->info.size.width, image->info.size.height};
                original_color = attachment.texture;
                attachment.texture = redirect.texture;
            }
        }
    }
    for (u32 cb = state.num_colors; cb < state.colors.size(); ++cb) {
        state.colors[cb] = {};
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc, GlDepthPass(regs));
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        if (static const bool rt_log = std::getenv("BB_RT_LOG") != nullptr; rt_log) {
            // diagnostics: each distinct (address, size, layers, slice) depth target once
            static std::set<std::tuple<u64, u32, u32, u32>> seen_db;
            if (seen_db.emplace(image.info.guest_address,
                                image.info.size.width << 16 | image.info.size.height,
                                image.info.resources.layers, slice).second &&
                seen_db.size() < 600) {
                const auto* vs = pipeline->GetStages()[u32(Shader::SwStage::Vertex)];
                std::fprintf(stderr, "[DBFMT] vs %08x addr %#llx %ux%u layers %u slice %u fmt %s view %s\n",
                             vs ? u32(vs->pgm_hash) : 0u,
                             (unsigned long long)image.info.guest_address, image.info.size.width,
                             image.info.size.height, image.info.resources.layers, slice,
                             vk::to_string(image.info.pixel_format).c_str(),
                             vk::to_string(image_view.info.format).c_str());
            }
        }
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        u32 stencil_clear_value = regs.stencil_clear;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        // The pass reads and may write depth and stencil (as the Vulkan attachment access did).
        if (!reduced) {
            runtime.AccessImage(&image, true);
        }

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);

        // The attachments match the pipeline's formats (PipelineCache::RefreshGraphicsKey): both
        // come from BoundDepthAttachment (vk_depth_attachment.h). A stencil without depth is
        // bound as the stencil attachment only.
        const DepthAttachment bound = BoundDepthAttachment(regs);
        auto& attachment = state.depth;
        attachment = {};
        attachment.texture = image_view.Texture();
        state.has_depth = bound.HasDepth();
        state.has_stencil = bound.HasStencil();
        state.stencil_only = state.has_stencil && !state.has_depth;
        // fifa-hybrid: a never-written stencil plane starts at 0 like PS4 memory (the UI's
        // stencil masks are written relative to it; undefined = 0xff inverted them: the
        // scoreboard's shine outside its shape). BB_STENCIL_ZERO=0 disables.
        static const bool stencil_zero = [] {
            const char* e = std::getenv("BB_STENCIL_ZERO");
            return !(e && e[0] == '0');
        }();
        if (stencil_zero && state.has_stencil && image.stencil_undefined) {
            image.stencil_undefined = false;
            if (!is_stencil_clear) {
                is_stencil_clear = true;
                stencil_clear_value = 0;
            }
            LOG_INFO(Render_Vulkan, "Zeroing the fresh stencil of depth target {:#x} {}x{}",
                     image.info.guest_address, image.info.size.width, image.info.size.height);
        }
        if (state.has_depth) {
            attachment.clear = is_depth_clear;
            attachment.clear_value.depthStencil.depth = is_depth_clear ? regs.depth_clear : 0.0f;
        }
        if (state.has_stencil) {
            attachment.clear_stencil = is_stencil_clear;
            attachment.clear_value.depthStencil.stencil = is_stencil_clear ? stencil_clear_value : 0u;
        }

        if (reduced) {
            const auto target = scene_targets->Attachment(image_id, desc.view_info);
            attachment.texture = target.texture;
        }
        image.usage.depth_target = true;
        // fifa-hybrid diagnostics (BB_DB_WATCH=<width>): draws into depth targets of that width, 40 in a row
        // out of every 1500, with the state that decides what lands in the map.
        if (static const u32 dbw = [] { const char* e = std::getenv("BB_DB_WATCH"); return e ? u32(std::atoi(e)) : 0u; }();
            dbw == 1 && image.info.size.width != 1920) {
            // BB_DB_WATCH=1: every depth target that is not 1080p, one line per distinct kind of draw with counts.
            const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
            const auto* vs = pipeline->GetStages()[u32(Shader::SwStage::Vertex)];
            const std::string key = fmt::format(
                "db {:#x} {}x{} vs {:08x} ps {:08x} ncol {} cb0 {:#x} clear {} val {} dtest {} dwrite {} dfunc {}",
                image.info.guest_address, image.info.size.width, image.info.size.height, vs ? u32(vs->pgm_hash) : 0u,
                ps ? u32(ps->pgm_hash) : 0u, state.num_colors,
                state.num_colors ? u64(regs.color_buffers[0].Address()) : 0ull, int(is_depth_clear), regs.depth_clear,
                int(regs.depth_control.depth_enable), int(regs.depth_control.depth_write_enable),
                int(regs.depth_control.depth_func));
            static std::map<std::string, u64> counts;
            const u64 c = ++counts[key];
            if (counts.size() < 600 && (c == 1 || c % 2000 == 0)) {
                std::fprintf(stderr, "[DBW1] %s count %llu\n", key.c_str(), (unsigned long long)c);
            }
        } else if (dbw > 1 && image.info.size.width == dbw) {
            static u32 n = 0;
            if (n++ % 1500 < 40) {
                const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
                const auto* vs = pipeline->GetStages()[u32(Shader::SwStage::Vertex)];
                std::fprintf(stderr, "[DBW] #%u db %#llx w %#llx %ux%u vs %08x ps %08x idx %u ncol %u cmode %d clear %d (en %d meta %d val %g) dtest %d dwrite %d dfunc %d clip %d vpz %g/%g vp %gx%g fzd %d fzv %d\n",
                             n, (unsigned long long)regs.depth_buffer.DepthAddress(),
                             (unsigned long long)regs.depth_buffer.DepthWriteAddress(), image.info.size.width,
                             image.info.size.height, vs ? u32(vs->pgm_hash) : 0u, ps ? u32(ps->pgm_hash) : 0u,
                             u32(regs.num_indices), state.num_colors, int(regs.color_control.mode), int(is_depth_clear),
                             int(regs.depth_render_control.depth_clear_enable),
                             int(texture_cache.IsMetaCleared(htile_address, slice)), regs.depth_clear,
                             int(regs.depth_control.depth_enable), int(regs.depth_control.depth_write_enable),
                             int(regs.depth_control.depth_func), int(regs.clipper_control.clip_space),
                             regs.viewport_control.zoffset_enable ? regs.viewports[0].zoffset : 0.f,
                             regs.viewport_control.zscale_enable ? regs.viewports[0].zscale : 1.f,
                             regs.viewports[0].xscale * 2.f, regs.viewports[0].yscale * 2.f,
                             int(regs.depth_render_override.force_z_dirty), int(regs.depth_render_override.force_z_valid));
            }
        }
        if (upscaler->Enabled()) {
            TemporalUpscaler::Target depth_redirect;
            depth_redirected = upscaler->RedirectDepth(image_id, depth_redirect);
            if (depth_redirected) {
                guest_extent = {image.info.size.width, image.info.size.height};
                attachment.texture = depth_redirect.texture;
                redirect = depth_redirect;
            }
        }
    } else {
        state.depth = {};
        state.has_depth = state.has_stencil = state.stencil_only = false;
    }

    // bbport: a pass drawn into the upscaler's output-size images (UI, display pass): every
    // attachment must be redirected, viewports and scissors are scaled.
    target_scale = {1.0f, 1.0f};
    if (color_redirected || depth_redirected) {
        u32 color_targets = 0;
        for (u32 cb = 0; cb < state.num_colors; ++cb) {
            color_targets += state.colors[cb].texture ? 1 : 0;
        }
        const bool has_depth = bool(db_desc.first);
        if ((color_targets > 0 && !color_redirected) || (has_depth && !depth_redirected) ||
            color_targets > 1) {
            static u32 warned = 0;
            if (warned < 8) {
                ++warned;
                const auto describe = [&](VideoCore::ImageId id) {
                    if (!id) {
                        return std::string{"-"};
                    }
                    const auto& info = texture_cache.GetImage(id).info;
                    return fmt::format("{:#x} {}x{} {}", info.guest_address, info.size.width,
                                       info.size.height, vk::to_string(info.pixel_format));
                };
                std::printf("Upscaler: pass not redirected (color %s: %s, depth %s: %s, %u "
                            "targets)\n",
                            color_redirected ? "yes" : "no", describe(cb_descs[0].first).c_str(),
                            depth_redirected ? "yes" : "no", describe(db_desc.first).c_str(),
                            state.num_colors);
            }
            if (color_redirected) {
                state.colors[0].texture = original_color;
            }
            if (depth_redirected) {
                state.depth.texture =
                    texture_cache.FindDepthTarget(db_desc.first, db_desc.second, GlDepthPass(regs)).Texture();
            }
        } else {
            state.width = redirect.width;
            state.height = redirect.height;
            // The UI movie viewport is now 1920x1080 even when the scene is 960x540.
            // A native viewport (or native window-space UI vertices) must not be doubled
            // again. The final display copy still uses the guest render-size coordinates.
            const auto& viewport = regs.viewports[0];
            const bool native_coordinates = redirect.native_ui &&
                (UiComposition::NativeViewport(viewport.xscale * 2, viewport.yscale * 2) ||
                 (regs.IsClipDisabled() && !regs.viewport_control.xscale_enable &&
                  !regs.viewport_control.yscale_enable));
            target_scale = UiComposition::Scale(guest_extent.first, guest_extent.second,
                                                redirect.width, redirect.height,
                                                native_coordinates);
        }
    }

    if (reduced) {
        // Half-resolution passes keep the scene's factor (proxy / native = scene / 1920).
        const auto size = scene_targets->Size();
        const auto proxy = pass_size.value_or(size);
        state.width = proxy.width;
        state.height = proxy.height;
        target_scale = {float(size.width) / 1920.0f, float(size.height) / 1080.0f};
    }
    if (FrameCapture::Active()) {
        std::array<const VideoCore::ImageInfo*, AmdGpu::NUM_COLOR_BUFFERS> colors{};
        for (u32 cb = 0; cb < state.num_colors; ++cb) {
            if (cb_descs[cb].first) {
                colors[cb] = &texture_cache.GetImage(cb_descs[cb].first).info;
            }
        }
        const auto* depth =
            db_desc.first ? &texture_cache.GetImage(db_desc.first).info : nullptr;
        FrameCapture::BeginPass(colors.data(), state.num_colors, depth);
    }
    // bbport: object motion vector attachment of G-buffer pipelines.
    if (key.motion_vectors) {
        object_motion->Attach(state, state.width, state.height);
    }
    // bbport: draws with no pixel shader and no attachments (vertex shaders that only write
    // memory) produce no fragments anyone can see, but the render area stayed at the device
    // maximum, 16384x16384: Apple's tiler walked ~260k tiles for each such pass, ~0.3 ms apiece,
    // ~6 ms per frame in the city (Metal System Trace, tools/pass_trace.py). A tiny area keeps the
    // vertex work and drops the tiling.
    if (state.num_colors == 0 && !state.has_depth && !state.has_stencil &&
        !pipeline->GetStages()[u32(Shader::SwStage::Fragment)] &&
        !BbToggle::Disabled(BbToggle::NoOutputPassArea)) {
        state.width = 16;
        state.height = 16;
    }
    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = CbExtent(0);
    const auto& mrt1_hint = CbExtent(1);
    VideoCore::TextureCache::ImageDesc mrt0_desc{Regs().color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{Regs().color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 Regs().color_buffers[0].Address(),
                                 Regs().color_buffers[1].Address()));
    runtime.ResolveImage(&mrt0_image, &mrt1_image, mrt0_desc.view_info.range,
                         mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = Regs();

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), DbExtent(), false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), DbExtent(), true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = Regs().depth_view.slice_start;
    sub_range.extent.layers = Regs().depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    runtime.CopyDepthStencil(&read_image, &write_image, sub_range);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    DrainDrawPipe();
    ASSERT_MSG(address % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer address and size must be a multiple of 4 bytes");
    if (!is_gds && !deferred_dispatches.empty() &&
        (deferred_conflict_site = "fill", DeferredDispatchTouches(address, num_bytes, true))) {
        FlushDeferredDispatches(); // fifa-hybrid
    }
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (num_bytes == 0) {
            return; // bbport: nothing to fill; no empty blit
        }
        if (!buffer_cache.IsRegionGpuModified(address, num_bytes)) {
            texture_cache.DropOndemandCovered(address, num_bytes);
            NoteHostWrite(address, num_bytes, "fill");
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + (num_bytes / sizeof(u32)), value);
            return;
        }
    }
    const auto [buffer, offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (is_gds) {
            return {buffer_cache.GetGdsBuffer(), address};
        }
        return buffer_cache.ObtainBuffer(address, num_bytes, true, false, true,
                                         BbArenaRuns::Caller::Fill);
    }();
    if (num_bytes == 0) {
        return; // GDS: nothing to fill
    }
    runtime.FillBuffer(*buffer, offset, num_bytes, value);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (VideoCore::WatchOverlaps(dst, num_bytes) || VideoCore::WatchOverlaps(src, num_bytes)) {
        std::fprintf(stderr, "[WATCH] dma copy %#llx <- %#llx %u bytes (gds %d/%d) dst gpu_modified=%d src gpu_modified=%d\n",
                     (unsigned long long)dst, (unsigned long long)src, num_bytes, int(dst_gds), int(src_gds),
                     int(buffer_cache.IsRegionGpuModified(dst, num_bytes)),
                     int(buffer_cache.IsRegionGpuModified(src, num_bytes)));
        if (!dst_gds && VideoCore::WatchOverlaps(src, num_bytes)) {
            // Follow the table to its destination (the probe would only re-arm at the next bind).
            VideoCore::g_watch_lo.store(dst, std::memory_order_relaxed);
            VideoCore::g_watch_hi.store(dst + num_bytes, std::memory_order_relaxed);
            std::fprintf(stderr, "[WATCH] re-armed on the dma destination %#llx+%u\n", (unsigned long long)dst, num_bytes);
        }
    }
    DrainDrawPipe();
    if (num_bytes == 0) {
        return; // bbport: nothing to copy; no empty blit
    }
    if (!deferred_dispatches.empty() &&
        (deferred_conflict_site = "dma",
         (!src_gds && DeferredDispatchTouches(src, num_bytes, false)) ||
             (!dst_gds && DeferredDispatchTouches(dst, num_bytes, true)))) {
        FlushDeferredDispatches(); // fifa-hybrid: a pending dispatch reaches the copy's ranges
    }
    if (static const bool dlog = std::getenv("BB_HOSTW_LOG") != nullptr; dlog && num_bytes >= 65536) {
        static std::set<std::pair<VAddr, u32>> seen;
        if (seen.size() < 400 && seen.emplace(dst, num_bytes).second) {
            std::fprintf(stderr, "[DMA] copy %#llx <- %#llx %u bytes gds %d/%d dst gpu %d src gpu %d src image %d\n",
                         (unsigned long long)dst, (unsigned long long)src, num_bytes, int(dst_gds), int(src_gds),
                         int(!dst_gds && buffer_cache.IsRegionGpuModified(dst, num_bytes)),
                         int(!src_gds && buffer_cache.IsRegionGpuModified(src, num_bytes)),
                         int(!src_gds && bool(texture_cache.FindImageFromRange(src, num_bytes))));
        }
    }
    if (!dst_gds && !buffer_cache.IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !buffer_cache.IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            texture_cache.DropOndemandCovered(dst, num_bytes);
            NoteHostWrite(dst, num_bytes, "copy");
            std::memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    const auto* gds_buffer = buffer_cache.GetGdsBuffer();
    const auto [src_buffer, src_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (src_gds) {
            return {gds_buffer, src};
        }
        return buffer_cache.ObtainBuffer(src, num_bytes, false, true, true,
                                         BbArenaRuns::Caller::Copy);
    }();
    const auto [dst_buffer, dst_offset] = [&] -> std::pair<const VideoCore::Buffer*, u64> {
        if (dst_gds) {
            return {gds_buffer, dst};
        }
        return buffer_cache.ObtainBuffer(dst, num_bytes, true, true, true,
                                         BbArenaRuns::Caller::Copy);
    }();
    const VideoCore::BufferCopy copy = {
        .src_offset = src_offset,
        .dst_offset = dst_offset,
        .size = num_bytes,
    };
    runtime.CopyBuffer(*src_buffer, *dst_buffer, std::span{&copy, 1});
}

void Rasterizer::NoteHostWrite(VAddr addr, u64 size, const char* what) {
    // BB_HOST_WRITE_NOTIFY=0: the behaviour before 0.5.1-g (the caches are not told).
    static const bool notify = [] {
        const char* e = std::getenv("BB_HOST_WRITE_NOTIFY");
        return !e || e[0] != '0';
    }();
    // The buffer cache is told only on request (BB_HOST_WRITE_NOTIFY_BUF=1): it re-uploads the pages
    // every frame and the game's own writes to them then fault every frame.
    static const bool notify_buf = [] {
        const char* e = std::getenv("BB_HOST_WRITE_NOTIFY_BUF");
        return e && e[0] == '1';
    }();
    static const bool log = std::getenv("BB_HOSTW_LOG") != nullptr;
    if (size == 0) {
        return;
    }
    u32 clean_pages = 0;
    std::string first;
    if (log) {
        const VAddr end = addr + size;
        u32 looked = 0;
        for (VAddr page = addr & ~0xfffull; page < end && looked < 8192; page += 0x1000, ++looked) {
            if (!buffer_cache.IsRegionCpuModified(page, 0x1000) &&
                !buffer_cache.IsRegionGpuModified(page, 0x1000)) {
                ++clean_pages;
            }
        }
    }
    const u32 images = texture_cache.NoteHostWrite(addr, size, notify, log ? &first : nullptr);
    if (notify && notify_buf) {
        buffer_cache.NoteHostWrite(addr, size);
    }
    if (log) {
        static u64 calls = 0, bytes = 0, hit_calls = 0, hit_images = 0, hit_pages = 0;
        static u32 lines = 0;
        ++calls;
        bytes += size;
        if (images || clean_pages) {
            ++hit_calls;
            hit_images += images;
            hit_pages += clean_pages;
            if (lines < 400 || images) {
                ++lines;
                std::fprintf(stderr, "[HOSTW] %s %#llx+%llu: %u clean buffer pages, %u clean images%s%s notify=%d\n",
                             what, (unsigned long long)addr, (unsigned long long)size, clean_pages, images,
                             first.empty() ? "" : " first ", first.c_str(), int(notify));
            }
        }
        if ((calls & (calls - 1)) == 0 || (calls % 5000) == 0) {
            std::fprintf(stderr, "[HOSTW] totals: %llu writes, %.1f MB, %llu touched clean data (%llu images, %llu buffer pages)\n",
                         (unsigned long long)calls, bytes / 1e6, (unsigned long long)hit_calls,
                         (unsigned long long)hit_images, (unsigned long long)hit_pages);
        }
    }
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    DrainDrawPipe();
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

// fifa-hybrid diagnostics (BB_FAULT_STATS): the on-demand resolve of write faults, split into the
// drop path (droppable image forgotten, no GPU round trip) and the download path.
static std::atomic<u64> g_odw_drop_n{0}, g_odw_drop_us{0}, g_odw_dl_n{0}, g_odw_dl_us{0};
static std::atomic<u64> g_odw_tail_n{0}, g_odw_tail_us{0}; // 0.5.1-e: tail releases
static std::atomic<u64> g_odw_none_n{0}; // write faults on a watched page with nothing pending

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, bool assume_locks) {
    DrainDrawPipe();
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    ResolveOndemandImages(addr, size, true, assume_locks);
    buffer_cache.InvalidateMemory(addr, size, assume_locks);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

void Rasterizer::OnCpuWrite(VAddr addr, u64 size, VAddr fault) {
    // The game already wrote: no readback may land on this range now. Pages the host tracked for
    // upload become CPU-modified (as ExtendWriteFault); on the written page a pending GPU write
    // (a label written back asynchronously) is dropped, the CPU's later write wins.
    DrainDrawPipe();
    if (!IsMapped(addr, size)) {
        return;
    }
    // Images: only the written bytes, as the relayed path did (InvalidateMemory(fault, 8)). The
    // whole window dirtied GPU-only render targets sharing it (the MSAA scene depth was then
    // re-uploaded from stale guest memory and the depth resolve saw the clear value).
    texture_cache.InvalidateMemory(fault, 8);
    buffer_cache.OnCpuWrite(addr, size, fault);
}

bool Rasterizer::OnWriteFault(VAddr addr, bool assume_locks) {
    // fifa-hybrid diagnostics: BB_FAULT_STATS=1 times the phases of a write fault (host side).
    static const bool stats = std::getenv("BB_FAULT_STATS") != nullptr;
    if (!stats) {
        DrainDrawPipe();
        if (!InvalidateMemory(addr, 8, assume_locks)) {
            return false;
        }
        buffer_cache.ExtendWriteFault(addr);
        return true;
    }
    using Clock = std::chrono::steady_clock;
    static std::atomic<u64> n{0}, unmapped{0}, t_drain{0}, t_ondemand{0}, t_buffer{0}, t_texture{0},
        t_extend{0};
    static std::atomic<s64> window_start{0};
    const auto t0 = Clock::now();
    DrainDrawPipe();
    const auto t1 = Clock::now();
    bool mapped = IsMapped(addr, 8);
    auto t2 = t1, t3 = t1, t4 = t1, t5 = t1;
    if (mapped) {
        ResolveOndemandImages(addr, 8, true, assume_locks);
        t2 = Clock::now();
        buffer_cache.InvalidateMemory(addr, 8, assume_locks);
        t3 = Clock::now();
        texture_cache.InvalidateMemory(addr, 8);
        t4 = Clock::now();
        buffer_cache.ExtendWriteFault(addr);
        t5 = Clock::now();
    } else {
        unmapped.fetch_add(1, std::memory_order_relaxed);
    }
    const auto us = [](auto a, auto b) {
        return u64(std::chrono::duration_cast<std::chrono::microseconds>(b - a).count());
    };
    n.fetch_add(1, std::memory_order_relaxed);
    t_drain.fetch_add(us(t0, t1), std::memory_order_relaxed);
    t_ondemand.fetch_add(us(t1, t2), std::memory_order_relaxed);
    t_buffer.fetch_add(us(t2, t3), std::memory_order_relaxed);
    t_texture.fetch_add(us(t3, t4), std::memory_order_relaxed);
    t_extend.fetch_add(us(t4, t5), std::memory_order_relaxed);
    const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now().time_since_epoch()).count();
    s64 ws = window_start.load(std::memory_order_relaxed);
    if (ws == 0) {
        window_start.compare_exchange_strong(ws, now);
    } else if (now - ws >= 2000000000 && window_start.compare_exchange_strong(ws, now)) {
        const u64 cnt = n.exchange(0);
        std::printf("[WFAULT] n=%llu unmapped=%llu drain=%.1fms ondemand=%.1fms buffer=%.1fms "
                    "(flushes %llu) texture=%.1fms extend=%.1fms | odw drop %llu (%.1fms) "
                    "download %llu (%.1fms) tail %llu (%.1fms) unresolved %llu\n",
                    (unsigned long long)cnt, (unsigned long long)unmapped.exchange(0),
                    t_drain.exchange(0) / 1e3, t_ondemand.exchange(0) / 1e3,
                    t_buffer.exchange(0) / 1e3,
                    (unsigned long long)buffer_cache.write_flushes.exchange(0),
                    t_texture.exchange(0) / 1e3, t_extend.exchange(0) / 1e3,
                    (unsigned long long)g_odw_drop_n.exchange(0), g_odw_drop_us.exchange(0) / 1e3,
                    (unsigned long long)g_odw_dl_n.exchange(0), g_odw_dl_us.exchange(0) / 1e3,
                    (unsigned long long)g_odw_tail_n.exchange(0), g_odw_tail_us.exchange(0) / 1e3,
                    (unsigned long long)g_odw_none_n.exchange(0));
        texture_cache.PrintOndemandWriteCounts();
        std::fflush(stdout);
    }
    return mapped;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size, bool assume_locks) {
    if (g_bb_diag_window.load(std::memory_order_relaxed)) {
        std::fprintf(stderr, "[RFD] read %#llx+%llu locks=%d gpumod=%d ondemand=%d\n",
                     (unsigned long long)addr, (unsigned long long)size, int(assume_locks),
                     int(buffer_cache.IsRegionGpuModified(addr, size)),
                     int(texture_cache.HasOndemandIn(addr, size)));
    }
    static const bool log_reads = std::getenv("BB_LOG_READFAULTS") != nullptr;
    if (log_reads) {
        static std::atomic<u64> count{0};
        const u64 n = count.fetch_add(1);
        if (n < 300 || (n % 500) == 0) {
            std::fprintf(stderr, "[RF] #%llu addr=%#llx size=%llu locks=%d gpumod=%d ondemand=%d\n",
                         (unsigned long long)n, (unsigned long long)addr, (unsigned long long)size,
                         int(assume_locks), int(buffer_cache.IsRegionGpuModified(addr, size)),
                         int(texture_cache.HasOndemandIn(addr, size)));
        }
    }
    DrainDrawPipe();
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    ResolveOndemandImages(addr, size, false, assume_locks);
    buffer_cache.ReadMemory(addr, size, false, assume_locks);
    return true;
}

// fifa-hybrid: images awaiting an on-demand readback that overlap a CPU access are downloaded
// now, on the GPU command thread (the download records a tiling dispatch and drains the GPU).
void Rasterizer::ResolveOndemandImages(VAddr addr, u64 size, bool for_write, bool assume_locks) {
    // Most faults touch no pending image; the round trip to the GPU thread below waited for its
    // whole command list (several ms per write fault in matches).
    // 0.5.1-e: a write into the image bytes of a released tail page (no read watcher there any
    // more) still downloads the image first.
    const bool tail_hit = for_write && texture_cache.OndemandTailHit(addr, size);
    if (!tail_hit && !texture_cache.HasOndemandIn(addr, size)) {
        return;
    }
    static const bool stats = std::getenv("BB_FAULT_STATS") != nullptr;
    if (stats && for_write) {
        texture_cache.LogOndemandWrite(addr, size);
    }
    static const bool drop_on_write = [] {
        const char* env = std::getenv("BB_ONDEMAND_WRITE_DROP");
        return !env || env[0] != '0';
    }();
    const auto t_start = std::chrono::steady_clock::now();
    const auto note = [&](std::atomic<u64>& n, std::atomic<u64>& us) {
        if (stats) {
            n.fetch_add(1, std::memory_order_relaxed);
            us.fetch_add(u64(std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - t_start)
                                 .count()),
                         std::memory_order_relaxed);
        }
    };
    // 0.5.1-e: a write past the end of a pending image on its last page releases that page
    // instead of waiting for the GPU (TextureCache::ReleaseOndemandTails).
    if (for_write && !tail_hit && texture_cache.ReleaseOndemandTails(addr, size) &&
        !texture_cache.HasOndemandIn(addr, size)) {
        note(g_odw_tail_n, g_odw_tail_us);
        return;
    }
    if (for_write && drop_on_write && !tail_hit) {
        texture_cache.ForgetOndemandImagesIn(addr, size, true);
        if (!texture_cache.HasOndemandIn(addr, size)) {
            note(g_odw_drop_n, g_odw_drop_us);
            return;
        } // block-compressed images and their shadows below: downloaded before the write
    }
    struct DlNote {
        bool on;
        std::atomic<u64>& n;
        std::atomic<u64>& us;
        decltype(note)& f;
        ~DlNote() {
            if (on) {
                f(n, us);
            }
        }
    } dl_note{stats && for_write, g_odw_dl_n, g_odw_dl_us, note};
    // fifa-hybrid: prefetched images need no round trip (TextureCache::ResolvePrefetchedOndemand).
    if (!assume_locks && texture_cache.ResolvePrefetchedOndemand(addr, size)) {
        return;
    }
    if (assume_locks) {
        texture_cache.DownloadImagesForCpuRead(addr, size, for_write, true);
        return;
    }
    // fifa-hybrid: the GPU thread records the downloads and submits; this (fault) thread waits
    // for the device and writes guest memory, so frames keep flowing meanwhile. The warm-up
    // drill's asset loading faulted 10-40 times per 2 s with a full drain each on the GPU
    // thread. BB_READBACK_GPUWAIT=1 restores the drain on the GPU thread.
    static const bool gpu_wait = [] {
        const char* env = std::getenv("BB_READBACK_GPUWAIT");
        return env && env[0] == '1';
    }();
    if (gpu_wait) {
        liverpool->SendCommand<true>([this, addr, size, for_write] {
            DrainDrawPipe();
            texture_cache.DownloadImagesForCpuRead(addr, size, for_write, false);
        });
        return;
    }
    VideoCore::TextureCache::CpuReadWork work;
    liverpool->SendCommand<true>([this, addr, size, for_write, &work] {
        DrainDrawPipe();
        texture_cache.DownloadImagesForCpuRead(addr, size, for_write, false, &work);
    });
    if (stats && for_write && work.copies.empty() && work.unprotect.empty()) {
        g_odw_none_n.fetch_add(1, std::memory_order_relaxed);
    }
    texture_cache.FinishCpuReadWork(work);
}

void Rasterizer::ProcessDownloadImages() {
    DrainDrawPipe();
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::RegisterMemory(VAddr addr, u64 size) {
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    DrainDrawPipe();
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

Metal::Draw::DrawState Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline,
                                                      const bool is_indexed,
                                                      const RenderState& state) const {
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    // The draw helper takes the whole state with each draw (DynamicState::CommitWith keeps it
    // current); tests of a depth or stencil aspect the pass lacks are off.
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetAttachments(state.has_depth, state.has_stencil);
    Metal::Draw::DrawState draw_state;
    dynamic_state.CommitWith(scheduler.GetDepthStencilCache(),
                             [&](const Metal::Draw::DrawState& s) { draw_state = s; });
    draw_state.target_width = state.width;
    draw_state.target_height = state.height;
    return draw_state;
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = Regs();

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        // fifa-hybrid diagnostics: BB_VP_LOG=1 prints each distinct viewport depth mapping once.
        if (static const bool vp_log = std::getenv("BB_VP_LOG") != nullptr; vp_log) {
            static std::set<std::tuple<float, float, int>> seen;
            if (seen.emplace(zoffset, zscale, int(regs.clipper_control.clip_space)).second &&
                seen.size() < 40) {
                std::fprintf(stderr, "[VP] zoffset=%g zscale=%g clip_space=%d -> min=%g max=%g\n",
                             zoffset, zscale, int(regs.clipper_control.clip_space), viewport.minDepth,
                             viewport.maxDepth);
            }
        }
        // Metal has no unrestricted depth range (nor had MoltenVK): restrict to [0, 1].
        viewport.minDepth = std::max(viewport.minDepth, 0.f);
        viewport.maxDepth = std::min(viewport.maxDepth, 1.f);

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            // Metal's (and MoltenVK's) largest viewport is 16384.
            viewport.width = float(16_KB);
            viewport.height = float(16_KB);
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            // bbport: sub-pixel jitter of scene geometry for the temporal upscaler; the same
            // shift as jittering the projection.
            viewport.x = (xoffset - xscale) * target_scale[0] + draw_jitter[0];
            viewport.y = (yoffset - yscale) * target_scale[1] + draw_jitter[1];
            viewport.width = xscale * 2.0f * target_scale[0];
            viewport.height = yscale * 2.0f * target_scale[1];
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        if (target_scale[0] != 1.0f || target_scale[1] != 1.0f) {
            const auto scale = [](s32 v, float f) { return s32(std::lround(float(v) * f)); };
            const s32 x0 = scale(vp_scsr.top_left_x, target_scale[0]);
            const s32 y0 = scale(vp_scsr.top_left_y, target_scale[1]);
            const s32 x1 = scale(vp_scsr.top_left_x + s32(vp_scsr.GetWidth()), target_scale[0]);
            const s32 y1 = scale(vp_scsr.top_left_y + s32(vp_scsr.GetHeight()), target_scale[1]);
            scissors.push_back({
                .offset = {x0, y0},
                .extent = {u32(std::max(x1 - x0, 0)), u32(std::max(y1 - y0, 0))},
            });
        } else {
            scissors.push_back({
                .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
                .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
            });
        }
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = Regs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid() && !(BbShadowForce() & 1);
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        // fifa-hybrid A/B: BB_BIAS_SCALE=<f> scales the polygon offset (0 disables, -1 flips).
        static const float bias_scale = [] {
            const char* env = std::getenv("BB_BIAS_SCALE");
            return env ? float(std::atof(env)) : 1.0f;
        }();
        static const bool bias_log = std::getenv("BB_BIAS_LOG") != nullptr;
        const float constant = front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset;
        const float slope = (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f;
        if (bias_log) {
            static std::set<std::pair<float, float>> seen;
            if (seen.emplace(constant, slope).second && seen.size() < 40) {
                std::fprintf(stderr, "[BIAS] constant=%g slope=%g clamp=%g front=%d depth fmt %u\n",
                             constant, slope, regs.poly_offset.depth_bias, int(front),
                             u32(regs.depth_buffer.z_info.format));
            }
        }
        dynamic_state.SetDepthBias(constant * bias_scale, regs.poly_offset.depth_bias,
                                   slope * bias_scale);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        // GCN REPLACE_OP writes DB_STENCILREFMASK.STENCILOPVAL, so a face whose stencil ops
        // include ReplaceOp takes its Vulkan reference from op_val.
        const auto& sc = regs.stencil_control;
        const auto uses_op_val = [](AmdGpu::StencilFunc fail, AmdGpu::StencilFunc zpass,
                                    AmdGpu::StencilFunc zfail) {
            return fail == AmdGpu::StencilFunc::ReplaceOp ||
                   zpass == AmdGpu::StencilFunc::ReplaceOp ||
                   zfail == AmdGpu::StencilFunc::ReplaceOp;
        };
        const bool front_op =
            uses_op_val(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
        const bool back_op =
            regs.depth_control.backface_enable
                ? uses_op_val(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                : front_op;
        const auto ref_conflict = [](AmdGpu::CompareFunc func, const AmdGpu::StencilRefMask& ref) {
            return func != AmdGpu::CompareFunc::Always && func != AmdGpu::CompareFunc::Never &&
                   ref.stencil_test_val != ref.stencil_op_val;
        };
        if ((front_op && ref_conflict(regs.depth_control.stencil_ref_func, front)) ||
            (back_op && regs.depth_control.backface_enable &&
             ref_conflict(regs.depth_control.stencil_bf_func, back))) {
            LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                       "op_val; the stencil test will use op_val");
        }
        dynamic_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                           back_op ? back.stencil_op_val : back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = Regs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    // What MoltenVK reported (VK_EXT_primitive_topology_list_restart): restart in lists, not in
    // patch lists.
    constexpr bool list_restart = true, patch_list_restart = false;
    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (list_restart || !is_list_topology(regs.primitive_type)) &&
        (patch_list_restart || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill && !(BbShadowForce() & 4));
    dynamic_state.SetCullMode((BbShadowForce() & 2) ? vk::CullModeFlagBits::eNone : cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = Regs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = Regs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    DrainDrawPipe();
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    scheduler.PushDebugGroup(std::string(str));
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    DrainDrawPipe();
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    scheduler.PopDebugGroup();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    DrainDrawPipe();
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    scheduler.InsertDebugLabel(std::string(str));
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    DrainDrawPipe();
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    // Metal debug groups have no color.
    (void)color;
    scheduler.InsertDebugLabel(std::string(str));
}

std::thread::id Rasterizer::GetGpuCommandProcessorThread() {
    return liverpool->GetGpuCommandProcessorThread();
}

#ifdef __linux__
u32 Rasterizer::GetGpuCommandProcessorThreadId() {
    return liverpool->GetGpuCommandProcessorThreadId();
}
#endif

} // namespace Vulkan

namespace Vulkan {

// bbport: with BB_PASS_LOG=<file>, BB_PASS_LABELS=1 logs every render pass (size, targets,
// shaders) under its numbered command buffer, in recording order: the k-th render encoder of
// Metal command buffer "cb#N" in a Metal System Trace is the k-th pass logged for N
// (tools/pass_trace.py). BB_PASS_LABELS=2 also ends the pass after every draw, so each draw gets
// its own encoder (GPU time per draw; diagnostic only, the extra pass breaks cost time).
static int PassLabels() {
    static const int mode = [] {
        const char* env = std::getenv("BB_PASS_LABELS");
        return env ? std::atoi(env) : 0;
    }();
    return mode;
}

void Rasterizer::MidFrameFlush(const RenderState& state) {
    static const double every_ms = [] {
        const char* env = std::getenv("BB_MIDFRAME_FLUSH_MS");
        return env ? std::atof(env) : 0.0;
    }();
    if (every_ms <= 0.0 || !scheduler.WillBeginRendering(state)) {
        return;
    }
    static u64 last_tick = 0;
    static auto last_time = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (scheduler.CurrentTick() != last_tick) { // someone else committed since
        last_tick = scheduler.CurrentTick();
        last_time = now;
        return;
    }
    if (std::chrono::duration<double, std::milli>(now - last_time).count() < every_ms) {
        return;
    }
    scheduler.Flush();
    last_tick = scheduler.CurrentTick();
    last_time = now;
    BbStats::midframe_flushes.fetch_add(1, std::memory_order_relaxed);
}

void Rasterizer::MarkPass(const GraphicsPipeline* pipeline, const RenderState& state) {
    const bool pass_labels = PassLabels() != 0 && BbPassLog::Enabled();
    auto* profiler = GpuProfiler::Get();
    if ((!profiler && !pass_labels) || !scheduler.WillBeginRendering(state)) {
        return;
    }
    if (pass_labels) {
        std::string targets;
        for (u32 cb = 0; cb < state.num_colors; ++cb) {
            if (cb_descs[cb].first) {
                const auto& info = texture_cache.GetImage(cb_descs[cb].first).info;
                targets += fmt::format("{}@{:x}", vk::to_string(info.pixel_format),
                                       info.guest_address);
            } else {
                targets += '-';
            }
            targets += ' ';
        }
        const auto* ps_stage = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        auto name = std::make_shared<std::string>(fmt::format(
            "pass {}x{} [{}] depth {} vs {:016x} ps {:016x}", state.width, state.height, targets,
            db_desc.first ? vk::to_string(texture_cache.GetImage(db_desc.first).info.pixel_format)
                          : std::string{"-"},
            pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash, ps_stage ? ps_stage->pgm_hash : 0));
        // Commands recorded now go into the submission that signals the current tick.
        scheduler.Record([name, tick = scheduler.CurrentTick()](Metal::Encoder&) {
            BbPassLog::Pass(tick, *name);
        });
        if (!profiler) {
            return;
        }
    }
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    std::array<u64, AmdGpu::NUM_COLOR_BUFFERS + 6> parts{};
    parts[0] = state.width | u64(state.height) << 32;
    parts[1] = vs.pgm_hash;
    parts[2] = ps ? ps->pgm_hash : 0;
    parts[3] = db_desc.first ? u64(texture_cache.GetImage(db_desc.first).info.pixel_format) : 0;
    for (u32 cb = 0; cb < state.num_colors && cb < AmdGpu::NUM_COLOR_BUFFERS; ++cb) {
        parts[4 + cb] = cb_descs[cb].first
                            ? u64(texture_cache.GetImage(cb_descs[cb].first).info.pixel_format) + 1
                            : 0;
    }
    const u64 key = XXH3_64bits(parts.data(), sizeof(parts));
    profiler->Mark(key, [&] {
        std::string targets;
        for (u32 cb = 0; cb < state.num_colors; ++cb) {
            targets += cb_descs[cb].first
                           ? vk::to_string(texture_cache.GetImage(cb_descs[cb].first).info.pixel_format)
                           : std::string{"-"};
            targets += ' ';
        }
        return fmt::format("pass {}x{} [{}] depth {} vs {:016x} ps {:016x}", state.width,
                           state.height, targets,
                           db_desc.first ? vk::to_string(
                                               texture_cache.GetImage(db_desc.first).info.pixel_format)
                                         : std::string{"-"},
                           vs.pgm_hash, ps ? ps->pgm_hash : 0);
    });
}

void Rasterizer::NoteFrameStart() {
    if (auto* profiler = GpuProfiler::Get()) {
        profiler->BeginFrame();
    }
    const u64 frame = BbStats::gpu_frames.fetch_add(1, std::memory_order_relaxed) + 1;
    Metal::EncoderStatsFrame(); // BB_ENCODER_STATS
    // BB_BUFFER_STATS=1: how many earlier frames the GPU has not finished when the GPU thread
    // starts a frame — the lag that decides whether guest memory can be read in place.
    static const bool stats = [] {
        const char* value = std::getenv("BB_BUFFER_STATS");
        return value && value[0] == '1';
    }();
    if (!stats) {
        return;
    }
    static std::array<u64, 16> frame_ticks{};
    static std::array<u64, 17> lag_histogram{};
    u32 lag = 0;
    for (u32 back = 1; back < frame_ticks.size() && back < frame; ++back) {
        const u64 tick = frame_ticks[(frame - back) % frame_ticks.size()];
        if (tick && !scheduler.IsFree(tick)) {
            lag = back;
        }
    }
    ++lag_histogram[lag];
    frame_ticks[frame % frame_ticks.size()] = scheduler.CurrentTick();
    if (frame % 600 == 0) {
        std::printf("GPU lag at frame start (frames not finished: count):");
        for (u32 i = 0; i < lag_histogram.size(); ++i) {
            if (lag_histogram[i]) {
                std::printf(" %u:%llu", i, static_cast<unsigned long long>(lag_histogram[i]));
            }
        }
        std::printf("\n");
        lag_histogram = {};
    }
}


// 0.5.1-c: the copy dispatch as a fragment pass (info.h InfoPersistent::CopyDispatch,
// copy_dispatch_detect_pass.cpp). BB_COPY_CS_PASS=0 keeps the dispatch, BB_COPY_CS_LOG=1 logs
// why one stays a dispatch.
bool Rasterizer::PlanCopyDispatchPass(const Shader::Info& cs, u32 dim_x, u32 dim_y, u32 dim_z,
                                      CopyDispatchPlan& plan) {
    static const bool off = [] {
        const char* e = std::getenv("BB_COPY_CS_PASS");
        return e && e[0] == '0';
    }();
    const auto& cd = cs.copy_dispatch;
    static const bool log = std::getenv("BB_COPY_CS_LOG") != nullptr;
    static std::atomic<u32> logged{0};
    // valid 1 is a stored 0.5.1-c..f record whose swizzle belongs to another permutation.
    if (off || cd.valid != 2) {
        if (log && !off && u64(dim_x) * dim_y >= 4096 && logged.fetch_add(1) < 400) {
            std::fprintf(stderr, "[COPYCS] cs %016llx %ux%ux%u: no copy_dispatch in this info (%p, images %zu)\n",
                         (unsigned long long)cs.pgm_hash, dim_x, dim_y, dim_z, (const void*)&cs, cs.images.size());
        }
        return false;
    }
    const auto decline = [&](const char* why) {
        const auto dst_id = cd.dst_image < bound_image_ids.size() ? bound_image_ids[cd.dst_image] : VideoCore::ImageId{};
        const auto src_id = cd.src_image < bound_image_ids.size() ? bound_image_ids[cd.src_image] : VideoCore::ImageId{};
        // Small grids (mip chains) are not logged.
        if (log && u64(dim_x) * dim_y >= 4096 && logged.fetch_add(1) < 400) {
            const auto* dst = dst_id ? &texture_cache.GetImage(dst_id) : nullptr;
            void* const dv = cd.dst_image < bound_image_views.size() ? bound_image_views[cd.dst_image] : nullptr;
            void* const sv = cd.src_image < bound_image_views.size() ? bound_image_views[cd.src_image] : nullptr;
            std::fprintf(stderr,
                         "[COPYCS] cs %016llx %ux%ux%u stays a dispatch: %s (images %u/%u bound %zu, dst image %ux%u type "
                         "%u L%u M%u fmt %s; dst view %ux%u 2d=%d, src view %ux%u 2d=%d, same view %d)\n",
                         (unsigned long long)cs.pgm_hash, dim_x, dim_y, dim_z, why, dst_id.index, src_id.index,
                         bound_image_ids.size(), dst ? dst->info.size.width : 0, dst ? dst->info.size.height : 0,
                         dst ? u32(dst->info.type) : 0, dst ? dst->info.resources.layers : 0,
                         dst ? dst->info.resources.levels : 0,
                         dst ? vk::to_string(dst->info.pixel_format).c_str() : "-",
                         dv ? Metal::TextureWidth(dv) : 0, dv ? Metal::TextureHeight(dv) : 0, dv ? int(Metal::TextureIs2D(dv)) : -1,
                         sv ? Metal::TextureWidth(sv) : 0, sv ? Metal::TextureHeight(sv) : 0, sv ? int(Metal::TextureIs2D(sv)) : -1,
                         int(dv == sv));
        }
        return false;
    };
    if (dim_z != 1 || dim_x == 0 || dim_y == 0 || !bound_images_simple) {
        return decline("grid or image bindings");
    }
    if (UseDrawPipe() || FrameCapture::Active() || upscaler->RasterScaling()) {
        return decline("draw pipe, capture or upscaler scene");
    }
    if (cd.dst_image >= bound_image_ids.size() || cd.src_image >= bound_image_ids.size() ||
        cd.dst_image >= cs.images.size() || cd.src_image >= cs.images.size() ||
        cd.src_sampler >= cs.samplers.size() ||
        cd.src_sampler >= bound_samplers.size() || cd.scale_buffer >= cs.buffers.size() ||
        bound_image_views.size() != bound_image_ids.size()) {
        return decline("binding counts");
    }
    const auto dst_id = bound_image_ids[cd.dst_image], src_id = bound_image_ids[cd.src_image];
    void* const dst_view = bound_image_views[cd.dst_image];
    void* const src_view = bound_image_views[cd.src_image];
    void* const sampler = bound_samplers[cd.src_sampler];
    // One image for both is fine when the views differ (another level or layer; the compute
    // shader read and wrote those same views); the same view would be a copy onto itself.
    if (!dst_id || !src_id || !dst_view || !src_view || !sampler || dst_view == src_view) {
        return decline("null binding or same view");
    }
    const auto& dst = texture_cache.GetImage(dst_id);
    if (dst.info.props.is_depth || dst.info.props.is_block || dst.info.num_samples > 1 ||
        !Metal::TextureIs2D(dst_view) || !Metal::TextureIs2D(src_view)) {
        return decline("destination or source kind");
    }
    // The view's own size (a level or layer view of a larger image is smaller than level 0).
    const u32 dst_width = Metal::TextureWidth(dst_view), dst_height = Metal::TextureHeight(dst_view);
    if (u64(dim_x) * cd.wg_x != dst_width || u64(dim_y) * cd.wg_y != dst_height) {
        return decline("grid does not cover the destination");
    }
    const char* numeric = vk::componentNumericFormat(dst.info.pixel_format, 0);
    if (std::strcmp(numeric, "UINT") == 0 || std::strcmp(numeric, "SINT") == 0) {
        return decline("integer destination");
    }
    // The record was detected on one permutation of the program. What depends on the bound
    // resources is taken from them here: the destination's channel select (the dispatch stores
    // the shader's value through its inverse) and the conversions the recompiler would add,
    // which the pass does not do.
    const auto dst_sharp = cs.images[cd.dst_image].GetSharp(cs);
    const auto src_sharp = cs.images[cd.src_image].GetSharp(cs);
    if (dst_sharp.GetNumberConversion() != AmdGpu::NumberConversion::None ||
        src_sharp.GetNumberConversion() != AmdGpu::NumberConversion::None) {
        return decline("number conversion");
    }
    if (cs.samplers[cd.src_sampler].GetSharp(cs).force_degamma &&
        src_sharp.GetNumberFmt() != AmdGpu::NumberFormat::Srgb) {
        return decline("forced degamma");
    }
    u8 stored[4];
    if (!cd.Compose(dst_sharp.DstSelect(), stored)) {
        return decline("channel select");
    }
    const auto sharp = cs.buffers[cd.scale_buffer].GetSharp(cs);
    const VAddr base = sharp.base_address;
    if (base < 0x400000 ||
        sharp.GetSize() < (u64(std::max(cd.scale_dword_x, cd.scale_dword_y)) + 1) * 4) {
        return decline("scale buffer");
    }
    float sx = 0.0f, sy = 0.0f;
    std::memcpy(&sx, reinterpret_cast<const void*>(base + u64(cd.scale_dword_x) * 4), sizeof sx);
    std::memcpy(&sy, reinterpret_cast<const void*>(base + u64(cd.scale_dword_y) * 4), sizeof sy);
    if (!(sx > 0.0f) || !(sy > 0.0f) || !std::isfinite(sx) || !std::isfinite(sy)) {
        return decline("scale values");
    }
    plan = {
        .dst = dst_view,
        .src = src_view,
        .sampler = sampler,
        .width = dst_width,
        .height = dst_height,
        .scale = {sx, sy},
        .remap = cd.src_image < 32 && ((push_data.depth_remap >> cd.src_image) & 1u) != 0,
        .swizzle = u32(stored[0]) | (u32(stored[1]) << 4) | (u32(stored[2]) << 8) |
                   (u32(stored[3]) << 12),
    };
    return true;
}

// fifa-hybrid: a guest dispatch recorded ahead of its submission (BB_HOIST_CS).
namespace {
u32 HoistCsMaxGroups() {
    static const u32 v = [] {
        const char* e = std::getenv("BB_HOIST_CS");
        return e ? u32(std::strtoul(e, nullptr, 10)) : 64u;
    }();
    return v;
}
} // namespace

bool Rasterizer::TryHoistDispatch(const ComputePipeline* pipeline, u32 dim_x, u32 dim_y,
                                  u32 dim_z, u64 cs_hash) {
    const u32 max_groups = HoistCsMaxGroups();
    if (max_groups == 0) {
        return false;
    }
    static const bool log_declines = std::getenv("BB_DEFER_CS_LOG") != nullptr;
    static std::atomic<u32> logged{0};
    const auto decline = [&](const char* why) {
        // 0.5.1-d: grids over the cap are not logged (they filled the 400 lines at boot).
        if (log_declines && u64(dim_x) * dim_y * dim_z <= max_groups && logged.fetch_add(1) < 400) {
            std::fprintf(stderr, "[HOIST] cs %016llx %ux%ux%u not hoisted: %s\n",
                         (unsigned long long)cs_hash, dim_x, dim_y, dim_z, why);
        }
        return false;
    };
    if (!runtime.Ordering()) {
        return decline("no arena ordering");
    }
    if (!scheduler.IsRendering()) {
        return decline("no open pass"); // in place costs nothing
    }
    if (UseDrawPipe() || FrameCapture::Active() || upscaler->RasterScaling()) {
        return decline("draw pipe, capture or upscaler scene");
    }
    if (u64(dim_x) * dim_y * dim_z > max_groups || dim_x == 0 || dim_y == 0 || dim_z == 0) {
        return decline("size");
    }
    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs.uses_dma) {
        return decline("dma");
    }
    // Images: Metal orders the ahead command buffer against the in-order one itself (tracked
    // textures), so only this submission's own commands matter: nothing in it may have bound
    // the image (last_bound_tick), written it (last_write_tick before our own access: uploads,
    // copies, targets), and it may not be an attachment of the open pass or a block-shadow pair
    // (their copies are in order). Depth and multisampled images stay in place.
    {
        const u64 tick = scheduler.CurrentTick();
        for (size_t i = 0; i < bound_images.size(); ++i) {
            const auto id = bound_images[i];
            const auto& image = texture_cache.GetImage(id);
            if (id == db_desc.first) {
                return decline("depth attachment");
            }
            for (const auto& [cb_id, cb_desc] : cb_descs) {
                if (cb_id == id) {
                    return decline("colour attachment");
                }
            }
            if (image.shadow_of || image.block_shadow || image.info.props.is_depth ||
                image.info.num_samples > 1) {
                return decline("image kind");
            }
            if (image.last_bound_tick >= tick) {
                return decline("image used this submission");
            }
            const u64 prev_write = i < bound_images_prev_write.size() ? bound_images_prev_write[i] : tick;
            if (prev_write >= tick) {
                return decline("image written this submission");
            }
            const auto& base = image;
            if (DeferredDispatchTouches(base.info.guest_address, base.info.guest_size,
                                        i < bound_images_written.size() && bound_images_written[i])) {
                return decline("deferred dispatch ranges");
            }
        }
    }
    for (const auto& desc : cs.buffers) {
        if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
            return decline("gds");
        }
    }
    // Its ranges against the dispatches deferred to the pass end (they would run after it).
    for (const auto& b : bound_buffers) {
        if (DeferredDispatchTouches(b.buffer->CpuAddr() + b.offset, b.size, b.is_written)) {
            return decline("deferred dispatch ranges");
        }
    }
    for (const auto& b : bound_buffers) {
        if (!runtime.AheadSafe(b.buffer, b.offset, b.size, b.is_written)) {
            return decline(b.is_written ? "output in use" : "input in use");
        }
    }
    for (const auto& r : read_ranges) {
        if (!runtime.AheadSafe(r.buffer, r.offset, r.size, false)) {
            return decline("read range in use");
        }
    }
    if (!pipeline->IsReady()) {
        return decline("pipeline compiling"); // the recording thread waits for it, this thread not
    }
    const Metal::Draw::ComputeView view = ComputeView(*pipeline);
    if (!view.state || !view.shader) {
        return decline("no pipeline");
    }
    if (!ahead_nulls) {
        ahead_nulls.emplace(device);
    }
    const Metal::Size3 groups{dim_x, dim_y, dim_z};
    const Metal::Draw::Nulls nulls = ahead_nulls->Get();
    // Recorded by the recording thread (this thread is the command processor; in cutscenes
    // it is the limit) into the ahead encoder, under its lock. Uploads hoisted meanwhile may
    // land before it in that stream, so its ranges are noted as used by this submission: a
    // later upload over them is not hoisted (HoistCopy's unused() check) and runs in order.
    scheduler.Record([this, view, groups, nulls, bound_copy = bound, push_copy = push_data](
                         Metal::Encoder&) {
        const std::span<const Metal::Draw::Bound> bound_span{bound_copy};
        const std::span<const u8> push{reinterpret_cast<const u8*>(&push_copy), sizeof(push_copy)};
        scheduler.RecordAhead([&](Metal::Encoder& e) {
            e.SetComputePipeline(view.state, view.buffers);
            Metal::Draw::BindStage(e, Metal::Stage::Compute, *view.shader, bound_span, push, nulls);
            e.Dispatch(groups, view.threads_per_group);
        });
    });
    // Its accesses: by the ahead stream (later commands of this submission that touch them
    // wait for the uploads' fence; HoistCopy marks the same way).
    runtime.BeginAheadCommand();
    for (const auto& b : bound_buffers) {
        runtime.NoteSubmissionUse(b.buffer, b.offset, b.size);
        runtime.MarkAccess(b.buffer, b.offset, b.size, b.is_written, Metal::EncoderAhead);
    }
    for (const auto& r : read_ranges) {
        runtime.NoteSubmissionUse(r.buffer, r.offset, r.size);
        runtime.MarkAccess(r.buffer, r.offset, r.size, false, Metal::EncoderAhead);
    }
    for (const auto id : bound_images) {
        texture_cache.GetImage(id).binding = {}; // not last_bound_tick: it ran ahead
    }
    bound_images.clear();
    bound_buffers.clear();
    read_ranges.clear();
    bound_images_written.clear();
    bound_images_prev_write.clear();
    bound_image_ids.clear();
    bound_image_views.clear();
    bound_samplers.clear();
    bound_images_simple = true;
    BbStats::hoisted_dispatches.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// fifa-hybrid: dispatches deferred to the end of their render pass (BB_DEFER_CS).
namespace {
u32 DeferCsMaxGroups() {
    static const u32 v = [] {
        const char* e = std::getenv("BB_DEFER_CS");
        return e ? u32(std::strtoul(e, nullptr, 10)) : 64u;
    }();
    return v;
}
bool RangesMeet(VAddr a, u64 a_size, VAddr b_begin, VAddr b_end) {
    return a < b_end && b_begin < a + a_size;
}
/// BB_DEFER_CS_LOG=1: the first conflicts that flush deferred dispatches, with what met what.
void LogDeferConflict(const char* where, VAddr address, u64 size, bool write, VAddr r_begin,
                      VAddr r_end, bool r_write) {
    static const bool on = std::getenv("BB_DEFER_CS_LOG") != nullptr;
    static std::atomic<u32> logged{0};
    if (on && logged.fetch_add(1) < 300) {
        std::fprintf(stderr, "[DEFER] conflict %s: %s %#llx+%#llx meets deferred %s %#llx..%#llx\n",
                     where, write ? "write" : "read", (unsigned long long)address,
                     (unsigned long long)size, r_write ? "write" : "read",
                     (unsigned long long)r_begin, (unsigned long long)r_end);
    }
}
} // namespace

bool Rasterizer::DeferredDispatchTouches(VAddr address, u64 size, bool write) const {
    if (deferred_cs_ranges.empty() || size == 0) {
        return false;
    }
    for (const auto& r : deferred_cs_ranges) {
        if ((r.write || write) && RangesMeet(address, size, r.begin, r.end)) {
            LogDeferConflict(deferred_conflict_site ? deferred_conflict_site : "?", address, size,
                             write, r.begin, r.end, r.write);
            return true;
        }
    }
    return false;
}

bool Rasterizer::DeferredDispatchTouchesImage(const VideoCore::Image& image, bool write) {
    if (deferred_cs_ranges.empty()) {
        return false;
    }
    const VideoCore::Image& base = image.shadow_of ? texture_cache.GetImage(image.shadow_of) : image;
    return DeferredDispatchTouches(base.info.guest_address, base.info.guest_size, write);
}

bool Rasterizer::DeferredDispatchesConflict(const Pipeline* pipeline) const {
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        for (const auto& desc : stage->buffers) {
            if (desc.IsSpecial()) {
                continue; // not a guest range
            }
            const auto sharp = desc.GetSharp(*stage);
            if (DeferredDispatchTouches(sharp.base_address, sharp.GetSize(), desc.is_written)) {
                return true;
            }
        }
        for (const auto& desc : stage->images) {
            const auto sharp = desc.GetSharp(*stage);
            const VAddr address = sharp.Address();
            if (address == 0) {
                continue;
            }
            // An upper bound of the image's bytes: whole mip chain and all layers.
            const u64 texels = u64(sharp.width + 1) * (sharp.height + 1) *
                               std::max<u64>(sharp.depth + 1, 1) * sharp.NumLayers();
            const u64 bytes =
                texels * AmdGpu::NumBitsPerBlock(sharp.GetDataFmt()) / 8 * 4 / 3 + 256;
            if (DeferredDispatchTouches(address, bytes, desc.is_written)) {
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::TryDeferDispatch(const ComputePipeline* pipeline, u32 dim_x, u32 dim_y,
                                  u32 dim_z, u64 cs_hash) {
    const u32 max_groups = DeferCsMaxGroups();
    if (max_groups == 0) {
        return false;
    }
    // BB_DEFER_CS_LOG=1: the first declines, with their reason.
    static const bool log_declines = std::getenv("BB_DEFER_CS_LOG") != nullptr;
    static std::atomic<u32> logged{0};
    const auto decline = [&](const char* why) {
        if (log_declines && u64(dim_x) * dim_y * dim_z <= max_groups && logged.fetch_add(1) < 400) {
            std::fprintf(stderr, "[DEFER] cs %016llx %ux%ux%u not deferred: %s\n",
                         (unsigned long long)cs_hash, dim_x, dim_y, dim_z, why);
        }
        return false;
    };
    if (flushing_deferred) {
        return decline("flushing");
    }
    if (!scheduler.IsRendering()) {
        return decline("no open pass");
    }
    if (UseDrawPipe()) {
        return decline("draw pipe");
    }
    // The temporal upscaler's scene (reduced-size proxies, its trigger dispatch) keeps the
    // in-place order; a merely enabled upscaler (the default settings object) does not.
    if (FrameCapture::Active() || upscaler->RasterScaling()) {
        return decline("capture or upscaler scene");
    }
    if (DeferredCopiesPending()) {
        return decline("copies pending");
    }
    if (u64(dim_x) * dim_y * dim_z > max_groups || dim_x == 0 || dim_y == 0 || dim_z == 0) {
        return decline("size");
    }
    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    if (cs.uses_dma) {
        return decline("dma");
    }
    for (const auto& desc : cs.buffers) {
        // GDS is not a guest range (the read-constant and shared-memory stand-ins are private
        // to the dispatch).
        if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
            return decline("gds");
        }
    }
    // A dispatch that reads or writes an attachment of the open pass stays in place.
    for (const auto id : bound_images) {
        if (id == db_desc.first) {
            return decline("depth attachment");
        }
        for (const auto& [cb_id, cb_desc] : cb_descs) {
            if (cb_id == id) {
                return decline("colour attachment");
            }
        }
    }
    DeferredDispatch d{pipeline, bound, push_data, bound_buffers, read_ranges, dim_x, dim_y,
                       dim_z, cs_hash};
    for (const auto& b : bound_buffers) {
        const VAddr begin = b.buffer->CpuAddr() + b.offset;
        deferred_cs_ranges.push_back({begin, begin + b.size, b.is_written});
    }
    for (const auto& r : read_ranges) {
        const VAddr begin = r.buffer->CpuAddr() + r.offset;
        deferred_cs_ranges.push_back({begin, begin + r.size, false});
    }
    for (size_t i = 0; i < bound_images.size(); ++i) {
        const auto& image = texture_cache.GetImage(bound_images[i]);
        const auto& base = image.shadow_of ? texture_cache.GetImage(image.shadow_of) : image;
        const bool written = i < bound_images_written.size() && bound_images_written[i];
        deferred_cs_ranges.push_back(
            {base.info.guest_address, base.info.guest_address + base.info.guest_size, written});
    }
    deferred_dispatches.push_back(std::move(d));
    BbStats::deferred_dispatches.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void Rasterizer::FlushDeferredDispatches() {
    if (deferred_dispatches.empty() || flushing_deferred) {
        return;
    }
    if (scheduler.IsRendering()) {
        scheduler.EndRendering(); // the pass-end hook comes back here
        if (deferred_dispatches.empty()) {
            return;
        }
    }
    flushing_deferred = true;
    std::vector<DeferredDispatch> list;
    list.swap(deferred_dispatches);
    deferred_cs_ranges.clear();
    // The binding state of the command being bound (the pass-end hook runs between its
    // BindResources and its recording) is kept.
    auto saved_bound = std::move(bound);
    const auto saved_push = push_data;
    const auto saved_buffers = bound_buffers;
    const auto saved_reads = read_ranges;
    const auto saved_images = bound_images;
    const auto saved_images_written = bound_images_written;
    const bool saved_compute = pipeline_is_compute;
    const u8 saved_encoders = draw_encoders;
    bound_images.clear();
    bound_images_written.clear();
    for (auto& d : list) {
        bound = std::move(d.bound);
        push_data = d.push;
        bound_buffers = d.buffers;
        read_ranges = d.reads;
        pipeline_is_compute = true;
        draw_encoders = Metal::EncoderCompute;
        if (auto* profiler = GpuProfiler::Get()) {
            profiler->Mark(d.cs_hash ^ 0xD15Aull, [&] {
                return fmt::format("dispatch cs {:016x} ({}x{}x{} groups, deferred)", d.cs_hash,
                                   d.dim_x, d.dim_y, d.dim_z);
            });
        }
        OrderAccesses();
        const Metal::Size3 groups{d.dim_x, d.dim_y, d.dim_z};
        const Metal::Draw::Resources resources = RecordResources(false);
        scheduler.RecordDraw(
            [=, runs = &scheduler.GetDrawRuns(), pipeline = d.pipeline](Metal::Encoder& e) {
                runs->Dispatch(e, ComputeView(*pipeline), resources.bound, resources.push, groups);
            },
            false);
        ResetBindings(); // marks the accesses where the dispatch really runs
    }
    bound = std::move(saved_bound);
    push_data = saved_push;
    bound_buffers = saved_buffers;
    read_ranges = saved_reads;
    bound_images = saved_images;
    bound_images_written = saved_images_written;
    pipeline_is_compute = saved_compute;
    draw_encoders = saved_encoders;
    flushing_deferred = false;
    scheduler.KickRecording();
}

} // namespace Vulkan
