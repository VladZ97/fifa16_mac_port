// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
// SPDX-License-Identifier: GPL-2.0-or-later

// bbport (Metal-native M4): the presenter on the window's CAMetalLayer. The post-process pass
// draws the guest's display buffer into a frame (the game image fitted into the window, so the
// present is a 1:1 copy) on the draw scheduler; Present draws the frame letterboxed into a
// drawable on a second command queue that waits for the frame's device tick. The shape follows
// Huax-1839/bloodborne-macos's mtl_presenter.mm (GPL-2.0-or-later), without frame interpolation.

#include <array>
#include <ctime>
#include "bbport_pacing.h"
#include "bbport_cpu.h"
#include "bbport_gpu_capture.h"
#include "bbport_overlay.h"
#include "bbport_settings.h"
#include "bbport_toggles.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "sdl_window.h"
#include "video_core/renderer_metal/mtl_pso_cache.h"
#include "video_core/renderer_metal/mtl_texture.h"
#include "video_core/renderer_metal/vk_to_mtl.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_temporal_upscaler.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <optional>
#include <unistd.h>

namespace Vulkan {

namespace {

/// MTLPixelFormatBGRA8Unorm: frames and drawables (the post-process pass writes gamma-encoded
/// values, as into MoltenVK's swapchain before).
constexpr u32 FrameFormat = 80;

u64 NowNs() {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}

Metal::LayerConfig LayerConfigFromEnv() {
    Metal::LayerConfig config{};
    config.pixel_format = FrameFormat;
    // MoltenVK has no Mailbox and 0.4.0 fell back to Fifo: only Immediate turns display sync off.
    config.display_sync = EmulatorSettings.GetPresentMode() != "Immediate";
    // Two drawables up to a 60 FPS limit, as the Vulkan swapchain had (measured 2026-10-14:
    // 48.5 -> 31.7 ms present latency at the same frame rate as three). Above it, three: with two
    // on a 120 Hz display every frame longer than one refresh waits a whole extra refresh for a
    // drawable, which locked 90 and 120 FPS limits at 60 (2026-10-17). BB_SWAPCHAIN_IMAGES
    // overrides, CAMetalLayer takes 2..3.
    const u32 limit = EmulatorSettings.GetFrameLimit();
    config.max_drawables = limit == 0 || limit > 60 ? 3 : 2;
    if (const char* env = std::getenv("BB_SWAPCHAIN_IMAGES")) {
        config.max_drawables = u32(std::clamp(std::atoi(env), 2, 3));
    }
    return config;
}

// Full screen: three. macOS shows a full-screen layer's frames directly, without the window
// compositor that otherwise absorbs a late frame, so with two a frame that misses a refresh
// also holds up the next nextDrawable: 10-18 late frames per run against 0-2 with three at the
// 60 cap on an M4 Max, and 30-40 FPS on a MacBook Air M5 (#25, 2026-10-21). BB_SWAPCHAIN_IMAGES
// overrides here too.
/// fifa-hybrid: the game's display tick is 100 Hz or more (BB_VBLANK_HZ=120, no frame cap).
bool FastTick() {
    return EmulatorSettings.GetFrameLimit() == 0 && EmulatorSettings.GetVblankFrequency() >= 100;
}

u32 FullscreenDrawables(const Metal::LayerConfig& config) {
    return std::getenv("BB_SWAPCHAIN_IMAGES") ? config.max_drawables : 3;
}

vk::Format GetFrameViewFormat(const Libraries::VideoOut::PixelFormat format) {
    switch (format) {
    case Libraries::VideoOut::PixelFormat::A8B8G8R8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A8R8G8B8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A2R10G10B10:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq:
        return vk::Format::eA2R10G10B10UnormPack32;
    default:
        break;
    }
    UNREACHABLE_MSG("Unknown format={}", static_cast<u32>(format));
    return {};
}

} // namespace

Frame* Presenter::FramePool::Take() {
    std::unique_lock lock{mutex};
    cv.wait(lock, [this] { return !free.empty(); });
    Frame* const frame = free.front();
    free.pop_front();
    return frame;
}

void Presenter::FramePool::Free(Frame* frame) {
    {
        std::scoped_lock lock{mutex};
        free.push_back(frame);
    }
    cv.notify_one();
}

Presenter::Presenter(Frontend::WindowSDL& window_, AmdGpu::Liverpool* liverpool_)
    : window{window_}, liverpool{liverpool_}, draw_scheduler{device, true},
      runtime{device, draw_scheduler},
      rasterizer{std::make_unique<Rasterizer>(device, draw_scheduler, runtime, liverpool)},
      texture_cache{rasterizer->GetTextureCache()},
      layer{window.GetWindowInfo().render_surface}, pp_pass{device, FrameFormat, FrameFormat},
      present_queue{std::make_unique<Metal::PresentQueue>(device)},
      pool{std::make_shared<FramePool>()} {
    ASSERT_MSG(layer != nullptr, "Presenter: the window has no CAMetalLayer");
    if (const char* env = std::getenv("BB_SNAPSHOT_TRIGGER"); env && *env) {
        snapshot_trigger = env;
    }
    Metal::LayerConfig config = LayerConfigFromEnv();
    // Snapshots read the drawable back.
    config.framebuffer_only = snapshot_trigger.empty();
    Metal::ConfigureLayer(device, layer, config);
    // 0.5.1-f: keep the display on a steady refresh grid (see StartSteadyRefresh).
    // BB_REFRESH_BEAT=0 leaves the display alone, BB_DISPLAY_60=1 uses its 60 Hz mode.
    // fifa-hybrid (0.5.1-g): with a 120 Hz game tick (BB_VBLANK_HZ=120) the frames themselves
    // keep the display at its full rate, and the heartbeat's composited path would cap them:
    // three drawables go round in about four refreshes there (80 to 90 frames a second,
    // measured 80 with a full queue against 116 without the heartbeat).
    const bool fast_tick = FastTick();
    if (const char* env = std::getenv("BB_REFRESH_BEAT");
        env ? env[0] != '0' : !fast_tick) {
        const char* fixed = std::getenv("BB_DISPLAY_60");
        // BB_REFRESH_BEAT=link: a display link asks for the full rate instead of the layer.
        Metal::StartSteadyRefresh(layer, fixed && fixed[0] == '1', env && env[0] == 'l');
    }
    windowed_drawables = layer_drawables = config.max_drawables;
    fullscreen_drawables = FullscreenDrawables(config);
    // The settings menu and FPS counter, drawn into the drawable (Present).
    BbOverlay::Init(device, FrameFormat);
    for (u32 i = 0; i < FramePool::Count; ++i) {
        pool->frames[i].id = i;
        pool->free.push_back(&pool->frames[i]);
    }
    fsr_settings.enable = EmulatorSettings.IsFsrEnabled();
    fsr_settings.use_rcas = EmulatorSettings.IsRcasEnabled();
    fsr_settings.rcas_attenuation = static_cast<float>(EmulatorSettings.GetRcasAttenuation() / 1000.f);
    if (fsr_settings.enable) {
        fsr_pass = std::make_unique<HostPasses::FsrPass>(device);
    }
    std::printf("Presenter: CAMetalLayer, %u drawables (%u in full screen), display sync %s, "
                "frame limit %u\n",
                config.max_drawables, fullscreen_drawables, config.display_sync ? "on" : "off",
                EmulatorSettings.GetFrameLimit());
}

Presenter::~Presenter() {
    // Bounded: a present stuck behind a device tick is logged and left behind (its completion
    // handler only touches the frame pool, which it keeps alive).
    present_queue.reset();
    draw_scheduler.Finish();
    if (rasterizer) {
        rasterizer->GetPipelineCache().Sync();
    }
}

bool Presenter::FlushPipelineArchive() {
    return rasterizer->GetPipelineCache().Psos().Flush();
}

bool Presenter::IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const {
    return std::ranges::find(vo_buffers_addr, color_buffer.Address()) != vo_buffers_addr.cend();
}

Frame* Presenter::GetRenderFrame(u32 width, u32 height) {
    Frame* const frame = pool->Take();
    if (!frame->texture || frame->width != width || frame->height != height) {
        // Command buffers that still use the old texture hold references of their own
        // (Encoder::KeepAlive in the post-process pass; the present command buffer).
        Metal::TextureDesc desc{};
        desc.pixel_format = FrameFormat;
        desc.width = width;
        desc.height = height;
        desc.render_target = true;
        frame->texture = Metal::CreateTexture(device, desc, "Frame", nullptr);
        ASSERT_MSG(frame->texture, "Presenter: no {}x{} frame texture", width, height);
        frame->width = width;
        frame->height = height;
    }
    frame->is_hdr = false;
    return frame;
}

Frame* Presenter::PrepareLastFrame() {
    std::scoped_lock lock{pool->mutex};
    Frame* const frame = pool->last_presented;
    const auto it = std::ranges::find(pool->free, frame);
    if (frame == nullptr || it == pool->free.end()) {
        return nullptr;
    }
    pool->free.erase(it);
    return frame;
}

Frame* Presenter::PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                               VAddr cpu_address) {
    // bbport: scaled upscaler presets: the output-size display buffer drawn by the port.
    TemporalUpscaler::Display display{};
    const bool upscaled = rasterizer->GetUpscaler().DisplayOverride(cpu_address, display);
    const vk::Format view_format = GetFrameViewFormat(attribute.attrib.pixel_format);

    // The source, viewed with its sRGB format and alpha One (the frame must not blend with UI).
    Metal::Object input;
    u32 image_width = 0;
    u32 image_height = 0;
    if (upscaled) {
        // The display buffer has the guest render target's format; the video-out format views
        // it as Vulkan's mutable-format view did (BGRA view of an RGBA buffer: the texture has
        // PixelFormatView usage).
        const u32 format =
            Metal::SafeViewPixelFormat(Metal::TexturePixelFormat(display.texture),
                                       Metal::VkToMtl::PixelFormat(view_format), true);
        input = Metal::CreateTextureView(
            display.texture, format, Metal::TextureType::Tex2D, 0, 1, 0, 1,
            {Metal::VkToMtl::SwizzleRed, Metal::VkToMtl::SwizzleGreen,
             Metal::VkToMtl::SwizzleBlue, Metal::VkToMtl::SwizzleOne});
        ASSERT_MSG(input, "Presenter: no view of the upscaled display buffer");
        image_width = display.width;
        image_height = display.height;
    } else {
        auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
        const VideoCore::ImageId image_id = texture_cache.FindImage(desc);
        texture_cache.UpdateImage(image_id);
        auto& image = texture_cache.GetImage(image_id);
        VideoCore::ImageViewInfo view_info{};
        view_info.format = view_format;
        view_info.mapping.a = vk::ComponentSwizzle::eOne;
        // The view belongs to the image's view cache: one more reference for the command buffer.
        input = Metal::Object::Retain(image.FindView(view_info).Texture());
        image_width = image.info.size.width;
        image_height = image.info.size.height;
        runtime.AccessImage(&image, false);
    }

    // The frame is the image fitted into the window (pixels), so Present copies it 1:1.
    const Metal::FitRect fit = Metal::FitImage(image_width, image_height,
                                               u32(std::max(window.GetWidth(), 0)),
                                               u32(std::max(window.GetHeight(), 0)));
    const u32 frame_width = fit.width ? fit.width : std::max(image_width, 1u);
    const u32 frame_height = fit.height ? fit.height : std::max(image_height, 1u);
    Frame* const frame = GetRenderFrame(frame_width, frame_height);

    // Vulkan had no sRGB variant of the 10-bit format, so an A2R10G10B10Srgb buffer reaches the
    // post-process pass still sRGB encoded and is decoded there.
    pp_settings.srgb_input =
        attribute.attrib.pixel_format == Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb;
    // 0.5.1-e: edge anti-aliasing of the game's picture at its own resolution, before any
    // scaling (BB_AA=0 turns it off; BB_AA_SUBPIX / BB_AA_EDGE / BB_AA_EDGE_MIN tune it;
    // BB_AA_SPLIT=<0..1> leaves the left part of the picture untouched for comparison).
    struct AaConfig {
        bool enabled{true};
        HostPasses::PostProcessingPass::AaSettings settings{};
    };
    static const AaConfig aa = [] {
        AaConfig config{};
        const auto read = [](const char* name, float& value, float lo, float hi) {
            if (const char* env = std::getenv(name)) {
                value = std::clamp(std::strtof(env, nullptr), lo, hi);
            }
        };
        if (const char* env = std::getenv("BB_AA")) {
            config.enabled = env[0] != '0';
        }
        read("BB_AA_SUBPIX", config.settings.subpix, 0.0f, 1.0f);
        read("BB_AA_EDGE", config.settings.edge_threshold, 0.03f, 0.5f);
        read("BB_AA_EDGE_MIN", config.settings.edge_threshold_min, 0.0f, 0.2f);
        read("BB_AA_SPLIT", config.settings.split, 0.0f, 1.0f);
        if (const char* env = std::getenv("BB_AA_QUALITY")) {
            config.settings.quality = u32(std::clamp(std::atoi(env), 0, 2));
        }
        return config;
    }();
    // Measurement aid: BB_AA_ALT=<seconds> cycles the pass off / fast / quality every that many seconds
    // and logs each switch, so one run gives paired GPU times at the same content.
    static const double aa_alt = [] {
        const char* env = std::getenv("BB_AA_ALT");
        return env ? std::max(0.0, std::strtod(env, nullptr)) : 0.0;
    }();
    bool run_aa = aa.enabled;
    auto aa_settings = aa.settings;
    if (aa_alt > 0.0) {
        // Three phases: off, 5 search steps, 12 search steps.
        static int last_phase = -1;
        const double now = double(bbport::CyclesToNs(bbport::CpuCycles())) * 1e-9;
        const int phase = int(u64(now / aa_alt) % 3u);
        if (phase != last_phase) {
            last_phase = phase;
            std::fprintf(stderr, "[AA] %s\n", phase == 0 ? "off" : phase == 1 ? "q0" : "q2");
        }
        run_aa = run_aa && phase != 0;
        aa_settings.quality = phase == 1 ? 0u : 2u;
    }
    if (run_aa && !upscaled && !frame->is_hdr) {
        input = pp_pass.EdgeAa(draw_scheduler, std::move(input), image_width, image_height,
                               aa_settings);
    }
    if (fsr_pass) {
        input = fsr_pass->Render(draw_scheduler, std::move(input), {image_width, image_height},
                                 {frame->width, frame->height}, fsr_settings, frame->is_hdr);
    }
    // fifa-hybrid: BB_GAMMA=<factor> scales the game's display gamma (above 1 brightens).
    static const float user_gamma = [] {
        const char* env = std::getenv("BB_GAMMA");
        const float value = env ? std::strtof(env, nullptr) : 1.0f;
        return value > 0.1f && value < 3.0f ? value : 1.0f;
    }();
    auto settings = pp_settings;
    settings.gamma *= user_gamma;
    pp_pass.Render(draw_scheduler, std::move(input), Metal::Object::Retain(frame->texture.Get()),
                   frame->width, frame->height, settings);

    // Flush frame creation commands.
    frame->ready_tick = draw_scheduler.CurrentTick();
    const u64 flush_start = bbport::CpuCycles();
    // fifa-hybrid: the GPU profiler's frame boundary (NoteFrameStart only runs with the motion
    // passes on, which FIFA has off): one frame per guest flip.
    if (auto* profiler = GpuProfiler::Get()) {
        profiler->BeginFrame();
    }
    draw_scheduler.Flush();
    const u64 flush_end = bbport::CpuCycles();
    BbStats::flip_submit_ns.fetch_add(bbport::CyclesToNs(flush_end - flush_start),
                                      std::memory_order_relaxed);
    if (BbStats::enabled) {
        WatchFrameCompletion(frame->ready_tick, flush_end);
    }

    // bbport: the GPU command thread runs at most BB_FRAMES_AHEAD (default 1) guest frames
    // ahead of the GPU: it waits here for the frame that many flips back. When the GPU is the
    // bottleneck it finishes frames at an even rate; without this bound the command thread ran
    // ahead and then blocked wherever a resource ran out, so flips (and the guest's frame
    // timing) came in bursts: 12.5/25 ms alternation at 80 FPS. 0 turns it off.
    static const u32 frames_ahead = [] {
        const char* env = std::getenv("BB_FRAMES_AHEAD");
        return env ? u32(std::max(0, std::atoi(env))) : 1u;
    }();
    if (frames_ahead) {
        recent_frame_ticks.push_back(frame->ready_tick);
        while (recent_frame_ticks.size() > frames_ahead) {
            const u64 tick = recent_frame_ticks.front();
            recent_frame_ticks.pop_front();
            if (recent_frame_ticks.size() == frames_ahead) {
                const u64 wait_start = bbport::CpuCycles();
                draw_scheduler.Wait(tick);
                BbStats::frames_ahead_ns.fetch_add(
                    bbport::CyclesToNs(bbport::CpuCycles() - wait_start), std::memory_order_relaxed);
            }
        }
    }
    return frame;
}

// bbport: frame timeline (BB_FRAME_STATS): a helper thread waits for each flip's submission
// and adds the time from the commit's return to the GPU finishing it.
void Presenter::WatchFrameCompletion(u64 tick, u64 submitted) {
    std::scoped_lock lk{watch_mutex};
    if (!watch_thread.joinable()) {
        watch_thread = std::jthread([this](std::stop_token stop) {
            Common::SetCurrentThreadName("bb:FrameWatch");
            u64 last_done = 0;
            while (!stop.stop_requested()) {
                std::pair<u64, u64> item;
                {
                    std::unique_lock wl{watch_mutex};
                    watch_cv.wait(wl, stop, [this] { return !watch_queue.empty(); });
                    if (stop.stop_requested()) {
                        return;
                    }
                    item = watch_queue.front();
                    watch_queue.pop_front();
                }
                draw_scheduler.GetTimeline().Wait(item.first);
                const u64 done = bbport::CpuCycles();
                BbStats::gpu_latency_ns.fetch_add(bbport::CyclesToNs(done - item.second),
                                                  std::memory_order_relaxed);
                if (last_done) {
                    BbStats::gpu_done_gap_ns.fetch_add(bbport::CyclesToNs(done - last_done),
                                                       std::memory_order_relaxed);
                    BbStats::gpu_done_frames.fetch_add(1, std::memory_order_relaxed);
                }
                last_done = done;
            }
        });
    }
    if (watch_queue.size() < 16) {
        watch_queue.emplace_back(tick, submitted);
        watch_cv.notify_one();
    }
}

Frame* Presenter::PrepareBlankFrame(bool) {
    // No content: Present clears the drawable black. No scheduler work, so either thread may ask.
    Frame* const frame = pool->Take();
    frame->ready_tick = 0;
    return frame;
}

bool Presenter::SnapshotRequested() const {
    return !snapshot_trigger.empty() && access(snapshot_trigger.c_str(), F_OK) == 0;
}

std::string Presenter::TakeSnapshotRequest() {
    if (!SnapshotRequested()) {
        return {};
    }
    std::string path;
    if (FILE* file = std::fopen(snapshot_trigger.c_str(), "r")) {
        char line[1024]{};
        if (std::fgets(line, sizeof(line), file)) {
            path = line;
            while (!path.empty() && (path.back() == '\n' || path.back() == '\r')) {
                path.pop_back();
            }
        }
        std::fclose(file);
    }
    std::remove(snapshot_trigger.c_str());
    return path.empty() ? snapshot_trigger + ".png" : path;
}

namespace {
/// bbport: frame pacing (bbport.ini even_frames, default on since 0.5.1-f, on displays whose
/// steady refresh is 120 or 240 Hz): the minimum time each frame stays
/// on screen, the frame cap's period. On variable-refresh displays (ProMotion) frames at a steady
/// cap otherwise alternate between neighbouring refresh slots (12.5 / 20.8 ms at 60 FPS on a
/// MacBook Pro); with it they are shown for one period each. It holds frames until their slot:
/// about one more frame of input latency (measured +23 ms at 60 FPS). Slightly under the period
/// (97%), so a frame on time is never held back to the next refresh.
double PacingSeconds() {
    // BB_FRAME_PACING=<0|1> overrides the setting, BB_PACING_FACTOR the share of the period.
    static const int forced = [] {
        const char* env = std::getenv("BB_FRAME_PACING");
        return env ? int(std::atoi(env) != 0) : -1;
    }();
    static const double factor = [] {
        const char* env = std::getenv("BB_PACING_FACTOR");
        const double v = env ? std::atof(env) : 0.0;
        return v > 0.1 && v <= 1.0 ? v : 0.97;
    }();
    if (forced == 0) {
        return 0;
    }
    if (forced < 0 && FastTick()) {
        return 0; // 120 Hz mode: every refresh may show a new frame, nothing to hold back
    }
    if (forced < 0) {
        // On a 60 Hz display every refresh shows one frame anyway; on a 120 Hz grid (see
        // Metal::StartSteadyRefresh) the minimum duration is what keeps a frame for two.
        const int hz = Metal::SteadyRefreshHz();
        if (!BbSettings::Get().frame_pacing.load(std::memory_order_relaxed) || hz < 100 ||
            hz % 60 != 0) {
            return 0;
        }
    }
    // Without a frame cap (FIFA 16: frame limit 0) the game's own rate is the vblank's.
    const u32 limit = EmulatorSettings.GetFrameLimit();
    const u32 fps = limit ? limit : EmulatorSettings.GetVblankFrequency();
    return fps ? factor / fps : 0;
}

/// fifa-hybrid (0.5.1-f): keeps finished frames from piling up in front of the display.
/// The game is paced by the host's own 60 Hz tick, the display by its own clock. When the game
/// runs a hair faster than the display (or delivers two frames close together after a slow one)
/// finished frames queue up, the queue never drains, and every later frame reaches the screen
/// 80 to 98 ms after its flip. The lock measures how long finished frames wait for their slot
/// (end of GPU work to shown) and moves the tick by at most 0.2 ms a frame until the shortest
/// recent wait sits at the target: one spare frame in front of the display, no more.
///
/// Runs with paced presents (120 Hz grid) and on 60 Hz displays, when "Even frame times" is on.
/// History: a target of 8 ms was below what the display path can do and slowed the game for
/// good; 26 ms worked on paper, but every frame the game delivered a few ms late then missed
/// its slot. Screen captures make the window server skip a refresh (the test harness took one
/// every 3.3 s, which looked like a display fault for a day): measure in capture-free stretches.
/// BB_PACING_LOCK=0 disables (1 forces it without paced presents), BB_PACING_TARGET_MS sets the
/// target (default 40), BB_PACING_TRACE=1 logs every frame.
class PacingLock {
public:
    void Note(u64 shown_ns, u64 gpu_end_ns, u64 flip_ns, bool paced) {
        if (!shown_ns) {
            return;
        }
        const double kPeriodMs = PeriodMs();
        const bool from_flip = FromFlip();
        const float flip_wait_ms =
            flip_ns && shown_ns >= flip_ns ? float(double(shown_ns - flip_ns) * 1e-6) : 0.f;
        std::scoped_lock lock{mutex};
        // By default with paced presents (120 Hz grid) and on 60 Hz displays, where a full
        // drawable queue is the same 50 ms of latency without any pacing.
        active = Mode() == 1 ||
                 (Mode() < 0 && (paced || ((Metal::SteadyRefreshHz() == 60 || FastTick()) &&
                                           BbSettings::Get().frame_pacing.load(
                                               std::memory_order_relaxed))));
        if (!active && BbPacing::g_slew_ns.load(std::memory_order_relaxed) != 0) {
            BbPacing::SetTickSlew(0);
        }
        if (Tracing()) {
            // BB_PACING_TRACE=1: shown interval / wait / GPU-end interval / tick slew, per frame.
            trace.push_back({last_shown_ns ? float((shown_ns - last_shown_ns) * 1e-6) : 0.f,
                             gpu_end_ns ? float((double(shown_ns) - double(gpu_end_ns)) * 1e-6) : 0.f,
                             last_gpu_end_ns && gpu_end_ns
                                 ? float((double(gpu_end_ns) - double(last_gpu_end_ns)) * 1e-6)
                                 : 0.f,
                             float(BbPacing::g_slew_ns.load(std::memory_order_relaxed) * 1e-6),
                             flip_wait_ms});
        }
        if (!active) {
            last_shown_ns = shown_ns;
            last_gpu_end_ns = gpu_end_ns;
            return;
        }
        const u64 last = last_shown_ns;
        last_shown_ns = shown_ns;
        last_gpu_end_ns = gpu_end_ns;
        ++frames;
        const bool late = last && shown_ns > last &&
                          double(shown_ns - last) * 1e-6 > kPeriodMs * 1.3;
        const bool valid = from_flip ? flip_wait_ms > 0.f && flip_wait_ms < 250.f
                                     : gpu_end_ns && shown_ns >= gpu_end_ns &&
                                           shown_ns - gpu_end_ns < 200'000'000ull;
        if (late || !valid) {
            if (late) {
                ++late_frames;
            }
            ring_n = 0; // the slots moved: earlier waits say nothing about the next frames
            BbPacing::SetTickSlew(0);
            return;
        }
        const double wait_ms =
            from_flip ? double(flip_wait_ms) : double(shown_ns - gpu_end_ns) * 1e-6;
        ring[ring_at] = wait_ms;
        ring_at = (ring_at + 1) % kRing;
        ring_n = std::min<u32>(ring_n + 1, kRing);
        wait_sum_ms += wait_ms;
        wait_min_ms = std::min(wait_min_ms, wait_ms);
        ++wait_n;
        if (ring_n < kRing) {
            BbPacing::SetTickSlew(0);
            return;
        }
        double least = ring[0];
        for (u32 i = 1; i < kRing; ++i) {
            least = std::min(least, ring[i]);
        }
        const double error_ms = least - BaseTarget();
        double slew_ms = 0;
        if (std::abs(error_ms) > 0.4) {
            // Later quickly (a queue costs latency on every frame), earlier only gently: a
            // short wait is usually a slow scene, which an earlier tick does not help.
            slew_ms = std::clamp(error_ms * 0.08, -0.1, 0.2);
        }
        BbPacing::SetTickSlew(s64(slew_ms * 1e6));
    }

    void Print() {
        std::scoped_lock lock{mutex};
        if (!trace.empty()) {
            timespec now{};
            clock_gettime(CLOCK_REALTIME, &now);
            std::printf("PTT: %lld.%03ld\nPT:", (long long)now.tv_sec, now.tv_nsec / 1'000'000);
            for (const auto& t : trace) {
                std::printf(" %.1f/%.1f/%.1f/%.2f/%.1f", t[0], t[1], t[2], t[3], t[4]);
            }
            std::printf("\n");
            trace.clear();
        }
        if (!active || !frames) {
            return;
        }
        std::printf("Pacing lock: target %.1f ms %s, wait of %u on-time frames "
                    "least %.1f mean %.1f ms, late %u of %u, ticks moved +%.1f / -%.1f ms\n",
                    BaseTarget(), FromFlip() ? "from the flip" : "from GPU end", wait_n, wait_n ? wait_min_ms : 0.0,
                    wait_n ? wait_sum_ms / wait_n : 0.0, late_frames, frames,
                    BbPacing::g_slewed_fwd_ns.exchange(0) * 1e-6,
                    BbPacing::g_slewed_back_ns.exchange(0) * 1e-6);
        wait_n = 0;
        wait_sum_ms = 0;
        wait_min_ms = 1e9;
        late_frames = frames = 0;
    }

    /// True when the lock runs without paced presents too (needs the presented callback).
    static bool Wanted() {
        return Mode() == 1 || Tracing() ||
               (Mode() < 0 && (Metal::SteadyRefreshHz() == 60 || FastTick()) &&
                BbSettings::Get().frame_pacing.load(std::memory_order_relaxed));
    }

private:
    /// BB_PACING_LOCK: 0 never, 1 always, unset (-1) with paced presents only.
    static int Mode() {
        static const int mode = [] {
            const char* env = std::getenv("BB_PACING_LOCK");
            return !env ? -1 : env[0] == '0' ? 0 : 1;
        }();
        return mode;
    }
    static bool Tracing() {
        static const bool on = [] {
            const char* env = std::getenv("BB_PACING_TRACE");
            return env && env[0] == '1';
        }();
        return on;
    }
    /// The display path needs a finished frame about one refresh plus 12 to 16 ms ahead of its
    /// slot (measured: frames slip under 21 to 24 ms on the 120 Hz grid, under 28 ms at 60 Hz).
    /// 40 ms leaves a whole game frame of room on the 120 Hz grid, so a frame the game delivers
    /// one tick late is still shown in its slot. Measured in capture-free match play: 26 ms
    /// 97.6% of frames on cadence (1.4 late a second), 34 ms 99.9%, 40 ms 100.0%, for 44, 52
    /// and 53 ms from flip to screen (a full queue: 78 to 88 ms).
    /// 0.5.1-g, measured from the flip (120 s capture-free match stretches; flip to screen
    /// includes about 5 ms before the flip packet). Late frames a second per run:
    /// 50 ms: 0.05, 0.03, 0.03 at 54 ms from flip to screen; 44 ms: 0.08, 0.27, 0.13, 0.08 at
    /// 49 ms; 40 ms: 0.88 at 46 ms; 36 ms: 0.38 at 42 ms; the old measure: 0.14 at 73 ms.
    /// 50 is both more even and 19 ms shorter than the old measure; 44 trades the evenness
    /// gain for 5 ms more (BB_PACING_TARGET_MS=44). 120 Hz tick: 26 ms, 30 to 31 ms.
    static double BaseTarget() {
        static const double forced = [] {
            const char* env = std::getenv("BB_PACING_TARGET_MS");
            const double v = env ? std::atof(env) : 0.0;
            return v >= 4.0 && v <= 60.0 ? v : 0.0;
        }();
        if (forced > 0) {
            return forced;
        }
        if (!FromFlip()) {
            return 40.0;
        }
        return FastTick() ? 26.0 : 50.0;
    }
    /// 0.5.1-g: the wait is measured from the game's flip reaching the host, not from the end
    /// of the GPU work. Three drawables hold a finished frame for at most about 40 ms at 60
    /// frames a second, so the old target sat exactly where the layer is full: the lock then
    /// read "on target" (or a little under, and moved the tick earlier) while the surplus
    /// frames waited unseen for a drawable in front of it, 8 to 32 ms more (flip to screen
    /// wandered between 53 and 87 ms in one match). From the flip the whole queue is in the
    /// measurement. BB_PACING_FROM=gpu restores the old measurement.
    static bool FromFlip() {
        static const bool on = [] {
            const char* env = std::getenv("BB_PACING_FROM");
            return !(env && env[0] == 'g');
        }();
        return on;
    }
    static double PeriodMs() {
        static const double ms = [] {
            const u32 limit = EmulatorSettings.GetFrameLimit();
            const u32 fps = limit ? limit : EmulatorSettings.GetVblankFrequency();
            return fps ? 1000.0 / fps : 1000.0 / 60.0;
        }();
        return ms;
    }
    static constexpr u32 kRing = 8;
    std::mutex mutex;
    std::array<double, kRing> ring{};
    std::vector<std::array<float, 5>> trace;
    u32 ring_at{}, ring_n{};
    u64 last_shown_ns{}, last_gpu_end_ns{};
    bool active{};
    u32 frames{}, late_frames{}, wait_n{};
    double wait_sum_ms{}, wait_min_ms{1e9};
};
PacingLock g_pacing_lock;

class PresentWait {
public:
    /// Wait of each shown frame between the end of its GPU work and the display (0.5.1-f).
    /// Measurement only (printed with the frame stats); PacingLock steers by the same wait.
    void Note(u64 shown_ns, u64 gpu_end_ns) {
        if (!shown_ns) {
            return;
        }
        std::scoped_lock lock{mutex};
        const u64 last = last_shown_ns;
        last_shown_ns = shown_ns;
        ++frames;
        if (last && shown_ns > last && double(shown_ns - last) * 1e-9 > 1.3 / 60.0) {
            ++late_frames;
        }
        if (gpu_end_ns && shown_ns >= gpu_end_ns && shown_ns - gpu_end_ns < 200'000'000ull) {
            const double wait_ms = double(shown_ns - gpu_end_ns) * 1e-6;
            wait_sum_ms += wait_ms;
            wait_min_ms = std::min(wait_min_ms, wait_ms);
            ++wait_n;
        }
    }

    void Print() {
        std::scoped_lock lock{mutex};
        if (!frames) {
            return;
        }
        std::printf("Present wait: GPU end to shown least %.1f mean %.1f ms over %u frames, "
                    "%u of %u shown late\n",
                    wait_n ? wait_min_ms : 0.0, wait_n ? wait_sum_ms / wait_n : 0.0, wait_n,
                    late_frames, frames);
        wait_n = 0;
        wait_sum_ms = 0;
        wait_min_ms = 1e9;
        late_frames = frames = 0;
    }

private:
    std::mutex mutex;
    u64 last_shown_ns{};
    u32 frames{}, late_frames{}, wait_n{};
    double wait_sum_ms{}, wait_min_ms{1e9};
};
PresentWait g_present_wait;
} // namespace

void Presenter::Present(Frame* frame, bool is_reusing_frame, bool is_game_frame, u32 present_id) {
    last_present = {NowNs(), 0, 0, 0};
    // nextDrawable and the present passes create autoreleased objects, and this thread
    // (bb:Present) has no pool of its own: without one, drawables never return to the layer.
    Metal::WithAutoreleasePool([&] {
        // Minimised, hidden or occluded: nextDrawable would block up to 1 s. The frame returns
        // to the pool at once, so PrepareFrame's Take never waits on a skipped present.
        if (!Metal::ShouldAcquireDrawable(window.IsVisible(), window.GetWidth(),
                                          window.GetHeight(), SnapshotRequested())) {
            BbOverlay::Render(nullptr, nullptr, 0, 0); // FPS timing, the menu's pending save
            pool->Free(frame);
            return;
        }
        if (const u32 wanted = window.IsFullscreen() ? fullscreen_drawables : windowed_drawables;
            wanted != layer_drawables) {
            Metal::SetMaximumDrawables(layer, wanted);
            layer_drawables = wanted;
            std::printf("Presenter: %u drawables (%s)\n", wanted,
                        window.IsFullscreen() ? "full screen" : "window");
        }
        Metal::Drawable drawable = Metal::NextDrawable(layer);
        if (!drawable.drawable) {
            static std::atomic<bool> logged{false};
            if (!logged.exchange(true)) {
                LOG_WARNING(Render_Vulkan, "No drawable from the layer: skipping frame(s)");
            }
            BbOverlay::Render(nullptr, nullptr, 0, 0); // FPS timing, the menu's pending save
            pool->Free(frame);
            return;
        }
        last_present.acquired = NowNs();

        void* const command_buffer = present_queue->Begin();
        void* source = nullptr;
        if (frame->ready_tick != 0) {
            // PrepareFrame flushed the tick, so it has a device tick (0: already complete).
            const std::optional<u64> device_tick =
                draw_scheduler.GetTimeline().DeviceTickOf(frame->ready_tick);
            ASSERT_MSG(device_tick.has_value(), "Present: frame tick {} was not committed",
                       frame->ready_tick);
            present_queue->WaitForDeviceTick(command_buffer, *device_tick);
            source = frame->texture.Get();
        }
        // A blank frame clears the drawable. Every acquired drawable is presented: one that is
        // not returns to the layer only after about a display refresh.
        pp_pass.Passes().Present(command_buffer, source, frame->width, frame->height,
                                 drawable.texture, drawable.width, drawable.height);
        // The settings menu / FPS counter over the presented image, before the snapshot (which
        // shows it) and the commit. Called every present: it also times the FPS readout and
        // saves menu changes while nothing is drawn.
        BbOverlay::Render(command_buffer, drawable.texture, drawable.width, drawable.height);
        std::string snapshot_path = TakeSnapshotRequest();
        std::shared_ptr<Metal::Object> snapshot;
        if (!snapshot_path.empty()) {
            snapshot = std::make_shared<Metal::Object>(Metal::EncodeSnapshot(
                device, command_buffer, drawable.texture, drawable.width, drawable.height));
        }
        const double pacing = PacingSeconds();
        const u64 flip_ns = is_game_frame ? frame->flip_ns : 0;
        present_queue->Commit(
            command_buffer, drawable.drawable.Get(),
            [pool = pool, frame, snapshot, path = std::move(snapshot_path),
             width = drawable.width, height = drawable.height] {
                pool->Free(frame);
                if (snapshot && *snapshot) {
                    // Off the completion thread: it is serial, and the PNG takes ~150 ms that
                    // later presents would wait behind.
                    std::thread([snapshot, path, width, height] {
                        const bool ok = Metal::WriteSnapshotPng(path.c_str(), snapshot->Get(),
                                                                width, height);
                        std::printf("Snapshot: %s %ux%u %s\n", path.c_str(), width, height,
                                    ok ? "written" : "FAILED");
                        std::fflush(stdout);
                    }).detach();
                }
            }, pacing,
            present_id || pacing > 0 || PacingLock::Wanted()
                ? std::function<void(u64, u64)>(
                      [this, present_id, pacing, flip_ns](u64 shown_ns, u64 gpu_end_ns) {
                          g_present_wait.Note(shown_ns, gpu_end_ns);
                          g_pacing_lock.Note(shown_ns, gpu_end_ns, flip_ns, pacing > 0);
                          if (!present_id) {
                              return;
                          }
                          std::scoped_lock lock{past_timings_mutex};
                          if (past_timings.size() < 1024) {
                              past_timings.push_back(vk::PastPresentationTimingGOOGLE{
                                  .presentID = present_id, .desiredPresentTime = 0,
                                  .actualPresentTime = shown_ns, .earliestPresentTime = 0,
                                  .presentMargin = 0});
                          }
                      })
                : std::function<void(u64, u64)>{});
        last_present.submitted = NowNs();
        last_present.presented = last_present.submitted;
    });
    BbGpuCapture::OnPresent(device.NativeDevice()); // bbport: BB_GPU_CAPTURE one-frame capture
    { // bbport: [FSTAT] present-interval stats every 2 s (fps, p99, worst, hitches > 50 ms)
        static std::vector<double> iv;
        static auto prev = std::chrono::steady_clock::now(), win = prev;
        static u64 busy0 = device.GpuBusyNs();
        const auto now = std::chrono::steady_clock::now();
        iv.push_back(std::chrono::duration<double, std::milli>(now - prev).count());
        prev = now;
        if (now - win >= std::chrono::seconds(2)) {
            const double secs = std::chrono::duration<double>(now - win).count();
            std::sort(iv.begin(), iv.end());
            const size_t hitches = iv.end() - std::upper_bound(iv.begin(), iv.end(), 50.0);
            const u64 busy = device.GpuBusyNs();
            std::printf("[FSTAT] fps=%.1f p50=%.1f p99=%.1f max=%.1f hitches=%zu gpu=%.1fms\n",
                        iv.size() / secs, iv[iv.size() / 2], iv[iv.size() * 99 / 100], iv.back(),
                        hitches, (busy - busy0) / 1e6 / iv.size());
            busy0 = busy;
            g_present_wait.Print();
            g_pacing_lock.Print();
            std::fflush(stdout);
            iv.clear();
            win = now;
        }
    }

    if (!is_reusing_frame) {
        std::scoped_lock lock{pool->mutex};
        pool->last_presented = frame;
    }
    if (!is_reusing_frame && is_game_frame) {
        DebugState.IncFlipFrameNum();
    }
}

} // namespace Vulkan
