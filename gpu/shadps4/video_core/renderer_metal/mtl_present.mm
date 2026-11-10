// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: presenting to a CAMetalLayer (Metal-native renderer, M4). See mtl_present.h.

#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>
#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/CATransaction.h>
#import <QuartzCore/CADisplayLink.h>
#include <pthread.h>
#include <thread>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <memory>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <utility>
#include <vector>
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_metal/mtl_blit_shaders.h"
#include "video_core/renderer_metal/mtl_device.h"
#include "video_core/renderer_metal/mtl_device_impl.h"
#include "video_core/renderer_metal/mtl_encoder.h"
#include "video_core/renderer_metal/mtl_present.h"
#include "video_core/renderer_metal/mtl_render_pipeline.h"
#include "video_core/renderer_metal/mtl_texture.h"

namespace Metal {

FitRect FitImage(u32 src_width, u32 src_height, u32 dst_width, u32 dst_height) {
    if (src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0) {
        return {0, 0, 0, 0};
    }
    // vk_presenter.cpp FitImage: float aspects, truncation towards zero.
    const float src_aspect = static_cast<float>(src_width) / static_cast<float>(src_height);
    const float dst_aspect = static_cast<float>(dst_width) / static_cast<float>(dst_height);
    u32 width = dst_width;
    u32 height = dst_height;
    if (src_aspect > dst_aspect) {
        height = static_cast<u32>(static_cast<float>(dst_width) / src_aspect);
    } else {
        width = static_cast<u32>(static_cast<float>(dst_height) * src_aspect);
    }
    return {static_cast<s32>((dst_width - width) / 2), static_cast<s32>((dst_height - height) / 2),
            width, height};
}

bool ShouldAcquireDrawable(bool window_visible, s32 width, s32 height, bool snapshot_requested) {
    if (width <= 0 || height <= 0) {
        return false;
    }
    return window_visible || snapshot_requested;
}

struct PresentPasses::Impl {
    Object library;
    Object post_process; ///< id<MTLRenderPipelineState>, colour frame_format
    Object present;      ///< id<MTLRenderPipelineState>, colour drawable_format
    Object sampler;      ///< linear, clamp to edge
    Device* device{};
    std::vector<std::pair<u32, Object>> edge_aa; ///< per output pixel format (0.5.1-e)
};

PresentPasses::PresentPasses(Device& device, u32 frame_format, u32 drawable_format)
    : impl{std::make_unique<Impl>()} {
    impl->device = &device;
    impl->library = CompileLibrary(device, BlitShaders::Source, "Present passes");
    impl->post_process = CreateRenderPipeline(device,
                                              RenderPipelineDesc{
                                                  .library = impl->library.Get(),
                                                  .vertex_function = BlitShaders::FullscreenTriangle,
                                                  .fragment_function = BlitShaders::PostProcess,
                                                  .color_format = frame_format,
                                              },
                                              "Post-process");
    impl->present = CreateRenderPipeline(device,
                                         RenderPipelineDesc{
                                             .library = impl->library.Get(),
                                             .vertex_function = BlitShaders::FullscreenTriangle,
                                             .fragment_function = BlitShaders::Present,
                                             .color_format = drawable_format,
                                         },
                                         "Present");
    impl->sampler = CreateSampler(device, SamplerDesc{.min_filter = MTLSamplerMinMagFilterLinear,
                                                      .mag_filter = MTLSamplerMinMagFilterLinear,
                                                      .mip_filter = MTLSamplerMipFilterNotMipmapped,
                                                      .address_u = MTLSamplerAddressModeClampToEdge,
                                                      .address_v = MTLSamplerAddressModeClampToEdge,
                                                      .address_w = MTLSamplerAddressModeClampToEdge});
    ASSERT_MSG(impl->sampler, "Present passes: no sampler");
}

PresentPasses::~PresentPasses() = default;

void PresentPasses::PostProcess(Encoder& encoder, void* input, void* output, u32 out_width,
                                u32 out_height, const PostProcessSettings& settings) {
    static_assert(sizeof(PostProcessSettings) == 12);
    RenderPassDesc pass{};
    pass.colors[0].texture = output; // every pixel is written: no clear, no load
    pass.colors[0].discard = true;
    pass.width = out_width;
    pass.height = out_height;
    encoder.BeginRenderPass(pass);
    // Fullscreen-triangle state: whatever the rasterizer left is reset.
    encoder.SetRenderPipeline(impl->post_process.Get(), Access::None); // textures only
    encoder.SetDepthStencil(nullptr);
    encoder.SetCullMode(CullMode::None);
    encoder.SetDepthBias(0.0f, 0.0f, 0.0f);
    encoder.SetFillMode(FillMode::Fill);
    encoder.SetDepthClip(DepthClip::Clip);
    encoder.SetViewport({0.0f, 0.0f, float(out_width), float(out_height), 0.0f, 1.0f});
    encoder.SetScissor({0, 0, out_width, out_height});
    encoder.SetTexture(Stage::Fragment, 0, input);
    encoder.SetSampler(Stage::Fragment, 0, impl->sampler.Get());
    encoder.SetBytes(Stage::Fragment, 0, &settings, sizeof(settings));
    encoder.Draw(Primitive::Triangle, 3);
    encoder.EndRenderPass();
}

void* PresentPasses::EdgeAaPipeline(u32 pixel_format) {
    for (const auto& [format, pipeline] : impl->edge_aa) {
        if (format == pixel_format) {
            return pipeline.Get();
        }
    }
    Object pipeline = CreateRenderPipeline(*impl->device,
                                           RenderPipelineDesc{
                                               .library = impl->library.Get(),
                                               .vertex_function = BlitShaders::FullscreenTriangle,
                                               .fragment_function = BlitShaders::EdgeAa,
                                               .color_format = pixel_format,
                                           },
                                           "Edge AA");
    return impl->edge_aa.emplace_back(pixel_format, std::move(pipeline)).second.Get();
}

void PresentPasses::EdgeAa(Encoder& encoder, void* pipeline, void* input, void* output, u32 width,
                           u32 height, const EdgeAaSettings& settings) {
    struct Params {
        float rcp_frame[2];
        float subpix;
        float edge_threshold;
        float edge_threshold_min;
        u32 linear_input;
        float split;
        u32 quality;
    };
    static_assert(sizeof(Params) == 32);
    const MTLPixelFormat format = ((__bridge id<MTLTexture>)input).pixelFormat;
    const Params params{
        .rcp_frame = {1.0f / float(width), 1.0f / float(height)},
        .subpix = settings.subpix,
        .edge_threshold = settings.edge_threshold,
        .edge_threshold_min = settings.edge_threshold_min,
        .linear_input = format == MTLPixelFormatBGRA8Unorm_sRGB ||
                        format == MTLPixelFormatRGBA8Unorm_sRGB,
        .split = settings.split,
        .quality = settings.quality,
    };
    RenderPassDesc pass{};
    pass.colors[0].texture = output; // every pixel is written: no clear, no load
    pass.colors[0].discard = true;
    pass.width = width;
    pass.height = height;
    encoder.BeginRenderPass(pass);
    encoder.SetRenderPipeline(pipeline, Access::None); // textures only
    encoder.SetDepthStencil(nullptr);
    encoder.SetCullMode(CullMode::None);
    encoder.SetDepthBias(0.0f, 0.0f, 0.0f);
    encoder.SetFillMode(FillMode::Fill);
    encoder.SetDepthClip(DepthClip::Clip);
    encoder.SetViewport({0.0f, 0.0f, float(width), float(height), 0.0f, 1.0f});
    encoder.SetScissor({0, 0, width, height});
    encoder.SetTexture(Stage::Fragment, 0, input);
    encoder.SetSampler(Stage::Fragment, 0, impl->sampler.Get());
    encoder.SetBytes(Stage::Fragment, 0, &params, sizeof(params));
    encoder.Draw(Primitive::Triangle, 3);
    encoder.EndRenderPass();
}

void PresentPasses::Present(void* command_buffer, void* frame, u32 frame_width, u32 frame_height,
                            void* target, u32 target_width, u32 target_height) {
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)command_buffer;
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = (__bridge id<MTLTexture>)target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> e = [cmd renderCommandEncoderWithDescriptor:pass];
    e.label = @"Present";
    const FitRect fit = FitImage(frame_width, frame_height, target_width, target_height);
    if (frame != nullptr && fit.width != 0 && fit.height != 0) {
        [e setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)impl->present.Get()];
        [e setViewport:(MTLViewport){double(fit.x), double(fit.y), double(fit.width),
                                     double(fit.height), 0.0, 1.0}];
        [e setFragmentTexture:(__bridge id<MTLTexture>)frame atIndex:0];
        [e setFragmentSamplerState:(__bridge id<MTLSamplerState>)impl->sampler.Get() atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [e endEncoding];
}

void ConfigureLayer(Device& device, void* layer, const LayerConfig& config) {
    CAMetalLayer* l = (__bridge CAMetalLayer*)layer;
    l.device = (__bridge id<MTLDevice>)device.NativeDevice();
    l.pixelFormat = MTLPixelFormat(config.pixel_format);
    CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    l.colorspace = srgb;
    CGColorSpaceRelease(srgb);
    l.maximumDrawableCount = config.max_drawables;
    l.displaySyncEnabled = config.display_sync;
    l.framebufferOnly = config.framebuffer_only;
}

namespace {
/// The display's 60 Hz mode with the size of the current one, or null.
CGDisplayModeRef CopySixtyHertzTwin(CGDirectDisplayID display) {
    CGDisplayModeRef now = CGDisplayCopyDisplayMode(display);
    if (!now) {
        return nullptr;
    }
    NSDictionary* options = @{(__bridge NSString*)kCGDisplayShowDuplicateLowResolutionModes : @YES};
    CFArrayRef modes = CGDisplayCopyAllDisplayModes(display, (__bridge CFDictionaryRef)options);
    CGDisplayModeRef found = nullptr;
    for (CFIndex i = 0; modes && i < CFArrayGetCount(modes); ++i) {
        CGDisplayModeRef m = (CGDisplayModeRef)CFArrayGetValueAtIndex(modes, i);
        if (CGDisplayModeGetWidth(m) == CGDisplayModeGetWidth(now) &&
            CGDisplayModeGetHeight(m) == CGDisplayModeGetHeight(now) &&
            CGDisplayModeGetPixelWidth(m) == CGDisplayModeGetPixelWidth(now) &&
            CGDisplayModeGetPixelHeight(m) == CGDisplayModeGetPixelHeight(now) &&
            std::abs(CGDisplayModeGetRefreshRate(m) - 60.0) < 0.01) {
            found = CGDisplayModeRetain(m);
            break;
        }
    }
    if (modes) {
        CFRelease(modes);
    }
    CGDisplayModeRelease(now);
    return found;
}
} // namespace

namespace {
std::atomic<int> g_steady_hz{0};
}
} // namespace Metal

/// BB_REFRESH_BEAT=link: a display link that asks for the display's full rate and does nothing
/// else. Unlike the heartbeat layer it changes no content, so the window server has no reason
/// to composite a full-screen game layer.
@interface BBRefreshLinkTarget : NSObject
- (void)tick:(CADisplayLink*)link;
@end
@implementation BBRefreshLinkTarget {
    unsigned _ticks;
    CFTimeInterval _first;
}
- (void)tick:(CADisplayLink*)link {
    if (_ticks == 0) {
        _first = link.timestamp;
    }
    if (++_ticks == 600) {
        const double secs = link.timestamp - _first;
        std::printf("Steady refresh: display link ran %.1f callbacks a second\n",
                    secs > 0 ? 599.0 / secs : 0.0);
        std::fflush(stdout);
    }
}
@end

namespace Metal {

int SteadyRefreshHz() {
    return g_steady_hz.load(std::memory_order_relaxed);
}

void StartSteadyRefresh(void* layer, bool fixed_60, bool by_link) {
    static std::atomic<bool> started{false};
    if (started.exchange(true)) {
        return;
    }
    CAMetalLayer* l = (__bridge CAMetalLayer*)layer;
    dispatch_async(dispatch_get_main_queue(), ^{
        NSScreen* screen = nil;
        id owner = l.delegate;
        if ([owner isKindOfClass:[NSView class]]) {
            screen = ((NSView*)owner).window.screen;
        }
        if (screen == nil) {
            screen = [NSScreen mainScreen];
        }
        const long max_fps = long(screen.maximumFramesPerSecond);
        // A 60 fps stream fits a 120 or 240 Hz grid; on other rates (144, 165) no grid shows
        // it evenly and the display's own adaptive refresh is the better choice.
        if (max_fps < 100 || max_fps % 60 != 0) {
            g_steady_hz = int(max_fps);
            std::printf("Steady refresh: nothing to do (display %ld Hz)\n", max_fps);
            std::fflush(stdout);
            return;
        }
        if (fixed_60) {
            NSNumber* number = screen.deviceDescription[@"NSScreenNumber"];
            const CGDirectDisplayID display =
                number ? CGDirectDisplayID(number.unsignedIntValue) : CGMainDisplayID();
            if (CGDisplayModeRef mode = CopySixtyHertzTwin(display)) {
                const CGError err = CGDisplaySetDisplayMode(display, mode, nullptr);
                CGDisplayModeRelease(mode);
                if (err == kCGErrorSuccess) {
                    g_steady_hz = 60;
                    std::printf("Steady refresh: display in its 60 Hz mode while the game runs "
                                "(was %ld Hz)\n", max_fps);
                    std::fflush(stdout);
                    return;
                }
                std::printf("Steady refresh: 60 Hz mode refused (error %d)\n", int(err));
            } else {
                std::printf("Steady refresh: the display has no 60 Hz mode of this size\n");
            }
        }
        if (by_link) {
            static BBRefreshLinkTarget* target = [BBRefreshLinkTarget new];
            CADisplayLink* link = [screen displayLinkWithTarget:target selector:@selector(tick:)];
            const float rate = float(max_fps);
            link.preferredFrameRateRange = CAFrameRateRangeMake(rate, rate, rate);
            [link addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
            g_steady_hz = int(max_fps);
            std::printf("Steady refresh: display link (display %ld Hz)\n", max_fps);
            std::fflush(stdout);
            return;
        }
        CALayer* dot = [CALayer layer];
        [CATransaction begin];
        [CATransaction setDisableActions:YES];
        dot.frame = CGRectMake(0, 0, 1, 1);
        dot.zPosition = 1000;
        dot.backgroundColor = CGColorGetConstantColor(kCGColorBlack);
        [l addSublayer:dot];
        [CATransaction commit];
        g_steady_hz = int(max_fps);
        std::printf("Steady refresh: heartbeat layer (display %ld Hz)\n", max_fps);
        std::fflush(stdout);
        // Faster than the refresh, so every refresh sees a change.
        const useconds_t period_us = useconds_t(850'000 / max_fps);
        std::thread([dot, period_us] {
            pthread_setname_np("bb-refresh-beat");
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
            CGColorRef dark = CGColorCreateGenericGray(0.02, 1.0);
            bool flip = false;
            for (;;) {
                usleep(period_us);
                @autoreleasepool {
                    [CATransaction begin];
                    [CATransaction setDisableActions:YES];
                    dot.backgroundColor = flip ? CGColorGetConstantColor(kCGColorBlack) : dark;
                    [CATransaction commit];
                    [CATransaction flush];
                }
                flip = !flip;
            }
        }).detach();
    });
}

void SetMaximumDrawables(void* layer, u32 count) {
    ((__bridge CAMetalLayer*)layer).maximumDrawableCount = count;
}

Drawable NextDrawable(void* layer) {
    id<CAMetalDrawable> drawable = [(__bridge CAMetalLayer*)layer nextDrawable];
    if (drawable == nil) {
        return {};
    }
    id<MTLTexture> texture = drawable.texture;
    Drawable result;
    result.texture = (__bridge void*)texture;
    result.width = u32(texture.width);
    result.height = u32(texture.height);
    result.drawable = Object::Adopt((__bridge_retained void*)drawable);
    return result;
}

Object EncodeSnapshot(Device& device, void* command_buffer, void* texture, u32 width,
                      u32 height) {
    if (width == 0 || height == 0) {
        return {};
    }
    // The PNG writer takes 4-byte BGRA texels; an HDR layer (RGBA16Float, BGR10A2) would need
    // a conversion pass.
    const MTLPixelFormat format = ((__bridge id<MTLTexture>)texture).pixelFormat;
    if (format != MTLPixelFormatBGRA8Unorm && format != MTLPixelFormatBGRA8Unorm_sRGB) {
        LOG_WARNING(Render, "Snapshot: drawable pixel format {} is not BGRA8; no snapshot",
                    static_cast<u32>(format));
        return {};
    }
    id<MTLBuffer> buffer =
        [(__bridge id<MTLDevice>)device.NativeDevice() newBufferWithLength:NSUInteger(width) * height * 4
                                                                   options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        return {};
    }
    buffer.label = @"Snapshot";
    id<MTLBlitCommandEncoder> blit =
        [(__bridge id<MTLCommandBuffer>)command_buffer blitCommandEncoder];
    blit.label = @"Snapshot";
    [blit copyFromTexture:(__bridge id<MTLTexture>)texture
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(width, height, 1)
                        toBuffer:buffer
               destinationOffset:0
          destinationBytesPerRow:NSUInteger(width) * 4
        destinationBytesPerImage:NSUInteger(width) * height * 4];
    [blit endEncoding];
    return Object::Adopt((__bridge_retained void*)buffer);
}

bool WriteSnapshotPng(const char* path, void* buffer, u32 width, u32 height) {
    @autoreleasepool {
        id<MTLBuffer> source = (__bridge id<MTLBuffer>)buffer;
        if (source == nil || source.length < NSUInteger(width) * height * 4) {
            return false;
        }
        CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGDataProviderRef provider = CGDataProviderCreateWithData(
            nullptr, source.contents, size_t(width) * height * 4, nullptr);
        // BGRA8 bytes = little-endian 32-bit pixels with alpha first (ARGB); alpha is 1.
        CGImageRef image = CGImageCreate(width, height, 8, 32, size_t(width) * 4, srgb,
                                         kCGBitmapByteOrder32Little | kCGImageAlphaNoneSkipFirst,
                                         provider, nullptr, false, kCGRenderingIntentDefault);
        bool ok = false;
        NSURL* url = [NSURL fileURLWithPath:@(path)];
        CGImageDestinationRef destination = CGImageDestinationCreateWithURL(
            (__bridge CFURLRef)url, CFSTR("public.png"), 1, nullptr);
        if (image != nullptr && destination != nullptr) {
            CGImageDestinationAddImage(destination, image, nullptr);
            ok = CGImageDestinationFinalize(destination);
        }
        if (destination != nullptr) {
            CFRelease(destination);
        }
        CGImageRelease(image);
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(srgb);
        return ok;
    }
}

void WithAutoreleasePool(const std::function<void()>& body) {
    @autoreleasepool {
        body();
    }
}

struct PresentQueue::Impl {
    id<MTLCommandQueue> queue;
    id<MTLSharedEvent> event; ///< the device's (Device::Commit signals it with each tick)
    Device* device{};         ///< counts the present command buffers' GPU time (GpuBusyNs)
    /// What completion handlers touch; the destructor waits for them, so a handler never
    /// outlives it.
    std::mutex mutex;
    std::condition_variable idle;
    u32 in_flight{}; ///< under `mutex`
    u32 shutdown_timeout_ms{};
};

PresentQueue::PresentQueue(Device& device, u32 shutdown_timeout_ms)
    : impl{std::make_unique<Impl>()} {
    impl->shutdown_timeout_ms = shutdown_timeout_ms;
    impl->queue = [(__bridge id<MTLDevice>)device.NativeDevice() newCommandQueue];
    ASSERT_MSG(impl->queue != nil, "No Metal present queue");
    impl->queue.label = @"Present";
    impl->event = device.GetImpl().event;
    impl->device = &device;
}

PresentQueue::~PresentQueue() {
    if (!WaitIdleFor(impl->shutdown_timeout_ms)) {
        LOG_ERROR(Render, "Metal present queue: {} command buffer(s) still in flight after {} ms; "
                          "leaving them behind",
                  InFlight(), impl->shutdown_timeout_ms);
        (void)impl.release(); // their completion handlers still use it
    }
}

void* PresentQueue::Begin() {
    id<MTLCommandBuffer> cmd = [impl->queue commandBuffer];
    ASSERT_MSG(cmd != nil, "No present command buffer");
    // A reference of our own until Commit (the caller may have no autorelease pool).
    return (__bridge_retained void*)cmd;
}

void PresentQueue::WaitForDeviceTick(void* command_buffer, u64 device_tick) {
    if (device_tick == 0) {
        return;
    }
    [(__bridge id<MTLCommandBuffer>)command_buffer encodeWaitForEvent:impl->event
                                                                value:device_tick];
}

void PresentQueue::Commit(void* command_buffer, void* drawable,
                          std::function<void()> on_complete, double min_duration,
                          std::function<void(u64, u64)> on_presented) {
    id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)command_buffer;
    // The GPU end time of this command buffer for the presented handler (it normally runs after
    // the completion handler: the picture cannot be shown before it is drawn).
    const auto gpu_end_ns = std::make_shared<std::atomic<u64>>(0);
    if (drawable != nullptr && on_presented) {
        // fifa-hybrid: the time the drawable was shown (presentedTime is CACurrentMediaTime
        // based = CLOCK_UPTIME_RAW seconds), 0 when it was dropped.
        [(__bridge id<CAMetalDrawable>)drawable addPresentedHandler:^(id<MTLDrawable> shown) {
          on_presented(u64(shown.presentedTime * 1e9),
                       gpu_end_ns->load(std::memory_order_relaxed));
        }];
    }
    if (drawable != nullptr && min_duration > 0) {
        [cmd presentDrawable:(__bridge id<CAMetalDrawable>)drawable
            afterMinimumDuration:min_duration];
    } else if (drawable != nullptr) {
        [cmd presentDrawable:(__bridge id<CAMetalDrawable>)drawable];
    }
    {
        std::scoped_lock lock{impl->mutex};
        ++impl->in_flight;
    }
    impl->device->TrackGpuTime((__bridge void*)cmd);
    Impl* const state = impl.get();
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
      if (done.status == MTLCommandBufferStatusError) {
          LOG_ERROR(Render, "Metal present command buffer failed: {}",
                    done.error ? done.error.localizedDescription.UTF8String : "unknown error");
      }
      gpu_end_ns->store(u64(done.GPUEndTime * 1e9), std::memory_order_relaxed);
      if (on_complete) {
          on_complete();
      }
      std::scoped_lock lock{state->mutex};
      if (--state->in_flight == 0) {
          state->idle.notify_all();
      }
    }];
    [cmd commit];
}

void PresentQueue::WaitIdle() {
    std::unique_lock lock{impl->mutex};
    impl->idle.wait(lock, [this] { return impl->in_flight == 0; });
}

bool PresentQueue::WaitIdleFor(u32 milliseconds) {
    std::unique_lock lock{impl->mutex};
    return impl->idle.wait_for(lock, std::chrono::milliseconds(milliseconds),
                               [this] { return impl->in_flight == 0; });
}

u32 PresentQueue::InFlight() const {
    std::scoped_lock lock{impl->mutex};
    return impl->in_flight;
}

} // namespace Metal
