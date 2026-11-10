//  SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
//  SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/host_passes/pp_pass.h"

#include <utility>
#include "video_core/renderer_metal/mtl_encoder.h"
#include "video_core/renderer_metal/mtl_texture.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan::HostPasses {

PostProcessingPass::PostProcessingPass(Metal::Device& device, u32 frame_format,
                                       u32 drawable_format)
    : passes{std::make_unique<Metal::PresentPasses>(device, frame_format, drawable_format)},
      device{device} {}

PostProcessingPass::~PostProcessingPass() = default;

void PostProcessingPass::Render(Scheduler& scheduler, Metal::Object input, Metal::Object output,
                                u32 width, u32 height, const Settings& settings) {
    scheduler.EndRendering();
    scheduler.Record([passes = passes.get(), input = std::move(input), output = std::move(output),
                      width, height, settings](Metal::Encoder& e) mutable {
        void* const in = input.Get();
        void* const out = output.Get();
        e.KeepAlive(std::move(input));
        e.KeepAlive(std::move(output));
        passes->PostProcess(e, in, out, width, height, settings);
    });
    // The guest pipelines' dynamic state must be emitted again after this pipeline.
    scheduler.GetDynamicState().Invalidate();
}

Metal::Object PostProcessingPass::EdgeAa(Scheduler& scheduler, Metal::Object input, u32 width,
                                         u32 height, const AaSettings& settings) {
    const u32 format = Metal::TexturePixelFormat(input.Get());
    if (format == 0 || width == 0 || height == 0) {
        return input;
    }
    AaTarget& target = aa_targets[aa_next];
    aa_next = (aa_next + 1) % u32(aa_targets.size());
    if (!target.texture || target.width != width || target.height != height ||
        target.format != format) {
        const Metal::TextureDesc desc{
            .pixel_format = format,
            .width = width,
            .height = height,
            .render_target = true,
        };
        target = {Metal::CreateTexture(device, desc, "Edge AA Image", nullptr), width, height,
                  format};
        if (!target.texture) {
            return input;
        }
    }
    void* const pipeline = passes->EdgeAaPipeline(format);
    Metal::Object output = Metal::Object::Retain(target.texture.Get());
    Metal::Object result = Metal::Object::Retain(target.texture.Get());
    scheduler.EndRendering();
    scheduler.Record([passes = passes.get(), pipeline, input = std::move(input),
                      output = std::move(output), width, height,
                      settings](Metal::Encoder& e) mutable {
        void* const in = input.Get();
        void* const out = output.Get();
        e.KeepAlive(std::move(input));
        e.KeepAlive(std::move(output));
        passes->EdgeAa(e, pipeline, in, out, width, height, settings);
    });
    scheduler.GetDynamicState().Invalidate();
    return result;
}

} // namespace Vulkan::HostPasses
