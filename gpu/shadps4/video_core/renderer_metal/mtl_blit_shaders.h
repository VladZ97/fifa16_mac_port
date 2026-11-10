// SPDX-FileCopyrightText: Copyright 2026 Mark Hall
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the texture cache's and the presenter's blit programs in MSL (Metal-native renderer),
// translated by hand from host_shaders/fs_tri.vert, color_to_ms_depth.frag, ms_image_blit.frag,
// depth_resample.frag and post_process.frag, plus the scale helper's nearest colour copy. One
// library; inputs at textures 0/1, push constants at buffer 0.

#pragma once

namespace Metal::BlitShaders {

// Function names in Source.
inline constexpr const char* FullscreenTriangle = "fs_tri_vs";
/// color_to_ms_depth.frag: sample N of the depth target takes channel N of the colour texel.
inline constexpr const char* ColorToMsDepth = "color_to_ms_depth_fs";
/// ms_image_blit.frag: non-MSAA source (texture2d) and MSAA source (texture2d_ms, per sample),
/// in float, uint and int flavours (Metal wants the shader's type to match the formats').
inline constexpr const char* MsImageBlit[3] = {"ms_image_blit_fs", "ms_image_blit_uint_fs",
                                               "ms_image_blit_int_fs"};
inline constexpr const char* MsImageBlitMsaa[3] = {
    "ms_image_blit_msaa_fs", "ms_image_blit_msaa_uint_fs", "ms_image_blit_msaa_int_fs"};
/// depth_resample.frag: depth only; HAS_STENCIL (stencil exported from texture 1);
/// STENCIL_BITS (keeps pixels whose stencil (texture 1) has the bit at buffer 0; bit 0 keeps all).
inline constexpr const char* DepthResample = "depth_resample_fs";
inline constexpr const char* DepthResampleStencil = "depth_resample_stencil_fs";
inline constexpr const char* DepthResampleStencilBits = "depth_resample_stencil_bits_fs";
/// post_process.frag (the presenter's gamma pass): texture 0, sampler 0, PpSettings at buffer 0.
inline constexpr const char* PostProcess = "post_process_fs";
/// The present blit: a linear sample of texture 0 (sampler 0) over the viewport (also the scale
/// helper's linear colour copy).
inline constexpr const char* Present = "present_fs";
/// Scaled colour copy (mtl_scale.h): the texel of texture 0 nearest to the pixel's centre, as
/// Vulkan's blitImage with eNearest (floor((x + 0.5) * src / dst)); float, uint and int flavours.
inline constexpr const char* ScaleNearest[3] = {"scale_nearest_fs", "scale_nearest_uint_fs",
                                                "scale_nearest_int_fs"};
/// 0.5.1-e: edge anti-aliasing of the display image before it is scaled to the window (FXAA 3.11
/// quality, 5 to 12 search steps): texture 0, sampler 0 (linear), EdgeAaParams at buffer 0.
/// Ported from NVIDIA FXAA 3.11 (BSD-3-Clause): the notice is in gpu/third_party/fxaa/LICENSE.txt.
inline constexpr const char* EdgeAa = "edge_aa_fs";
/// 0.5.1-c: a full-screen copy dispatch as a fragment pass (BlitHelper::CopyDispatch).
inline constexpr const char* CopyDispatch = "copy_dispatch_fs";

inline constexpr const char* Source = R"msl(
#include <metal_stdlib>
using namespace metal;

struct FsTriOut {
    float4 position [[position]];
    float2 uv;
};

// fs_tri.vert: one triangle covering the target. Metal's clip space has y up, Vulkan's y down:
// y is negated so that uv (0, 0) is still the top-left texel.
vertex FsTriOut fs_tri_vs(uint vertex_id [[vertex_id]]) {
    const float2 pos = float2(float((vertex_id & 1u) << 2u), float((vertex_id & 2u) << 1u));
    FsTriOut out;
    out.position = float4(pos.x - 1.0, 1.0 - pos.y, 0.0, 1.0);
    out.uv = pos * 0.5;
    return out;
}

struct DepthOut {
    float depth [[depth(any)]];
};

struct DepthStencilOut {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

static uint2 TexelOf(float2 uv, uint2 size) {
    return min(uint2(uv * float2(size)), size - 1u);
}

// color_to_ms_depth.frag: runs per sample ([[sample_id]]); a colour texel holds up to four
// samples' depths (one per channel).
fragment DepthOut color_to_ms_depth_fs(FsTriOut in [[stage_in]], uint sample_id [[sample_id]],
                                       texture2d<float> color [[texture(0)]]) {
    const uint2 coord = TexelOf(in.uv, uint2(color.get_width(), color.get_height()));
    DepthOut out;
    out.depth = color.read(coord, 0)[min(sample_id, 3u)];
    return out;
}

// ms_image_blit.frag, without and with SRC_MSAA.
#define MS_IMAGE_BLIT(T, SUFFIX)                                                                  \
    fragment vec<T, 4> ms_image_blit##SUFFIX##_fs(float4 position [[position]],                  \
                                                  texture2d<T> in_tex [[texture(0)]]) {           \
        return in_tex.read(uint2(position.xy), 0);                                               \
    }                                                                                            \
    fragment vec<T, 4> ms_image_blit_msaa##SUFFIX##_fs(float4 position [[position]],             \
                                                       uint sample_id [[sample_id]],              \
                                                       texture2d_ms<T> in_tex [[texture(0)]]) {   \
        return in_tex.read(uint2(position.xy), sample_id);                                       \
    }
MS_IMAGE_BLIT(float, )
MS_IMAGE_BLIT(uint, _uint)
MS_IMAGE_BLIT(int, _int)

// depth_resample.frag: nearest resample of depth (and stencil) between native and reduced scene
// targets.
fragment DepthOut depth_resample_fs(FsTriOut in [[stage_in]],
                                    depth2d<float> src_depth [[texture(0)]]) {
    const uint2 coord = TexelOf(in.uv, uint2(src_depth.get_width(), src_depth.get_height()));
    DepthOut out;
    out.depth = src_depth.read(coord, 0);
    return out;
}

fragment DepthStencilOut depth_resample_stencil_fs(FsTriOut in [[stage_in]],
                                                   depth2d<float> src_depth [[texture(0)]],
                                                   texture2d<uint> src_stencil [[texture(1)]]) {
    const uint2 coord = TexelOf(in.uv, uint2(src_depth.get_width(), src_depth.get_height()));
    DepthStencilOut out;
    out.depth = src_depth.read(coord, 0);
    out.stencil = src_stencil.read(coord, 0).r;
    return out;
}

// Fixed-function stencil replacement writes one bit per draw. The first draw (bit 0, write mask
// 0) copies depth everywhere, including zero-stencil pixels.
fragment DepthOut depth_resample_stencil_bits_fs(FsTriOut in [[stage_in]],
                                                 depth2d<float> src_depth [[texture(0)]],
                                                 texture2d<uint> src_stencil [[texture(1)]],
                                                 constant uint& bit [[buffer(0)]]) {
    const uint2 coord = TexelOf(in.uv, uint2(src_depth.get_width(), src_depth.get_height()));
    if (bit != 0u && (src_stencil.read(coord, 0).r & bit) == 0u) {
        discard_fragment();
    }
    DepthOut out;
    out.depth = src_depth.read(coord, 0);
    return out;
}

// post_process.frag: re-encodes the frame with the user's gamma (pp_gamma at gamma 1 is the sRGB
// encode); srgb_input decodes sRGB-encoded values first (10-bit frames without an sRGB view); hdr
// passes the frame through. GLSL mix(x, y, bvec) = select(x, y, cond): y where true.
struct PpSettings {
    float gamma;
    uint hdr;
    uint srgb_input;
};
constant float pp_cutoff = 0.0031308, pp_a = 1.055, pp_b = 0.055, pp_d = 12.92;
static float3 pp_gamma(float3 c, float g) {
    return select(pp_a * pow(c, float3(1.0 / (2.4 + 1.0 - g))) - pp_b, pp_d * c / g,
                  c < pp_cutoff);
}
static float3 pp_degamma(float3 c) {
    return select(pow(max(c + pp_b, 0.0) / pp_a, float3(2.4)), c / pp_d, c < pp_d * pp_cutoff);
}
fragment float4 post_process_fs(FsTriOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                                sampler s [[sampler(0)]], constant PpSettings& pp [[buffer(0)]]) {
    float4 c = tex.sample(s, in.uv);
    if (pp.hdr != 0u) {
        return c;
    }
    if (pp.srgb_input != 0u) {
        c.rgb = pp_degamma(c.rgb);
    }
    return float4(pp_gamma(c.rgb, pp.gamma), c.a);
}

// 0.5.1-e edge anti-aliasing: a port of the quality path of FXAA 3.11 by Timothy Lottes
// (Copyright (c) 2014-2015, NVIDIA CORPORATION; BSD-3-Clause, gpu/third_party/fxaa/LICENSE.txt),
// luma computed per tap. FIFA's 2-sample scene leaves full-pixel stair-steps on the 1080p display image, which
// the scale to a Retina window then magnifies; this pass runs at the game's resolution, before
// the scaler. `linear_input`: the texture view decodes sRGB, so luma takes a square root.
struct EdgeAaParams {
    float2 rcp_frame;
    float subpix;
    float edge_threshold;
    float edge_threshold_min;
    uint linear_input;
    float split;
    uint quality; ///< 0: 5 search steps, 1: 8, 2: 12
};
static float aa_luma(float3 c, uint linear_input) {
    const float l = dot(c, float3(0.299, 0.587, 0.114));
    return linear_input != 0u ? sqrt(max(l, 0.0)) : l;
}
fragment float4 edge_aa_fs(FsTriOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                           sampler s [[sampler(0)]], constant EdgeAaParams& aa [[buffer(0)]]) {
    float2 posM = in.uv;
    const float4 rgbM = tex.sample(s, posM, level(0.0));
    if (in.uv.x < aa.split) {
        return rgbM;
    }
    const uint lin = aa.linear_input;
    const float lumaM = aa_luma(rgbM.rgb, lin);
    float lumaS = aa_luma(tex.sample(s, posM, level(0.0), int2(0, 1)).rgb, lin);
    const float lumaE = aa_luma(tex.sample(s, posM, level(0.0), int2(1, 0)).rgb, lin);
    float lumaN = aa_luma(tex.sample(s, posM, level(0.0), int2(0, -1)).rgb, lin);
    const float lumaW = aa_luma(tex.sample(s, posM, level(0.0), int2(-1, 0)).rgb, lin);
    const float rangeMax = max(max(lumaN, lumaW), max(lumaE, max(lumaS, lumaM)));
    const float rangeMin = min(min(lumaN, lumaW), min(lumaE, min(lumaS, lumaM)));
    const float range = rangeMax - rangeMin;
    if (range < max(aa.edge_threshold_min, rangeMax * aa.edge_threshold)) {
        return rgbM;
    }
    const float lumaNW = aa_luma(tex.sample(s, posM, level(0.0), int2(-1, -1)).rgb, lin);
    const float lumaSE = aa_luma(tex.sample(s, posM, level(0.0), int2(1, 1)).rgb, lin);
    const float lumaNE = aa_luma(tex.sample(s, posM, level(0.0), int2(1, -1)).rgb, lin);
    const float lumaSW = aa_luma(tex.sample(s, posM, level(0.0), int2(-1, 1)).rgb, lin);
    const float lumaNS = lumaN + lumaS;
    const float lumaWE = lumaW + lumaE;
    const float subpixRcpRange = 1.0 / range;
    const float subpixNSWE = lumaNS + lumaWE;
    const float edgeHorz1 = (-2.0 * lumaM) + lumaNS;
    const float edgeVert1 = (-2.0 * lumaM) + lumaWE;
    const float lumaNESE = lumaNE + lumaSE;
    const float lumaNWNE = lumaNW + lumaNE;
    const float edgeHorz2 = (-2.0 * lumaE) + lumaNESE;
    const float edgeVert2 = (-2.0 * lumaN) + lumaNWNE;
    const float lumaNWSW = lumaNW + lumaSW;
    const float lumaSWSE = lumaSW + lumaSE;
    const float edgeHorz4 = (abs(edgeHorz1) * 2.0) + abs(edgeHorz2);
    const float edgeVert4 = (abs(edgeVert1) * 2.0) + abs(edgeVert2);
    const float edgeHorz3 = (-2.0 * lumaW) + lumaNWSW;
    const float edgeVert3 = (-2.0 * lumaS) + lumaSWSE;
    const float edgeHorz = abs(edgeHorz3) + edgeHorz4;
    const float edgeVert = abs(edgeVert3) + edgeVert4;
    const float subpixNWSWNESE = lumaNWSW + lumaNESE;
    const bool horzSpan = edgeHorz >= edgeVert;
    float lengthSign = horzSpan ? aa.rcp_frame.y : aa.rcp_frame.x;
    const float subpixA = subpixNSWE * 2.0 + subpixNWSWNESE;
    if (!horzSpan) {
        lumaN = lumaW;
        lumaS = lumaE;
    }
    const float subpixB = (subpixA * (1.0 / 12.0)) - lumaM;
    const float gradientN = lumaN - lumaM;
    const float gradientS = lumaS - lumaM;
    float lumaNN = lumaN + lumaM;
    const float lumaSS = lumaS + lumaM;
    const bool pairN = abs(gradientN) >= abs(gradientS);
    const float gradient = max(abs(gradientN), abs(gradientS));
    if (pairN) {
        lengthSign = -lengthSign;
    }
    const float subpixC = saturate(abs(subpixB) * subpixRcpRange);
    float2 posB = posM;
    const float2 offNP = horzSpan ? float2(aa.rcp_frame.x, 0.0) : float2(0.0, aa.rcp_frame.y);
    if (horzSpan) {
        posB.y += lengthSign * 0.5;
    } else {
        posB.x += lengthSign * 0.5;
    }
    // The search steps along the edge, in texels: FXAA quality presets 12, 15 and 39.
    const float step_table[3][12] = {
        {1.0, 1.5, 2.0, 4.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 12.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0}};
    const int step_counts[3] = {5, 8, 12};
    const uint quality = min(aa.quality, 2u);
    const int step_count = step_counts[quality];
    float steps[12];
    for (int i = 0; i < 12; ++i) {
        steps[i] = step_table[quality][i];
    }
    float2 posN = posB - offNP * steps[0];
    float2 posP = posB + offNP * steps[0];
    const float subpixD = (-2.0 * subpixC) + 3.0;
    float lumaEndN = aa_luma(tex.sample(s, posN, level(0.0)).rgb, lin);
    const float subpixE = subpixC * subpixC;
    float lumaEndP = aa_luma(tex.sample(s, posP, level(0.0)).rgb, lin);
    if (!pairN) {
        lumaNN = lumaSS;
    }
    const float gradientScaled = gradient * 0.25;
    const float lumaMM = lumaM - lumaNN * 0.5;
    const float subpixF = subpixD * subpixE;
    const bool lumaMLTZero = lumaMM < 0.0;
    lumaEndN -= lumaNN * 0.5;
    lumaEndP -= lumaNN * 0.5;
    bool doneN = abs(lumaEndN) >= gradientScaled;
    bool doneP = abs(lumaEndP) >= gradientScaled;
    if (!doneN) {
        posN -= offNP * steps[1];
    }
    if (!doneP) {
        posP += offNP * steps[1];
    }
    for (int i = 2; i < step_count && (!doneN || !doneP); ++i) {
        if (!doneN) {
            lumaEndN = aa_luma(tex.sample(s, posN, level(0.0)).rgb, lin) - lumaNN * 0.5;
            doneN = abs(lumaEndN) >= gradientScaled;
            if (!doneN) {
                posN -= offNP * steps[i];
            }
        }
        if (!doneP) {
            lumaEndP = aa_luma(tex.sample(s, posP, level(0.0)).rgb, lin) - lumaNN * 0.5;
            doneP = abs(lumaEndP) >= gradientScaled;
            if (!doneP) {
                posP += offNP * steps[i];
            }
        }
    }
    const float dstN = horzSpan ? posM.x - posN.x : posM.y - posN.y;
    const float dstP = horzSpan ? posP.x - posM.x : posP.y - posM.y;
    const bool goodSpanN = (lumaEndN < 0.0) != lumaMLTZero;
    const float spanLength = dstP + dstN;
    const bool goodSpanP = (lumaEndP < 0.0) != lumaMLTZero;
    const bool directionN = dstN < dstP;
    const float dst = min(dstN, dstP);
    const bool goodSpan = directionN ? goodSpanN : goodSpanP;
    const float subpixG = subpixF * subpixF;
    const float pixelOffset = (dst * (-1.0 / spanLength)) + 0.5;
    const float subpixH = subpixG * aa.subpix;
    const float pixelOffsetSubpix = max(goodSpan ? pixelOffset : 0.0, subpixH);
    if (horzSpan) {
        posM.y += pixelOffsetSubpix * lengthSign;
    } else {
        posM.x += pixelOffsetSubpix * lengthSign;
    }
    return float4(tex.sample(s, posM, level(0.0)).rgb, rgbM.a);
}

// The present blit: the frame over the viewport (the aspect-fit rect), linear.
fragment float4 present_fs(FsTriOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                           sampler s [[sampler(0)]]) {
    return tex.sample(s, in.uv);
}

// Scaled colour copies (mtl_scale.h): nearest texel over the viewport. Metal blits cannot scale.
#define SCALE_NEAREST(T, SUFFIX)                                                                  \
    fragment vec<T, 4> scale_nearest##SUFFIX##_fs(FsTriOut in [[stage_in]],                      \
                                                  texture2d<T> src [[texture(0)]]) {             \
        return src.read(TexelOf(in.uv, uint2(src.get_width(), src.get_height())), 0);           \
    }
SCALE_NEAREST(float, )
SCALE_NEAREST(uint, _uint)
SCALE_NEAREST(int, _int)

// copy_dispatch.frag (0.5.1-c): the compute copy shader as a fragment pass. The dispatch sampled
// (gid + 0.5) * scale at LOD 0 and wrote a component swizzle of it at gid; position.xy is gid + 0.5.
// remap: the host's depth remap of the sampled image (PushData::depth_remap); swizzle: 4-bit codes
// per output component, 0-3 the sample's components, 4 zero, 5 one.
struct CopyDispatchParams {
    float2 scale;
    uint remap;
    uint swizzle;
};
fragment float4 copy_dispatch_fs(float4 position [[position]], texture2d<float> src [[texture(0)]],
                                 sampler smp [[sampler(0)]],
                                 constant CopyDispatchParams& p [[buffer(0)]]) {
    float4 v = src.sample(smp, position.xy * p.scale, level(0.0f));
    if (p.remap != 0u) {
        v = v * 2.0f - 1.0f;
    }
    const float c[6] = {v.x, v.y, v.z, v.w, 0.0f, 1.0f};
    return float4(c[p.swizzle & 7u], c[(p.swizzle >> 4u) & 7u], c[(p.swizzle >> 8u) & 7u],
                  c[(p.swizzle >> 12u) & 7u]);
}
)msl";

} // namespace Metal::BlitShaders
