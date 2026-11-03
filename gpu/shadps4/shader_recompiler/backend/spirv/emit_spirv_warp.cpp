#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

Id SubgroupScope(EmitContext& ctx) {
    return ctx.ConstU32(static_cast<u32>(spv::Scope::Subgroup));
}

Id EmitWarpId(EmitContext& ctx) {
    UNREACHABLE();
}

Id EmitLaneId(EmitContext& ctx) {
    return ctx.OpLoad(ctx.U32[1], ctx.subgroup_local_invocation_id);
}

Id EmitQuadBroadcast(EmitContext& ctx, Id value, Id index) {
    // bbport: Apple GPUs do not feed helper lanes into quad shuffles, so a fragment shader that
    // builds its own derivatives from quad lanes (FIFA 16 player normals) reads garbage on partial
    // quads. Rebuild the other lane's value from the hardware fine derivatives instead; with all
    // four lanes known through dx, dy and dxy the bilinear fit is exact.
    static const bool no_quad_fix = std::getenv("BB_NO_QUAD_FIX") != nullptr;
    static const char* only_pgm = std::getenv("BB_QUAD_FIX_PGM");
    const bool pgm_ok = !only_pgm || std::strstr(only_pgm, fmt::format("{:08x}", ctx.info.pgm_hash).c_str());
    if (!no_quad_fix && pgm_ok && ctx.sw_stage == SwStage::Fragment && Sirit::ValidId(ctx.frag_coord)) {
        const Id f32 = ctx.F32[1];
        const Id v = ctx.OpBitcast(f32, value);
        const Id dx = ctx.OpDPdxFine(f32, v);
        const Id dy = ctx.OpDPdyFine(f32, v);
        const Id dxy = ctx.OpDPdyFine(f32, dx);
        const Id coord = ctx.OpLoad(ctx.F32[4], ctx.frag_coord);
        const auto parity = [&](u32 comp) {
            const Id c = ctx.OpConvertFToU(ctx.U32[1], ctx.OpCompositeExtract(f32, coord, comp));
            return ctx.OpConvertUToF(f32, ctx.OpBitwiseAnd(ctx.U32[1], c, ctx.ConstU32(1U)));
        };
        const auto bit = [&](u32 b) {
            const Id x = ctx.OpBitFieldUExtract(ctx.U32[1], index, ctx.ConstU32(b), ctx.ConstU32(1U));
            return ctx.OpConvertUToF(f32, x);
        };
        static const bool swap = std::getenv("BB_QUAD_SWAP") != nullptr;
        static const bool flipy = std::getenv("BB_QUAD_FLIPY") != nullptr;
        const Id ox = ctx.OpFSub(f32, bit(swap ? 1 : 0), parity(0));
        Id oy = ctx.OpFSub(f32, bit(swap ? 0 : 1), parity(1));
        if (flipy) {
            oy = ctx.OpFNegate(f32, oy);
        }
        Id r = ctx.OpFAdd(f32, v, ctx.OpFMul(f32, ox, dx));
        r = ctx.OpFAdd(f32, r, ctx.OpFMul(f32, oy, dy));
        r = ctx.OpFAdd(f32, r, ctx.OpFMul(f32, ctx.OpFMul(f32, ox, oy), dxy));
        return ctx.OpBitcast(ctx.U32[1], r);
    }
    return ctx.OpGroupNonUniformQuadBroadcast(ctx.U32[1], SubgroupScope(ctx), value, index);
}

Id EmitReadFirstLane(EmitContext& ctx, Id value) {
    return ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), value);
}

Id EmitShuffle(EmitContext& ctx, Id value, Id index) {
    return ctx.OpGroupNonUniformShuffle(ctx.U32[1], SubgroupScope(ctx), value, index);
}

Id EmitShuffleXor(EmitContext& ctx, Id value, Id mask) {
    return ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, mask);
}

Id EmitReadLane(EmitContext& ctx, Id value, Id lane) {
    return ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value, lane);
}

Id EmitWriteLane(EmitContext& ctx, Id value, Id write_value, u32 lane) {
    return ctx.u32_zero_value;
}

Id EmitBallot(EmitContext& ctx, Id bit) {
    const Id ballot{ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), bit)};
    return ctx.OpBitcast(ctx.U64, ctx.OpVectorShuffle(ctx.U32[2], ballot, ballot, 0, 1));
}

Id EmitBallotFindLsb(EmitContext& ctx, Id mask) {
    const Id value{ctx.OpCompositeConstruct(ctx.U32[4], ctx.OpBitcast(ctx.U32[2], mask),
                                            ctx.u32_zero_value, ctx.u32_zero_value)};
    return ctx.OpGroupNonUniformBallotFindLSB(ctx.U32[1], SubgroupScope(ctx), value);
}

Id EmitInverseBallot(EmitContext& ctx, Id mask) {
    const Id value{ctx.OpCompositeConstruct(ctx.U32[4], ctx.OpBitcast(ctx.U32[2], mask),
                                            ctx.u32_zero_value, ctx.u32_zero_value)};
    return ctx.OpGroupNonUniformInverseBallot(ctx.U1[1], SubgroupScope(ctx), value);
}

Id EmitGroupAny(EmitContext& ctx, Id bit) {
    return ctx.OpGroupNonUniformAny(ctx.U1[1], SubgroupScope(ctx), bit);
}

} // namespace Shader::Backend::SPIRV
