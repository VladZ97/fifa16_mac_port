// SPDX-FileCopyrightText: Copyright 2026 fifa-hybrid contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// 0.5.1-c: recognises a compute program that copies one sampled 2D image into one storage image
// texel for texel (info.h InfoPersistent::CopyDispatch). The renderer then draws the copy as a
// fragment pass (vk_rasterizer.cpp PlanCopyDispatchPass) instead of a compute dispatch: on Apple
// GPUs a full-screen storage-image write costs several times a colour attachment store.
// Matched shape (FIFA 16's 240x135 and 120x67 copies): one ImageSampleExplicitLod at LOD 0 of
// ((wg * size + lid) + 0.5) * (two dwords of a constant buffer), one ImageWrite at (wg * size +
// lid), the written colour a component swizzle of the sample (constants 0 and 1 allowed), no other
// memory access and no control flow. BB_COPY_CS_DETECT=0 disables, BB_COPY_CS_LOG=1 logs.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/abstract_syntax_list.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Optimization {
namespace {

bool Allowed(IR::Opcode op) {
    switch (op) {
    case IR::Opcode::Void:
    case IR::Opcode::ConditionRef:
    case IR::Opcode::Prologue:
    case IR::Opcode::Epilogue:
    case IR::Opcode::GetAttributeU32:
    case IR::Opcode::GetUserData:
    case IR::Opcode::ReadConst:
    case IR::Opcode::ReadConstBuffer:
    case IR::Opcode::BitCastF32U32:
    case IR::Opcode::BitCastU32F32:
    case IR::Opcode::ConvertF32U32:
    case IR::Opcode::ConvertF32S32:
    case IR::Opcode::FPAdd32:
    case IR::Opcode::FPMul32:
    case IR::Opcode::FPFma32:
    case IR::Opcode::IAdd32:
    case IR::Opcode::IMul32:
    case IR::Opcode::ShiftLeftLogical32:
    case IR::Opcode::ShiftRightLogical32:
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::BitwiseOr32:
    case IR::Opcode::BitFieldUExtract:
    case IR::Opcode::CompositeConstructU32x2:
    case IR::Opcode::CompositeConstructU32x3:
    case IR::Opcode::CompositeConstructU32x4:
    case IR::Opcode::CompositeExtractU32x2:
    case IR::Opcode::CompositeExtractU32x3:
    case IR::Opcode::CompositeExtractU32x4:
    case IR::Opcode::CompositeConstructF32x2:
    case IR::Opcode::CompositeConstructF32x3:
    case IR::Opcode::CompositeConstructF32x4:
    case IR::Opcode::CompositeExtractF32x2:
    case IR::Opcode::CompositeExtractF32x3:
    case IR::Opcode::CompositeExtractF32x4:
    case IR::Opcode::CompositeShuffleF32x4:
    case IR::Opcode::CompositeShuffleU32x4:
    case IR::Opcode::ImageSampleExplicitLod:
    case IR::Opcode::ImageWrite:
        return true;
    default:
        return false;
    }
}

IR::Value Strip(IR::Value v) {
    return v; // this IR has no identity instructions after SSA rewriting
}

IR::Inst* InstOf(const IR::Value& v0) {
    const IR::Value v = Strip(v0);
    return v.IsEmpty() || v.IsImmediate() ? nullptr : v.Inst();
}

bool IsImmU32(const IR::Value& v0, u32 x) {
    const IR::Value v = Strip(v0);
    return !v.IsEmpty() && v.IsImmediate() && v.Type() == IR::Type::U32 && v.U32() == x;
}

bool IsImmF32(const IR::Value& v0, f32 x) {
    const IR::Value v = Strip(v0);
    return !v.IsEmpty() && v.IsImmediate() && v.Type() == IR::Type::F32 && v.F32() == x;
}

bool IsAttribute(const IR::Value& v, IR::Attribute attribute, u32 comp) {
    IR::Inst* const inst = InstOf(v);
    if (!inst || inst->GetOpcode() != IR::Opcode::GetAttributeU32) {
        return false;
    }
    const IR::Value attr = inst->Arg(0);
    return attr.IsImmediate() && attr.Type() == IR::Type::Attribute &&
           attr.Attribute() == attribute && IsImmU32(inst->Arg(1), comp);
}

u32 Log2Exact(u32 size) {
    for (u32 i = 0; i < 32; ++i) {
        if ((1u << i) == size) {
            return i;
        }
    }
    return 32;
}

/// wg * size + lid (either operand order; the multiply as a shift or an IMul32).
bool IsGlobalId(const IR::Value& v, u32 comp, u32 size) {
    IR::Inst* const add = InstOf(v);
    if (!add || add->GetOpcode() != IR::Opcode::IAdd32) {
        return false;
    }
    for (int order = 0; order < 2; ++order) {
        if (!IsAttribute(add->Arg(1 - order), IR::Attribute::LocalInvocationId, comp)) {
            continue;
        }
        IR::Inst* const mul = InstOf(add->Arg(order));
        if (!mul) {
            continue;
        }
        if (mul->GetOpcode() == IR::Opcode::ShiftLeftLogical32 &&
            IsAttribute(mul->Arg(0), IR::Attribute::WorkgroupId, comp) &&
            Log2Exact(size) < 32 && IsImmU32(mul->Arg(1), Log2Exact(size))) {
            return true;
        }
        if (mul->GetOpcode() == IR::Opcode::IMul32) {
            for (int o2 = 0; o2 < 2; ++o2) {
                if (IsAttribute(mul->Arg(o2), IR::Attribute::WorkgroupId, comp) &&
                    IsImmU32(mul->Arg(1 - o2), size)) {
                    return true;
                }
            }
        }
    }
    return false;
}

/// float(gid) + 0.5
bool IsCenter(const IR::Value& v, u32 comp, u32 size) {
    IR::Inst* const add = InstOf(v);
    if (!add || add->GetOpcode() != IR::Opcode::FPAdd32) {
        return false;
    }
    for (int order = 0; order < 2; ++order) {
        if (!IsImmF32(add->Arg(1 - order), 0.5f)) {
            continue;
        }
        IR::Inst* const cvt = InstOf(add->Arg(order));
        if (cvt && cvt->GetOpcode() == IR::Opcode::ConvertF32U32 &&
            IsGlobalId(cvt->Arg(0), comp, size)) {
            return true;
        }
    }
    return false;
}

/// bitcast<f32>(constant buffer dword): the sample scale.
bool IsScale(const IR::Value& v, u32& buffer, u32& dword) {
    IR::Inst* const cast = InstOf(v);
    if (!cast || cast->GetOpcode() != IR::Opcode::BitCastF32U32) {
        return false;
    }
    IR::Inst* const load = InstOf(cast->Arg(0));
    if (!load || load->GetOpcode() != IR::Opcode::ReadConstBuffer) {
        return false;
    }
    const IR::Value handle = Strip(load->Arg(0)), index = Strip(load->Arg(1));
    if (handle.IsEmpty() || index.IsEmpty() || !handle.IsImmediate() || !index.IsImmediate() ||
        handle.Type() != IR::Type::U32 || index.Type() != IR::Type::U32) {
        return false;
    }
    buffer = handle.U32();
    dword = index.U32();
    return true;
}

/// (float(gid) + 0.5) * scale
bool IsCoord(const IR::Value& v, u32 comp, u32 size, u32& buffer, u32& dword) {
    IR::Inst* const mul = InstOf(v);
    if (!mul || mul->GetOpcode() != IR::Opcode::FPMul32) {
        return false;
    }
    for (int order = 0; order < 2; ++order) {
        if (IsCenter(mul->Arg(order), comp, size) && IsScale(mul->Arg(1 - order), buffer, dword)) {
            return true;
        }
    }
    return false;
}

/// Component `idx` of the written colour: a sample component (0-3), zero (4) or one (5).
bool ResolveComponent(const IR::Value& v, u32 idx, const IR::Inst* sample, u8& code, int depth) {
    if (depth > 8) {
        return false;
    }
    IR::Inst* const inst = InstOf(v);
    if (!inst) {
        return false;
    }
    if (inst == sample) {
        code = u8(idx);
        return true;
    }
    switch (inst->GetOpcode()) {
    case IR::Opcode::CompositeConstructF32x4: {
        const IR::Value c = Strip(inst->Arg(idx));
        if (!c.IsEmpty() && c.IsImmediate()) {
            if (c.Type() != IR::Type::F32) {
                return false;
            }
            if (c.F32() == 0.0f) {
                code = 4;
                return true;
            }
            if (c.F32() == 1.0f) {
                code = 5;
                return true;
            }
            return false;
        }
        IR::Inst* const ci = InstOf(c);
        if (ci && ci->GetOpcode() == IR::Opcode::CompositeExtractF32x4) {
            const IR::Value j = Strip(ci->Arg(1));
            if (j.IsEmpty() || !j.IsImmediate() || j.U32() > 3) {
                return false;
            }
            return ResolveComponent(ci->Arg(0), j.U32(), sample, code, depth + 1);
        }
        return false;
    }
    case IR::Opcode::CompositeShuffleF32x4: {
        const IR::Value j = Strip(inst->Arg(2 + idx));
        if (j.IsEmpty() || !j.IsImmediate() || j.U32() > 7) {
            return false;
        }
        const u32 k = j.U32();
        return k < 4 ? ResolveComponent(inst->Arg(0), k, sample, code, depth + 1)
                     : ResolveComponent(inst->Arg(1), k - 4, sample, code, depth + 1);
    }
    default:
        return false;
    }
}

} // namespace

void CopyDispatchDetectPass(IR::Program& program, const RuntimeInfo& runtime_info) {
    Info& info = program.info;
    info.copy_dispatch = {};
    if (info.sw_stage != SwStage::Compute) {
        return;
    }
    static const bool off = [] {
        const char* e = std::getenv("BB_COPY_CS_DETECT");
        return e && e[0] == '0';
    }();
    static const bool log = std::getenv("BB_COPY_CS_LOG") != nullptr;
    if (off) {
        return;
    }
    const auto decline = [&](const char* why) {
        if (log) {
            std::fprintf(stderr, "[COPYCS] cs %016llx: not a copy dispatch (%s)\n",
                         (unsigned long long)info.pgm_hash, why);
        }
    };
    const auto& wg = runtime_info.hw.cs.workgroup_size;
    if (wg[2] != 1 || wg[0] == 0 || wg[1] == 0 || wg[0] > 1024 || wg[1] > 1024) {
        return decline("workgroup size");
    }
    for (const auto& node : program.syntax_list) {
        if (node.type != IR::AbstractSyntaxNode::Type::Block &&
            node.type != IR::AbstractSyntaxNode::Type::Return &&
            node.type != IR::AbstractSyntaxNode::Type::Unreachable) {
            return decline("control flow");
        }
    }
    IR::Inst* write = nullptr;
    IR::Inst* sample = nullptr;
    for (IR::Block* block : program.blocks) {
        for (IR::Inst& inst : *block) {
            const IR::Opcode op = inst.GetOpcode();
            if (!Allowed(op)) {
                if (log) {
                    std::fprintf(stderr, "[COPYCS] cs %016llx: not a copy dispatch (opcode %d)\n",
                                 (unsigned long long)info.pgm_hash, int(op));
                }
                return;
            }
            if (op == IR::Opcode::ImageWrite) {
                if (write) {
                    return decline("two writes");
                }
                write = &inst;
            } else if (op == IR::Opcode::ImageSampleExplicitLod) {
                if (sample) {
                    return decline("two samples");
                }
                sample = &inst;
            }
        }
    }
    if (!write || !sample) {
        return decline("no write or no sample");
    }
    const IR::Value write_handle = Strip(write->Arg(0));
    if (write_handle.IsEmpty() || !write_handle.IsImmediate()) {
        return decline("write handle");
    }
    const u32 dst = write_handle.U32() & 0xFFFF;
    IR::Inst* const coords = InstOf(write->Arg(1));
    if (!coords || coords->GetOpcode() != IR::Opcode::CompositeConstructU32x2) {
        return decline("write coordinates");
    }
    if (!IsGlobalId(coords->Arg(0), 0, wg[0]) || !IsGlobalId(coords->Arg(1), 1, wg[1])) {
        return decline("write coordinates are not the global id");
    }
    const IR::Value write_lod = Strip(write->Arg(2)), write_ms = Strip(write->Arg(3));
    if ((!write_lod.IsEmpty() && !IsImmU32(write_lod, 0)) ||
        (!write_ms.IsEmpty() && !IsImmU32(write_ms, 0))) {
        return decline("write level or sample");
    }
    const IR::Value sample_handle = Strip(sample->Arg(0));
    if (sample_handle.IsEmpty() || !sample_handle.IsImmediate()) {
        return decline("sample handle");
    }
    const u32 src = sample_handle.U32() & 0xFFFF, smp = sample_handle.U32() >> 16;
    IR::Inst* const body = InstOf(sample->Arg(1));
    if (!body || body->GetOpcode() != IR::Opcode::CompositeConstructF32x2) {
        return decline("sample coordinates");
    }
    u32 buffer_x = 0, dword_x = 0, buffer_y = 0, dword_y = 0;
    if (!IsCoord(body->Arg(0), 0, wg[0], buffer_x, dword_x) ||
        !IsCoord(body->Arg(1), 1, wg[1], buffer_y, dword_y) || buffer_x != buffer_y) {
        return decline("sample coordinates are not (gid + 0.5) * scale");
    }
    if (!IsImmF32(sample->Arg(2), 0.0f)) {
        return decline("sample level");
    }
    const IR::Value offset = Strip(sample->Arg(3));
    if (!offset.IsEmpty() && !(offset.IsImmediate() && offset.Type() == IR::Type::U32 && offset.U32() == 0)) {
        return decline("sample offset");
    }
    if (src == dst || dst >= info.images.size() || src >= info.images.size() ||
        smp >= info.samplers.size() || buffer_x >= info.buffers.size() || dst > 255 || src > 255 ||
        smp > 255 || dword_x > 0xFFFF || dword_y > 0xFFFF) {
        return decline("resource indices");
    }
    if (!info.images[dst].is_written || info.images[src].is_written) {
        return decline("image roles");
    }
    u8 swizzle[4] = {0, 1, 2, 3};
    for (u32 i = 0; i < 4; ++i) {
        if (!ResolveComponent(write->Arg(4), i, sample, swizzle[i], 0)) {
            return decline("written colour is not a swizzle of the sample");
        }
    }
    // The write was patched for this permutation's destination (its channel select, inverted:
    // resource_patching_pass.cpp). Other permutations share this record, so it keeps the
    // shader's own swizzle and the renderer applies the bound destination's select.
    const AmdGpu::CompMapping dst_select = info.images[dst].GetSharp(info).DstSelect();
    auto& cd = info.copy_dispatch;
    if (!cd.Decompose(dst_select, swizzle)) {
        cd = {};
        return decline("written colour does not fit the destination's channel select");
    }
    cd.valid = 2;
    cd.dst_image = u8(dst);
    cd.src_image = u8(src);
    cd.src_sampler = u8(smp);
    cd.scale_buffer = u8(buffer_x);
    cd.scale_dword_x = u16(dword_x);
    cd.scale_dword_y = u16(dword_y);
    cd.wg_x = u16(wg[0]);
    cd.wg_y = u16(wg[1]);
    if (log) {
        std::fprintf(stderr,
                     "[COPYCS] cs %016llx: copy dispatch img%u <- img%u (sampler %u), scale buffer "
                     "%u dwords %u/%u, workgroup %ux%u, written %u%u%u%u for select %u%u%u%u, "
                     "shader swizzle %u%u%u%u\n",
                     (unsigned long long)info.pgm_hash, dst, src, smp, buffer_x, dword_x, dword_y,
                     wg[0], wg[1], swizzle[0], swizzle[1], swizzle[2], swizzle[3],
                     u32(dst_select.r), u32(dst_select.g), u32(dst_select.b), u32(dst_select.a),
                     cd.swizzle[0], cd.swizzle[1], cd.swizzle[2], cd.swizzle[3]);
    }
}

} // namespace Shader::Optimization
