// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <unordered_set>
#include "common/alignment.h"
#include "common/logging/log.h"
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"

namespace Shader::Optimization {

static bool IsLoadShared(const IR::Inst& inst) {
    return inst.GetOpcode() == IR::Opcode::LoadSharedU16 ||
           inst.GetOpcode() == IR::Opcode::LoadSharedU32 ||
           inst.GetOpcode() == IR::Opcode::LoadSharedU64;
}

static bool IsLdsAtomic(const IR::Inst& inst) {
    const IR::Opcode opcode = inst.GetOpcode();
    return opcode >= IR::Opcode::SharedAtomicIAdd32 &&
           opcode <= IR::Opcode::SharedAtomicCmpSwap64 && !inst.Flags<bool>();
}

static bool IsWriteShared(const IR::Inst& inst) {
    const IR::Opcode opcode = inst.GetOpcode();
    if (opcode >= IR::Opcode::SharedAtomicIAdd32 && opcode <= IR::Opcode::SharedAtomicCmpSwap64) {
        return !inst.Flags<bool>();
    }
    return opcode == IR::Opcode::WriteSharedU16 || opcode == IR::Opcode::WriteSharedU32 ||
           opcode == IR::Opcode::WriteSharedU64;
}

// Inserts barriers when a shared memory write and read occur in the same basic block.
// subgroup: the block may run in divergent control flow, only a SIMD-group memory barrier is legal.
static void EmitBarrierInBlock(IR::Block* block, bool subgroup = false) {
    const auto emit_barrier = [subgroup](IR::IREmitter& ir) {
        if (subgroup) {
            ir.SubgroupBarrier();
        } else {
            ir.Barrier();
        }
    };
    enum class BarrierAction : u32 {
        None,
        BarrierOnWrite,
        BarrierOnRead,
    };
    BarrierAction action{};
    for (IR::Inst& inst : block->Instructions()) {
        if (IsLoadShared(inst)) {
            if (action == BarrierAction::BarrierOnRead) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                emit_barrier(ir);
            }
            action = BarrierAction::BarrierOnWrite;
            continue;
        }
        if (IsWriteShared(inst)) {
            if (action == BarrierAction::BarrierOnWrite) {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                emit_barrier(ir);
            }
            action = BarrierAction::BarrierOnRead;
        }
    }
    if (action != BarrierAction::None) {
        // fifa-hybrid: blocks carry no terminator, so the closing barrier goes after the last
        // instruction (upstream placed it before it, leaving a final LDS write unordered).
        IR::IREmitter ir{*block, block->end()};
        emit_barrier(ir);
    }
}

using NodeSet = std::unordered_set<const IR::Block*>;

static void EmitBarrierAtBlockStart(IR::Block* block) {
    auto insert_point = std::ranges::find_if_not(block->Instructions(), IR::IsPhi);
    IR::IREmitter ir{*block, insert_point};
    ir.Barrier();
}

static bool IsDivergent(const IR::U1& cond) {
    if (cond.IsImmediate()) {
        return false;
    }
    return IR::BreadthFirstSearch(cond, [](IR::Inst* inst) -> std::optional<bool> {
               switch (inst->GetOpcode()) {
               case IR::Opcode::LaneId:
                   return true;
               case IR::Opcode::GetAttributeU32:
                   if (inst->Arg(0).Attribute() == IR::Attribute::LocalInvocationId) {
                       return true;
                   }
                   break;
               default:
                   break;
               }
               return std::nullopt;
           }) == true;
}

// Inserts a barrier after divergent conditional blocks to avoid undefined
// behavior when some threads write and others read from shared memory.
static void EmitBarrierInMergeBlock(const IR::AbstractSyntaxNode::Data& data,
                                    NodeSet& divergence_end, u32& divergence_depth) {
    const IR::U1 cond = data.if_node.cond;
    if (IsDivergent(cond)) {
        if (divergence_depth == 0) {
            EmitBarrierAtBlockStart(data.if_node.merge);
        }
        ++divergence_depth;
        divergence_end.emplace(data.if_node.merge);
    }
}

// A barrier inside a loop is invalid when different invocations leave on different iterations.
// Mark such loops so their shared-memory synchronization can be deferred to the merge block.
static NodeSet FindDivergentLoops(const IR::AbstractSyntaxList& syntax_list) {
    NodeSet divergent_loops;
    for (const IR::AbstractSyntaxNode& node : syntax_list) {
        switch (node.type) {
        case IR::AbstractSyntaxNode::Type::Repeat:
            if (IsDivergent(node.data.repeat.cond)) {
                divergent_loops.emplace(node.data.repeat.merge);
            }
            break;
        case IR::AbstractSyntaxNode::Type::Break:
            if (IsDivergent(node.data.break_node.cond)) {
                divergent_loops.emplace(node.data.break_node.merge);
            }
            break;
        default:
            break;
        }
    }
    return divergent_loops;
}

static bool SelectableType(IR::Type type) {
    return type == IR::Type::U1 || type == IR::Type::U32 || type == IR::Type::F32 ||
           type == IR::Type::U64;
}

// fifa-hybrid: GCN code inside a branch on the thread id (`if (tid < 64) { ... }`: the barrier-free
// tail of an LDS reduction) relies on the whole 64-wide wave running in lockstep. On a host whose
// SIMD group is narrower, half of that wave can run ahead of the other half, and no workgroup
// barrier may be placed inside the branch. When the body is a single block whose only side effects
// are LDS accesses, make the branch uniform instead: every thread executes the body, the formerly
// inactive threads address a scratch LDS word, and the merge block's phis keep their old value for
// them. Workgroup barriers are then legal in the body. Returns false (nothing changed) when the
// shape does not qualify.
static bool ConvertDivergentIf(IR::Program& program, size_t index, u32 dummy_offset) {
    using Type = IR::AbstractSyntaxNode::Type;
    auto& syntax = program.syntax_list;
    if (index + 2 >= syntax.size()) {
        return false;
    }
    auto& if_node = syntax[index].data.if_node;
    const auto& body_node = syntax[index + 1];
    const auto& end_node = syntax[index + 2];
    if (body_node.type != Type::Block || body_node.data.block != if_node.body ||
        end_node.type != Type::EndIf || end_node.data.end_if.merge != if_node.merge) {
        return false;
    }
    IR::Block* body = if_node.body;
    IR::Block* merge = if_node.merge;
    if (body->ImmPredecessors().size() != 1 || merge->ImmPredecessors().size() != 2) {
        return false;
    }
    bool has_read = false;
    bool has_write = false;
    std::vector<IR::Inst*> lds_insts;
    for (IR::Inst& inst : body->Instructions()) {
        if (IR::IsPhi(inst)) {
            return false;
        }
        if (IsLoadShared(inst)) {
            has_read = true;
            lds_insts.push_back(&inst);
            continue;
        }
        if (IsLdsAtomic(inst)) {
            has_read = true;
            has_write = true;
            lds_insts.push_back(&inst);
            continue;
        }
        if (inst.GetOpcode() == IR::Opcode::WriteSharedU16 ||
            inst.GetOpcode() == IR::Opcode::WriteSharedU32 ||
            inst.GetOpcode() == IR::Opcode::WriteSharedU64) {
            has_write = true;
            lds_insts.push_back(&inst);
            continue;
        }
        if (inst.MayHaveSideEffects()) {
            return false;
        }
    }
    if (!has_read || !has_write) {
        return false;
    }
    // The merge block's phis: exactly (body, other) pairs of a selectable type.
    struct PhiFix {
        IR::Inst* phi;
        size_t body_arg;
        size_t other_arg;
    };
    std::vector<PhiFix> phi_fixes;
    for (IR::Inst& inst : merge->Instructions()) {
        if (!IR::IsPhi(inst)) {
            break;
        }
        if (inst.NumArgs() != 2) {
            return false;
        }
        const size_t body_arg = inst.PhiBlock(0) == body ? 0 : inst.PhiBlock(1) == body ? 1 : 2;
        if (body_arg == 2) {
            return false;
        }
        const size_t other_arg = 1 - body_arg;
        const IR::Type type = inst.Arg(body_arg).Type();
        if (!SelectableType(type) || inst.Arg(other_arg).Type() != type) {
            return false;
        }
        phi_fixes.push_back({&inst, body_arg, other_arg});
    }

    const IR::U1 cond = if_node.cond;
    for (IR::Inst* inst : lds_insts) {
        IR::IREmitter ir{*body, IR::Block::InstructionList::s_iterator_to(*inst)};
        const IR::U32 address{inst->Arg(0)};
        inst->SetArg(0, ir.Select(cond, address, ir.Imm32(dummy_offset)));
    }
    for (const PhiFix& fix : phi_fixes) {
        IR::IREmitter ir{*body, body->end()};
        fix.phi->SetArg(fix.body_arg,
                        ir.Select(cond, fix.phi->Arg(fix.body_arg), fix.phi->Arg(fix.other_arg)));
    }
    if_node.cond = IR::IREmitter{}.Imm1(true);
    return true;
}

static constexpr u32 GcnSubgroupSize = 64;

void SharedMemoryBarrierPass(IR::Program& program, const RuntimeInfo& runtime_info,
                             const Profile& profile) {
    if (program.info.hw_stage != HwStage::Compute) {
        return;
    }
    const auto& cs_info = runtime_info.hw.cs;
    const u32 shared_memory_size = cs_info.shared_memory_size;
    const u32 threadgroup_size =
        cs_info.workgroup_size[0] * cs_info.workgroup_size[1] * cs_info.workgroup_size[2];
    if (shared_memory_size == 0) {
        return;
    }
    // fifa-hybrid: the host SIMD group is narrower than a GCN wave (Apple: 32). Any workgroup
    // larger than it can have lockstep-dependent LDS code that spans two SIMD groups.
    const bool narrow_host_wave =
        profile.subgroup_size != 0 && profile.subgroup_size < GcnSubgroupSize;
    const bool only_subgroup_barriers =
        narrow_host_wave ? threadgroup_size <= profile.subgroup_size
                         : (threadgroup_size != GcnSubgroupSize || !profile.needs_lds_barriers);
    if (only_subgroup_barriers) {
        // (upstream shadPS4) GCN waves can use shared memory between invocations without explicit
        // barriers because they run in lockstep. Host compilers can reorder these accesses, so
        // keep their order with subgroup memory barriers.
        for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
            if (node.type == IR::AbstractSyntaxNode::Type::Block) {
                EmitBarrierInBlock(node.data.block, true);
            }
        }
        return;
    }
    // Scratch LDS word for the if-conversion (after the wave64 lowering's scratch area, which
    // LowerWave64BallotPass allocates at the same place).
    u32 dummy_offset = 0;
    u32 scratch_needed = 0;
    bool convert = narrow_host_wave;
    if (const char* env = std::getenv("BB_LDS_IFCONV"); env && env[0] == '0') {
        convert = false;
    }
    if (convert) {
        const u32 wave64_scratch =
            (Common::AlignUp(threadgroup_size, GcnSubgroupSize) / profile.subgroup_size) *
            static_cast<u32>(sizeof(u32));
        dummy_offset = Common::AlignUp(shared_memory_size, u32(sizeof(u64))) + wave64_scratch;
        scratch_needed = dummy_offset + u32(sizeof(u64)) - shared_memory_size;
        if (profile.max_shared_memory_size != 0 &&
            shared_memory_size + scratch_needed > profile.max_shared_memory_size) {
            LOG_WARNING(Render_Recompiler,
                        "Shader {:#x}: no LDS left for the lockstep if-conversion ({} + {} bytes)",
                        program.info.pgm_hash, shared_memory_size, scratch_needed);
            convert = false;
        }
    }
    using Type = IR::AbstractSyntaxNode::Type;
    u32 divergence_depth{};
    u32 converted{};
    NodeSet divergence_end;
    const NodeSet divergent_loops = FindDivergentLoops(program.syntax_list);
    auto& syntax = program.syntax_list;
    for (size_t i = 0; i < syntax.size(); ++i) {
        const IR::AbstractSyntaxNode& node = syntax[i];
        if (node.type == Type::EndIf) {
            if (divergence_end.contains(node.data.end_if.merge)) {
                --divergence_depth;
            }
            continue;
        }
        if (node.type == Type::If) {
            if (convert && divergence_depth == 0 && IsDivergent(node.data.if_node.cond) &&
                ConvertDivergentIf(program, i, dummy_offset)) {
                // Now uniform: its body block is handled below at depth 0 with full barriers.
                ++converted;
                continue;
            }
            // Check if branch depth is zero, we don't want to insert barrier in potentially
            // divergent code.
            EmitBarrierInMergeBlock(node.data, divergence_end, divergence_depth);
            continue;
        }
        if (node.type == Type::Loop && divergent_loops.contains(node.data.loop.merge)) {
            ++divergence_depth;
            continue;
        }
        if (node.type == Type::Repeat && divergent_loops.contains(node.data.repeat.merge)) {
            ASSERT(divergence_depth > 0);
            --divergence_depth;
            if (divergence_depth == 0) {
                EmitBarrierAtBlockStart(node.data.repeat.merge);
            }
            continue;
        }
        if (node.type == Type::Block) {
            // Uniform: workgroup barriers; inside divergent control flow only SIMD-group memory
            // barriers are legal (best effort for lockstep code the conversion cannot reach).
            EmitBarrierInBlock(node.data.block, divergence_depth != 0);
        }
    }
    if (converted != 0) {
        program.info.shared_memory_scratch_size =
            std::max(program.info.shared_memory_scratch_size, scratch_needed);
        LOG_INFO(Render_Recompiler,
                 "Shader {:#x}: {} divergent LDS branch(es) made uniform (threadgroup {}, scratch "
                 "word at {:#x})",
                 program.info.pgm_hash, converted, threadgroup_size, dummy_offset);
    }
}

} // namespace Shader::Optimization
