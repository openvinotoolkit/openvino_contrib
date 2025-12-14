// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"

#include <sstream>
#include <unordered_set>

#include "openvino/core/except.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace ov {
namespace metal_plugin {
namespace {

struct EltwisePattern {
    mlir::scf::ForOp loop;
    mlir::Operation* arith_op = nullptr;
};

EltwisePattern find_eltwise_pattern(mlir::func::FuncOp func) {
    EltwisePattern p{};
    func.walk([&](mlir::scf::ForOp op) {
        if (!p.loop) p.loop = op;
    });
    if (!p.loop) return p;
    for (auto& op : p.loop.getBody()->getOperations()) {
        if (mlir::isa<mlir::arith::AddFOp, mlir::arith::SubFOp, mlir::arith::MulFOp>(op)) {
            p.arith_op = &op;
            break;
        }
    }
    return p;
}

std::string op_to_msl(KernelOpKind k) {
    switch (k) {
        case KernelOpKind::ElementwiseAdd: return " + ";
        case KernelOpKind::ElementwiseSub: return " - ";
        case KernelOpKind::ElementwiseMul: return " * ";
        case KernelOpKind::ElementwiseDiv: return " / ";
        case KernelOpKind::ElementwisePow: return "pow";  // handled specially
        case KernelOpKind::ElementwiseMod: return "fmod";  // handled specially
        case KernelOpKind::ElementwiseFloorMod: return "floormod";  // placeholder
        default:
            OPENVINO_THROW("Eltwise codegen: unsupported op kind");
    }
}

std::string emit_eltwise_msl(const EltwiseCodegenDesc& d) {
    const bool is_int32 = d.element_type == ov::element::i32;
    const bool is_int64 = d.element_type == ov::element::i64;
    const bool is_int = is_int32 || is_int64;
    const std::string scalar_ty = is_int64 ? "long" : (is_int32 ? "int" : "float");
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    if (is_int && d.eltwise_kind == KernelOpKind::ElementwisePow) {
        ss << "inline " << scalar_ty << " ipow(" << scalar_ty << " base, " << scalar_ty
           << " exp) {\n";
        ss << "    if (exp < 0) return 0;\n";
        ss << "    " << scalar_ty << " result = 1;\n";
        ss << "    " << scalar_ty << " b = base;\n";
        ss << "    int e = static_cast<int>(exp);\n";
        ss << "    while (e > 0) {\n";
        ss << "        if (e & 1) result *= b;\n";
        ss << "        e >>= 1;\n";
        ss << "        if (e) b *= b;\n";
        ss << "    }\n";
        ss << "    return result;\n";
        ss << "}\n";
    }
    const bool use_half = d.use_half_compute;
    auto emit_op = [&](const std::string& a, const std::string& b) -> std::string {
        if (d.eltwise_kind == KernelOpKind::ElementwisePow) {
            if (is_int) {
                return "ipow(" + a + ", " + b + ")";
            }
            return "pow(" + a + "," + b + ")";
        }
        if (d.eltwise_kind == KernelOpKind::ElementwiseMod ||
            d.eltwise_kind == KernelOpKind::ElementwiseFloorMod) {
            // floor_mod = fmod(a, b) adjusted for sign (same as numpy)
            return "";  // handled as a statement block below
        }
        return a + op_to_msl(d.eltwise_kind) + b;
    };

    auto emit_floor_mod_block = [&](const std::string& a, const std::string& b, const std::string& dst) {
        if (is_int) {
            // numpy-style mod: result has the sign of the divisor.
            ss << "    " << scalar_ty << " _a = " << a << ";\n";
            ss << "    " << scalar_ty << " _b = " << b << ";\n";
            ss << "    " << scalar_ty << " _r = _a % _b;\n";
            ss << "    if (_r != 0 && ((_r < 0) != (_b < 0))) _r += _b;\n";
            ss << "    " << dst << " = _r;\n";
        } else {
            ss << "    float _a = " << a << ";\n";
            ss << "    float _b = " << b << ";\n";
            ss << "    float _r = fmod(_a, _b);\n";
            ss << "    if (_r != 0.0f && ((_r < 0.0f) != (_b < 0.0f))) _r += _b;\n";
            ss << "    " << dst << " = _r;\n";
        }
    };

    ss << "kernel void eltwise_kernel(\n";
    ss << "  device const " << scalar_ty << "* A [[buffer(0)]],\n";
    ss << "  device const " << scalar_ty << "* B [[buffer(1)]],\n";
    ss << "  device " << scalar_ty << "* C [[buffer(2)]],\n";
    ss << "  constant uint& NUM_ELEMS [[buffer(3)]],\n";
    ss << "  constant uint& RANK [[buffer(4)]],\n";
    ss << "  constant int* out_dims [[buffer(5)]],\n";
    ss << "  constant int* stride0 [[buffer(6)]],\n";
    ss << "  constant int* stride1 [[buffer(7)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= NUM_ELEMS) return;\n";
    if (!d.is_broadcast) {
        if (d.eltwise_kind == KernelOpKind::ElementwiseFloorMod ||
            d.eltwise_kind == KernelOpKind::ElementwiseMod) {
            emit_floor_mod_block("A[gid]", "B[gid]", "C[gid]");
        } else if (!is_int && use_half && d.eltwise_kind == KernelOpKind::ElementwiseDiv) {
            ss << "    half a = half(A[gid]);\n";
            ss << "    half b = half(B[gid]);\n";
            ss << "    C[gid] = float(a / b);\n";
        } else if (is_int && d.eltwise_kind == KernelOpKind::ElementwiseDiv) {
            ss << "    " << scalar_ty << " b = B[gid];\n";
            ss << "    C[gid] = b == 0 ? 0 : (A[gid] / b);\n";
        } else {
            ss << "    C[gid] = " << emit_op("A[gid]", "B[gid]") << ";\n";
        }
        ss << "}\n";
        return ss.str();
    }
    ss << "    uint idx = gid;\n";
    ss << "    int off0 = 0; int off1 = 0;\n";
    ss << "    for (int d = (int)RANK - 1; d >= 0; --d) {\n";
    ss << "        int coord = idx % out_dims[d];\n";
    ss << "        idx /= out_dims[d];\n";
    ss << "        off0 += (stride0[d] == 0 ? 0 : coord * stride0[d]);\n";
    ss << "        off1 += (stride1[d] == 0 ? 0 : coord * stride1[d]);\n";
    ss << "    }\n";
    if (d.eltwise_kind == KernelOpKind::ElementwiseFloorMod ||
        d.eltwise_kind == KernelOpKind::ElementwiseMod) {
        emit_floor_mod_block("A[off0]", "B[off1]", "C[gid]");
    } else if (!is_int && use_half && d.eltwise_kind == KernelOpKind::ElementwiseDiv) {
        ss << "    half a = half(A[off0]);\n";
        ss << "    half b = half(B[off1]);\n";
        ss << "    C[gid] = float(a / b);\n";
    } else if (is_int && d.eltwise_kind == KernelOpKind::ElementwiseDiv) {
        ss << "    " << scalar_ty << " b = B[off1];\n";
        ss << "    C[gid] = b == 0 ? 0 : (A[off0] / b);\n";
    } else {
        ss << "    C[gid] = " << emit_op("A[off0]", "B[off1]") << ";\n";
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace

std::string generate_msl_for_eltwise(const EltwiseCodegenDesc& d, mlir::ModuleOp module) {
    if (module) {
        auto func_it = module.getOps<mlir::func::FuncOp>().begin();
        if (func_it != module.getOps<mlir::func::FuncOp>().end()) {
            auto pat = find_eltwise_pattern(*func_it);
            OPENVINO_ASSERT(pat.loop && pat.arith_op, "Eltwise MLIR pattern not recognized");
        }
    }
    return emit_eltwise_msl(d);
}

}  // namespace metal_plugin
}  // namespace ov
