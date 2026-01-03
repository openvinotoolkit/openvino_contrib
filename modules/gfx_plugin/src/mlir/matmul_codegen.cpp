// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "openvino/core/except.hpp"
#include "runtime/gfx_activation.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

struct LoopInfo {
    int64_t lower = 0;
    int64_t upper = -1;
    int64_t step = 1;
    bool has_static_bounds = false;
    mlir::scf::ForOp op;
};

std::optional<int64_t> get_constant_int(mlir::Value v) {
    if (auto c = v.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        return c.value();
    }
    if (auto c = v.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) {
            return int_attr.getInt();
        }
    }
    return std::nullopt;
}

LoopInfo make_loop_info(mlir::scf::ForOp for_op) {
    LoopInfo info;
    info.op = for_op;
    if (auto lb = get_constant_int(for_op.getLowerBound()))
        info.lower = *lb;
    if (auto ub = get_constant_int(for_op.getUpperBound()))
        info.upper = *ub;
    if (auto st = get_constant_int(for_op.getStep()))
        info.step = *st;
    info.has_static_bounds = (info.upper >= 0);
    return info;
}

std::vector<LoopInfo> collect_loop_nest(mlir::scf::ForOp root) {
    std::vector<LoopInfo> loops;
    mlir::scf::ForOp cur = root;
    while (cur) {
        loops.push_back(make_loop_info(cur));
        auto inner = cur.getBody()->getOps<mlir::scf::ForOp>();
        if (inner.empty())
            break;
        cur = *inner.begin();
    }
    return loops;
}

mlir::func::FuncOp find_kernel_func(mlir::ModuleOp module) {
    if (auto func = module.lookupSymbol<mlir::func::FuncOp>("matmul_main"))
        return func;
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        return func;
    }
    return nullptr;
}

void validate_against_desc(const std::vector<LoopInfo>& loops, const MatMulCodegenDesc& desc) {
    auto check_dim = [&](size_t idx, int64_t expected) {
        if (idx >= loops.size())
            return;
        const auto& li = loops[idx];
        if (li.has_static_bounds) {
            const int64_t span = (li.upper - li.lower + (li.step - li.step %  li.step)) / li.step;
            if (span != expected) {
                OPENVINO_THROW("MLIR MatMul loop bounds mismatch at level ", idx, ": expected ", expected, " got ", span);
            }
        }
    };
    check_dim(0, desc.M);
    check_dim(1, desc.N);
    check_dim(2, desc.K);
}

std::string emit_matmul_msl(const MatMulCodegenDesc& desc, const std::string& scalar) {
    const bool use_half = (scalar == "half");
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "constant uint M = " << desc.M << ";\n";
    ss << "constant uint N = " << desc.N << ";\n";
    ss << "constant uint K = " << desc.K << ";\n";
    ss << "constant uint BATCH = " << desc.batch << ";\n";
    ss << "constant uint BATCH_A = " << desc.batch_a << ";\n";
    ss << "constant uint BATCH_B = " << desc.batch_b << ";\n";
    ss << "constant bool B_IS_NK = " << (desc.b_is_nk_layout ? "true" : "false") << ";\n";
    ss << "constant bool A_TRANSPOSE = " << (desc.a_transpose ? "true" : "false") << ";\n";
    if (desc.has_bias) {
        ss << "constant uint BIAS_B = " << desc.bias_dims[0] << ";\n";
        ss << "constant uint BIAS_M = " << desc.bias_dims[1] << ";\n";
        ss << "constant uint BIAS_N = " << desc.bias_dims[2] << ";\n";
    }
    ss << "kernel void matmul_kernel(\n";
    ss << "  device const " << scalar << "* A [[buffer(0)]],\n";
    ss << "  device const " << scalar << "* B [[buffer(1)]],\n";
    if (desc.has_bias) {
        ss << "  device const " << scalar << "* bias [[buffer(2)]],\n";
        ss << "  device " << scalar << "* C [[buffer(3)]],\n";
    } else {
        ss << "  device " << scalar << "* C [[buffer(2)]],\n";
    }
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    uint total = BATCH * M * N;\n";
    ss << "    if (gid >= total) return;\n";
    ss << "    uint batch = gid / (M * N);\n";
    ss << "    uint idx = gid - batch * M * N;\n";
    ss << "    uint row = idx / N;\n";
    ss << "    uint col = idx - row * N;\n";
    ss << "    if (row < M && col < N) {\n";
    ss << "        uint batch_a = (BATCH_A == 1) ? 0 : batch;\n";
    ss << "        uint batch_b = (BATCH_B == 1) ? 0 : batch;\n";
        ss << "        device const " << scalar << "* Ap = A + batch_a * M * K;\n";
        ss << "        device const " << scalar << "* Bp = B + batch_b * K * N;\n";
    ss << "        float acc = 0.0f;\n";
    ss << "        for (uint k = 0; k < K; ++k) {\n";
    ss << "            float a = static_cast<float>(A_TRANSPOSE ? Ap[k * M + row] : Ap[row * K + k]);\n";
    ss << "            float b = static_cast<float>(B_IS_NK ? Bp[col * K + k] : Bp[k * N + col]);\n";
    ss << "            acc += a * b;\n";
    ss << "        }\n";
    if (desc.has_bias) {
        ss << "        uint bb = (BIAS_B == 1) ? 0 : batch;\n";
        ss << "        uint bm = (BIAS_M == 1) ? 0 : row;\n";
        ss << "        uint bn = (BIAS_N == 1) ? 0 : col;\n";
        ss << "        uint bias_idx = (bb * BIAS_M + bm) * BIAS_N + bn;\n";
        ss << "        acc += static_cast<float>(bias[bias_idx]);\n";
    }
    if (desc.has_activation) {
        auto act = [&]() -> std::string {
            switch (desc.activation) {
                case ActivationKind::Relu: return "max(x, 0.0f)";
                case ActivationKind::Sigmoid: return "1.0f / (1.0f + exp(-x))";
                case ActivationKind::Tanh: return "tanh(x)";
                case ActivationKind::Elu:
                    return "(x >= 0.0f) ? x : " + std::to_string(desc.alpha) + " * (exp(x) - 1.0f)";
                case ActivationKind::Prelu:
                    return "(x >= 0.0f) ? x : x * " + std::to_string(desc.alpha);
                case ActivationKind::Gelu:
                    return "0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)))";
                case ActivationKind::Swish:
                    return "x / (1.0f + exp(-x))";
                case ActivationKind::HSwish:
                    return "x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
                case ActivationKind::HSigmoid:
                    return "clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
                case ActivationKind::Abs:
                    return "fabs(x)";
                case ActivationKind::Sign:
                    return "(x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)";
                default:
                    return "x";
            }
        }();
        ss << "        float x = acc;\n";
        ss << "        acc = " << act << ";\n";
    }
    if (use_half) {
        ss << "        C[(batch * M + row) * N + col] = static_cast<" << scalar << ">(acc);\n";
    } else {
        ss << "        C[(batch * M + row) * N + col] = acc;\n";
    }
    ss << "    }\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace

std::string generate_msl_for_matmul(const MatMulCodegenDesc& desc, mlir::ModuleOp module) {
    OPENVINO_ASSERT(desc.M > 0 && desc.N > 0 && desc.K > 0, "MatMul dims must be positive");
    std::string scalar = "float";
    if (auto func = get_entry_func(module)) {
        auto ft = func.getFunctionType();
        if (ft.getNumInputs() >= 1) {
            scalar = msl_type_from_mlir(ft.getInput(0));
        }
    } else {
        scalar = (desc.element_type == ov::element::f16) ? "half" : "float";
    }
    if (!module) {
        return emit_matmul_msl(desc, scalar);
    }

    auto func = find_kernel_func(module);
    if (!func) {
        return emit_matmul_msl(desc, scalar);
    }

    mlir::scf::ForOp outer_for = nullptr;
    func.walk([&](mlir::scf::ForOp for_op) {
        if (!outer_for)
            outer_for = for_op;
    });
    if (!outer_for) {
        return emit_matmul_msl(desc, scalar);
    }

    auto loops = collect_loop_nest(outer_for);
    if (loops.size() >= 2) {
        validate_against_desc(loops, desc);
    }

    return emit_matmul_msl(desc, scalar);
}

}  // namespace gfx_plugin
}  // namespace ov
