// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/codegen_common.hpp"
#include "mlir_codegen/index_expr_utils.hpp"

#include <sstream>
#include <vector>

#include "openvino/core/except.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace ov {
namespace gfx_plugin {
namespace {

mlir::func::FuncOp find_func(mlir::ModuleOp module) {
    for (auto f : module.getOps<mlir::func::FuncOp>())
        return f;
    return nullptr;
}

std::string emit_softmax_msl(const SoftmaxCodegenDesc& d,
                             const std::string& scalar,
                             const std::vector<std::string>& input_idx,
                             const std::vector<std::string>& output_idx,
                             uint32_t rank) {
    const bool use_half = (scalar == "half");
    std::vector<std::string> dims;
    if (rank == 3)
        dims = {"p.cols", "p.inner"};
    else
        dims = {"p.cols"};
    const std::string in_flat = flatten_indices(input_idx, dims);
    const std::string out_flat = flatten_indices(output_idx, dims);

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "struct SoftmaxParams { uint rows; uint cols; uint inner; };\n";
    ss << "kernel void softmax_kernel(\n";
    ss << "  device const " << scalar << "* input [[buffer(0)]],\n";
    ss << "  device " << scalar << "* output [[buffer(1)]],\n";
    ss << "  constant SoftmaxParams& p [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    uint row = gid / p.cols;\n";
    ss << "    uint col = gid - row * p.cols;\n";
    ss << "    if (row >= p.rows || col >= p.cols) return;\n";
    ss << "    uint outer = row / p.inner;\n";
    ss << "    uint inner_i = row - outer * p.inner;\n";
    ss << "    uint base_outer = outer * p.cols * p.inner;\n";
    ss << "    // compute max\n";
    ss << "    float m = -INFINITY;\n";
    ss << "    for (uint c = 0; c < p.cols; ++c) {\n";
    ss << "        uint idx = base_outer + c * p.inner + inner_i;\n";
    ss << "        float v = static_cast<float>(input[idx]);\n";
    ss << "        m = m > v ? m : v;\n";
    ss << "    }\n";
    ss << "    float sum = 0.0f;\n";
    ss << "    for (uint c = 0; c < p.cols; ++c) {\n";
    ss << "        uint idx = base_outer + c * p.inner + inner_i;\n";
    ss << "        float v = static_cast<float>(input[idx]);\n";
    ss << "        sum += exp(v - m);\n";
    ss << "    }\n";
    ss << "    uint out_idx = base_outer + col * p.inner + inner_i;\n";
    ss << "    float v = static_cast<float>(input[out_idx]);\n";
    if (d.log_softmax) {
        ss << "    float logsum = log(sum);\n";
        if (use_half) {
            ss << "    output[out_idx] = static_cast<" << scalar << ">((v - m) - logsum);\n";
        } else {
            ss << "    output[out_idx] = (v - m) - logsum;\n";
        }
    } else {
        ss << "    float inv = 1.0f / sum;\n";
        if (use_half) {
            ss << "    output[out_idx] = static_cast<" << scalar << ">(exp(v - m) * inv);\n";
        } else {
            ss << "    output[out_idx] = exp(v - m) * inv;\n";
        }
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace

std::string generate_msl_for_softmax(const SoftmaxCodegenDesc& d, mlir::ModuleOp module) {
    OPENVINO_ASSERT(d.rows > 0 && d.cols > 0, "Softmax: rows/cols must be positive");
    std::string scalar = msl_type_from_element(d.element_type);
    if (scalar.empty()) {
        scalar = "float";
    }
    const bool has_inner = d.inner > 1;
    if (!module) {
        OPENVINO_THROW("Softmax MLIR module is null");
    }

    auto func = find_func(module);
    if (!func) {
        OPENVINO_THROW("Softmax MLIR: function not found");
    }
    (void)func;
    if (has_inner) {
        return emit_softmax_msl(d, scalar, {"row", "col", "inner_i"}, {"row", "col", "inner_i"}, 3);
    }
    return emit_softmax_msl(d, scalar, {"row", "col"}, {"row", "col"}, 2);
}

}  // namespace gfx_plugin
}  // namespace ov
