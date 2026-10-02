// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen_common.hpp"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <sstream>
#include <vector>

#include "llvm/Support/Casting.h"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
struct SplitParams {
    uint32_t outer;
    uint32_t inner;
    uint32_t axis_offset;
    uint32_t axis_len;
    uint32_t axis_total;
};

std::vector<int64_t> invert_permutation(const std::vector<int64_t>& permutation) {
    std::vector<int64_t> inverse(permutation.size(), -1);
    for (size_t i = 0; i < permutation.size(); ++i) {
        const auto axis = permutation[i];
        OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(permutation.size()),
                        "Split codegen: permutation axis out of range");
        OPENVINO_ASSERT(inverse[static_cast<size_t>(axis)] < 0,
                        "Split codegen: permutation axis repeated");
        inverse[static_cast<size_t>(axis)] = static_cast<int64_t>(i);
    }
    return inverse;
}
}  // namespace

std::string generate_msl_for_split(const SplitCodegenDesc& d, mlir::ModuleOp module) {
    (void)module;
    const int64_t axis_norm = d.axis < 0 ? d.axis + static_cast<int64_t>(d.input_shape.size()) : d.axis;
    OPENVINO_ASSERT(axis_norm >= 0 && axis_norm < static_cast<int64_t>(d.input_shape.size()),
                    "Split codegen: axis out of range");
    std::string scalar_t = msl_type_from_element(d.element_type);
    if (scalar_t.empty()) {
        scalar_t = "float";
    }
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "struct SplitParams {\n";
    ss << "  uint outer;\n";
    ss << "  uint inner;\n";
    ss << "  uint axis_offset;\n";
    ss << "  uint axis_len;\n";
    ss << "  uint axis_total;\n";
    ss << "};\n";
    ss << "kernel void split_kernel(device const scalar_t* src [[buffer(0)]],\n";
    ss << "                           device scalar_t* dst [[buffer(1)]],\n";
    ss << "                           constant SplitParams& p [[buffer(2)]],\n";
    ss << "                           uint gid [[thread_position_in_grid]]) {\n";
    if (d.input_permutation.empty()) {
        ss << "  uint split_size = p.axis_len;\n";
        ss << "  uint axis_offset = p.axis_offset;\n";
        ss << "  uint total = p.outer * split_size * p.inner;\n";
        ss << "  if (gid >= total) return;\n";
        ss << "  uint tmp = gid;\n";
        ss << "  uint outer_idx = tmp / (split_size * p.inner);\n";
        ss << "  tmp -= outer_idx * split_size * p.inner;\n";
        ss << "  uint axis_idx = tmp / p.inner;\n";
        ss << "  uint inner_idx = tmp - axis_idx * p.inner;\n";
        ss << "  uint src_idx = (outer_idx * (p.axis_total * p.inner)) + ((axis_offset + axis_idx) * p.inner) + inner_idx;\n";
        ss << "  dst[gid] = src[src_idx];\n";
    } else {
        OPENVINO_ASSERT(d.source_input_shape.size() == d.input_shape.size(),
                        "Split codegen: source/logical rank mismatch");
        OPENVINO_ASSERT(d.input_permutation.size() == d.input_shape.size(),
                        "Split codegen: permutation rank mismatch");
        const auto inverse_permutation = invert_permutation(d.input_permutation);
        std::vector<uint64_t> source_strides(d.source_input_shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(d.source_input_shape.size()) - 2; i >= 0; --i) {
            source_strides[static_cast<size_t>(i)] =
                source_strides[static_cast<size_t>(i + 1)] *
                static_cast<uint64_t>(d.source_input_shape[static_cast<size_t>(i + 1)]);
        }
        ss << "  uint dims[" << d.input_shape.size() << "];\n";
        for (size_t i = 0; i < d.input_shape.size(); ++i) {
            if (static_cast<int64_t>(i) == axis_norm) {
                ss << "  dims[" << i << "] = p.axis_len;\n";
            } else {
                ss << "  dims[" << i << "] = " << static_cast<uint32_t>(d.input_shape[i]) << "u;\n";
            }
        }
        ss << "  uint total = 1u;\n";
        for (size_t i = 0; i < d.input_shape.size(); ++i) {
            ss << "  total *= dims[" << i << "];\n";
        }
        ss << "  if (gid >= total) return;\n";
        ss << "  uint logical_idx[" << d.input_shape.size() << "];\n";
        ss << "  uint rem = gid;\n";
        for (int64_t i = static_cast<int64_t>(d.input_shape.size()) - 1; i >= 0; --i) {
            ss << "  logical_idx[" << i << "] = rem % dims[" << i << "];\n";
            ss << "  rem /= dims[" << i << "];\n";
        }
        ss << "  logical_idx[" << axis_norm << "] += p.axis_offset;\n";
        ss << "  uint src_idx = 0u;\n";
        for (size_t src_dim = 0; src_dim < d.source_input_shape.size(); ++src_dim) {
            ss << "  src_idx += logical_idx[" << inverse_permutation[src_dim] << "] * "
               << static_cast<uint32_t>(source_strides[src_dim]) << "u;\n";
        }
        ss << "  dst[gid] = src[src_idx];\n";
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
