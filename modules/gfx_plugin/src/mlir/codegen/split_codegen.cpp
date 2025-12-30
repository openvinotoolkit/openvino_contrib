// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/codegen/codegen_common.hpp"

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
}  // namespace

std::string generate_msl_for_split(const SplitCodegenDesc& d, mlir::ModuleOp module) {
    (void)module;
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
    ss << "}\n";
    return ss.str();
}

}  // namespace gfx_plugin
}  // namespace ov
