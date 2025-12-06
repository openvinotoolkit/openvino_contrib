// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "slice_msl.hpp"

#include <sstream>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

// Generate a simple rank-N slice kernel using precomputed starts/steps/strides in the op.
std::string generate_msl_for_slice(const KernelOp& op) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Slice, "MSL slice generator: wrong op kind");
    OPENVINO_ASSERT(op.input0 && op.output, "MSL slice: invalid tensor wiring");
    const size_t rank = op.slice.out_shape.size();
    OPENVINO_ASSERT(rank == op.slice.in_shape.size(), "MSL slice: shape rank mismatch");
    OPENVINO_ASSERT(rank <= 6, "MSL slice: rank > 6 not supported");

    std::ostringstream ss;
    ss << "using namespace metal;\n";
    ss << "constant uint RANK = " << rank << ";\n";

    auto dump_vec = [&](const char* name, const std::vector<int64_t>& v) {
        ss << "constant int " << name << "[" << v.size() << "] = {";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) ss << ",";
            ss << v[i];
        }
        ss << "};\n";
    };
    dump_vec("out_shape", op.slice.out_shape);
    dump_vec("in_strides", op.slice.in_strides);
    dump_vec("starts", op.slice.starts);
    dump_vec("steps", op.slice.steps);

    ss << "kernel void slice_kernel(\n";
    ss << "  device const float* in0 [[buffer(0)]],\n";
    ss << "  device float* out [[buffer(1)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    uint idx = gid;\n";
    ss << "    int coord[" << rank << "];\n";
    ss << "    for (int d = " << static_cast<int>(rank) - 1 << "; d >= 0; --d) {\n";
    ss << "        int dim = out_shape[d];\n";
    ss << "        coord[d] = idx % dim;\n";
    ss << "        idx /= dim;\n";
    ss << "    }\n";
    ss << "    int in_offset = 0;\n";
    ss << "    for (uint d = 0; d < RANK; ++d) {\n";
    ss << "        int in_coord = starts[d] + coord[d] * steps[d];\n";
    ss << "        in_offset += in_coord * in_strides[d];\n";
    ss << "    }\n";
    ss << "    out[gid] = in0[in_offset];\n";
    ss << "}\n";

    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov

