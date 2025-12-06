// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "split_msl.hpp"

#include <sstream>
#include <numeric>

namespace ov {
namespace metal_plugin {

static std::string dtype_name(const MetalDType& dt) {
    switch (dt.compute) {
        case MetalDType::ComputeType::F32: return "float";
        case MetalDType::ComputeType::I32: return "int";
        case MetalDType::ComputeType::I64: return "long";
        default: return "float";
    }
}

std::string generate_split_msl(const KernelOp& op) {
    std::ostringstream ss;
    const auto& desc = op.split;
    const auto& shape = desc.input_shape;
    const size_t rank = shape.size();
    const size_t outputs = desc.split_sizes.size();
    std::vector<int64_t> strides(rank, 1);
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    std::vector<int64_t> prefix(outputs, 0);
    for (size_t i = 1; i < outputs; ++i) {
        prefix[i] = prefix[i - 1] + static_cast<int64_t>(desc.split_sizes[i - 1]);
    }
    const int64_t total_elems = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
    const auto dtype = dtype_name(desc.dtype);

    ss << "#include <metal_stdlib>\nusing namespace metal;\n\n";
    ss << "kernel void split_kernel(\n";
    ss << "    const device " << dtype << "* src [[buffer(0)]],\n";
    for (size_t i = 0; i < outputs; ++i) {
        ss << "    device " << dtype << "* out" << i << " [[buffer(" << (i + 1) << ")]],\n";
    }
    ss << "    uint gid [[thread_position_in_grid]]) {\n";
    ss << "  const uint total = " << total_elems << "u;\n";
    ss << "  if (gid >= total) return;\n";
    // Compute axis coordinate
    ss << "  const uint stride_axis = " << strides[desc.axis] << "u;\n";
    ss << "  const uint dim_axis = " << shape[desc.axis] << "u;\n";
    ss << "  const uint inner = gid % stride_axis;\n";
    ss << "  const uint outer = gid / (dim_axis * stride_axis);\n";
    ss << "  uint axis_coord = (gid / stride_axis) % dim_axis;\n";
    // Find split index and prefix
    ss << "  uint split_idx = 0;\n";
    ss << "  uint axis_offset = 0;\n";
    for (size_t i = 0; i < outputs; ++i) {
        uint64_t pref = static_cast<uint64_t>(prefix[i]);
        uint64_t size = static_cast<uint64_t>(desc.split_sizes[i]);
        ss << "  if (axis_coord >= " << pref << "u && axis_coord < " << (pref + size)
           << "u) { split_idx = " << i << "; axis_offset = " << pref << "u; }\n";
    }
    ss << "  uint local_axis = axis_coord - axis_offset;\n";
    ss << "  uint out_axis = 0;\n";
    for (size_t i = 0; i < outputs; ++i) {
        ss << "  if (split_idx == " << i << ") out_axis = " << desc.split_sizes[i] << "u;\n";
    }
    ss << "  uint dst_index = outer * out_axis * stride_axis + local_axis * stride_axis + inner;\n";
    ss << "  " << dtype << " v = src[gid];\n";
    for (size_t i = 0; i < outputs; ++i) {
        ss << "  if (split_idx == " << i << ") out" << i << "[dst_index] = v;\n";
    }
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
