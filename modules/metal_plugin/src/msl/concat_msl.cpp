// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msl/concat_msl.hpp"

#include <sstream>

namespace ov {
namespace metal_plugin {

namespace {
std::string scalar_type(const MetalDType& dt) {
    switch (dt.storage) {
        case MetalDType::StorageType::F16: return "half";
        case MetalDType::StorageType::F32: return "float";
        case MetalDType::StorageType::I32: return "int";
        case MetalDType::StorageType::I64: return "long";  // not supported in GPU, but avoid crash
        default: return "float";
    }
}
}  // namespace

std::string generate_msl_for_concat(const KernelOp& op) {
    std::ostringstream ss;
    const auto t = scalar_type(op.concat.dtype);
    ss << R"(
using namespace metal;

struct ConcatParams {
    uint outer;
    uint inner;
    uint axis_offset;
    uint axis_len;
};

kernel void concat_kernel(device const )" << t << R"(* src      [[buffer(0)]],
                          device )" << t << R"(*       dst      [[buffer(1)]],
                          constant ConcatParams& p [[buffer(2)]],
                          uint gid                [[thread_position_in_grid]]) {
    uint total = p.outer * p.axis_len * p.inner;
    if (gid >= total) return;
    uint tmp = gid;
    uint outer = tmp / (p.axis_len * p.inner);
    tmp -= outer * p.axis_len * p.inner;
    uint axis = tmp / p.inner;
    uint inner = tmp - axis * p.inner;

    uint dst_idx = ((outer * (p.axis_len + p.axis_offset) + (p.axis_offset + axis)) * p.inner) + inner;
    uint src_idx = ((outer * p.axis_len + axis) * p.inner) + inner;
    dst[dst_idx] = src[src_idx];
}
)";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov

