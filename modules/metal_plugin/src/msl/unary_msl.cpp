// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msl/unary_msl.hpp"

#include <sstream>

#include "msl/msl_common.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_unary(const KernelOp& op) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Unary, "MSL generator supports only Unary here");
    OPENVINO_ASSERT(op.input0 && op.output, "Invalid KernelOp wiring for Unary");

    std::ostringstream ss;
    ss << "using namespace metal;\n";
    ss << "kernel void unary_kernel(\n";
    ss << "  device const float* in0 [[buffer(0)]],\n";
    ss << "  device float* out [[buffer(1)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    float x = in0[gid];\n";
    ss << "    out[gid] = " << msl::activation_expr(op.activation, op.alpha) << ";\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov

