// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "msl/mul_msl.hpp"

#include <sstream>

#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_elementwise_mul(const KernelOp& op) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::ElementwiseMul, "MSL generator supports only ElementwiseMul");
    OPENVINO_ASSERT(op.input0 && op.output, "Invalid KernelOp wiring for Mul");

    std::ostringstream ss;
    ss << "using namespace metal;\n";

    if (!op.is_broadcast) {
        ss << "kernel void mul_kernel(\n";
        ss << "  device const float* in0 [[buffer(0)]],\n";
        ss << "  device const float* in1 [[buffer(1)]],\n";
        ss << "  device float* out [[buffer(2)]],\n";
        ss << "  uint gid [[thread_position_in_grid]]) {\n";
        ss << "    out[gid] = in0[gid] * in1[gid];\n";
        ss << "}\n";
        return ss.str();
    }

    ss << "constant uint RANK = " << op.out_shape.size() << ";\n";
    ss << "constant int out_dims[" << op.out_shape.size() << "] = {";
    for (size_t i = 0; i < op.out_shape.size(); ++i) {
        if (i) ss << ",";
        ss << op.out_shape[i];
    }
    ss << "};\n";

    auto dump_array = [&](const char* name, const std::vector<int64_t>& v) {
        ss << "constant int " << name << "[" << v.size() << "] = {";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) ss << ",";
            ss << v[i];
        }
        ss << "};\n";
    };
    dump_array("stride0", op.stride0);
    dump_array("stride1", op.stride1);

    ss << "kernel void mul_broadcast_kernel(\n";
    ss << "  device const float* in0 [[buffer(0)]],\n";
    ss << "  device const float* in1 [[buffer(1)]],\n";
    ss << "  device float* out [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    uint idx = gid;\n";
    ss << "    int offsets0[" << op.out_shape.size() << "];\n";
    ss << "    int offsets1[" << op.out_shape.size() << "];\n";
    ss << "    for (int d = " << static_cast<int>(op.out_shape.size()) - 1 << "; d >= 0; --d) {\n";
    ss << "        int dim = out_dims[d];\n";
    ss << "        int coord = idx % dim;\n";
    ss << "        idx /= dim;\n";
    ss << "        offsets0[d] = (stride0[d] == 0) ? 0 : coord * stride0[d];\n";
    ss << "        offsets1[d] = (stride1[d] == 0) ? 0 : coord * stride1[d];\n";
    ss << "    }\n";
    ss << "    int off0 = 0;\n";
    ss << "    int off1 = 0;\n";
    ss << "    for (uint d = 0; d < RANK; ++d) {\n";
    ss << "        off0 += offsets0[d];\n";
    ss << "        off1 += offsets1[d];\n";
    ss << "    }\n";
    ss << "    out[gid] = in0[off0] * in1[off1];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov

