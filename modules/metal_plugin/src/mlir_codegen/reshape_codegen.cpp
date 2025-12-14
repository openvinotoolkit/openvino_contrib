// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir_codegen/reshape_codegen.hpp"
#include "kernel_ir/kernel_ir_common.hpp"

#include <numeric>
#include <sstream>
#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

std::string generate_msl_for_reshape(const KernelOp& op) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Reshape, "MSL generator: expected Reshape op");
    OPENVINO_ASSERT(op.input0 && op.output, "Reshape: invalid wiring");

    size_t elems = 0;
    if (op.output) {
        elems = 1;
        for (auto d : op.output->shape) elems *= static_cast<size_t>(d);
    }

    std::string scalar_ty = "float";
    switch (op.output->dtype.ov_type) {
        case ov::element::f16: scalar_ty = "half"; break;
        case ov::element::i32: scalar_ty = "int"; break;
        case ov::element::i64: scalar_ty = "long"; break;
        default: scalar_ty = "float"; break;
    }

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "constant uint TOTAL = " << elems << ";\n";
    ss << "kernel void reshape_copy(\n";
    ss << "  device const " << scalar_ty << "* in0 [[buffer(0)]],\n";
    ss << "  device " << scalar_ty << "* out [[buffer(1)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= TOTAL) return;\n";
    ss << "    out[gid] = in0[gid];\n";
    ss << "}\n";
    return ss.str();
}

}  // namespace metal_plugin
}  // namespace ov
