// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_softmax_kernel(const KernelOp& op, std::string& log) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Softmax, "compile_softmax_kernel expects Softmax");
    SoftmaxCodegenDesc desc;
    desc.kind = KernelOpKind::Softmax;
    desc.rows = op.rows;
    desc.cols = op.cols;
    // If inner is unknown at compile-time (dynamic shapes), prefer rank-3 kernel by using a non-zero value.
    desc.inner = (op.inner == 0 ? 2 : op.inner);
    if (op.output && op.output->dtype.ov_type != ov::element::dynamic) {
        desc.element_type = op.output->dtype.ov_type;
    }
    auto source = generate_msl_for_softmax(desc, /*module*/ nullptr);
    return compile_msl_from_source(source, "softmax_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
