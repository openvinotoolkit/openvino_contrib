// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_unary_kernel(const KernelOp& op, std::string& log) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Unary, "compile_unary_kernel expects Unary");
    UnaryCodegenDesc desc;
    desc.kind = KernelOpKind::Unary;
    desc.activation = op.activation;
    desc.alpha = op.alpha;
    ov::element::Type et = op.dtype.ov_type;
    if (op.output && op.output->dtype.ov_type != ov::element::dynamic)
        et = op.output->dtype.ov_type;
    else if (op.input0 && op.input0->dtype.ov_type != ov::element::dynamic)
        et = op.input0->dtype.ov_type;
    if (et == ov::element::dynamic)
        et = ov::element::f32;
    desc.element_type = et;
    auto source = generate_msl_for_unary(desc);
    return compile_msl_from_source(source, "unary_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
