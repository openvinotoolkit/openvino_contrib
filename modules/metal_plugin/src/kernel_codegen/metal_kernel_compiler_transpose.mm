// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"

#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_transpose_kernel(const KernelOp& op, std::string& log) {
    TransposeCodegenDesc desc;
    desc.kind = KernelOpKind::Transpose;
    desc.element_type = op.output ? op.output->dtype.ov_type : ov::element::f32;
    desc.in_shape.assign(op.transpose.in_shape.begin(), op.transpose.in_shape.end());
    desc.out_shape.assign(op.transpose.out_shape.begin(), op.transpose.out_shape.end());
    for (auto p : op.transpose.perm) desc.perm.push_back(static_cast<uint32_t>(p));
    desc.use_half = (desc.element_type == ov::element::f16);
    desc.use_int = (desc.element_type == ov::element::i32);

    auto source = generate_msl_for_transpose(desc, nullptr);
    return compile_msl_from_source(source, "transpose_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
