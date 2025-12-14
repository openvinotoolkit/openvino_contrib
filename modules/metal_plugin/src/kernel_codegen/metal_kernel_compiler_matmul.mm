// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_matmul_kernel(const KernelOp& op, std::string& log) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::MatMul, "compile_matmul_kernel expects MatMul");
    MatMulCodegenDesc desc;
    desc.kind = KernelOpKind::MatMul;
    desc.M = op.M;
    desc.N = op.N;
    desc.K = op.K;
    desc.batch = op.batch;
    desc.batch_a = op.batch_a;
    desc.batch_b = op.batch_b;
    desc.a_transpose = op.a_transpose;
    desc.b_transpose = op.b_transpose;
    desc.b_is_nk_layout = op.b_is_nk_layout;
    auto source = generate_msl_for_matmul(desc, /*module*/ nullptr);
    return compile_msl_from_source(source, "matmul_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
