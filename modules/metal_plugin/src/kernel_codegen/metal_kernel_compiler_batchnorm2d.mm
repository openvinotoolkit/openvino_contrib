// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_batchnorm2d_kernel(const KernelOp& op, std::string& log) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::BatchNorm2D, "compile_batchnorm2d_kernel expects BatchNorm2D");
    OPENVINO_ASSERT(op.batchnorm.C && op.batchnorm.H && op.batchnorm.W && op.batchnorm.N, "BatchNorm2D dims missing");
    OPENVINO_ASSERT(op.bn_params.size() == 4 * op.batchnorm.C + 1, "BatchNorm params size mismatch");

    BatchNorm2DCodegenDesc desc;
    desc.kind = KernelOpKind::BatchNorm2D;
    desc.N = op.batchnorm.N;
    desc.C = op.batchnorm.C;
    desc.H = op.batchnorm.H;
    desc.W = op.batchnorm.W;
    desc.element_type = op.output ? op.output->dtype.ov_type : ov::element::f32;
    auto source = generate_msl_for_batchnorm2d(desc);
    return compile_msl_from_source(source, "batchnorm2d_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
