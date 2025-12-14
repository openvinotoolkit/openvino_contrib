// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_avgpool2d_kernel(const KernelOp& op, std::string& log) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::AvgPool2D, "compile_avgpool2d_kernel expects AvgPool2D");
    Pool2DCodegenDesc desc;
    desc.kind = KernelOpKind::AvgPool2D;
    desc.N = op.pool.N;
    desc.C = op.pool.C;
    desc.H = op.pool.H;
    desc.W = op.pool.W;
    desc.kH = op.pool.kernelH;
    desc.kW = op.pool.kernelW;
    desc.strideH = op.pool.strideH;
    desc.strideW = op.pool.strideW;
    desc.padTop = op.pool.padTop;
    desc.padLeft = op.pool.padLeft;
    desc.padBottom = op.pool.padBottom;
    desc.padRight = op.pool.padRight;
    desc.outH = op.pool.outH;
    desc.outW = op.pool.outW;
    desc.is_avg = true;
    desc.exclude_pad = op.pool.exclude_pad;
    auto source = generate_msl_for_avgpool2d(desc, /*module*/ nullptr);
    return compile_msl_from_source(source, "pool2d_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
