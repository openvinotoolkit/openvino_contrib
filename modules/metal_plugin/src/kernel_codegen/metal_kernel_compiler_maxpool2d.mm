// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_maxpool2d_kernel(const KernelOp& op, std::string& log) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::MaxPool2D, "compile_maxpool2d_kernel expects MaxPool2D");
    Pool2DCodegenDesc desc;
    desc.kind = KernelOpKind::MaxPool2D;
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
    desc.is_avg = false;
    desc.exclude_pad = op.pool.exclude_pad;
    ov::element::Type et = op.dtype.ov_type;
    if (op.output && op.output->dtype.ov_type != ov::element::dynamic)
        et = op.output->dtype.ov_type;
    else if (op.input0 && op.input0->dtype.ov_type != ov::element::dynamic)
        et = op.input0->dtype.ov_type;
    if (et == ov::element::dynamic)
        et = ov::element::f32;
    desc.element_type = et;
    auto source = generate_msl_for_maxpool2d(desc, /*module*/ nullptr);
    return compile_msl_from_source(source, "pool2d_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
