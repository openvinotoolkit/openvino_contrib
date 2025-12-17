// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_conv3d_kernel(const KernelOp& op, std::string& log) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Conv3D, "compile_conv3d_kernel expects Conv3D op");
    const auto& c = op.conv3d;
    Conv3DCodegenDesc desc;
    desc.kind = KernelOpKind::Conv3D;
    desc.N = c.N;
    desc.C_in = c.C_in;
    desc.D = c.D;
    desc.H = c.H;
    desc.W = c.W;
    desc.C_out = c.C_out;
    desc.kD = c.kernelD;
    desc.kH = c.kernelH;
    desc.kW = c.kernelW;
    desc.strideD = c.strideD;
    desc.strideH = c.strideH;
    desc.strideW = c.strideW;
    desc.dilationD = c.dilationD;
    desc.dilationH = c.dilationH;
    desc.dilationW = c.dilationW;
    desc.padFront = c.padFront;
    desc.padTop = c.padTop;
    desc.padLeft = c.padLeft;
    desc.padBack = c.padBack;
    desc.padBottom = c.padBottom;
    desc.padRight = c.padRight;
    desc.outD = c.outD;
    desc.outH = c.outH;
    desc.outW = c.outW;
    if (op.output && op.output->dtype.ov_type != ov::element::dynamic) {
        desc.element_type = op.output->dtype.ov_type;
    }

    auto source = generate_msl_for_conv3d(desc, /*module*/ nullptr);
    return compile_msl_from_source(source, "conv3d_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
