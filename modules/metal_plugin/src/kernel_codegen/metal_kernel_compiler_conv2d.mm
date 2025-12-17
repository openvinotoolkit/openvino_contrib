// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir_codegen/codegen_common.hpp"
#include <fstream>

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_conv2d_kernel(const KernelOp& op, std::string& log) {
    OPENVINO_ASSERT(op.kind == KernelOpKind::Conv2D, "compile_conv2d_kernel expects Conv2D op");
    Conv2DCodegenDesc desc;
    desc.kind = KernelOpKind::Conv2D;
    desc.N = op.conv2d.N;
    desc.C_in = op.conv2d.C_in;
    desc.H = op.conv2d.H;
    desc.W = op.conv2d.W;
    desc.C_out = op.conv2d.C_out;
    desc.groups = op.conv2d.groups;
    desc.C_in_pg = op.conv2d.C_in_per_group ? op.conv2d.C_in_per_group
                                            : (desc.groups ? desc.C_in / desc.groups : desc.C_in);
    desc.C_out_pg = op.conv2d.C_out_per_group ? op.conv2d.C_out_per_group
                                              : (desc.groups ? desc.C_out / desc.groups : desc.C_out);
    desc.kH = op.conv2d.kernelH;
    desc.kW = op.conv2d.kernelW;
    desc.strideH = op.conv2d.strideH;
    desc.strideW = op.conv2d.strideW;
    desc.dilationH = op.conv2d.dilationH;
    desc.dilationW = op.conv2d.dilationW;
    desc.padTop = op.conv2d.padTop;
    desc.padLeft = op.conv2d.padLeft;
    desc.padBottom = op.conv2d.padBottom;
    desc.padRight = op.conv2d.padRight;
    desc.outH = op.conv2d.outH;
    desc.outW = op.conv2d.outW;
    if (op.output && op.output->dtype.ov_type != ov::element::dynamic) {
        desc.element_type = op.output->dtype.ov_type;
    }
    if (desc.outH == 0) {
        int64_t eff_kh = static_cast<int64_t>(desc.dilationH) * (static_cast<int64_t>(desc.kH) - 1) + 1;
        desc.outH = static_cast<uint32_t>((static_cast<int64_t>(desc.H) + desc.padTop + desc.padBottom - eff_kh) / desc.strideH + 1);
    }
    if (desc.outW == 0) {
        int64_t eff_kw = static_cast<int64_t>(desc.dilationW) * (static_cast<int64_t>(desc.kW) - 1) + 1;
        desc.outW = static_cast<uint32_t>((static_cast<int64_t>(desc.W) + desc.padLeft + desc.padRight - eff_kw) / desc.strideW + 1);
    }
    desc.has_bias = op.conv2d.has_bias;
    desc.has_activation = op.conv2d.has_activation;
    desc.activation = op.conv2d.activation;
    desc.alpha = op.conv2d.alpha;
    desc.has_bn = op.conv2d.has_bn;
    desc.epsilon = op.conv2d.epsilon;
    desc.clamp_min = op.conv2d.clamp_min;
    desc.clamp_max = op.conv2d.clamp_max;
    desc.gamma = op.conv2d.gamma;
    desc.beta = op.conv2d.beta;
    desc.mean = op.conv2d.mean;
    desc.var = op.conv2d.var;
    {
        std::ofstream ofs("/tmp/metal_conv.log", std::ios::app);
        ofs << "[Compile Conv2D] N=" << desc.N << " C_in=" << desc.C_in << " H=" << desc.H << " W=" << desc.W
            << " C_out=" << desc.C_out << " pad=" << desc.padTop << "," << desc.padLeft << ","
            << desc.padBottom << "," << desc.padRight << " out=" << desc.outH << "x" << desc.outW
            << " stride=" << desc.strideH << "x" << desc.strideW << std::endl;
    }

    auto source = generate_msl_for_conv2d(desc, /*module*/ nullptr);
    return compile_msl_from_source(source, "conv2d_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
