// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir_codegen/codegen_common.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_passes.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_interpolate_kernel(const KernelOp& op, std::string& log) {
    mlir::MLIRContext ctx;
    auto module = build_mlir_interpolate_from_op(op, ctx);
    run_mlir_pipeline(module);

    InterpolateCodegenDesc desc;
    desc.kind = KernelOpKind::Interpolate;
    desc.element_type = static_cast<ov::element::Type>(op.interpolate.dtype.ov_type);
    desc.N = op.interpolate.N;
    desc.C = op.interpolate.C;
    desc.H_in = op.interpolate.H_in;
    desc.W_in = op.interpolate.W_in;
    desc.H_out = op.interpolate.H_out;
    desc.W_out = op.interpolate.W_out;
    desc.scale_h = op.interpolate.scale_h;
    desc.scale_w = op.interpolate.scale_w;
    desc.align_corners = op.interpolate.align_corners;
    desc.nearest = op.interpolate.nearest;

    auto source = generate_msl_from_mlir(module, desc);
    return compile_msl_from_source(source, "interpolate_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
