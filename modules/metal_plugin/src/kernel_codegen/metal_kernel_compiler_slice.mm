// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_passes.hpp"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_slice_kernel(const KernelOp& op, std::string& log) {
    mlir::MLIRContext ctx;
    auto module = build_mlir_split_from_op(op, ctx);  // reuse split builder for a single slice chunk
    run_mlir_pipeline(module);

    SplitCodegenDesc desc;
    desc.kind = KernelOpKind::Split;
    desc.axis = op.slice.axes.empty() ? 0 : op.slice.axes[0];
    desc.input_shape = op.slice.in_shape;
    desc.split_sizes = {static_cast<size_t>(op.slice.out_shape.empty() ? 0 : op.slice.out_shape[desc.axis])};
    desc.element_type = static_cast<ov::element::Type>(op.output ? op.output->dtype.ov_type : ov::element::f32);
    auto source = generate_msl_from_mlir(module, desc);
    return compile_msl_from_source(source, "slice_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
