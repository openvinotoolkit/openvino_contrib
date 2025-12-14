// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_passes.hpp"
#include "mlir_codegen/codegen_common.hpp"

#include "mlir/IR/MLIRContext.h"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_concat_kernel(const KernelOp& op, std::string& log) {
    mlir::MLIRContext ctx;
    auto module = build_mlir_concat_from_op(op, ctx);
    run_mlir_pipeline(module);

    ConcatCodegenDesc desc;
    desc.kind = KernelOpKind::Concat;
    desc.element_type = static_cast<ov::element::Type>(op.output ? op.output->dtype.ov_type : ov::element::f32);
    desc.outer = op.concat.outer;
    desc.inner = op.concat.inner;
    desc.axis_offset = op.concat.axis_offsets.empty() ? 0 : op.concat.axis_offsets.front();
    desc.axis_len = op.concat.axis_sizes.empty() ? 0 : op.concat.axis_sizes.front();
    desc.axis_total = op.concat.axis_total;
    auto source = generate_msl_from_mlir(module, desc);
    return compile_msl_from_source(source, "concat_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
