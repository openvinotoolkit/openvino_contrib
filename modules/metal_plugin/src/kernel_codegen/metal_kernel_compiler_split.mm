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

id<MTLComputePipelineState> MetalKernelCompiler::compile_split_kernel(const KernelOp& op, std::string& log) {
    mlir::MLIRContext ctx;
    auto module = build_mlir_split_from_op(op, ctx);
    run_mlir_pipeline(module);

    SplitCodegenDesc desc;
    desc.kind = KernelOpKind::Split;
    desc.axis = static_cast<int64_t>(op.split.axis);
    desc.inner = op.split.inner;
    uint64_t outer = 1;
    for (size_t k = 0; k < static_cast<size_t>(op.split.axis); ++k) outer *= static_cast<uint64_t>(op.split.input_shape[k]);
    desc.outer = outer;
    desc.input_shape = op.split.input_shape;
    desc.split_sizes = op.split.split_sizes;
    ov::element::Type et = static_cast<ov::element::Type_t>(op.split.element_type);
    if (et == ov::element::Type_t::dynamic && op.input0)
        et = static_cast<ov::element::Type_t>(op.input0->dtype.ov_type);
    desc.element_type = et;
    auto source = generate_msl_from_mlir(module, desc);
    return compile_msl_from_source(source, "split_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
