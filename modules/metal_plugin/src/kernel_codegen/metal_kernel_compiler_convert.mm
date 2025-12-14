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

id<MTLComputePipelineState> MetalKernelCompiler::compile_convert_kernel(const KernelOp& op, std::string& log) {
    mlir::MLIRContext ctx;
    auto module = build_mlir_convert_from_op(op, ctx);
    run_mlir_pipeline(module);

    ConvertCodegenDesc desc;
    desc.kind = KernelOpKind::Convert;
    desc.src_type = static_cast<ov::element::Type>(op.convert.src_dtype.ov_type);
    desc.dst_type = static_cast<ov::element::Type>(op.convert.dst_dtype.ov_type);
    auto source = generate_msl_from_mlir(module, desc);
    return compile_msl_from_source(source, "convert_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov

