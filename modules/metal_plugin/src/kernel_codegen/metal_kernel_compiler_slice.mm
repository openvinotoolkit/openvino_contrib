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
    // Generic slice kernel does not rely on MLIR module; use dtype info only.
    ConvertCodegenDesc desc;
    desc.kind = KernelOpKind::Slice;  // treated as slice in dispatcher
    desc.dst_type = static_cast<ov::element::Type>(op.output ? op.output->dtype.ov_type : ov::element::f32);
    auto source = generate_msl_for_slice_generic(desc, nullptr);
    return compile_msl_from_source(source, "slice_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
