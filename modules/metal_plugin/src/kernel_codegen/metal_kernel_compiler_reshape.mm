// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"

#include "mlir_codegen/reshape_codegen.hpp"
#include "runtime/metal_logger.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_reshape_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_reshape(op);
    return compile_msl_from_source(source, "reshape_copy", log);
}

}  // namespace metal_plugin
}  // namespace ov
