// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "msl/unary_msl.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_unary_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_unary(op);
    return compile_msl_from_source(source, "unary_kernel", log);
}

}  // namespace metal_plugin
}  // namespace ov
