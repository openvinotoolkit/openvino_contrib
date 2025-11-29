// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"

#include "kernel_codegen/msl_generator.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_add_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_elementwise_add(op);
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL for Add: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"add_kernel"];
    if (!fn) {
        log = "function add_kernel not found";
        OPENVINO_THROW("MLIR backend: add_kernel function missing in compiled library");
    }

    id<MTLComputePipelineState> pso = [m_device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    [lib release];
    if (!pso || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to create pipeline state: ", err);
    }

    return pso;
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_matmul_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_matmul(op);
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL for MatMul: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"matmul_kernel"];
    if (!fn) {
        log = "function matmul_kernel not found";
        OPENVINO_THROW("MLIR backend: matmul_kernel function missing in compiled library");
    }

    id<MTLComputePipelineState> pso = [m_device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    [lib release];
    if (!pso || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to create matmul pipeline state: ", err);
    }

    return pso;
}

}  // namespace metal_plugin
}  // namespace ov
