// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "kernel_codegen/metal_kernel_compiler.hpp"

#include "msl/add_msl.hpp"
#include "msl/mul_msl.hpp"
#include "msl/matmul_msl.hpp"
#include "msl/pool_avg_msl.hpp"
#include "msl/pool_max_msl.hpp"
#include "msl/softmax_msl.hpp"
#include "msl/unary_msl.hpp"
#include "msl/conv_msl.hpp"
#include "msl/batchnorm_msl.hpp"
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

    id<MTLFunction> fn = [lib newFunctionWithName:(op.is_broadcast ? @"add_broadcast_kernel" : @"add_kernel")];
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

id<MTLComputePipelineState> MetalKernelCompiler::compile_mul_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_elementwise_mul(op);
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL for Mul: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:(op.is_broadcast ? @"mul_broadcast_kernel" : @"mul_kernel")];
    if (!fn) {
        log = "function mul_kernel not found";
        OPENVINO_THROW("MLIR backend: mul_kernel function missing in compiled library");
    }

    id<MTLComputePipelineState> pso = [m_device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    [lib release];
    if (!pso || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to create mul pipeline state: ", err);
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

id<MTLComputePipelineState> MetalKernelCompiler::compile_unary_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_unary(op);
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL for Unary: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"unary_kernel"];
    if (!fn) {
        log = "function unary_kernel not found";
        OPENVINO_THROW("MLIR backend: unary_kernel function missing in compiled library");
    }

    id<MTLComputePipelineState> pso = [m_device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    [lib release];
    if (!pso || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to create unary pipeline state: ", err);
    }

    return pso;
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_softmax_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_softmax(op);
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL for Softmax: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"softmax_kernel"];
    if (!fn) {
        log = "function softmax_kernel not found";
        OPENVINO_THROW("MLIR backend: softmax_kernel function missing in compiled library");
    }

    id<MTLComputePipelineState> pso = [m_device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    [lib release];
    if (!pso || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to create softmax pipeline state: ", err);
    }

    return pso;
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_maxpool2d_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_maxpool2d(op);
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL for MaxPool2D: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"maxpool2d_kernel"];
    if (!fn) {
        log = "function maxpool2d_kernel not found";
        OPENVINO_THROW("MLIR backend: maxpool2d_kernel function missing in compiled library");
    }

    id<MTLComputePipelineState> pso = [m_device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    [lib release];
    if (!pso || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to create maxpool2d pipeline state: ", err);
    }

    return pso;
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_avgpool2d_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_avgpool2d(op);
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL for AvgPool2D: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"avgpool2d_kernel"];
    if (!fn) {
        log = "function avgpool2d_kernel not found";
        OPENVINO_THROW("MLIR backend: avgpool2d_kernel function missing in compiled library");
    }

    id<MTLComputePipelineState> pso = [m_device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    [lib release];
    if (!pso || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to create avgpool2d pipeline state: ", err);
    }

    return pso;
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_conv2d_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_conv2d(op);
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL for Conv2D: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"conv2d_kernel"];
    if (!fn) {
        log = "function conv2d_kernel not found";
        OPENVINO_THROW("MLIR backend: conv2d_kernel function missing in compiled library");
    }

    id<MTLComputePipelineState> pso = [m_device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    [lib release];
    if (!pso || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to create conv2d pipeline state: ", err);
    }
    return pso;
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_batchnorm2d_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_batchnorm2d(op);
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL for BatchNorm2D: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"batchnorm2d_kernel"];
    if (!fn) {
        log = "function batchnorm2d_kernel not found";
        OPENVINO_THROW("MLIR backend: batchnorm2d_kernel function missing in compiled library");
    }

    id<MTLComputePipelineState> pso = [m_device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    [lib release];
    if (!pso || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to create batchnorm2d pipeline state: ", err);
    }
    return pso;
}

}  // namespace metal_plugin
}  // namespace ov
