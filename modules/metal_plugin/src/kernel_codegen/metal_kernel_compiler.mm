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
#include "msl/conv_msl.hpp"  // conv2d hand-written
#include "kernel_codegen/conv3d_msl.hpp"
#include "msl/batchnorm_msl.hpp"
#include "msl/slice_msl.hpp"
#include "msl/split_msl.hpp"
#include "msl/interpolate_msl.hpp"
#include "msl/sub_msl.hpp"
#include "msl/concat_msl.hpp"
#include "mlir_codegen/codegen_common.hpp"
#include "runtime/metal_dtype.hpp"
#include "runtime/metal_logger.hpp"
#include <numeric>
#include "openvino/core/except.hpp"

namespace ov {
namespace metal_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_msl_from_source(const std::string& source,
                                                                         const char* entry_point,
                                                                         std::string& log) {
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    if (metal_log_config().level >= MetalLogLevel::Trace) {
        METAL_LOG_TRACE("msl", "[METAL MSL] source dump (" << (entry_point ? entry_point : "") << "):\n" << source);
    }
    opts.fastMathEnabled = NO;
    const char* dump_env = std::getenv("OV_METAL_DEBUG_MSL");
    if (dump_env && std::string(dump_env) != "0") {
        METAL_LOG_TRACE("msl", "[METAL MSL] source dump (" << (entry_point ? entry_point : "") << "):\n" << source);
    }
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL source: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:entry_point]];
    if (!fn) {
        log = std::string("function ") + entry_point + " not found";
        [lib release];
        OPENVINO_THROW("MLIR backend: entry point missing in compiled library");
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

id<MTLComputePipelineState> MetalKernelCompiler::compile_add_kernel(const KernelOp& op, std::string& log) {
    EltwiseCodegenDesc desc;
    desc.kind = op.kind;
    desc.eltwise_kind = op.kind;
    if (op.output) {
        desc.element_type = op.output->dtype.ov_type;
    }
    desc.num_elements = static_cast<uint32_t>(op.output ? std::accumulate(op.output->shape.begin(), op.output->shape.end(), int64_t{1}, std::multiplies<int64_t>()) : 0);
    if (desc.num_elements == 0 && !op.out_shape.empty()) {
        desc.num_elements = static_cast<uint32_t>(std::accumulate(op.out_shape.begin(), op.out_shape.end(), int64_t{1}, std::multiplies<int64_t>()));
    }
    if (desc.num_elements == 0 && !op.out_shape.empty()) {
        desc.num_elements = static_cast<uint32_t>(std::accumulate(op.out_shape.begin(), op.out_shape.end(), int64_t{1}, std::multiplies<int64_t>()));
    }
    if (desc.num_elements == 0 && !op.out_shape.empty()) {
        desc.num_elements = static_cast<uint32_t>(std::accumulate(op.out_shape.begin(), op.out_shape.end(), int64_t{1}, std::multiplies<int64_t>()));
    }
    desc.is_broadcast = op.is_broadcast;
    desc.out_shape = op.out_shape;
    desc.stride0 = op.stride0;
    desc.stride1 = op.stride1;
    auto source = generate_msl_from_mlir(nullptr, desc);
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_sub_kernel(const KernelOp& op, std::string& log) {
    EltwiseCodegenDesc desc;
    desc.kind = op.kind;
    desc.eltwise_kind = op.kind;
    if (op.output) {
        desc.element_type = op.output->dtype.ov_type;
    }
    auto product = [](const std::vector<int64_t>& s) {
        return static_cast<uint32_t>(std::accumulate(s.begin(), s.end(), int64_t{1}, std::multiplies<int64_t>()));
    };
    desc.num_elements = (op.output && !op.output->shape.empty()) ? product(op.output->shape) : 0u;
    if (desc.num_elements == 0 && !op.out_shape.empty()) {
        desc.num_elements = product(op.out_shape);
    }
    if (desc.num_elements == 0 && op.input0 && !op.input0->shape.empty()) {
        desc.num_elements = product(op.input0->shape);
    }
    desc.is_broadcast = op.is_broadcast;
    desc.out_shape = op.out_shape;
    desc.stride0 = op.stride0;
    desc.stride1 = op.stride1;
    auto source = generate_msl_from_mlir(nullptr, desc);
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_div_kernel(const KernelOp& op, std::string& log) {
    EltwiseCodegenDesc desc;
    desc.kind = op.kind;
    desc.eltwise_kind = op.kind;
    if (op.output) {
        desc.element_type = op.output->dtype.ov_type;
    }
    desc.num_elements = static_cast<uint32_t>(op.output ? std::accumulate(op.output->shape.begin(),
                                                                            op.output->shape.end(),
                                                                            int64_t{1},
                                                                            std::multiplies<int64_t>()) :
                                                          0);
    if (desc.num_elements == 0 && !op.out_shape.empty()) {
        desc.num_elements = static_cast<uint32_t>(std::accumulate(op.out_shape.begin(),
                                                                  op.out_shape.end(),
                                                                  int64_t{1},
                                                                  std::multiplies<int64_t>()));
    }
    desc.is_broadcast = op.is_broadcast;
    desc.out_shape = op.out_shape;
    desc.stride0 = op.stride0;
    desc.stride1 = op.stride1;
    // Keep compute in float32 even for f16 storage to avoid precision loss / mismatch.
    auto source = generate_msl_from_mlir(nullptr, desc);
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_mod_kernel(const KernelOp& op, std::string& log) {
    EltwiseCodegenDesc desc;
    desc.kind = op.kind;
    desc.eltwise_kind = op.kind;
    if (op.output) {
        desc.element_type = op.output->dtype.ov_type;
    }
    desc.num_elements = static_cast<uint32_t>(op.output ? std::accumulate(op.output->shape.begin(), op.output->shape.end(), int64_t{1}, std::multiplies<int64_t>()) : 0);
    if (desc.num_elements == 0 && !op.out_shape.empty()) {
        desc.num_elements = static_cast<uint32_t>(std::accumulate(op.out_shape.begin(), op.out_shape.end(), int64_t{1}, std::multiplies<int64_t>()));
    }
    desc.is_broadcast = op.is_broadcast;
    desc.out_shape = op.out_shape;
    desc.stride0 = op.stride0;
    desc.stride1 = op.stride1;
    // Compute in float32 even when storing f16.
    auto source = generate_msl_for_eltwise(desc, nullptr);
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_floor_mod_kernel(const KernelOp& op, std::string& log) {
    EltwiseCodegenDesc desc;
    desc.kind = op.kind;
    desc.eltwise_kind = op.kind;
    if (op.output) {
        desc.element_type = op.output->dtype.ov_type;
    }
    desc.num_elements = static_cast<uint32_t>(op.output ? std::accumulate(op.output->shape.begin(), op.output->shape.end(), int64_t{1}, std::multiplies<int64_t>()) : 0);
    if (desc.num_elements == 0 && !op.out_shape.empty()) {
        desc.num_elements = static_cast<uint32_t>(std::accumulate(op.out_shape.begin(), op.out_shape.end(), int64_t{1}, std::multiplies<int64_t>()));
    }
    desc.is_broadcast = op.is_broadcast;
    desc.out_shape = op.out_shape;
    desc.stride0 = op.stride0;
    desc.stride1 = op.stride1;
    // Always compute in float32; downcast handled in runtime if needed.
    auto source = generate_msl_for_eltwise(desc, nullptr);
    const char* dump_env = std::getenv("OV_METAL_DEBUG_MSL");
    if (dump_env && std::string(dump_env) != "0") {
        METAL_LOG_TRACE("msl", "[METAL MSL] FloorMod source:\n" << source);
    }
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_pow_kernel(const KernelOp& op, std::string& log) {
    EltwiseCodegenDesc desc;
    desc.kind = op.kind;
    desc.eltwise_kind = op.kind;
    if (op.output) {
        desc.element_type = op.output->dtype.ov_type;
    }
    desc.num_elements = static_cast<uint32_t>(op.output ? std::accumulate(op.output->shape.begin(),
                                                                            op.output->shape.end(),
                                                                            int64_t{1},
                                                                            std::multiplies<int64_t>()) :
                                                          0);
    if (desc.num_elements == 0 && !op.out_shape.empty()) {
        desc.num_elements = static_cast<uint32_t>(std::accumulate(op.out_shape.begin(),
                                                                  op.out_shape.end(),
                                                                  int64_t{1},
                                                                  std::multiplies<int64_t>()));
    }
    desc.is_broadcast = op.is_broadcast;
    desc.out_shape = op.out_shape;
    desc.stride0 = op.stride0;
    desc.stride1 = op.stride1;
    auto source = generate_msl_from_mlir(nullptr, desc);
    return compile_msl_from_source(source, "eltwise_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_slice_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_slice(op);
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL for Slice: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"slice_kernel"];
    if (!fn) {
        log = "function slice_kernel not found";
        OPENVINO_THROW("MLIR backend: slice_kernel function missing in compiled library");
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

id<MTLComputePipelineState> MetalKernelCompiler::compile_split_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_split_msl(op);
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL for Split: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"split_kernel"];
    if (!fn) {
        log = "function split_kernel not found";
        OPENVINO_THROW("MLIR backend: split_kernel function missing in compiled library");
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

id<MTLComputePipelineState> MetalKernelCompiler::compile_concat_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_concat(op);
    return compile_msl_from_source(source, "concat_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_interpolate_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_interpolate(op);
    return compile_msl_from_source(source, "interpolate_kernel", log);
}

id<MTLComputePipelineState> MetalKernelCompiler::compile_mul_kernel(const KernelOp& op, std::string& log) {
    EltwiseCodegenDesc desc;
    desc.kind = op.kind;
    desc.eltwise_kind = op.kind;
    desc.num_elements = static_cast<uint32_t>(op.output ? std::accumulate(op.output->shape.begin(),
                                                                            op.output->shape.end(),
                                                                            int64_t{1},
                                                                            std::multiplies<int64_t>()) :
                                                          0);
    if (desc.num_elements == 0 && !op.out_shape.empty()) {
        desc.num_elements = static_cast<uint32_t>(std::accumulate(op.out_shape.begin(),
                                                                  op.out_shape.end(),
                                                                  int64_t{1},
                                                                  std::multiplies<int64_t>()));
    }
    // Force broadcast-capable kernel; it'll also handle equal-shape
    desc.is_broadcast = true;
    desc.out_shape = op.out_shape;
    desc.stride0 = op.stride0;
    desc.stride1 = op.stride1;
    if (op.output) {
        desc.element_type = op.output->dtype.ov_type;
    }
    auto source = generate_msl_from_mlir(nullptr, desc);
    return compile_msl_from_source(source, "eltwise_kernel", log);
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

id<MTLComputePipelineState> MetalKernelCompiler::compile_conv3d_kernel(const KernelOp& op, std::string& log) {
    auto source = generate_msl_for_conv3d(op);
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL for Conv3D: ", err);
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"conv3d_kernel"];
    if (!fn) {
        log = "function conv3d_kernel not found";
        [lib release];
        OPENVINO_THROW("MLIR backend: conv3d_kernel function missing in compiled library");
    }

    id<MTLComputePipelineState> pso = [m_device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    [lib release];
    if (!pso || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to create conv3d pipeline state: ", err);
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
