// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backends/metal/codegen/metal_compiler.hpp"
#include "runtime/gfx_logger.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_msl_from_source(const std::string& source,
                                                                         const char* entry_point,
                                                                         std::string& log) {
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    if (gfx_log_config().level >= GfxLogLevel::Trace) {
        GFX_LOG_TRACE("msl", "[GFX MSL] source dump (" << (entry_point ? entry_point : "") << "):\n" << source);
    }
    opts.fastMathEnabled = NO;
    const char* dump_env = std::getenv("OV_GFX_DEBUG_MSL");
    if (dump_env && std::string(dump_env) != "0") {
        GFX_LOG_TRACE("msl", "[GFX MSL] source dump (" << (entry_point ? entry_point : "") << "):\n" << source);
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

}  // namespace gfx_plugin
}  // namespace ov
