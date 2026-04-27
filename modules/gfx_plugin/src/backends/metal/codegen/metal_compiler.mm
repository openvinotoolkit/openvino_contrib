// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backends/metal/codegen/metal_compiler.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {

id<MTLComputePipelineState> MetalKernelCompiler::compile_msl_from_source(const std::string& source,
                                                                         const char* entry_point,
                                                                         std::string& log) {
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    if (gfx_log_config().level >= GfxLogLevel::Trace) {
        gfx_log_trace("msl") << "[GFX MSL] source dump (" << (entry_point ? entry_point : "") << "):\n" << source;
    }
    opts.fastMathEnabled = NO;
    const char* dump_env = std::getenv("OV_GFX_DEBUG_MSL");
    if (dump_env && std::string(dump_env) != "0") {
        gfx_log_trace("msl") << "[GFX MSL] source dump (" << (entry_point ? entry_point : "") << "):\n" << source;
    }
    const auto library_compile_start =
        current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    id<MTLLibrary> lib = [m_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                                 options:opts
                                                   error:&error];
    if (current_compile_trace()) {
        increment_compile_counter("metal_library_compile_count");
        add_compile_segment(
            "metal_library_compile",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - library_compile_start)
                                      .count()));
    }
    [opts release];
    if (!lib || error) {
        std::string err = error ? std::string([[error localizedDescription] UTF8String]) : "unknown";
        log = err;
        OPENVINO_THROW("MLIR backend: failed to compile MSL source: ", err);
    }

    const auto function_lookup_start =
        current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:entry_point]];
    if (current_compile_trace()) {
        increment_compile_counter("metal_function_lookup_count");
        add_compile_segment(
            "metal_function_lookup",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - function_lookup_start)
                                      .count()));
    }
    if (!fn) {
        log = std::string("function ") + entry_point + " not found";
        [lib release];
        OPENVINO_THROW("MLIR backend: entry point missing in compiled library");
    }

    const auto pipeline_state_start =
        current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    id<MTLComputePipelineState> pso = [m_device newComputePipelineStateWithFunction:fn error:&error];
    if (current_compile_trace()) {
        increment_compile_counter("metal_pipeline_state_create_count");
        add_compile_segment(
            "metal_pipeline_state_create",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - pipeline_state_start)
                                      .count()));
    }
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
