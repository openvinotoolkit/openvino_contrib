// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/mpsrt/mpsrt_context.hpp"

#import <Foundation/Foundation.h>

#include <chrono>
#include <sstream>

#include "openvino/core/except.hpp"
#include "runtime/gfx_compile_profiling.hpp"

namespace ov {
namespace gfx_plugin {
namespace metal {
namespace mpsrt {
namespace {

std::string ns_error_message(NSError* error, const char* fallback) {
    if (!error) {
        return fallback ? std::string(fallback) : std::string("unknown Metal error");
    }
    NSString* description = [error localizedDescription];
    return description ? std::string([description UTF8String]) : std::string("unknown Metal error");
}

std::string make_pipeline_cache_key(const MpsrtRuntimeStage& stage, const std::string& source) {
    std::ostringstream key;
    key << stage.dispatch_kernel_family_id << '|'
        << stage.dispatch_entry_point << '|'
        << stage.dispatch_threads_per_threadgroup << '|'
        << stage.dispatch_flags << '|'
        << std::hash<std::string>{}(source);
    return key.str();
}

bool fail(std::string* log, const std::string& message) {
    if (log) {
        *log = message;
    }
    return false;
}

}  // namespace

struct MpsrtContext::PipelineCacheEntry {
    std::string key;
    id<MTLComputePipelineState> pipeline = nil;
};

MpsrtContext::MpsrtContext(id<MTLDevice> device) : m_device([device retain]) {
    OPENVINO_ASSERT(m_device, "GFX MPSRT: Metal device is null");
    m_command_queue = [m_device newCommandQueue];
    OPENVINO_ASSERT(m_command_queue, "GFX MPSRT: failed to create Metal command queue");

    if (@available(macOS 11.0, iOS 14.0, *)) {
        NSError* error = nil;
        MTLBinaryArchiveDescriptor* descriptor = [[MTLBinaryArchiveDescriptor alloc] init];
        m_binary_archive = [m_device newBinaryArchiveWithDescriptor:descriptor error:&error];
        [descriptor release];
        if (!m_binary_archive) {
            increment_compile_counter("mpsrt_binary_archive_create_failed_count");
        }
    }
}

MpsrtContext::~MpsrtContext() {
    for (auto& entry : m_pipeline_cache) {
        [entry.pipeline release];
        entry.pipeline = nil;
    }
    [m_binary_archive release];
    [m_command_queue release];
    [m_device release];
}

size_t MpsrtContext::pipeline_cache_size() const {
    return m_pipeline_cache.size();
}

id<MTLComputePipelineState> MpsrtContext::get_or_create_pipeline(const MpsrtRuntimeStage& stage,
                                                                 const std::string& msl_source,
                                                                 bool& cache_hit,
                                                                 std::string* log) {
    cache_hit = false;
    if (stage.dispatch_entry_point.empty()) {
        (void)fail(log, "GFX MPSRT: MSL dispatch entry point is empty");
        return nil;
    }
    if (msl_source.empty()) {
        (void)fail(log, "GFX MPSRT: MSL source is empty");
        return nil;
    }

    const std::string key = make_pipeline_cache_key(stage, msl_source);
    for (const auto& entry : m_pipeline_cache) {
        if (entry.key == key) {
            cache_hit = true;
            ++m_pipeline_cache_hits;
            increment_compile_counter("mpsrt_pso_cache_hit_count");
            return entry.pipeline;
        }
    }

    ++m_pipeline_cache_misses;
    increment_compile_counter("mpsrt_pso_cache_miss_count");

    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];

    const auto library_start =
        current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    id<MTLLibrary> library = [m_device newLibraryWithSource:[NSString stringWithUTF8String:msl_source.c_str()]
                                                    options:options
                                                      error:&error];
    [options release];
    if (current_compile_trace()) {
        increment_compile_counter("mpsrt_library_compile_count");
        add_compile_segment(
            "mpsrt_library_compile",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - library_start)
                                      .count()));
    }
    if (!library || error) {
        const std::string message = "GFX MPSRT: failed to compile MSL library: " +
                                    ns_error_message(error, "library compile failed");
        [library release];
        (void)fail(log, message);
        return nil;
    }

    const auto function_start =
        current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:stage.dispatch_entry_point.c_str()]];
    if (current_compile_trace()) {
        increment_compile_counter("mpsrt_function_lookup_count");
        add_compile_segment(
            "mpsrt_function_lookup",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - function_start)
                                      .count()));
    }
    [library release];
    if (!function) {
        (void)fail(log, "GFX MPSRT: function " + stage.dispatch_entry_point + " not found in MSL library");
        return nil;
    }

    const auto pso_start =
        current_compile_trace() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    id<MTLComputePipelineState> pipeline = nil;
    if (m_binary_archive) {
        MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
        descriptor.computeFunction = function;
        descriptor.binaryArchives = @[ m_binary_archive ];
        pipeline = [m_device newComputePipelineStateWithDescriptor:descriptor
                                                           options:0
                                                        reflection:nil
                                                             error:&error];
        [descriptor release];
    } else {
        pipeline = [m_device newComputePipelineStateWithFunction:function error:&error];
    }
    [function release];
    if (current_compile_trace()) {
        increment_compile_counter("mpsrt_pipeline_state_create_count");
        add_compile_segment(
            "mpsrt_pipeline_state_create",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                      std::chrono::steady_clock::now() - pso_start)
                                      .count()));
    }
    if (!pipeline || error) {
        const std::string message = "GFX MPSRT: failed to create pipeline state: " +
                                    ns_error_message(error, "pipeline state creation failed");
        [pipeline release];
        (void)fail(log, message);
        return nil;
    }

    m_pipeline_cache.push_back({key, pipeline});
    return pipeline;
}

bool MpsrtContext::prepare_msl_dispatch(const MpsrtRuntimeStage& stage,
                                        const std::string& msl_source,
                                        MpsrtPreparedMslDispatch& out,
                                        std::string* log) {
    out = {};
    if (stage.kind != GfxMpsrtStageKind::MSLDispatch) {
        return fail(log, "GFX MPSRT: requested MSL preparation for non-MSL stage");
    }
    if (stage.dispatch_kernel_family_id == 0 || stage.msl_dispatch_desc.kernel_family == 0) {
        return fail(log, "GFX MPSRT: MSL dispatch kernel family is not set");
    }

    bool cache_hit = false;
    id<MTLComputePipelineState> pipeline = get_or_create_pipeline(stage, msl_source, cache_hit, log);
    if (!pipeline) {
        return false;
    }

    increment_compile_counter("mpsrt_prepare_msl_dispatch_count");
    out.stage_record_key = stage.stage_record_key;
    out.dispatch_entry_point = stage.dispatch_entry_point;
    out.dispatch_kernel_family_id = stage.dispatch_kernel_family_id;
    out.dispatch_threads_per_threadgroup = stage.dispatch_threads_per_threadgroup;
    out.thread_execution_width = static_cast<uint32_t>([pipeline threadExecutionWidth]);
    out.max_total_threads_per_threadgroup = static_cast<uint32_t>([pipeline maxTotalThreadsPerThreadgroup]);
    out.pipeline_cache_hit = cache_hit;
    out.pipeline = pipeline;
    return true;
}

bool MpsrtContext::prepare_model(const MpsrtModel& model,
                                 const std::string& msl_source,
                                 MpsrtPreparedModel& out,
                                 std::string* log) {
    out = {};
    for (size_t i = 0; i < model.stages.size(); ++i) {
        const auto& stage = model.stages[i];
        if (stage.kind != GfxMpsrtStageKind::MSLDispatch) {
            ++out.skipped_non_msl_stages;
            continue;
        }
        MpsrtPreparedMslDispatch prepared;
        if (!prepare_msl_dispatch(stage, msl_source, prepared, log)) {
            return false;
        }
        prepared.stage_index = i;
        out.msl_dispatches.push_back(prepared);
    }
    return true;
}

}  // namespace mpsrt
}  // namespace metal
}  // namespace gfx_plugin
}  // namespace ov
