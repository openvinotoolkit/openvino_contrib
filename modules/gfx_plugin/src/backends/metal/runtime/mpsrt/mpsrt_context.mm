// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/mpsrt/mpsrt_context.hpp"

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <chrono>
#include <iomanip>
#include <sstream>
#include <utility>

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

const MpsrtRuntimeTensor* find_tensor(const MpsrtModel& model, GfxMpsrtValue value) {
    for (const auto& tensor : model.tensors) {
        if (tensor.value == value) {
            return &tensor;
        }
    }
    return nullptr;
}

uint64_t tensor_dense_bytes(const GfxMpsrtTensorAbiDesc& desc) {
    const auto dtype = static_cast<GfxMpsrtDType>(desc.dtype);
    const uint32_t elem_bytes = gfx_mpsrt_element_size_bytes(dtype);
    if (elem_bytes == 0 || desc.rank == 0 || desc.rank > 8) {
        return 0;
    }
    uint64_t elements = 1;
    for (uint32_t i = 0; i < desc.rank; ++i) {
        if (desc.dims[i] == 0) {
            return 0;
        }
        elements *= desc.dims[i];
    }
    return elements * elem_bytes;
}

std::string make_const_tensor_cache_key(GfxMpsrtValue value,
                                        const GfxMpsrtTensorAbiDesc& desc,
                                        size_t bytes) {
    std::ostringstream stream;
    stream << "const|" << value
           << '|' << desc.dtype
           << '|' << desc.storage
           << '|' << desc.layout
           << '|' << desc.rank
           << '|' << bytes;
    for (uint32_t i = 0; i < desc.rank && i < 8; ++i) {
        stream << '|' << desc.dims[i];
    }
    return stream.str();
}

uint32_t matrix_count_or_one(const GfxMpsrtTensorAbiDesc& desc) {
    return desc.matrix_count == 0 ? 1 : desc.matrix_count;
}

bool matrix_count_can_feed_output(uint32_t input_count, uint32_t output_count) {
    return input_count == output_count || input_count == 1;
}

bool validate_matrix_desc(const GfxMpsrtTensorAbiDesc& desc,
                          const char* name,
                          std::string* log) {
    if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
        return fail(log, std::string("GFX MPSRT: MPS GEMM ") + name + " tensor is not matrix storage");
    }
    if (desc.matrix_rows == 0 || desc.matrix_columns == 0 || desc.matrix_row_bytes == 0) {
        return fail(log, std::string("GFX MPSRT: MPS GEMM ") + name + " matrix descriptor is incomplete");
    }
    if (desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32) &&
        desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F16)) {
        return fail(log, std::string("GFX MPSRT: MPS GEMM ") + name + " dtype is unsupported");
    }
    return true;
}

bool is_mps_conv2d_stage(GfxMpsrtStageKind kind) {
    return kind == GfxMpsrtStageKind::MPSConv2D ||
           kind == GfxMpsrtStageKind::MPSGroupConv2D;
}

bool is_first_class_mps_stage(GfxMpsrtStageKind kind) {
    switch (kind) {
        case GfxMpsrtStageKind::MPSConv2D:
        case GfxMpsrtStageKind::MPSGroupConv2D:
        case GfxMpsrtStageKind::MPSPool2D:
        case GfxMpsrtStageKind::MPSResize2D:
        case GfxMpsrtStageKind::MPSGemm:
        case GfxMpsrtStageKind::MPSSoftmax:
        case GfxMpsrtStageKind::MPSTopK:
            return true;
        case GfxMpsrtStageKind::MSLDispatch:
        case GfxMpsrtStageKind::SPIRVDispatch:
        case GfxMpsrtStageKind::Alias:
        case GfxMpsrtStageKind::Unknown:
        default:
            return false;
    }
}

bool validate_image_desc(const GfxMpsrtTensorAbiDesc& desc,
                         const char* stage_name,
                         const char* tensor_name,
                         std::string* log) {
    if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Image)) {
        return fail(log,
                    std::string("GFX MPSRT: ") + stage_name + " " +
                        tensor_name + " tensor is not image storage");
    }
    if (desc.rank != 4 || desc.image_width == 0 || desc.image_height == 0 ||
        desc.image_feature_channels == 0 || desc.image_batch == 0) {
        return fail(log,
                    std::string("GFX MPSRT: ") + stage_name + " " +
                        tensor_name + " image descriptor is incomplete");
    }
    if (desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32) &&
        desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F16)) {
        return fail(log,
                    std::string("GFX MPSRT: ") + stage_name + " " +
                        tensor_name + " dtype is unsupported");
    }
    return true;
}

bool validate_conv_weights_desc(const GfxMpsrtTensorAbiDesc& desc,
                                const MpsrtRuntimeStage& stage,
                                std::string* log) {
    const char* stage_name = gfx_mpsrt_stage_kind_name(stage.kind);
    if ((desc.flags & GfxMpsrtTensorFlagConst) == 0) {
        return fail(log, std::string("GFX MPSRT: ") + stage_name + " weights tensor must be const");
    }
    if (desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32) &&
        desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F16)) {
        return fail(log, std::string("GFX MPSRT: ") + stage_name + " weights dtype is unsupported");
    }
    if (stage.kind == GfxMpsrtStageKind::MPSConv2D && desc.rank != 4) {
        return fail(log, "GFX MPSRT: MPS Conv2D weights must be OIHW rank-4");
    }
    if (stage.kind == GfxMpsrtStageKind::MPSGroupConv2D && desc.rank != 5) {
        return fail(log, "GFX MPSRT: MPS GroupConv2D weights must be GOIHW rank-5");
    }
    if (stage.kind == GfxMpsrtStageKind::MPSGroupConv2D &&
        desc.dims[0] != stage.conv2d_desc.groups) {
        return fail(log, "GFX MPSRT: MPS GroupConv2D weights group count mismatch");
    }
    return true;
}

bool make_mps_gemm_cache_key(const MpsrtRuntimeStage& stage,
                             const GfxMpsrtTensorAbiDesc& lhs,
                             const GfxMpsrtTensorAbiDesc& rhs,
                             const GfxMpsrtTensorAbiDesc& output,
                             uint32_t result_rows,
                             uint32_t result_columns,
                             uint32_t interior_columns,
                             std::string& key,
                             std::string* log) {
    if (lhs.dtype != rhs.dtype || lhs.dtype != output.dtype) {
        return fail(log, "GFX MPSRT: MPS GEMM matrix dtype mismatch");
    }
    const uint32_t lhs_count = matrix_count_or_one(lhs);
    const uint32_t rhs_count = matrix_count_or_one(rhs);
    const uint32_t output_count = matrix_count_or_one(output);
    if (!matrix_count_can_feed_output(lhs_count, output_count) ||
        !matrix_count_can_feed_output(rhs_count, output_count)) {
        return fail(log, "GFX MPSRT: MPS GEMM batch matrix counts must be either 1 or output matrix count");
    }

    const uint32_t lhs_rows = stage.gemm_desc.transpose_lhs ? lhs.matrix_columns : lhs.matrix_rows;
    const uint32_t lhs_columns = stage.gemm_desc.transpose_lhs ? lhs.matrix_rows : lhs.matrix_columns;
    const uint32_t rhs_rows = stage.gemm_desc.transpose_rhs ? rhs.matrix_columns : rhs.matrix_rows;
    const uint32_t rhs_columns = stage.gemm_desc.transpose_rhs ? rhs.matrix_rows : rhs.matrix_columns;
    if (lhs_rows != result_rows || rhs_columns != result_columns || lhs_columns != interior_columns ||
        rhs_rows != interior_columns) {
        return fail(log, "GFX MPSRT: MPS GEMM matrix dimensions do not match transpose contract");
    }

    std::ostringstream stream;
    stream << "mps_gemm|" << stage.gemm_desc.transpose_lhs
           << '|' << stage.gemm_desc.transpose_rhs
           << '|' << std::setprecision(9) << stage.gemm_desc.alpha
           << '|' << std::setprecision(9) << stage.gemm_desc.beta
           << '|' << result_rows
           << '|' << result_columns
           << '|' << interior_columns
           << '|' << lhs.dtype
           << '|' << lhs_count
           << '|' << rhs_count
           << '|' << output_count;
    key = stream.str();
    return true;
}

std::string unsupported_mps_stage_message(const MpsrtRuntimeStage& stage) {
    std::ostringstream stream;
    stream << "GFX MPSRT: " << gfx_mpsrt_stage_kind_name(stage.kind)
           << " is a first-class AppleMps vendor stage, but its runtime encoder is not implemented yet";
    return stream.str();
}

}  // namespace

struct MpsrtContext::PipelineCacheEntry {
    std::string key;
    id<MTLComputePipelineState> pipeline = nil;
};

struct MpsrtContext::MpsGemmCacheEntry {
    std::string key;
    id kernel = nil;
};

struct MpsrtContext::ConstTensorCacheEntry {
    std::string key;
    GfxMpsrtValue value = 0;
    GfxMpsrtTensorAbiDesc desc{};
    size_t bytes = 0;
    id<MTLBuffer> buffer = nil;
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
    for (auto& entry : m_mps_gemm_cache) {
        [entry.kernel release];
        entry.kernel = nil;
    }
    for (auto& entry : m_const_tensor_cache) {
        [entry.buffer release];
        entry.buffer = nil;
    }
    [m_binary_archive release];
    [m_command_queue release];
    [m_device release];
}

size_t MpsrtContext::pipeline_cache_size() const {
    return m_pipeline_cache.size();
}

bool MpsrtContext::register_const_tensor_data(GfxMpsrtValue value,
                                              const GfxMpsrtTensorAbiDesc& desc,
                                              const void* data,
                                              size_t bytes,
                                              std::string* log) {
    if (value == 0 && (desc.flags & GfxMpsrtTensorFlagConst) == 0) {
        return fail(log, "GFX MPSRT: const tensor value must be nonzero unless descriptor is explicitly const");
    }
    if ((desc.flags & GfxMpsrtTensorFlagConst) == 0) {
        return fail(log, "GFX MPSRT: registered tensor descriptor is not marked const");
    }
    if (!data || bytes == 0) {
        return fail(log, "GFX MPSRT: const tensor data is empty");
    }
    const uint64_t expected_bytes = tensor_dense_bytes(desc);
    if (expected_bytes == 0 || expected_bytes != bytes) {
        std::ostringstream stream;
        stream << "GFX MPSRT: const tensor byte size mismatch for value " << value
               << " expected " << expected_bytes << " got " << bytes;
        return fail(log, stream.str());
    }

    const std::string key = make_const_tensor_cache_key(value, desc, bytes);
    for (const auto& entry : m_const_tensor_cache) {
        if (entry.value == value) {
            if (entry.key != key) {
                return fail(log, "GFX MPSRT: const tensor value was already registered with a different descriptor");
            }
            ++m_const_tensor_cache_hits;
            increment_compile_counter("mpsrt_const_tensor_cache_hit_count");
            return true;
        }
    }

    ++m_const_tensor_cache_misses;
    increment_compile_counter("mpsrt_const_tensor_cache_miss_count");

    id<MTLBuffer> staging = [m_device newBufferWithBytes:data
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared];
    if (!staging) {
        return fail(log, "GFX MPSRT: failed to create staging buffer for const tensor");
    }
    id<MTLBuffer> buffer = [m_device newBufferWithLength:bytes
                                                 options:MTLResourceStorageModePrivate];
    if (!buffer) {
        [staging release];
        return fail(log, "GFX MPSRT: failed to create private buffer for const tensor");
    }

    id<MTLCommandBuffer> command = [m_command_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromBuffer:staging
            sourceOffset:0
                toBuffer:buffer
       destinationOffset:0
                    size:bytes];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    [staging release];
    if ([command status] != MTLCommandBufferStatusCompleted) {
        [buffer release];
        return fail(log, "GFX MPSRT: failed to upload const tensor into GPU-owned const pack");
    }

    m_const_tensor_cache.push_back({key, value, desc, bytes, buffer});
    increment_compile_counter("mpsrt_const_tensor_upload_count");
    return true;
}

bool MpsrtContext::has_const_tensor(GfxMpsrtValue value) const {
    for (const auto& entry : m_const_tensor_cache) {
        if (entry.value == value) {
            return true;
        }
    }
    return false;
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

bool MpsrtContext::prepare_mps_gemm(const MpsrtModel& model,
                                    const MpsrtRuntimeStage& stage,
                                    MpsrtPreparedMpsGemm& out,
                                    std::string* log) {
    out = {};
    if (stage.kind != GfxMpsrtStageKind::MPSGemm) {
        return fail(log, "GFX MPSRT: requested MPS GEMM preparation for non-GEMM stage");
    }
    if (!MPSSupportsMTLDevice(m_device)) {
        return fail(log, "GFX MPSRT: Metal device does not support MPS");
    }
    if (stage.inputs.size() != 2 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(log, "GFX MPSRT: MPS GEMM requires two inputs and one output");
    }

    const auto* lhs_tensor = find_tensor(model, stage.inputs[0]);
    const auto* rhs_tensor = find_tensor(model, stage.inputs[1]);
    if (!lhs_tensor || !rhs_tensor) {
        return fail(log, "GFX MPSRT: MPS GEMM input tensor descriptor is missing");
    }
    const auto& lhs = lhs_tensor->desc;
    const auto& rhs = rhs_tensor->desc;
    const auto& output = stage.output_descs.front();
    if (!validate_matrix_desc(lhs, "lhs", log) ||
        !validate_matrix_desc(rhs, "rhs", log) ||
        !validate_matrix_desc(output, "output", log)) {
        return false;
    }

    const uint32_t result_rows = output.matrix_rows;
    const uint32_t result_columns = output.matrix_columns;
    const uint32_t interior_columns = stage.gemm_desc.transpose_lhs ? lhs.matrix_rows : lhs.matrix_columns;
    std::string key;
    if (!make_mps_gemm_cache_key(stage, lhs, rhs, output, result_rows, result_columns, interior_columns, key, log)) {
        return false;
    }

    for (const auto& entry : m_mps_gemm_cache) {
        if (entry.key == key) {
            ++m_mps_gemm_cache_hits;
            increment_compile_counter("mpsrt_mps_gemm_kernel_cache_hit_count");
            out.stage_record_key = stage.stage_record_key;
            out.gemm_desc = stage.gemm_desc;
            out.result_rows = result_rows;
            out.result_columns = result_columns;
            out.interior_columns = interior_columns;
            out.batch_count = matrix_count_or_one(output);
            out.lhs_batch_broadcast = matrix_count_or_one(lhs) == 1 && out.batch_count > 1;
            out.rhs_batch_broadcast = matrix_count_or_one(rhs) == 1 && out.batch_count > 1;
            out.data_type = output.dtype;
            out.kernel_cache_hit = true;
            out.kernel = entry.kernel;
            return true;
        }
    }

    ++m_mps_gemm_cache_misses;
    increment_compile_counter("mpsrt_mps_gemm_kernel_cache_miss_count");
    id kernel = [[MPSMatrixMultiplication alloc] initWithDevice:m_device
                                                  transposeLeft:(stage.gemm_desc.transpose_lhs != 0)
                                                 transposeRight:(stage.gemm_desc.transpose_rhs != 0)
                                                     resultRows:result_rows
                                                  resultColumns:result_columns
                                                interiorColumns:interior_columns
                                                          alpha:stage.gemm_desc.alpha
                                                           beta:stage.gemm_desc.beta];
    if (!kernel) {
        return fail(log, "GFX MPSRT: failed to create MPSMatrixMultiplication kernel");
    }

    m_mps_gemm_cache.push_back({key, kernel});
    increment_compile_counter("mpsrt_prepare_mps_gemm_count");
    out.stage_record_key = stage.stage_record_key;
    out.gemm_desc = stage.gemm_desc;
    out.result_rows = result_rows;
    out.result_columns = result_columns;
    out.interior_columns = interior_columns;
    out.batch_count = matrix_count_or_one(output);
    out.lhs_batch_broadcast = matrix_count_or_one(lhs) == 1 && out.batch_count > 1;
    out.rhs_batch_broadcast = matrix_count_or_one(rhs) == 1 && out.batch_count > 1;
    out.data_type = output.dtype;
    out.kernel_cache_hit = false;
    out.kernel = kernel;
    return true;
}

bool MpsrtContext::prepare_mps_conv2d(const MpsrtModel& model,
                                      const MpsrtRuntimeStage& stage,
                                      MpsrtPreparedMpsConv2D& out,
                                      std::string* log) {
    out = {};
    if (!is_mps_conv2d_stage(stage.kind)) {
        return fail(log, "GFX MPSRT: requested MPS Conv2D preparation for non-Conv2D stage");
    }
    if (!MPSSupportsMTLDevice(m_device)) {
        return fail(log, "GFX MPSRT: Metal device does not support MPS");
    }
    if (stage.inputs.size() != 2 || stage.outputs.size() != 1 || stage.output_descs.size() != 1) {
        return fail(log, "GFX MPSRT: MPS Conv2D requires input, weights and one output");
    }

    const auto* input_tensor = find_tensor(model, stage.inputs[0]);
    const auto* weights_tensor = find_tensor(model, stage.inputs[1]);
    if (!input_tensor || !weights_tensor) {
        return fail(log, "GFX MPSRT: MPS Conv2D input or weights tensor descriptor is missing");
    }
    const auto& input = input_tensor->desc;
    const auto& weights = weights_tensor->desc;
    const auto& output = stage.output_descs.front();
    const char* stage_name = gfx_mpsrt_stage_kind_name(stage.kind);
    if (!validate_image_desc(input, stage_name, "input", log) ||
        !validate_image_desc(output, stage_name, "output", log) ||
        !validate_conv_weights_desc(weights, stage, log)) {
        return false;
    }
    if (input.dtype != output.dtype || weights.dtype != output.dtype) {
        return fail(log, "GFX MPSRT: MPS Conv2D input, weights and output dtype must match for this ABI");
    }
    if (input.image_batch != output.image_batch) {
        return fail(log, "GFX MPSRT: MPS Conv2D input/output batch mismatch");
    }
    if (stage.conv2d_desc.groups == 0) {
        return fail(log, "GFX MPSRT: MPS Conv2D group count is zero");
    }
    const ConstTensorCacheEntry* weights_entry = nullptr;
    for (const auto& entry : m_const_tensor_cache) {
        if (entry.value == stage.inputs[1]) {
            weights_entry = &entry;
            break;
        }
    }
    if (!weights_entry || !weights_entry->buffer) {
        return fail(log,
                    "GFX MPSRT: MPS Conv2D weights tensor is not materialized in the MPSRT const-pack cache");
    }
    const uint64_t expected_weight_bytes = tensor_dense_bytes(weights);
    if (expected_weight_bytes == 0 || expected_weight_bytes != weights_entry->bytes) {
        return fail(log, "GFX MPSRT: MPS Conv2D const-pack weight size mismatch");
    }

    out.stage_record_key = stage.stage_record_key;
    out.conv2d_desc = stage.conv2d_desc;
    out.weights_value = stage.inputs[1];
    out.weights_byte_length = weights_entry->bytes;
    out.input_feature_channels = input.image_feature_channels;
    out.output_feature_channels = output.image_feature_channels;
    out.output_width = output.image_width;
    out.output_height = output.image_height;
    out.output_batch = output.image_batch;
    out.data_type = output.dtype;
    out.weights_cache_hit = true;
    out.weights_buffer = weights_entry->buffer;
    increment_compile_counter("mpsrt_prepare_mps_conv2d_count");
    return true;
}

bool MpsrtContext::prepare_model(const MpsrtModel& model,
                                 const std::string& msl_source,
                                 MpsrtPreparedModel& out,
                                 std::string* log) {
    out = {};
    for (size_t i = 0; i < model.stages.size(); ++i) {
        const auto& stage = model.stages[i];
        if (stage.kind == GfxMpsrtStageKind::MSLDispatch) {
            MpsrtPreparedMslDispatch prepared;
            if (!prepare_msl_dispatch(stage, msl_source, prepared, log)) {
                return false;
            }
            prepared.stage_index = i;
            out.msl_dispatches.push_back(prepared);
        } else if (stage.kind == GfxMpsrtStageKind::MPSGemm) {
            MpsrtPreparedMpsGemm prepared;
            if (!prepare_mps_gemm(model, stage, prepared, log)) {
                return false;
            }
            prepared.stage_index = i;
            out.mps_gemm_stages.push_back(prepared);
        } else if (is_mps_conv2d_stage(stage.kind)) {
            MpsrtPreparedMpsConv2D prepared;
            if (!prepare_mps_conv2d(model, stage, prepared, log)) {
                return false;
            }
            prepared.stage_index = i;
            out.mps_conv2d_stages.push_back(prepared);
        } else if (is_first_class_mps_stage(stage.kind)) {
            return fail(log, unsupported_mps_stage_message(stage));
        } else {
            ++out.skipped_non_msl_stages;
        }
    }
    return true;
}

}  // namespace mpsrt
}  // namespace metal
}  // namespace gfx_plugin
}  // namespace ov
