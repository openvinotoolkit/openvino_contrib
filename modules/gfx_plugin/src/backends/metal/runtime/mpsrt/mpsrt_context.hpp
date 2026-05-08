// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "backends/metal/runtime/mpsrt/mpsrt_model.hpp"
#include "runtime/gfx_mpsrt_storage_bridge.hpp"

namespace ov {
namespace gfx_plugin {
namespace metal {
namespace mpsrt {

struct MpsrtPreparedMslDispatch {
    size_t stage_index = 0;
    std::string stage_record_key;
    std::string dispatch_entry_point;
    uint32_t dispatch_kernel_family_id = 0;
    uint32_t dispatch_threads_per_threadgroup = 0;
    uint32_t thread_execution_width = 0;
    uint32_t max_total_threads_per_threadgroup = 0;
    bool pipeline_cache_hit = false;
    id<MTLComputePipelineState> pipeline = nil;
};

struct MpsrtPreparedMpsGemm {
    size_t stage_index = 0;
    std::string stage_record_key;
    GfxMpsrtGemmAbiDesc gemm_desc{};
    uint32_t result_rows = 0;
    uint32_t result_columns = 0;
    uint32_t interior_columns = 0;
    uint32_t batch_count = 1;
    bool lhs_batch_broadcast = false;
    bool rhs_batch_broadcast = false;
    uint32_t data_type = static_cast<uint32_t>(GfxMpsrtDType::Unknown);
    bool kernel_cache_hit = false;
    id kernel = nil;
};

struct MpsrtPreparedMpsConv2D {
    size_t stage_index = 0;
    std::string stage_record_key;
    GfxMpsrtConv2DAbiDesc conv2d_desc{};
    GfxMpsrtValue weights_value = 0;
    size_t weights_byte_length = 0;
    uint32_t input_feature_channels = 0;
    uint32_t output_feature_channels = 0;
    uint32_t output_width = 0;
    uint32_t output_height = 0;
    uint32_t output_batch = 0;
    uint32_t data_type = static_cast<uint32_t>(GfxMpsrtDType::Unknown);
    bool weights_cache_hit = false;
    bool kernel_cache_hit = false;
    id<MTLBuffer> weights_buffer = nil;
    id kernel = nil;
};

struct MpsrtPreparedMpsPool2D {
    size_t stage_index = 0;
    std::string stage_record_key;
    GfxMpsrtPool2DAbiDesc pool2d_desc{};
    uint32_t output_width = 0;
    uint32_t output_height = 0;
    uint32_t output_batch = 0;
    uint32_t output_feature_channels = 0;
    uint32_t data_type = static_cast<uint32_t>(GfxMpsrtDType::Unknown);
    bool kernel_cache_hit = false;
    id kernel = nil;
};

struct MpsrtPreparedMpsResize2D {
    size_t stage_index = 0;
    std::string stage_record_key;
    GfxMpsrtResize2DAbiDesc resize2d_desc{};
    uint32_t input_width = 0;
    uint32_t input_height = 0;
    uint32_t output_width = 0;
    uint32_t output_height = 0;
    uint32_t output_batch = 0;
    uint32_t output_feature_channels = 0;
    uint32_t data_type = static_cast<uint32_t>(GfxMpsrtDType::Unknown);
    bool kernel_cache_hit = false;
    id kernel = nil;
};

struct MpsrtPreparedMpsSoftmax {
    size_t stage_index = 0;
    std::string stage_record_key;
    GfxMpsrtSoftmaxAbiDesc softmax_desc{};
    uint32_t rows = 0;
    uint32_t columns = 0;
    uint32_t matrix_count = 1;
    uint32_t data_type = static_cast<uint32_t>(GfxMpsrtDType::Unknown);
    bool kernel_cache_hit = false;
    id kernel = nil;
};

struct MpsrtPreparedMpsTopK {
    size_t stage_index = 0;
    std::string stage_record_key;
    GfxMpsrtTopKAbiDesc topk_desc{};
    uint32_t rows = 0;
    uint32_t source_columns = 0;
    uint32_t k = 0;
    uint32_t matrix_count = 1;
    uint32_t data_type = static_cast<uint32_t>(GfxMpsrtDType::Unknown);
    uint32_t index_type = static_cast<uint32_t>(GfxMpsrtDType::Unknown);
    bool kernel_cache_hit = false;
    id kernel = nil;
};

struct MpsrtPreparedResource {
    uint32_t resource_index = 0;
    GfxMpsrtExternalBufferRole role = GfxMpsrtExternalBufferRole::Unknown;
    MpsrtRuntimeResourceLifetime lifetime = MpsrtRuntimeResourceLifetime::Unknown;
    bool has_tensor_value = false;
    GfxMpsrtValue value = 0;
    GfxMpsrtTensorAbiDesc tensor_desc{};
    size_t byte_length = 0;
    size_t offset = 0;
    size_t heap_allocation_size = 0;
    size_t heap_alignment = 0;
    size_t first_stage_index = 0;
    size_t last_stage_index = 0;
    id<MTLBuffer> buffer = nil;
    id<MTLTexture> texture = nil;
};

struct MpsrtPreparedImageBridgeResource {
    GfxMpsrtValue value = 0;
    GfxMpsrtStorageBridgeDirection direction = GfxMpsrtStorageBridgeDirection::Unknown;
    GfxMpsrtTensorAbiDesc tensor_desc{};
    size_t heap_allocation_size = 0;
    size_t heap_alignment = 0;
    id<MTLTexture> texture = nil;
};

struct MpsrtPreparedModel {
    id<MTLHeap> resource_heap = nil;
    size_t resource_heap_size = 0;
    size_t resource_heap_unaliased_size = 0;
    size_t resource_heap_aliasable_size = 0;
    size_t resource_heap_alias_reuse_count = 0;
    size_t transient_buffer_resource_count = 0;
    size_t transient_image_resource_count = 0;
    size_t image_bridge_resource_count = 0;
    std::vector<MpsrtPreparedResource> resources;
    std::vector<MpsrtPreparedImageBridgeResource> image_bridge_resources;
    std::vector<MpsrtPreparedMslDispatch> msl_dispatches;
    std::vector<MpsrtPreparedMpsGemm> mps_gemm_stages;
    std::vector<MpsrtPreparedMpsConv2D> mps_conv2d_stages;
    std::vector<MpsrtPreparedMpsPool2D> mps_pool2d_stages;
    std::vector<MpsrtPreparedMpsResize2D> mps_resize2d_stages;
    std::vector<MpsrtPreparedMpsSoftmax> mps_softmax_stages;
    std::vector<MpsrtPreparedMpsTopK> mps_topk_stages;
    uint32_t skipped_non_msl_stages = 0;
};

class MpsrtContext {
public:
    explicit MpsrtContext(id<MTLDevice> device);
    ~MpsrtContext();

    MpsrtContext(const MpsrtContext&) = delete;
    MpsrtContext& operator=(const MpsrtContext&) = delete;

    id<MTLDevice> device() const {
        return m_device;
    }

    id<MTLCommandQueue> command_queue() const {
        return m_command_queue;
    }

    bool register_const_tensor_data(GfxMpsrtValue value,
                                    const GfxMpsrtTensorAbiDesc& desc,
                                    const void* data,
                                    size_t bytes,
                                    std::string* log = nullptr);
    bool has_const_tensor(GfxMpsrtValue value) const;

    bool prepare_msl_dispatch(const MpsrtRuntimeStage& stage,
                              const std::string& msl_source,
                              MpsrtPreparedMslDispatch& out,
                              std::string* log = nullptr);

    bool prepare_mps_gemm(const MpsrtModel& model,
                          const MpsrtRuntimeStage& stage,
                          MpsrtPreparedMpsGemm& out,
                          std::string* log = nullptr);

    bool prepare_mps_conv2d(const MpsrtModel& model,
                            const MpsrtRuntimeStage& stage,
                            MpsrtPreparedMpsConv2D& out,
                            std::string* log = nullptr);

    bool prepare_mps_pool2d(const MpsrtModel& model,
                            const MpsrtRuntimeStage& stage,
                            MpsrtPreparedMpsPool2D& out,
                            std::string* log = nullptr);

    bool prepare_mps_resize2d(const MpsrtModel& model,
                              const MpsrtRuntimeStage& stage,
                              MpsrtPreparedMpsResize2D& out,
                              std::string* log = nullptr);

    bool prepare_mps_softmax(const MpsrtModel& model,
                             const MpsrtRuntimeStage& stage,
                             MpsrtPreparedMpsSoftmax& out,
                             std::string* log = nullptr);

    bool prepare_mps_topk(const MpsrtModel& model,
                          const MpsrtRuntimeStage& stage,
                          MpsrtPreparedMpsTopK& out,
                          std::string* log = nullptr);

    bool prepare_model(const MpsrtModel& model,
                       const std::string& msl_source,
                       MpsrtPreparedModel& out,
                       std::string* log = nullptr);

    size_t pipeline_cache_size() const;
    uint64_t pipeline_cache_hits() const {
        return m_pipeline_cache_hits;
    }
    uint64_t pipeline_cache_misses() const {
        return m_pipeline_cache_misses;
    }

    id<MTLComputePipelineState> get_or_create_pipeline(const MpsrtRuntimeStage& stage,
                                                       const std::string& msl_source,
                                                       bool& cache_hit,
                                                       std::string* log);
    bool prepare_model_resources(const MpsrtModel& model,
                                 MpsrtPreparedModel& out,
                                 std::string* log) const;

private:
    struct PipelineCacheEntry;
    struct MpsGemmCacheEntry;
    struct MpsConv2DCacheEntry;
    struct MpsPool2DCacheEntry;
    struct MpsResize2DCacheEntry;
    struct MpsSoftmaxCacheEntry;
    struct MpsTopKCacheEntry;
    struct ConstTensorCacheEntry;

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_command_queue = nil;
    id<MTLBinaryArchive> m_binary_archive = nil;
    std::vector<PipelineCacheEntry> m_pipeline_cache;
    std::vector<MpsGemmCacheEntry> m_mps_gemm_cache;
    std::vector<MpsConv2DCacheEntry> m_mps_conv2d_cache;
    std::vector<MpsPool2DCacheEntry> m_mps_pool2d_cache;
    std::vector<MpsResize2DCacheEntry> m_mps_resize2d_cache;
    std::vector<MpsSoftmaxCacheEntry> m_mps_softmax_cache;
    std::vector<MpsTopKCacheEntry> m_mps_topk_cache;
    std::vector<ConstTensorCacheEntry> m_const_tensor_cache;
    uint64_t m_pipeline_cache_hits = 0;
    uint64_t m_pipeline_cache_misses = 0;
    uint64_t m_mps_gemm_cache_hits = 0;
    uint64_t m_mps_gemm_cache_misses = 0;
    uint64_t m_mps_conv2d_cache_hits = 0;
    uint64_t m_mps_conv2d_cache_misses = 0;
    uint64_t m_mps_pool2d_cache_hits = 0;
    uint64_t m_mps_pool2d_cache_misses = 0;
    uint64_t m_mps_resize2d_cache_hits = 0;
    uint64_t m_mps_resize2d_cache_misses = 0;
    uint64_t m_mps_softmax_cache_hits = 0;
    uint64_t m_mps_softmax_cache_misses = 0;
    uint64_t m_mps_topk_cache_hits = 0;
    uint64_t m_mps_topk_cache_misses = 0;
    uint64_t m_const_tensor_cache_hits = 0;
    uint64_t m_const_tensor_cache_misses = 0;
};

}  // namespace mpsrt
}  // namespace metal
}  // namespace gfx_plugin
}  // namespace ov
