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
    id<MTLBuffer> weights_buffer = nil;
};

struct MpsrtPreparedModel {
    std::vector<MpsrtPreparedMslDispatch> msl_dispatches;
    std::vector<MpsrtPreparedMpsGemm> mps_gemm_stages;
    std::vector<MpsrtPreparedMpsConv2D> mps_conv2d_stages;
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

private:
    struct PipelineCacheEntry;
    struct MpsGemmCacheEntry;
    struct ConstTensorCacheEntry;

    id<MTLComputePipelineState> get_or_create_pipeline(const MpsrtRuntimeStage& stage,
                                                       const std::string& msl_source,
                                                       bool& cache_hit,
                                                       std::string* log);

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_command_queue = nil;
    id<MTLBinaryArchive> m_binary_archive = nil;
    std::vector<PipelineCacheEntry> m_pipeline_cache;
    std::vector<MpsGemmCacheEntry> m_mps_gemm_cache;
    std::vector<ConstTensorCacheEntry> m_const_tensor_cache;
    uint64_t m_pipeline_cache_hits = 0;
    uint64_t m_pipeline_cache_misses = 0;
    uint64_t m_mps_gemm_cache_hits = 0;
    uint64_t m_mps_gemm_cache_misses = 0;
    uint64_t m_const_tensor_cache_hits = 0;
    uint64_t m_const_tensor_cache_misses = 0;
};

}  // namespace mpsrt
}  // namespace metal
}  // namespace gfx_plugin
}  // namespace ov
