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

struct MpsrtPreparedModel {
    std::vector<MpsrtPreparedMslDispatch> msl_dispatches;
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

    bool prepare_msl_dispatch(const MpsrtRuntimeStage& stage,
                              const std::string& msl_source,
                              MpsrtPreparedMslDispatch& out,
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

    id<MTLComputePipelineState> get_or_create_pipeline(const MpsrtRuntimeStage& stage,
                                                       const std::string& msl_source,
                                                       bool& cache_hit,
                                                       std::string* log);

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_command_queue = nil;
    id<MTLBinaryArchive> m_binary_archive = nil;
    std::vector<PipelineCacheEntry> m_pipeline_cache;
    uint64_t m_pipeline_cache_hits = 0;
    uint64_t m_pipeline_cache_misses = 0;
};

}  // namespace mpsrt
}  // namespace metal
}  // namespace gfx_plugin
}  // namespace ov
