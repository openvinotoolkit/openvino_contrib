// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/event.hpp>
#include <cuda/runtime.hpp>
#include <cuda_perf_counts.hpp>

#include "openvino/runtime/profiling_info.hpp"
#include "ops/subgraph.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * Interface for Profiler class or other Delegators
 */
class IExecutionDelegator {
public:
    /**
     * Virtual destructor for the interface
     */
    virtual ~IExecutionDelegator() = default;

    /**
     * Start time measurement of stage
     */
    virtual void set_stream(const CUDA::Stream& stream) = 0;

    /**
     * Start time measurement of stage
     */
    virtual void start_stage() = 0;

    /**
     * Stop time measurement of stage
     * @param stage Stage for which time measurement was performed
     */
    virtual void stop_stage(PerfStages stage) = 0;

    /**
     * Execute sequence from SubGraph/TensorIterator class
     * @param subGraphPtr Pointer to SubGraph
     * @param memoryManager Reference to MemoryManager
     * @param buffer Reference to orkbuffers::mutable_buffer
     * @param context Reference to InferenceRequestContext
     */
    virtual void execute_sequence(const SubGraph* subGraphPtr,
                                  const MemoryManager& memoryManager,
                                  const Workbuffers::mutable_buffer& buffer,
                                  const InferenceRequestContext& context) = 0;

    /**
     * Capture sequence from SubGraph/TensorIterator class
     * @param subGraphPtr Pointer to SubGraph
     * @param memoryManager Reference to MemoryManager
     * @param buffer Reference to orkbuffers::mutable_buffer
     * @param context Reference to InferenceRequestContext
     */
    virtual void capture_sequence(const SubGraph* subGraphPtr,
                                  const MemoryManager& memoryManager,
                                  const Workbuffers::mutable_buffer& buffer,
                                  InferenceRequestContext& context) = 0;

    /**
     * Execute CUDA graph sequence from SubGraph class
     * @param subGraphPtr Pointer to SubGraph
     * @param memoryManager Reference to MemoryManager
     * @param buffer Reference to orkbuffers::mutable_buffer
     * @param context Reference to InferenceRequestContext
     */
    virtual void execute_graph_sequence(const SubGraph* subGraphPtr,
                                        const MemoryManager& memoryManager,
                                        const Workbuffers::mutable_buffer& buffer,
                                        InferenceRequestContext& context) = 0;

    /**
     * Returns performance counters
     * @return Performance counters
     */
    virtual const std::vector<ov::ProfilingInfo> get_performance_counts() const = 0;

    /**
     * Processes performance events into performance counters
     */
    virtual void process_events() = 0;

    /**
     * Set CUDA event record mode
     * @param mode Value of CUDA::Event::RecordMode to set
     */
    virtual void set_cuda_event_record_mode(CUDA::Event::RecordMode mode) = 0;
};

}  // namespace nvidia_gpu
}  // namespace ov
