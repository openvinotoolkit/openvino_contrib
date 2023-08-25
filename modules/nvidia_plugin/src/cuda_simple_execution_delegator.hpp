// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ops/parameter.hpp>
#include <ops/result.hpp>
#include <ops/tensor_iterator.hpp>

#include "cuda_iexecution_delegator.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * Creates profiler sequence and stores profiler results.
 */
class SimpleExecutionDelegator : public IExecutionDelegator {
public:
    /**
     * Constructor of Profiler class
     * @param perfCount Option that indicates if performance counters are enabled
     */
    SimpleExecutionDelegator() = default;

    /**
     * Start time measurement of stage
     */
    void set_stream(const CUDA::Stream& stream) override{};

    /**
     * Start time measurement of stage
     */
    void start_stage() override {}

    /**
     * Stop time measurement of stage
     * @param stage Stage for which time measurement was performed
     */
    virtual void stop_stage(PerfStages stage) override{};

    virtual void execute_sequence(const SubGraph* subGraphPtr,
                          const MemoryManager& memoryManager,
                          const Workbuffers::mutable_buffer& buffer,
                          const InferenceRequestContext& context) override {
        for (auto& op : subGraphPtr->getExecSequence()) {
            const auto& inputTensors = memoryManager.inputTensorPointers(*op, buffer);
            const auto& outputTensors = memoryManager.outputTensorPointers(*op, buffer);
            const auto& workBuffers = memoryManager.workBuffers(*op, buffer);
            op->Execute(context, inputTensors, outputTensors, workBuffers);
        }
    };

    virtual void capture_sequence(const SubGraph* subGraphPtr,
                          const MemoryManager& memoryManager,
                          const Workbuffers::mutable_buffer& buffer,
                          InferenceRequestContext& context) override {
        for (auto& op : subGraphPtr->getExecSequence()) {
            const auto& inputTensors = memoryManager.inputTensorPointers(*op, buffer);
            const auto& outputTensors = memoryManager.outputTensorPointers(*op, buffer);
            const auto& workBuffers = memoryManager.workBuffers(*op, buffer);
            op->Capture(context, inputTensors, outputTensors, workBuffers);
        }
    };

    /**
     * Returns performance counters
     * @return Performance counters
     */
    // [[nodiscard]]
    virtual const std::vector<ov::ProfilingInfo> get_performance_counts() const override{};

    /**
     * Processes performance events into performance counters
     */
    virtual void process_events() override{};

    virtual void set_cuda_event_record_mode(CUDA::Event::RecordMode mode) override{};
};

}  // namespace nvidia_gpu
}  // namespace ov
