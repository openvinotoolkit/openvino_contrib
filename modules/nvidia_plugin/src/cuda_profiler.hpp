// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ops/tensor_iterator.hpp>
#include <utils/perf_timing.hpp>

#include "cuda_iexecution_delegator.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * Creates profiler sequence and stores profiler results.
 */
class Profiler : public IExecutionDelegator {
public:
    using PerformaceCounters = std::map<std::string, ov::ProfilingInfo>;
    using Duration = std::chrono::duration<float, std::micro>;
    using Time = std::chrono::steady_clock;

    class ProfilerSequence;
    class ProfileExecStep;

    /**
     * Constructor of Profiler class
     * @param perfCount Option that indicates if performance counters are enabled
     */
    explicit Profiler(const SubGraph& graph);

    /**
     * Start time measurement of stage
     */
    void set_stream(const CUDA::Stream& stream) override { active_stream_ = &stream; }

    /**
     * Start time measurement of stage
     */
    void start_stage() override { start_ = Time::now(); }

    /**
     * Stop time measurement of stage
     * @param stage Stage for which time measurement was performed
     */
    void stop_stage(PerfStages stage) override { durations_[static_cast<std::size_t>(stage)] = Time::now() - start_; }

    /**
     * Execute sequence from SubGraph/TensorIterator class
     * @param subGraphPtr Pointer to SubGraph
     * @param memoryManager Reference to MemoryManager
     * @param buffer Reference to orkbuffers::mutable_buffer
     * @param context Reference to InferenceRequestContext
     */
    void execute_sequence(const SubGraph* subGraphPtr,
                          const MemoryManager& memoryManager,
                          const Workbuffers::mutable_buffer& buffer,
                          const InferenceRequestContext& context) override;

    /**
     * Capture sequence from SubGraph/TensorIterator class
     * @param subGraphPtr Pointer to SubGraph
     * @param memoryManager Reference to MemoryManager
     * @param buffer Reference to orkbuffers::mutable_buffer
     * @param context Reference to InferenceRequestContext
     */
    void capture_sequence(const SubGraph* subGraphPtr,
                          const MemoryManager& memoryManager,
                          const Workbuffers::mutable_buffer& buffer,
                          InferenceRequestContext& context) override;

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
                                        InferenceRequestContext& context) override;

    /**
     * Returns performance counters
     * @return Performance counters
     */
    [[nodiscard]] const std::vector<ov::ProfilingInfo> get_performance_counts() const override;

    /**
     * Processes performance events into performance counters
     */
    void process_events() override;

    /**
     * Set CUDA event record mode
     * @param mode Value of CUDA::Event::RecordMode to set
     */
    void set_cuda_event_record_mode(CUDA::Event::RecordMode mode) override { cuda_event_record_mode_ = mode; }

private:
    /**
     * Creates profiler sequence and increase infer request counter
     * @return ProfilerSequence for single InferRequest
     */
    Profiler::ProfilerSequence create_exec_sequence(const SubGraph* subGraphPtr);

    void collect_subgraphs(const SubGraph& graph, std::vector<OperationBase::Ptr>& vector);
    void collect_node_visitor(const OperationBase::Ptr& execStep,
                              std::vector<ProfileExecStep>& perfSteps,
                              std::vector<OperationBase::Ptr>& allExecSequence);

    const CUDA::Stream* active_stream_ = nullptr;
    std::vector<std::pair<const void*, std::vector<ProfileExecStep>>> subgraph_perf_steps_map_;
    PerformaceCounters perf_counters_{};
    PerformaceCounters stage_counters_{};
    std::vector<std::string> execution_order_{};
    utils::PerformaceTiming exec_timing_{};
    // for performance counters
    std::array<Duration, static_cast<std::size_t>(PerfStages::NumOfStages)> durations_;
    Time::time_point start_{};
    size_t infer_count_{};
    CUDA::Event::RecordMode cuda_event_record_mode_{CUDA::Event::RecordMode::Default};
};

class Profiler::ProfileExecStep {
public:
    /**
     * Constructor for profiler execution step
     * @param profiler Profiler class
     * @param execStep Executable step
     */
    ProfileExecStep(Profiler& profiler, const OperationBase& execStep) : profiler_{profiler}, exec_step_{execStep} {}

    /**
     * Execute method wrapper that wrap each element with time measurement
     * @tparam TArgs Additional arguments for Execute method of operation
     * @param args Additional arguments for Execute method of operation
     */
    template <typename... TArgs>
    void execute(TArgs&&... args) const {
        timing_.setStart(*this->profiler_.active_stream_, profiler_.cuda_event_record_mode_);
        exec_step_.Execute(std::forward<TArgs>(args)...);
        timing_.setStop(*this->profiler_.active_stream_, profiler_.cuda_event_record_mode_);
    }

    template <typename... TArgs>
    void capture(TArgs&&... args) const {
        timing_.setStart(*this->profiler_.active_stream_, profiler_.cuda_event_record_mode_);
        exec_step_.Capture(std::forward<TArgs>(args)...);
        timing_.setStop(*this->profiler_.active_stream_, profiler_.cuda_event_record_mode_);
    }

    template <typename... TArgs>
    void execute_graph(TArgs&&... args) const {
        timing_.setStart(*this->profiler_.active_stream_, profiler_.cuda_event_record_mode_);
        exec_step_.ExecuteGraph(std::forward<TArgs>(args)...);
        timing_.setStop(*this->profiler_.active_stream_, profiler_.cuda_event_record_mode_);
    }

    /**
     * Adapter method for pointer of operation
     * @return Reference to ProfileExecStep
     */
    const ProfileExecStep& operator*() const { return *this; }

    /**
     * Adapter method for pointer of operation
     * @return Pointer to ProfileExecStep
     */
    const ProfileExecStep* operator->() const { return this; }

    /**
     * Implicitly casts ProfileExecStep to OperationBase
     * @return
     */
    operator const OperationBase&() const { return static_cast<const OperationBase&>(exec_step_); }

    /**
     * measure time for this execution step
     * @return Time for this step
     */
    float measure() { return timing_.measure(); }

    /**
     * Get time for this execution step
     * @return Time for this step
     */
    [[nodiscard]] float duration() const noexcept { return timing_.duration(); }

    /**
     * Get name of the operation
     * @return Name of the operation
     */
    [[nodiscard]] const std::string& get_op_name() { return exec_step_.GetName(); }

private:
    Profiler& profiler_;
    const OperationBase& exec_step_;
    mutable utils::PerformaceTiming timing_;
};

class Profiler::ProfilerSequence {
public:
    /**
     * Constructor for profiler sequence
     * @param profiler Profiler class
     * @param stream CUDA stream
     */
    ProfilerSequence(Profiler& profiler, size_t index) : profiler_{profiler}, index_{index} {
        profiler_.exec_timing_.setStart(*profiler_.active_stream_, profiler.cuda_event_record_mode_);
    }

    /**
     * Destructor
     * Stops time measurement
     */
    ~ProfilerSequence() {
        profiler_.exec_timing_.setStop(*profiler_.active_stream_, profiler_.cuda_event_record_mode_);
    }

    /**
     * begin method for iterable class
     * @return Iterator
     */
    auto begin() const { return profiler_.subgraph_perf_steps_map_[index_].second.begin(); }

    /**
     * end method for iterable class
     * @return Iterator
     */
    auto end() const { return profiler_.subgraph_perf_steps_map_[index_].second.end(); }

    /**
     * begin method for iterable class
     * @return Constant iterator
     */
    auto cbegin() { return profiler_.subgraph_perf_steps_map_[index_].second.cbegin(); }

    /**
     * end method for iterable class
     * @return Constant iterator
     */
    auto cend() { return profiler_.subgraph_perf_steps_map_[index_].second.cend(); }

private:
    Profiler& profiler_;
    const size_t index_;
};

}  // namespace nvidia_gpu
}  // namespace ov
