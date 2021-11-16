// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <map>
#include <ops/tensor_iterator.hpp>
#include <utils/perf_timing.hpp>
#include <vector>

#include "cuda_graph.hpp"
#include "cuda_operation_base.hpp"

namespace CUDAPlugin {

/**
 * Creates profiler sequence and stores profiler results.
 */
class Profiler {
public:
    enum Stages { Preprocess, Postprocess, StartPipeline, WaitPipeline, NumOfStages };

    using PerformaceCounters = std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>;
    using Duration = std::chrono::duration<float, std::micro>;
    using Time = std::chrono::steady_clock;

    class ProfilerSequence;
    class ProfileExecStep;

    /**
     * Constructor of Profiler class
     * @param perfCount Option that indicates if performance counters are enabled
     */
    explicit Profiler(bool perfCount, const SubGraph& graph);

    /**
     * Start time measurement of stage
     */
    void SetStream(const CUDA::Stream& stream) { active_stream_ = &stream; }

    /**
     * Start time measurement of stage
     */
    void StartStage() { start_ = Time::now(); }

    /**
     * Stop time measurement of stage
     * @param stage Stage for which time measurement was performed
     */
    void StopStage(Stages stage) { durations_[stage] = Time::now() - start_; }

    /**
     * Creates profiler sequence and increase infer request counter
     * @return ProfilerSequence for single InferRequest
     */
    Profiler::ProfilerSequence CreateExecSequence(const SubGraph* subGraphPtr);

    /**
     * Returns performance counters
     * @return Performance counters
     */
    [[nodiscard]] const PerformaceCounters& GetPerformanceCounts() const { return perf_counters_; }

    /**
     * Processes performance events into performance counters
     */
    void ProcessEvents();

private:
    void CollectSubGraphs(const SubGraph& graph, std::vector<OperationBase::Ptr>& vector);
    void CollectSubGraphs(const TensorIteratorOp& graph, std::vector<OperationBase::Ptr>& allExecSequence);
    void CollectNodeVisitor(const OperationBase::Ptr& execStep,
                            std::vector<ProfileExecStep>& perfSteps,
                            std::vector<OperationBase::Ptr>& allExecSequence);

    const CUDA::Stream* active_stream_ = nullptr;
    const bool perf_count_;
    std::vector<std::pair<const void*, std::vector<ProfileExecStep>>> subgraph_perf_steps_map_;
    PerformaceCounters perf_counters_{};
    utils::PerformaceTiming exec_timing_{};
    // for performance counters
    std::array<Duration, NumOfStages> durations_;
    Time::time_point start_{};
    size_t infer_count_{};
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
    void Execute(TArgs&&... args) const {
        if (this->profiler_.perf_count_) {
            timing_.setStart(*this->profiler_.active_stream_);
            exec_step_.Execute(std::forward<TArgs>(args)...);
            timing_.setStop(*this->profiler_.active_stream_);
        } else {
            exec_step_.Execute(std::forward<TArgs>(args)...);
        }
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
     * Measure time for this execution step
     * @return Time for this step
     */
    float Measure() { return timing_.measure(); }

    /**
     * Get time for this execution step
     * @return Time for this step
     */
    [[nodiscard]] float Duration() const noexcept { return timing_.duration(); }

    /**
     * Get name of the operation
     * @return Name of the operation
     */
    [[nodiscard]] const std::string& GetOpName() { return exec_step_.GetName(); }

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
        if (profiler_.perf_count_) {
            profiler_.exec_timing_.setStart(*profiler_.active_stream_);
        }
    }

    /**
     * Destructor
     * Stops time measurement
     */
    ~ProfilerSequence() {
        if (profiler_.perf_count_) {
            profiler_.exec_timing_.setStop(*profiler_.active_stream_);
        }
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

}  // namespace CUDAPlugin
