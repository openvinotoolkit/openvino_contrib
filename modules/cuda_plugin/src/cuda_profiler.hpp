// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <utils/perf_timing.hpp>
#include <vector>

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
     * @param execSequence Execution sequence to profile
     */
    explicit Profiler(bool perfCount, const std::vector<OperationBase::Ptr>& execSequence);

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
     * Creates profiler sequence
     * @param stream CUDA stream for on which operation should be executed
     * @return ProfilerSequence
     */
    Profiler::ProfilerSequence CreateExecSequence(const CUDA::Stream& stream);

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
    const CUDA::Stream* active_stream_ = nullptr;
    bool perf_count_;
    std::vector<ProfileExecStep> perf_steps_{};
    PerformaceCounters perf_counters_{};
    utils::PerformaceTiming exec_timing_{};
    // for performance counters
    std::array<Duration, NumOfStages> durations_;
    Time::time_point start_;
    size_t infer_count_{};
};

class Profiler::ProfileExecStep {
public:
    /**
     * Constructor for profiler execution step
     * @param profiler Profiler class
     * @param execStep Executable step
     */
    ProfileExecStep(Profiler& profiler, OperationBase& execStep) : profiler_{profiler}, exec_step_{execStep} {}

    /**
     * Execute method wrapper that wrap each element with time measurement
     * @tparam TArgs Additional arguments for Execute method of operation
     * @param args Additional arguments for Execute method of operation
     */
    template <typename... TArgs>
    void Execute(TArgs&&... args) {
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
    ProfileExecStep& operator*() { return *this; }

    /**
     * Adapter method for pointer of operation
     * @return Pointer to ProfileExecStep
     */
    ProfileExecStep* operator->() { return this; }

    /**
     * Implicitly casts ProfileExecStep to OperationBase
     * @return
     */
    operator OperationBase&() const { return static_cast<OperationBase&>(exec_step_); }

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
    OperationBase& exec_step_;
    utils::PerformaceTiming timing_;
};

class Profiler::ProfilerSequence {
public:
    /**
     * Constructor for profiler sequence
     * @param profiler Profiler class
     * @param stream CUDA stream
     */
    ProfilerSequence(Profiler& profiler, const CUDA::Stream& stream) : profiler_{profiler}, stream_{stream} {
        profiler_.exec_timing_.setStart(stream_);
    }

    /**
     * Destructor
     * Stops time measurement
     */
    ~ProfilerSequence() { profiler_.exec_timing_.setStop(stream_); }

    /**
     * begin method for iterable class
     * @return Iterator
     */
    auto begin() { return profiler_.perf_steps_.begin(); }

    /**
     * end method for iterable class
     * @return Iterator
     */
    auto end() { return profiler_.perf_steps_.end(); }

    /**
     * begin method for iterable class
     * @return Constant iterator
     */
    auto cbegin() { return profiler_.perf_steps_.cbegin(); }

    /**
     * end method for iterable class
     * @return Constant iterator
     */
    auto cend() { return profiler_.perf_steps_.cend(); }

private:
    Profiler& profiler_;
    const CUDA::Stream& stream_;
};

}  // namespace CUDAPlugin
