// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_profiler.hpp"

namespace CUDAPlugin {

namespace {

InferenceEngine::InferenceEngineProfileInfo makeProfileInfo(const IOperationMeta& op, unsigned execution_index) {
    InferenceEngine::InferenceEngineProfileInfo result{};
    op.GetCategory().copy(result.exec_type, sizeof(result.exec_type) - 1);
    op.GetTypeName().copy(result.layer_type, sizeof(result.layer_type) - 1);
    result.execution_index = execution_index;
    return result;
}

InferenceEngine::InferenceEngineProfileInfo makeProfileInfo(const std::string& layer,
                                                            const std::string_view& exec_type) {
    InferenceEngine::InferenceEngineProfileInfo result{};
    exec_type.copy(result.exec_type, sizeof(result.exec_type) - 1);
    layer.copy(result.layer_type, sizeof(result.layer_type) - 1);
    result.execution_index = 0;
    return result;
}

constexpr InferenceEngine::InferenceEngineProfileInfo makeProfileInfo(long long realTime_uSec,
                                                                      long long cpuTime_uSec = 0) noexcept {
    return InferenceEngine::InferenceEngineProfileInfo{
        InferenceEngine::InferenceEngineProfileInfo::EXECUTED, realTime_uSec, cpuTime_uSec};
}

}  // namespace

Profiler::Profiler(bool perfCount, const std::vector<OperationBase::Ptr>& execSequence) : perf_count_{perfCount} {
    perf_steps_.reserve(execSequence.size());
    for (int i = 0; i < execSequence.size(); ++i) {
        perf_steps_.emplace_back(*this, *execSequence[i]);
    }

    if (perfCount) {
        for (int i = 0; i < execSequence.size(); ++i) {
            auto& op = *execSequence[i];
            const auto& name = op.GetName();
            const auto& type = op.GetTypeName();
            auto perf = perf_counters_.find(name);
            if (perf == perf_counters_.cend()) perf_counters_.emplace(name, makeProfileInfo(op, i));
            perf = perf_counters_.find(type);
            if (perf == perf_counters_.cend()) {
                perf_counters_.emplace(type, makeProfileInfo(op.GetTypeName(), op.GetCategory()));
            } else {
                // Layers of the same type may have different exec_type, in sych case we clear exec_type
                if (perf->second.exec_type[0] && op.GetCategory().compare(perf->second.exec_type) != 0)
                    perf->second.exec_type[0] = 0;
            }
        }
    }
}

void Profiler::ProcessEvents() {
    if (infer_count_ == 0) return;
    constexpr float ms2us = 1000.0;
    std::map<std::string, float> layer_timing{};
    for (auto& timing : perf_steps_) {
        timing.Measure();
        static const auto perf = perf_counters_.find(timing.GetOpName());
        if (perf != perf_counters_.cend()) {
            perf->second.realTime_uSec = timing.Duration() * ms2us / infer_count_;
            perf->second.status = InferenceEngine::InferenceEngineProfileInfo::EXECUTED;
            if (perf->second.layer_type[0]) {
                layer_timing[perf->second.layer_type] += timing.Duration();
            }
        }
    }
    for (auto const& timing : layer_timing) {
        const auto summary = perf_counters_.find(timing.first);
        if (summary != perf_counters_.cend()) {
            summary->second.realTime_uSec = timing.second * ms2us / infer_count_;
            summary->second.status = InferenceEngine::InferenceEngineProfileInfo::EXECUTED;
        }
    }

    auto param_timing = layer_timing.find("Parameter");
    auto result_timing = layer_timing.find("Result");
    // Adding some overall performance counters
    perf_counters_["1. input preprocessing"] = makeProfileInfo(0, durations_[Preprocess].count());
    perf_counters_["2. input transfer to a device"] = makeProfileInfo(
        // Sum of all Parameters divided by count of infer requests
        param_timing == layer_timing.cend() ? 0 : param_timing->second * ms2us / infer_count_);
    perf_counters_["3. execution time"] =
        makeProfileInfo(exec_timing_.measure() * ms2us / infer_count_, durations_[StartPipeline].count());
    perf_counters_["4. output transfer from a device"] = makeProfileInfo(
        // Sum of all Results divided by count of infer requests
        result_timing == layer_timing.cend() ? 0 : result_timing->second * ms2us / infer_count_);
    perf_counters_["5. output postprocessing"] = makeProfileInfo(0, durations_[Postprocess].count());
}

Profiler::ProfilerSequence Profiler::CreateExecSequence(const CUDA::Stream& stream) {
    ++infer_count_;
    active_stream_ = &stream;
    return ProfilerSequence{*this, *active_stream_};
}

}  // namespace CUDAPlugin
