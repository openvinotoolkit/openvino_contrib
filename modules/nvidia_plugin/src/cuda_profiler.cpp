// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_profiler.hpp"

#include <ops/parameter.hpp>
#include <ops/result.hpp>

namespace ov {
namespace nvidia_gpu {

namespace {

std::string get_primitive_type(const IOperationMeta& op) {
    std::stringstream ss;
    ss << op.GetCategory() << "_";
    ss << op.GetRuntimePrecision();
    return ss.str();
}

std::pair<std::string, ov::ProfilingInfo> make_profile_info(const IOperationMeta& op) {
    ov::ProfilingInfo result{};
    result.exec_type = get_primitive_type(op);
    result.node_type = op.GetTypeName();
    result.node_name = op.GetName();
    return {result.node_name, result};
}

std::pair<std::string, ov::ProfilingInfo> make_profile_info(std::string stage_name,
                                                            std::chrono::microseconds real_time,
                                                            std::chrono::microseconds cpu_time = std::chrono::microseconds(0)) noexcept {
    return {stage_name, ov::ProfilingInfo{ov::ProfilingInfo::Status::NOT_RUN, real_time, cpu_time, stage_name}};
}
}  // namespace

Profiler::Profiler(const SubGraph& graph) {
    std::vector<OperationBase::Ptr> execSequence;
    collect_subgraphs(graph, execSequence);

    for (size_t i = 0; i < execSequence.size(); ++i) {
        auto& op = *execSequence[i];
        perf_counters_.emplace(make_profile_info(op));
        execution_order_.push_back(op.GetName());
    }
}

void Profiler::process_events() {
    if (infer_count_ == 0) return;
    auto ms_to_us = [](float timing) {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<float, std::milli>{timing});
    };
    auto time_per_infer_us = [&](float timing) {
        return ms_to_us(timing / infer_count_);
    };
    std::map<std::string, float> layer_timing{};
    for (auto& timing_map : subgraph_perf_steps_map_) {
        auto graph = static_cast<const ov::nvidia_gpu::SubGraph*>(timing_map.first);
        OPENVINO_ASSERT(graph, "Performance counter graph is empty");
        auto& timings = timing_map.second;
        for (auto& timing : timings) {
            timing.measure();
            const auto perf = perf_counters_.find(timing.get_op_name());
            if (perf != perf_counters_.cend()) {
                perf->second.real_time = time_per_infer_us(timing.duration());
                perf->second.status = ov::ProfilingInfo::Status::EXECUTED;
                if (perf->second.node_type[0]) {
                    layer_timing[perf->second.node_type] += timing.duration();
                }
                auto ops = graph->getModel()->get_ops();
                const auto& op = std::find_if(ops.begin(), ops.end(),
                    [&timing](std::shared_ptr<ov::Node>& node) { return node->get_friendly_name() == timing.get_op_name(); });
                if (op != ops.end()) {
                    auto& info = (*op)->get_rt_info();
                    const auto& it = info.find(ov::nvidia_gpu::PERF_COUNTER_NAME);
                    OPENVINO_ASSERT(it != info.end(), "Operation ", (*op)->get_friendly_name(), " doesn't contain performance counter");
                    auto info_perf_count = it->second.as<std::shared_ptr<ov::nvidia_gpu::PerfCounts>>();
                    info_perf_count->total_duration += ms_to_us(timing.duration());
                    info_perf_count->num += infer_count_;
                    auto pos = perf->second.exec_type.find('_');
                    if (pos != std::string::npos) {
                        info_perf_count->impl_type = perf->second.exec_type.substr(0, pos);
                        info_perf_count->runtime_precision = perf->second.exec_type.substr(pos + 1);
                    } else {
                        info_perf_count->impl_type = perf->second.exec_type;
                        info_perf_count->runtime_precision = "undefined";
                    }
                }
            }
        }
    }
    for (auto const& timing : layer_timing) {
        const auto summary = perf_counters_.find(timing.first);
        if (summary != perf_counters_.cend()) {
            summary->second.real_time = time_per_infer_us(timing.second);
            summary->second.status = ov::ProfilingInfo::Status::EXECUTED;
        }
    }

    // Sum of all Parameters divided by count of infer requests
    std::chrono::microseconds zero_time(0);
    auto param_timing = layer_timing.find("Parameter");
    auto parameter_ms = param_timing == layer_timing.cend() ? zero_time : time_per_infer_us(param_timing->second);

    // Sum of all Results divided by count of infer requests
    auto result_timing = layer_timing.find("Result");
    auto result_ms = result_timing == layer_timing.cend() ? zero_time : time_per_infer_us(result_timing->second);

    // Adding some overall performance counters
    auto stage_time_ms = [this](PerfStages stage) {
        const auto i = static_cast<std::size_t>(stage);
        return std::chrono::microseconds(static_cast<long long>(durations_[i].count()));
    };

    auto insert_stage = [&](const std::pair<std::string, ov::ProfilingInfo>& value) {
        auto const result = stage_counters_.insert(value);
        if (!result.second) { result.first->second = value.second; }
    };
    insert_stage(make_profile_info("1. input preprocessing", zero_time, stage_time_ms(PerfStages::Preprocess)));
    insert_stage(make_profile_info("2. input transfer to a device", parameter_ms));
    insert_stage(make_profile_info("3. execution time", time_per_infer_us(exec_timing_.measure()), stage_time_ms(PerfStages::StartPipeline)));
    insert_stage(make_profile_info("4. output transfer from a device", result_ms));
    insert_stage(make_profile_info("5. output postprocessing", zero_time, stage_time_ms(PerfStages::Postprocess)));
}

void Profiler::execute_sequence(const SubGraph* subGraphPtr,
                                const MemoryManager& memoryManager,
                                const Workbuffers::mutable_buffer& buffer,
                                const InferenceRequestContext& context) {
    for (const auto& op : create_exec_sequence(subGraphPtr)) {
        const auto& inTensors = memoryManager.inputTensorPointers(*op, buffer);
        const auto& outTensors = memoryManager.outputTensorPointers(*op, buffer);
        const auto& workBuffers = memoryManager.workBuffers(*op, buffer);
        op->execute(context, inTensors, outTensors, workBuffers);
    }
}

void Profiler::capture_sequence(const SubGraph* subGraphPtr,
                                const MemoryManager& memoryManager,
                                const Workbuffers::mutable_buffer& buffer,
                                InferenceRequestContext& context) {
    for (const auto& op : create_exec_sequence(subGraphPtr)) {
        const auto& inputTensors = memoryManager.inputTensorPointers(*op, buffer);
        const auto& outputTensors = memoryManager.outputTensorPointers(*op, buffer);
        const auto& workBuffers = memoryManager.workBuffers(*op, buffer);
        op->capture(context, inputTensors, outputTensors, workBuffers);
    }
}

void Profiler::execute_graph_sequence(const SubGraph* subGraphPtr,
                                      const MemoryManager& memoryManager,
                                      const Workbuffers::mutable_buffer& buffer,
                                      InferenceRequestContext& context) {
    for (const auto& op : create_exec_sequence(subGraphPtr)) {
        const auto& inTensors = memoryManager.inputTensorPointers(*op, buffer);
        const auto& outTensors = memoryManager.outputTensorPointers(*op, buffer);
        const auto& workBuffers = memoryManager.workBuffers(*op, buffer);
        op->execute_graph(context, inTensors, outTensors, workBuffers);
    }
}

Profiler::ProfilerSequence Profiler::create_exec_sequence(const SubGraph* subGraphPtr) {
    OPENVINO_ASSERT(active_stream_);
    ++infer_count_;
    auto foundPerfStepsIter = std::find_if(subgraph_perf_steps_map_.begin(),
                                           subgraph_perf_steps_map_.end(),
                                           [subGraphPtr](const auto& ps) { return ps.first == subGraphPtr; });
    OPENVINO_ASSERT(foundPerfStepsIter != subgraph_perf_steps_map_.end());
    return ProfilerSequence{*this,
                            static_cast<size_t>(std::distance(subgraph_perf_steps_map_.begin(), foundPerfStepsIter))};
}

void Profiler::collect_subgraphs(const SubGraph& graph, std::vector<OperationBase::Ptr>& allExecSequence) {
    std::vector<ProfileExecStep> perfSteps;
    const auto& execSequence = graph.getExecSequence();
    for (const auto& execStep : execSequence) {
        collect_node_visitor(execStep, perfSteps, allExecSequence);
    }
    subgraph_perf_steps_map_.emplace_back(&graph, std::move(perfSteps));
}

void Profiler::collect_node_visitor(const OperationBase::Ptr& execStep,
                                    std::vector<ProfileExecStep>& perfSteps,
                                    std::vector<OperationBase::Ptr>& allExecSequence) {
    allExecSequence.push_back(execStep);
    const auto& op = *execStep;
    perfSteps.emplace_back(*this, op);
    if (const auto tensorIteratorPtr = dynamic_cast<const TensorIteratorOp*>(&op)) {
        collect_subgraphs(*tensorIteratorPtr, allExecSequence);
    }
}

const std::vector<ov::ProfilingInfo> Profiler::get_performance_counts() const {
    std::vector<ov::ProfilingInfo> result;
    // First insert common stage information
    for (auto& stage_info : stage_counters_) {
        result.push_back(stage_info.second);
    }
    // Peformance counters in right order
    for (auto& name : execution_order_) {
        if (perf_counters_.count(name)) {
            result.push_back(perf_counters_.at(name));
        }
    }
    return result;
}
}  // namespace nvidia_gpu
}  // namespace ov
