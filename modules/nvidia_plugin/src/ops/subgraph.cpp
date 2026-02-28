// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph.hpp"

#include <fmt/format.h>

#include <cuda_graph_topology_runner.hpp>
#include <cuda_op_buffers_extractor.hpp>
#include <cuda_operation_registry.hpp>
#include <cuda_iexecution_delegator.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/tensor_iterator.hpp>

#include <cuda_dynamic_operation.hpp>

#include "nop_op.hpp"
#include "parameter.hpp"
#include "result.hpp"

namespace ov {
namespace nvidia_gpu {

SubGraph::SubGraph(const CreationContext& context,
                   const SubGraphOp& op,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, op, std::move(inputIds), std::move(outputIds)),
      model_{op.get_function()},
      creation_context_{context} {
    const bool isStableParamsAndResultsNeeded = nullptr != dynamic_cast<const ov::op::v0::TensorIterator*>(&op);
    initExecuteSequence(isStableParamsAndResultsNeeded, isStableParamsAndResultsNeeded);
}

SubGraph::SubGraph(const CreationContext& context, const std::shared_ptr<const ov::Model>& model)
    : OperationBase(context, nullptr), model_{model}, creation_context_{context} {
    initExecuteSequence(false, false);
}

SubGraph::SubGraph(const CreationContext& context,
                   const std::shared_ptr<const ov::Model>& model,
                   const ExecSequence& sequence,
                   const std::shared_ptr<MemoryManager>& memoryManager)
    : OperationBase{context, nullptr},
      memory_manager_{memoryManager},
      exec_sequence_{sequence},
      model_{model},
      creation_context_{context} {}

const std::vector<OperationBase::Ptr>& SubGraph::getParams() const { return params_; }

const std::vector<OperationBase::Ptr>& SubGraph::getResults() const { return results_; }

bool SubGraph::isDynamicOp(const ov::Node& node) {
    if (node.is_dynamic()) {
        return true;
    }
    for (size_t i = 0; i < node.get_output_size(); ++i) {
        if (node.get_output_partial_shape(i).is_dynamic()) {
            return true;
        }
    }
    return false;
}

void SubGraph::initExecuteSequence(bool isStableParams, bool isStableResults) {
    static constexpr auto InitNeeded = IOperationExec::WorkbufferStatus::InitNeeded;

    if (!model_) {
        return;
    }
    const auto& orderedNodes = model_->get_ordered_ops();
    std::vector<Ptr> init_sequence{};
    OperationBuffersExtractor opBuffersExtractor{orderedNodes, isStableParams, isStableResults};
    const auto paramSize = model_->get_parameters().size();
    params_ = std::vector<OperationBase::Ptr>(paramSize);
    params_info_ = std::vector<OperationInfo>(paramSize);
    const auto resultSize = model_->get_results().size();
    results_ = std::vector<OperationBase::Ptr>(resultSize);
    results_info_ = std::vector<OperationInfo>(resultSize);
    std::unordered_map<int, int> node_to_exec_idx;
    for (unsigned node_idx = 0; node_idx < orderedNodes.size(); node_idx++) {
        const auto& node = orderedNodes[node_idx];
        if (!OperationRegistry::getInstance().hasOperation(node)) {
            throw_ov_exception(fmt::format("Node: name = {}, description = {}; Is not found in OperationRegistry",
                                         node->get_name(),
                                         node->description()));
        }
        auto inIds = opBuffersExtractor.inputTensorIds(*node);
        auto outIds = opBuffersExtractor.outputTensorIds(*node);

        const bool isDynamic = isDynamicOp(*node);
        const bool isParam = dynamic_cast<const ov::op::v0::Parameter*>(node.get()) != nullptr;
        const bool isResult = dynamic_cast<const ov::op::v0::Result*>(node.get()) != nullptr;

        OperationBase::Ptr operation;
        if (isDynamic) {
            operation = std::make_shared<DynamicOperation>(
                creation_context_, node, std::move(inIds), std::move(outIds));
        } else {
            operation = OperationRegistry::getInstance().createOperation(
                creation_context_, node, move(inIds), move(outIds));
        }
        if (dynamic_cast<NopOp*>(operation.get())) {
            continue;
        }
        operation->SetWorkbufferIds(
            opBuffersExtractor.processWorkbufferRequest(node_idx, operation->GetWorkBufferRequest()));
        if (InitNeeded == operation->SetWorkbufferIds(opBuffersExtractor.processWorkbufferRequest(
                              node_idx, operation->GetWorkBufferRequest()))) {
            init_sequence.push_back(operation);
        }
        if (isParam) {
            const auto paramIdx =
                model_->get_parameter_index(std::dynamic_pointer_cast<ov::op::v0::Parameter>(node));
            params_[paramIdx] = operation;
            if (!isDynamic) {
                params_info_[paramIdx].size_ = getTensorByteSize(*node);
                params_info_[paramIdx].shape_ = node->get_shape();
            }
            params_info_[paramIdx].type_ = node->get_element_type();
        } else if (isResult) {
            const auto resultIdx = model_->get_result_index(std::dynamic_pointer_cast<ov::op::v0::Result>(node));
            results_[resultIdx] = operation;
            if (!isDynamic) {
                results_info_[resultIdx].size_ = getTensorByteSize(*node);
                results_info_[resultIdx].shape_ = node->get_shape();
            }
            results_info_[resultIdx].type_ = node->get_element_type();
        }
        node_to_exec_idx[node_idx] = static_cast<int>(exec_sequence_.size());
        exec_sequence_.push_back(operation);
    }
    createDynamicReleaseSchedule(opBuffersExtractor, node_to_exec_idx);
    memory_manager_ = createMemoryManager(opBuffersExtractor);
    initSharedImmutableWorkbuffers(init_sequence);
}

void SubGraph::createDynamicReleaseSchedule(const OperationBuffersExtractor& opBuffersExtractor,
                                             const std::unordered_map<int, int>& node_to_exec_idx) {
    std::unordered_map<int, std::vector<BufferID>> release_schedule;
    for (auto id : opBuffersExtractor.mutableBuffersIds()) {
        if (opBuffersExtractor.mutableBufferSize(id) == 0) {
            int end_node_idx = opBuffersExtractor.mutableBufferLifespanEnd(id);
            auto it = node_to_exec_idx.find(end_node_idx);
            if (it != node_to_exec_idx.end()) {
                release_schedule[it->second].push_back(id);
            }
        }
    }
    for (auto& [exec_idx, ids] : release_schedule) {
        if (auto* dynOp = dynamic_cast<DynamicOperation*>(exec_sequence_[exec_idx].get())) {
            dynOp->setBuffersToRelease(std::move(ids));
        }
    }
}

std::unique_ptr<MemoryManager> SubGraph::createMemoryManager(const OperationBuffersExtractor& opBuffersExtractor) {
    // Build memory model for mutable memory block
    auto constants_model = opBuffersExtractor.createConstantMemoryModel();
    auto memory_model = opBuffersExtractor.createMutableMemoryModel();
    auto immutable_workbuffer_model = opBuffersExtractor.createImmutableMemoryModel();

    // Build shared constants memory block
    auto shared_constants_blob = std::make_shared<DeviceMemBlock>(constants_model);
    opBuffersExtractor.initConstantMemory(shared_constants_blob);

    auto immutable_workbuffers = std::make_shared<DeviceMemBlock>(immutable_workbuffer_model);
    // Later on, for each infer request
    return std::make_unique<MemoryManager>(shared_constants_blob, memory_model, immutable_workbuffers);
}

std::size_t SubGraph::GetCudaGraphsCount() const {
    if (!hasTopologyRunners()) {
        return graph_compatibility_ == CudaGraphCompatibility::NONE ? 0 : 1;
    }
    auto count = runner_ ? runner_->GetCudaGraphsCount() : static_cast<std::size_t>(0);
    for (const auto& op : exec_sequence_) {
        const auto sg = std::dynamic_pointer_cast<SubGraph>(op);
        if (sg) {
            count += sg->GetCudaGraphsCount();
        }
    }
    return count;
}

void SubGraph::initSharedImmutableWorkbuffers(const std::vector<OperationBase::Ptr>& init_sequence) {
    for (auto op : init_sequence) {
        op->InitSharedImmutableWorkbuffers(getSharedWorkbuffers(*op));
    }
}

std::vector<DevicePointer<void*>> SubGraph::getSharedWorkbuffers(const IOperationExec& operation) {
    const auto& ids = operation.GetWorkbufferIds();
    std::vector<DevicePointer<void*>> result;
    result.reserve(ids.immutableIds.size());
    for (const auto immutable_id : ids.immutableIds) {
        void* ptr = memory_manager_->immutableWorkbuffers().deviceBufferPtr(immutable_id);
        OPENVINO_ASSERT(ptr != nullptr, "Workbuffer not found. ID is " + std::to_string(immutable_id));
        result.emplace_back(ptr);
    }
    return result;
}

void SubGraph::Execute(const InferenceRequestContext& context, Inputs, Outputs, const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    const auto& memoryManager = *memory_manager_;
    auto& mutableBuffer = workbuffers.mutable_buffers.at(0);

    auto& executionDelegator = context.getExecutionDelegator();
    executionDelegator.set_stream(stream);
    executionDelegator.execute_sequence(this, memoryManager, mutableBuffer, context);
}

CudaGraphCompatibility SubGraph::GetCudaGraphCompatibility() const {
    if (!is_compatibility_analyzed_) {
        graph_compatibility_ = CudaGraphCompatibility::FULL;
        for (const auto& op : exec_sequence_) {
            auto opCompatability = op->GetCudaGraphCompatibility();
            if (opCompatability == CudaGraphCompatibility::SPECIAL) {
                graph_compatibility_ = opCompatability;
            } else if (opCompatability == CudaGraphCompatibility::NONE) {
                graph_compatibility_ = opCompatability;
                break;
            }
        }
        is_compatibility_analyzed_ = true;
    }
    return graph_compatibility_;
}

void SubGraph::Capture(InferenceRequestContext& context, Inputs, Outputs, const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    const auto& memoryManager = *memory_manager_;
    auto& mutableBuffer = workbuffers.mutable_buffers.at(0);

    auto& executionDelegator = context.getExecutionDelegator();
    executionDelegator.set_stream(stream);
    executionDelegator.capture_sequence(this, memoryManager, mutableBuffer, context);
}

void SubGraph::ExecuteGraph(InferenceRequestContext& context,
                            Inputs inputTensors,
                            Outputs outputTensors,
                            const Workbuffers& workbuffers) const {
    const auto& stream = context.getThreadContext().stream();
    const auto& memoryManager = *memory_manager_;
    auto& mutableBuffer = workbuffers.mutable_buffers.at(0);

    auto& executionDelegator = context.getExecutionDelegator();
    executionDelegator.set_stream(stream);
    executionDelegator.execute_graph_sequence(this, memoryManager, mutableBuffer, context);
}

void SubGraph::initializeRunner() {
    runner_ = std::make_shared<CudaGraphTopologyRunner>(creation_context_, model_, exec_sequence_, memory_manager_);
}

WorkbufferRequest SubGraph::GetWorkBufferRequest() const {
    const auto memoryBlockSize = memory_manager_->mutableTensorsMemoryModel()->deviceMemoryBlockSize();
    return {{}, {memoryBlockSize}};
}

}  // namespace nvidia_gpu
}  // namespace ov
