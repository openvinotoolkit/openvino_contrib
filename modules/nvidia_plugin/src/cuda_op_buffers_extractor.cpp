// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_op_buffers_extractor.hpp"

#include <fmt/format.h>

#include <error.hpp>
#include <gsl/span_ext>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/tensor_iterator.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/op/util/assign_base.hpp>
#include <stdexcept>
#include <transformer/nodes/concat_optimized.hpp>
#include <utility>

namespace ov {
namespace nvidia_gpu {

OperationBuffersExtractor::OperationBuffersExtractor(gsl::span<const NodePtr> ordered_nodes,
                                                     bool is_stable_params,
                                                     bool is_stable_results)
    : is_stable_params_{is_stable_params},
      is_stable_results_{is_stable_results},
      num_ordered_nodes_{static_cast<unsigned long>(ordered_nodes.size())} {
    for (int node_idx = 0; node_idx < num_ordered_nodes_; node_idx++) {
        const auto& node = ordered_nodes[node_idx];
        if (IsParameterNode(*node))
            extractParameterTensors(node, node_idx);
        else if (IsResultNode(*node) || IsAssignNode(*node))
            extractResultTensors(node);
        else if (IsConstantNode(*node))
            extractImmutableTensors(node);
        else if (IsConcatOptimizedNode(*node))
            mergeConcatMutableTensors(node, node_idx);
        else if (isReshapeOnlyNode(*node))
            extractReshapeTensors(node, node_idx);
        else
            extractMutableTensors(node, node_idx);
    }
    for (int node_idx = 0; node_idx < num_ordered_nodes_; node_idx++) {
        for (const auto& input : ordered_nodes[node_idx]->inputs()) {
            try {
                const auto tensorName = GetTensorNameInternal(input);
                const BufferID bufferId = tensor_names_.at(tensorName)->GetBuffer().GetId();
                const bool isImmutableBuffer = immutable_buffers_.find(bufferId) != immutable_buffers_.end();
                if (!isImmutableBuffer) {
                    auto& mutableBuffer = mutable_buffers_.at(bufferId);
                    if (node_idx > mutableBuffer.lifespan_end) {
                        mutableBuffer.lifespan_end = node_idx;
                    }
                    if (mutableBuffer.size != 0 &&
                        mutableBuffer.size < GetTensorByteSize(input)) {
                        ThrowBufferSizesAreNotMatchError(input);
                    }
                }
            } catch (std::out_of_range& e) {
                ThrowGraphIsBadFormedError(input);
            }
        }
    }
}

std::vector<TensorID> OperationBuffersExtractor::inputTensorIds(const ov::Node& node) const {
    std::vector<TensorID> result{};
    for (const auto& input : node.inputs()) {
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
        result.push_back(*tensorId);
    }
    return result;
}

std::vector<TensorID> OperationBuffersExtractor::outputTensorIds(const ov::Node& node) const {
    if (IsResultNode(node) || IsAssignNode(node)) return {};
    std::vector<TensorID> result{};
    for (const auto& output : node.outputs()) {
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(output));
        result.push_back(*tensorId);
    }
    return result;
}

int OperationBuffersExtractor::mutableBufferLifespanStart(BufferID buffer_id) const {
    try {
        return mutable_buffers_.at(buffer_id).lifespan_start;
    } catch (std::out_of_range& e) {
        throw_ov_exception(fmt::format("Buffer id {} is out of range.", buffer_id));
    }
}

int OperationBuffersExtractor::mutableBufferLifespanEnd(BufferID buffer_id) const {
    try {
        return mutable_buffers_.at(buffer_id).lifespan_end;
    } catch (std::out_of_range& e) {
        throw_ov_exception(fmt::format("Buffer id {} is out of range.", buffer_id));
    }
}

std::size_t OperationBuffersExtractor::mutableBufferSize(BufferID buffer_id) const {
    try {
        return mutable_buffers_.at(buffer_id).size;
    } catch (std::out_of_range& e) {
        throw_ov_exception(fmt::format("Buffer id {} is out of range.", buffer_id));
    }
}

gsl::span<const OperationBuffersExtractor::Byte> OperationBuffersExtractor::immutableBuffer(BufferID buffer_id) const {
    try {
        return immutable_buffers_.at(buffer_id);
    } catch (std::out_of_range& e) {
        throw_ov_exception(fmt::format("Buffer id {} is out of range.", buffer_id));
    }
}

std::vector<BufferID> OperationBuffersExtractor::mutableBuffersIds() const {
    std::vector<BufferID> result{};
    for (const auto& pair : mutable_buffers_) {
        result.push_back(pair.first);
    }
    return result;
}

std::vector<BufferID> OperationBuffersExtractor::immutableBuffersIds() const {
    std::vector<BufferID> result{};
    for (const auto& pair : immutable_buffers_) {
        result.push_back(pair.first);
    }
    return result;
}

void OperationBuffersExtractor::mergeConcatMutableTensors(const NodePtr& node, int node_idx) {
    std::vector<std::pair<std::string, TensorID::Ptr>> mergedTensors;
    mergedTensors.reserve(node->inputs().size());
    for (const auto& input : node->inputs()) {
        const auto& tensorName = GetTensorNameInternal(input.get_source_output());
        const auto& tensorId = tensor_names_.at(tensorName);
        OPENVINO_ASSERT(&tensorId->GetBuffer() == tensorId.get());
        mergedTensors.emplace_back(tensorName, tensorId);
    }
    OPENVINO_ASSERT(!mergedTensors.empty());

    std::vector<BufferID> mergedBufferIds;
    std::transform(mergedTensors.begin(), mergedTensors.end(), std::back_inserter(mergedBufferIds), [](const auto& nt) {
        return nt.second->GetBuffer().GetId();
    });

    int minLifespanStart = mutable_buffers_.at(mergedBufferIds.front()).lifespan_start;
    for (const auto& bufferId : mergedBufferIds) {
        const int lifespanStart = mutable_buffers_.at(bufferId).lifespan_start;
        if (lifespanStart < minLifespanStart) {
            minLifespanStart = lifespanStart;
        }
    }

    const auto& output = node->output(0);
    auto mergedTensorByteSize = GetTensorByteSize(output);
    auto parentTensor = std::make_shared<TensorID>(next_buffer_id_);
    next_buffer_id_ += 1;
    tensor_names_.emplace(GetTensorNameInternal(output), parentTensor);

    mutable_buffers_.emplace(std::make_pair(parentTensor->GetBuffer().GetId(),
                                            BufferDesc{minLifespanStart, node_idx, mergedTensorByteSize}));
    for (const auto& bufferId : mergedBufferIds) {
        mutable_buffers_.erase(bufferId);
    }

    size_t totalSize = 0;
    for (const auto& t : mergedTensors) {
        auto& tensor = tensor_names_.at(t.first);
        tensor->SetParent(parentTensor, totalSize);
        totalSize += mutable_tensor_sizes_.at(tensor->GetId());
    }
    mutable_tensor_sizes_[parentTensor->GetId()] = totalSize;
    OPENVINO_ASSERT(mergedTensorByteSize == 0 || mergedTensorByteSize == totalSize);
}

void OperationBuffersExtractor::extractReshapeTensors(const NodePtr& node, int node_idx) {
    try {
        OPENVINO_ASSERT(node->inputs().size() >= 1);
        OPENVINO_ASSERT(node->outputs().size() == 1);
        const auto input = node->inputs().at(0);
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
        const auto output = node->outputs().at(0);
        tensor_names_.emplace(GetTensorNameInternal(output), tensorId);
    } catch (std::out_of_range&) {
        throw_ov_exception(fmt::format("Failed to extract output buffer for reshape only node '{}'", node->get_name()));
    }
}

void OperationBuffersExtractor::extractMutableTensors(const NodePtr& node, int node_idx) {
    for (const auto& output : node->outputs()) {
        auto tensorByteSize = GetTensorByteSize(output);
        mutable_tensor_sizes_[next_buffer_id_] = tensorByteSize;
        mutable_buffers_.emplace(std::make_pair(next_buffer_id_, BufferDesc{node_idx, node_idx, tensorByteSize}));
        tensor_names_.emplace(GetTensorNameInternal(output), std::make_shared<TensorID>(next_buffer_id_));
        next_buffer_id_++;
    }
}

void OperationBuffersExtractor::extractParameterTensors(const NodePtr& node, int node_idx) {
    if (node->inputs().size() > 0) {
        OPENVINO_ASSERT(node->get_output_size() > 0);
        auto input = node->inputs().front().get_source_output();
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
        for (auto& output : node->outputs()) {
            tensor_names_.emplace(GetTensorNameInternal(output), tensorId);
        }
    } else {
        const int lastNodeIdx = is_stable_params_ ? num_ordered_nodes_ : node_idx;
        for (const auto& output : node->outputs()) {
            auto tensorByteSize = GetTensorByteSize(output);
            mutable_tensor_sizes_[next_buffer_id_] = tensorByteSize;
            mutable_buffers_.emplace(
                std::make_pair(next_buffer_id_, BufferDesc{node_idx, lastNodeIdx, tensorByteSize}));
            tensor_names_.emplace(GetTensorNameInternal(output), std::make_shared<TensorID>(next_buffer_id_));
            next_buffer_id_++;
        }
    }
}

void OperationBuffersExtractor::extractResultTensors(const NodePtr& node) {
    if (node->get_output_size() > 0) {
        auto input = node->inputs().front().get_source_output();
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
        for (auto& output : node->outputs()) {
            tensor_names_.emplace(GetTensorNameInternal(output), tensorId);
        }
    }
    if (is_stable_results_) {
        auto input = node->inputs().front().get_source_output();
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
        auto resultBuffer = std::find_if(mutable_buffers_.begin(), mutable_buffers_.end(), [&tensorId](const auto& mb) {
            return mb.first == tensorId->GetId();
        });
        if (resultBuffer == mutable_buffers_.end()) {
            throw_ov_exception(fmt::format("Cannot find mutable buffer for Result with name {}", node->get_name()));
        }
        resultBuffer->second.lifespan_end = num_ordered_nodes_;
    }
}

void OperationBuffersExtractor::extractImmutableTensors(const NodePtr& node) {
    auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node);
    const Byte* ptr = reinterpret_cast<const Byte*>(constant->get_data_ptr());
    auto span = gsl::make_span(ptr, GetTensorByteSize(node->output(0)));
    auto tensor = std::make_shared<TensorID>(next_buffer_id_);
    immutable_buffers_.emplace(std::make_pair(tensor->GetId(), span));
    tensor_names_.emplace(GetTensorNameInternal(node->output(0)), tensor);
    next_buffer_id_++;
}

WorkbufferIds OperationBuffersExtractor::processWorkbufferRequest(int node_idx, const WorkbufferRequest& request) {
    WorkbufferIds result{};
    for (auto size : request.immutable_sizes) {
        immutable_workbuffers_.emplace(next_buffer_id_, size);
        result.immutableIds.push_back(next_buffer_id_);
        next_buffer_id_++;
    }
    for (auto size : request.mutable_sizes) {
        // mutable workbuffers share the same memory space with mutable I/O buffers
        mutable_buffers_.emplace(std::make_pair(next_buffer_id_, BufferDesc{node_idx, node_idx, size}));
        result.mutableIds.push_back(next_buffer_id_);
        next_buffer_id_++;
    }
    return result;
}

void OperationBuffersExtractor::initConstantMemory(DeviceMemBlock::Ptr memory_block) const {
    for (const auto& buffer_id : memory_block->bufferIds()) {
        auto span = immutableBuffer(buffer_id);
        void* device_ptr = memory_block->deviceBufferPtr(buffer_id);
        OPENVINO_ASSERT(device_ptr != nullptr);
        throwIfError(::cudaMemcpy(device_ptr, span.data(), span.size_bytes(), cudaMemcpyHostToDevice));
    }
}

MemoryModel::Ptr OperationBuffersExtractor::createConstantMemoryModel() const {
    ImmutableMemoryModelBuilder constants_block_builder;
    // Process nGraph and add allocations
    for (auto id : immutableBuffersIds()) {
        auto span = immutableBuffer(id);
        constants_block_builder.addAllocation(id, span.size());
    }
    return constants_block_builder.build();
}

MemoryModel::Ptr OperationBuffersExtractor::createMutableMemoryModel() const {
    MemoryModelBuilder mutable_model_builder;
    for (auto id : mutableBuffersIds()) {
        auto size = mutableBufferSize(id);
        if (size == 0) continue;  // Dynamic tensors — allocated at runtime
        mutable_model_builder.addAllocation(
            id, mutableBufferLifespanStart(id), mutableBufferLifespanEnd(id), size);
    }
    return mutable_model_builder.build();
}

MemoryModel::Ptr OperationBuffersExtractor::createImmutableMemoryModel() const {
    ImmutableMemoryModelBuilder immutable_workbuffer_model_builder;
    const auto& immutable_workbufer_sizes = immutableWorkbufferSizes();
    for (const auto& index : immutable_workbufer_sizes) {
        immutable_workbuffer_model_builder.addAllocation(index.first, index.second);
    }
    return immutable_workbuffer_model_builder.build();
}

bool OperationBuffersExtractor::IsParameterNode(const ov::Node& node) {
    return dynamic_cast<const ov::op::v0::Parameter*>(&node) != nullptr;
}

bool OperationBuffersExtractor::IsResultNode(const ov::Node& node) {
    return dynamic_cast<const ov::op::v0::Result*>(&node) != nullptr;
}

bool OperationBuffersExtractor::IsConstantNode(const ov::Node& node) {
    return dynamic_cast<const ov::op::v0::Constant*>(&node) != nullptr;
}

bool OperationBuffersExtractor::IsAssignNode(const ov::Node& node) {
    return dynamic_cast<const ov::op::util::AssignBase*>(&node) != nullptr;
}

bool OperationBuffersExtractor::IsConcatOptimizedNode(const ov::Node& node) {
    return dynamic_cast<const nodes::ConcatOptimized*>(&node) != nullptr;
}

bool OperationBuffersExtractor::isReshapeOnlyNode(const ov::Node& node) {
    return ov::is_type<const ov::op::v1::Reshape>(&node) || ov::is_type<const ov::op::v0::Squeeze>(&node) ||
           ov::is_type<const ov::op::v0::Unsqueeze>(&node);
}

void OperationBuffersExtractor::ThrowBufferSizesAreNotMatchError(const ov::Input<ov::Node>& input) {
    throw_ov_exception(
        fmt::format("Buffer size of Input #{} of {} node and corresponding "
                    "output #{} of {} node are not equal.",
                    input.get_index(),
                    input.get_node()->get_name(),
                    input.get_source_output().get_index(),
                    input.get_source_output().get_node()->get_name()));
}

void OperationBuffersExtractor::ThrowGraphIsBadFormedError(const ov::Input<ov::Node>& input) {
    throw_ov_exception(
        fmt::format("Provided graph is bad formed. Input #{} of \"{}\" node "
                    "isn't connected to any output",
                    input.get_index(),
                    input.get_node()->get_name()));
}

}  // namespace nvidia_gpu
}  // namespace ov
