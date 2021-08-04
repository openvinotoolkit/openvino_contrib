// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_op_buffers_extractor.hpp"

#include <details/ie_exception.hpp>
#include <gsl/span_ext>
#include <stdexcept>
#include <transformer/nodes/concat_optimized.hpp>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/op/unsqueeze.hpp"

namespace CUDAPlugin {

OperationBuffersExtractor::OperationBuffersExtractor(gsl::span<const NodePtr> ordered_nodes) {
    const auto num_ordered_nodes = ordered_nodes.size();
    for (int node_idx = 0; node_idx < num_ordered_nodes; node_idx++) {
        const auto& node = ordered_nodes[node_idx];
        if (IsResultNode(*node))
            continue;
        else if (IsConstantNode(*node))
            extractImmutableTensors(node);
        else if (IsConcatOptimizedNode(*node))
            mergeConcatMutableTensors(node, node_idx);
        else
            extractMutableTensors(node, node_idx);
    }
    for (int node_idx = 0; node_idx < num_ordered_nodes; node_idx++) {
        for (const auto& input : ordered_nodes[node_idx]->inputs()) {
            try {
                const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
                if (!IsConstantNode(*input.get_source_output().get_node_shared_ptr())) {
                    mutable_buffers_.at(tensorId.buffer_id).lifespan_end = node_idx;
                    if (mutable_buffers_.at(tensorId.buffer_id).size <
                        GetTensorByteSize(input))
                        ThrowBufferSizesAreNotMatchError(input);
                }
            } catch (std::out_of_range& e) {
                ThrowGraphIsBadFormedError(input);
            }
        }
    }
}

std::vector<TensorID>
OperationBuffersExtractor::inputTensorIds(const ngraph::Node& node) const {
    std::vector<TensorID> result {};
    for (const auto& input : node.inputs()) {
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
        result.push_back(tensorId);
    }
    return result;
}

std::vector<TensorID>
OperationBuffersExtractor::outputTensorIds(const ngraph::Node& node) const {
    if (IsResultNode(node))
        return { };
    std::vector<TensorID> result {};
    for (const auto& output : node.outputs()) {
        const auto& tensorId = tensor_names_.at(GetTensorNameInternal(output));
        result.push_back(tensorId);
    }
    return result;
}

int OperationBuffersExtractor::mutableBufferLifespanStart(BufferID buffer_id) const {
    try {
        return mutable_buffers_.at(buffer_id).lifespan_start;
    } catch (std::out_of_range& e) {
        THROW_IE_EXCEPTION << "Buffer id " << buffer_id
                           << " is out of range.";
    }
}

int OperationBuffersExtractor::mutableBufferLifespanEnd(BufferID buffer_id) const {
    try {
        return mutable_buffers_.at(buffer_id).lifespan_end;
    } catch (std::out_of_range& e) {
        THROW_IE_EXCEPTION << "Buffer id " << buffer_id
                           << " is out of range.";
    }
}

std::size_t
OperationBuffersExtractor::mutableBufferSize(BufferID buffer_id) const {
    try {
        return mutable_buffers_.at(buffer_id).size;
    } catch (std::out_of_range& e) {
        THROW_IE_EXCEPTION << "Buffer id " << buffer_id
                           << " is out of range.";
    }
}

gsl::span<const OperationBuffersExtractor::Byte>
OperationBuffersExtractor::immutableBuffer(BufferID buffer_id) const {
    try {
        return immutable_buffers_.at(buffer_id);
    } catch (std::out_of_range& e) {
        THROW_IE_EXCEPTION << "Buffer id " << buffer_id
                           << " is out of range.";
    }
}

std::vector<BufferID>
OperationBuffersExtractor::mutableBuffersIds() const {
    std::vector<BufferID> result { };
    for (const auto& pair : mutable_buffers_) {
        result.push_back(pair.first);
    }
    return result;
}

std::vector<BufferID> OperationBuffersExtractor::immutableBuffersIds() const {
    std::vector<BufferID> result { };
    for (const auto& pair : immutable_buffers_) {
        result.push_back(pair.first);
    }
    return result;
}

std::size_t OperationBuffersExtractor::GetTensorByteSize(const ngraph::Output<ngraph::Node>& input) {
    return input.get_element_type().size() * shape_size(input.get_shape());
}

std::size_t OperationBuffersExtractor::GetTensorByteSize(const ngraph::Input<ngraph::Node>& input) {
    return input.get_element_type().size() * shape_size(input.get_shape());
}

void OperationBuffersExtractor::mergeConcatMutableTensors(const NodePtr& node, int node_idx) {
    std::vector<BufferID> mergedTensorIds;
    std::vector<std::pair<std::string, TensorID>> mergedTensors;
    mergedTensors.reserve(node->inputs().size());
    for (const auto& input : node->inputs()) {
        auto output = input.get_source_output();
        const auto& tensorName = GetTensorNameInternal(output);
        const auto& tensorId = tensor_names_.at(tensorName);
        mergedTensors.emplace_back(tensorName, tensorId);
        mergedTensorIds.push_back(tensorId.buffer_id);
    }

    std::vector<std::pair<BufferID, BufferDesc>> filteredBufferDescs;
    std::copy_if(mutable_buffers_.begin(), mutable_buffers_.end(), std::back_inserter(filteredBufferDescs),
                 [&mergedTensorIds](const auto& t){
                   return mergedTensorIds.end() != std::find(mergedTensorIds.begin(), mergedTensorIds.end(), t.first);
                 });

    const auto minBufferId = *std::min_element(mergedTensorIds.begin(), mergedTensorIds.end());
    unsigned totalSize = 0;
    for (const auto& tn : mergedTensors) {
        auto& tensorId = tensor_names_[tn.first];
        const auto origId = tensorId.buffer_id;
        tensorId.buffer_id = minBufferId;
        tensorId.offset += totalSize;
        auto foundBufferDesc = std::find_if(filteredBufferDescs.begin(), filteredBufferDescs.end(),
                                        [origId](const auto& t) {
                                          return t.first == origId;
                                        });
        totalSize += foundBufferDesc->second.size;
    }

    mutable_buffers_.at(minBufferId).size = totalSize;
    for (const auto& i : mergedTensorIds) {
        if (i == minBufferId) {
            continue;
        }
        mutable_buffers_.erase(i);
    }

    const auto& output = node->output(0);
    mutable_buffers_.emplace(std::make_pair(minBufferId,
                                            BufferDesc { node_idx, node_idx, totalSize }));
    tensor_names_.emplace(GetTensorNameInternal(output), minBufferId);
}

void OperationBuffersExtractor::extractMutableTensors(const NodePtr& node, int node_idx) {
    if (isReshapeOnlyNode(*node)) {
        try {
            Expects(node->inputs().size() >= 1);
            Expects(node->outputs().size() == 1);
            const auto input = node->inputs().at(0);
            const auto& tensorId = tensor_names_.at(GetTensorNameInternal(input));
            const auto output = node->outputs().at(0);
            tensor_names_.emplace(GetTensorNameInternal(output), tensorId);
        } catch (std::out_of_range&) {
            THROW_IE_EXCEPTION << "Failed to extract output buffer for reshape only node '"
                               << node->get_name() << "'";
        }
    } else {
        for (const auto& output : node->outputs()) {
            mutable_buffers_.emplace(std::make_pair(
                next_buffer_id_, BufferDesc { node_idx, node_idx, GetTensorByteSize(output) }));
            tensor_names_.emplace(GetTensorNameInternal(output), next_buffer_id_);
            next_buffer_id_++;
        }
    }
}

void OperationBuffersExtractor::extractImmutableTensors(const NodePtr& node) {
    auto constant = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(node);
    const Byte * ptr = reinterpret_cast<const Byte*>(constant->get_data_ptr());
    auto span = gsl::make_span(ptr, GetTensorByteSize(node->output(0)));
    immutable_buffers_.emplace(std::make_pair(next_buffer_id_, span));
    tensor_names_.emplace(GetTensorNameInternal(node->output(0)), next_buffer_id_);
    next_buffer_id_++;
}

WorkbufferIds OperationBuffersExtractor::processWorkbufferRequest(int node_idx, const WorkbufferRequest& request) {
    WorkbufferIds result {};
    for(auto size : request.immutable_sizes) {
        immutable_workbuffers_.emplace(std::make_pair(next_buffer_id_, size));
        result.immutableIds.push_back(next_buffer_id_);
        next_buffer_id_++;
    }
    for(auto size : request.mutable_sizes) {
        // mutable workbuffers share the same memory space with mutable I/O buffers
        mutable_buffers_.emplace(std::make_pair(
            next_buffer_id_, BufferDesc { node_idx, node_idx, size }));
        result.mutableIds.push_back(next_buffer_id_);
        next_buffer_id_++;
    }
    return result;
}

bool OperationBuffersExtractor::IsResultNode(const ngraph::Node& node) {
    return dynamic_cast<const ngraph::op::v0::Result*>(&node) != nullptr;
}

bool OperationBuffersExtractor::IsConstantNode(const ngraph::Node& node) {
    return dynamic_cast<const ngraph::op::v0::Constant*>(&node) != nullptr;
}

bool OperationBuffersExtractor::IsConcatOptimizedNode(const ngraph::Node& node) {
    return dynamic_cast<const nodes::ConcatOptimized*>(&node) != nullptr;
}

bool OperationBuffersExtractor::isReshapeOnlyNode(const ngraph::Node& node) {
    return ngraph::is_type<const ngraph::op::v1::Reshape>(&node)
        || ngraph::is_type<const ngraph::op::v0::Squeeze>(&node)
        || ngraph::is_type<const ngraph::op::v0::Unsqueeze>(&node);
}

void OperationBuffersExtractor::ThrowBufferSizesAreNotMatchError(const ngraph::Input<ngraph::Node>& input) {
    THROW_IE_EXCEPTION << "Buffer size of Input #"
                       << std::to_string(input.get_index()) << " of "
                       << input.get_node()->get_name()
                       << " node and corresponding output #"
                       << std::to_string(input.get_source_output().get_index()) << " of "
                       << input.get_source_output().get_node()->get_name()
                       << " node are not equal.";
}

void OperationBuffersExtractor::ThrowGraphIsBadFormedError(const ngraph::Input<ngraph::Node>& input) {
    THROW_IE_EXCEPTION << "Provided graph is bad formed. Input #"
                       << std::to_string(input.get_index()) << " of \""
                       << input.get_node()->get_name()
                       << "\" node isn't connected to any output";
}

} // namespace CUDAPlugin
