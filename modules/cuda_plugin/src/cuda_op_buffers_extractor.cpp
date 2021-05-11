// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_op_buffers_extractor.hpp"
#include <stdexcept>
#include <details/ie_exception.hpp>
#include "ngraph/op/result.hpp"
#include "ngraph/op/constant.hpp"

#include <gsl/span_ext>

namespace CUDAPlugin {

OperationBuffersExtractor::OperationBuffersExtractor(gsl::span<const NodePtr> ordered_nodes) {
    const auto num_ordered_nodes = ordered_nodes.size();
    unsigned buffer_idx { 0 };
    for (int node_idx = 0; node_idx < num_ordered_nodes; node_idx++) {
        const auto& node = ordered_nodes[node_idx];
        if (IsResultNode(*node))
            continue;
        else if (IsConstantNode(*node))
            extractImmutableBuffers(node, buffer_idx);
        else
            extractMutableBuffers(node, node_idx, buffer_idx);
    }
    for (int node_idx = 0; node_idx < num_ordered_nodes; node_idx++) {
        for (const auto& input : ordered_nodes[node_idx]->inputs()) {
            try {
                const auto buffer_idx = buffer_names_.at(GetBufferNameInternal(input));
                if (!IsConstantNode(*input.get_source_output().get_node_shared_ptr())) {
                    mutable_buffers_.at(buffer_idx).lifespan_end = node_idx;
                    if (mutable_buffers_.at(buffer_idx).size != GetBufferSize(input))
                        ThrowBufferSizesAreNotMatchError(input);
                }
            } catch (std::out_of_range& e) {
                ThrowGraphIsBadFormedError(input);
            }
        }
    }
}

std::vector<unsigned> OperationBuffersExtractor::inputBufferIndices(const ngraph::Node& node) const {
    std::vector<unsigned> result { };
    for (const auto& input : node.inputs()) {
        const auto buffer_idx = buffer_names_.at(GetBufferNameInternal(input));
        result.push_back(buffer_idx);
    }
    return result;
}

std::vector<unsigned> OperationBuffersExtractor::outputBufferIndices(const ngraph::Node& node) const {
    if (IsResultNode(node))
        return { };
    std::vector<unsigned> result {};
    for (const auto& output : node.outputs()) {
        const auto buffer_idx = buffer_names_.at(GetBufferNameInternal(output));
        result.push_back(buffer_idx);
    }
    return result;
}

int OperationBuffersExtractor::mutableBufferLifespanStart(unsigned buffer_index) const {
    try {
        return mutable_buffers_.at(buffer_index).lifespan_start;
    } catch (std::out_of_range& e) {
        THROW_IE_EXCEPTION << "Buffer index " << buffer_index << " is out of range.";
    }
}

int OperationBuffersExtractor::mutableBufferLifespanEnd(unsigned buffer_index) const {
    try {
        return mutable_buffers_.at(buffer_index).lifespan_end;
    } catch (std::out_of_range& e) {
        THROW_IE_EXCEPTION << "Buffer index " << buffer_index << " is out of range.";
    }
}

std::size_t OperationBuffersExtractor::mutableBufferSize(unsigned buffer_index) const {
    try {
        return mutable_buffers_.at(buffer_index).size;
    } catch (std::out_of_range& e) {
        THROW_IE_EXCEPTION << "Buffer index " << buffer_index << " is out of range.";
    }
}

gsl::span<const OperationBuffersExtractor::Byte> OperationBuffersExtractor::immutableBuffer(unsigned buffer_index) const {
    try {
        return immutable_buffers_.at(buffer_index);
    } catch (std::out_of_range& e) {
        THROW_IE_EXCEPTION << "Buffer index " << buffer_index << " is out of range.";
    }
}

std::vector<unsigned> OperationBuffersExtractor::mutableBuffersIndices() const {
    std::vector<unsigned> result { };
    for (const auto pair : mutable_buffers_)
        result.push_back(pair.first);
    return result;
}

std::vector<unsigned> OperationBuffersExtractor::immutableBuffersIndices() const {
    std::vector<unsigned> result { };
    for (const auto pair : immutable_buffers_)
        result.push_back(pair.first);
    return result;
}

std::size_t OperationBuffersExtractor::GetBufferSize(const ngraph::Output<ngraph::Node>& output) {
    return output.get_element_type().size() * shape_size(output.get_shape());
}

std::size_t OperationBuffersExtractor::GetBufferSize(const ngraph::Input<ngraph::Node>& output) {
    return output.get_element_type().size() * shape_size(output.get_shape());
}

void OperationBuffersExtractor::extractMutableBuffers(const NodePtr& node, int node_idx, unsigned& buffer_idx) {
    for (const auto& output : node->outputs()) {
        mutable_buffers_.emplace(std::make_pair(buffer_idx, BufferDesc { node_idx,
                node_idx, GetBufferSize(output) }));
        buffer_names_.emplace(GetBufferNameInternal(output), buffer_idx);
        buffer_idx++;
    }
}

void OperationBuffersExtractor::extractImmutableBuffers(const NodePtr& node, unsigned& buffer_idx) {
    auto constant = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(node);
    const Byte * ptr = reinterpret_cast<const Byte*>(constant->get_data_ptr());
    auto span = gsl::make_span(ptr, GetBufferSize(node->output(0)));
    immutable_buffers_.emplace(std::make_pair(buffer_idx, span));
    buffer_names_.emplace(GetBufferNameInternal(node->output(0)), buffer_idx);
    buffer_idx++;
}

bool OperationBuffersExtractor::IsResultNode(const ngraph::Node& node) {
    return dynamic_cast<const ngraph::op::v0::Result*>(&node) != nullptr;
}

bool OperationBuffersExtractor::IsConstantNode(const ngraph::Node& node) {
    return dynamic_cast<const ngraph::op::v0::Constant*>(&node) != nullptr;
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
