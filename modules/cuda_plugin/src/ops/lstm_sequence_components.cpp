// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_sequence_components.hpp"

#include <error.hpp>
#include <gsl/gsl_assert>
#include <ngraph/op/constant.hpp>
#include <typeinfo>

#include "constant_factory.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/op/unsqueeze.hpp"

namespace CUDAPlugin::RNN::Details {

namespace {

bool isReshapeOnlyNode(const ngraph::Node* node) {
    return ngraph::is_type<const ngraph::op::v1::Reshape>(node) ||
           ngraph::is_type<const ngraph::op::v0::Squeeze>(node) ||
           ngraph::is_type<const ngraph::op::v0::Unsqueeze>(node);
}

gsl::span<const uint8_t> findInputConstantBuffer(const ngraph::Node& inNode, int inputIdx) {
    const ngraph::Node* node = inNode.get_input_node_ptr(inputIdx);
    const ngraph::op::v0::Constant* constant = dynamic_cast<const ngraph::op::v0::Constant*>(node);
    while (!constant) {
        Expects(isReshapeOnlyNode(node));
        node = node->get_input_node_ptr(0);
        constant = dynamic_cast<const ngraph::op::v0::Constant*>(node);
    }
    Expects(constant);
    const size_t size_bytes =
        ngraph::shape_size(constant->get_output_shape(0)) * constant->get_output_element_type(0).size();
    return {constant->get_data_ptr<const uint8_t>(), size_bytes};
}

}  // namespace

LSTMSequenceParams::LSTMSequenceParams(const ngraph::op::v5::LSTMSequence& node)
    : element_type_{node.get_input_element_type(LSTMSequenceArgIndices::x)},
      direction_{node.get_direction()},
      activations_{node.get_activations()},
      activations_alpha_{node.get_activations_alpha()},
      activations_beta_{node.get_activations_beta()},
      clip_{node.get_clip()},
      hidden_size_{node.get_hidden_size()} {
    Expects(node.get_input_size() == 7);
    Expects(node.get_output_size() == 3);

    const auto& x_shape = node.get_input_shape(LSTMSequenceArgIndices::x);
    Expects(x_shape.size() == 3);
    batch_size_ = x_shape[0];
    max_seq_length_ = x_shape[1];
    input_size_ = x_shape[2];

    w_host_buffers_ = findInputConstantBuffer(node, LSTMSequenceArgIndices::weights);
    r_host_buffers_ = findInputConstantBuffer(node, LSTMSequenceArgIndices::recurrence_weights);
    b_host_buffers_ = findInputConstantBuffer(node, LSTMSequenceArgIndices::biases);

    validate(node);
}

LSTMSequenceParams::LSTMSequenceParams(const CUDAPlugin::nodes::LSTMSequenceOptimized& node)
    : element_type_{node.get_input_element_type(LSTMSequenceArgIndices::x)},
      direction_{node.get_direction()},
      activations_{node.get_activations()},
      activations_alpha_{node.get_activations_alpha()},
      activations_beta_{node.get_activations_beta()},
      clip_{node.get_clip()},
      hidden_size_{node.get_hidden_size()} {
    Expects(node.get_input_size() == 7);
    Expects(node.get_output_size() == 3);

    const auto& x_shape = node.get_input_shape(LSTMSequenceArgIndices::x);
    Expects(x_shape.size() == 3);
    using LSTMSequenceOptimized = CUDAPlugin::nodes::LSTMSequenceOptimized;
    switch (node.get_major_format()) {
        case LSTMSequenceOptimized::BatchMajor:
            batch_size_ = x_shape[0];
            max_seq_length_ = x_shape[1];
            input_size_ = x_shape[2];
            break;
        case LSTMSequenceOptimized::SequenceMajor:
            max_seq_length_ = x_shape[0];
            batch_size_ = x_shape[1];
            input_size_ = x_shape[2];
            break;
        default:
            Expects(false);
    }

    w_host_buffers_ = findInputConstantBuffer(node, LSTMSequenceArgIndices::weights);
    r_host_buffers_ = findInputConstantBuffer(node, LSTMSequenceArgIndices::recurrence_weights);
    b_host_buffers_ = findInputConstantBuffer(node, LSTMSequenceArgIndices::biases);

    validate(node);
}

void LSTMSequenceParams::validate(const ngraph::op::v5::LSTMSequence& node) {
    const auto& sl_shape = node.get_input_shape(LSTMSequenceArgIndices::sequence_lengths);
    Expects(sl_shape.size() == 1);
    Expects(sl_shape[0] == batch_size_);

    Expects(node.get_input_element_type(LSTMSequenceArgIndices::x) == element_type_ &&
            node.get_input_element_type(LSTMSequenceArgIndices::hidden_input) == element_type_ &&
            node.get_input_element_type(LSTMSequenceArgIndices::cell_input) == element_type_ &&
            node.get_input_element_type(LSTMSequenceArgIndices::weights) == element_type_ &&
            node.get_input_element_type(LSTMSequenceArgIndices::recurrence_weights) == element_type_ &&
            node.get_input_element_type(LSTMSequenceArgIndices::biases) == element_type_ &&
            node.get_output_element_type(LSTMSequenceArgIndices::y) == element_type_ &&
            node.get_output_element_type(LSTMSequenceArgIndices::hidden_output) == element_type_ &&
            node.get_output_element_type(LSTMSequenceArgIndices::cell_output) == element_type_);

    const size_t num_directions = (direction_ == direction::BIDIRECTIONAL) ? 2 : 1;

    const auto& w_shape = node.get_input_shape(LSTMSequenceArgIndices::weights);
    Expects(w_shape.size() == 3);
    Expects(w_shape[0] == num_directions);
    Expects(w_shape[1] == lin_layer_count * hidden_size_);
    Expects(w_shape[2] == input_size_);

    const auto& r_shape = node.get_input_shape(LSTMSequenceArgIndices::recurrence_weights);
    Expects(r_shape.size() == 3);
    Expects(r_shape[0] == num_directions);
    Expects(r_shape[1] == lin_layer_count * hidden_size_);
    Expects(r_shape[2] == hidden_size_);

    const auto& b_shape = node.get_input_shape(LSTMSequenceArgIndices::biases);
    Expects(b_shape.size() == 2);
    Expects(b_shape[0] == num_directions);
    Expects(b_shape[1] == lin_layer_count * hidden_size_);

    const auto element_type_size = element_type_.size();
    Expects(w_host_buffers_.size_bytes() == ngraph::shape_size(w_shape) * element_type_size);
    Expects(r_host_buffers_.size_bytes() == ngraph::shape_size(r_shape) * element_type_size);
    Expects(b_host_buffers_.size_bytes() == ngraph::shape_size(b_shape) * element_type_size);
}

TransposeTensorAdapterBase::TransposeTensorAdapterBase(cudaDataType_t element_type,
                                                       size_t element_size,
                                                       const std::vector<int64_t>& src_shape,
                                                       const std::vector<int64_t>& dst_shape,
                                                       const std::vector<int>& mode)
    : element_type_{element_type},
      element_size_{element_size},
      src_shape_{src_shape},
      dst_shape_{dst_shape},
      src_mode_(mode.size()),
      dst_mode_{mode} {
    std::iota(src_mode_.begin(), src_mode_.end(), 0);
    const auto num_elements = ngraph::shape_size(src_shape_);
    Expects(num_elements > 0);
    Expects(num_elements == ngraph::shape_size(dst_shape_));
    Expects(src_shape_.size() == dst_shape_.size());
    Expects(src_shape_.size() == src_mode_.size());
    Expects(src_mode_.size() == dst_mode_.size());
}

void TransposeTensorAdapterBase::requestWorkbuffer(std::vector<size_t>& workbuffers_sizes) {
    workbuffer_.addRequest(workbuffers_sizes, ngraph::shape_size(src_shape_) * element_size_);
}

void* TransposeTensorAdapterBase::dnnApiPtr(const std::vector<Workbuffers::mutable_buffer>& mutable_buffers) const {
    return workbuffer_.requiredPtr(mutable_buffers);
}

void TransposeTensorAdapterBase::execute(const InferenceRequestContext& context, const void* src, void* dst) const {
    cutensorTensorDescriptor_t src_desc{}, dst_desc{};
    initCuTensorDescriptor(context.getThreadContext().cuTensorHandle(), src_shape_, src_desc);
    initCuTensorDescriptor(context.getThreadContext().cuTensorHandle(), dst_shape_, dst_desc);
    throwIfError(::cutensorPermutation(&context.getThreadContext().cuTensorHandle().get(),
                                       &NumericConst<constants::one>(element_type_),
                                       src,
                                       &src_desc,
                                       src_mode_.data(),
                                       dst,
                                       &dst_desc,
                                       dst_mode_.data(),
                                       element_type_,
                                       context.getThreadContext().stream().get()));
}

void TransposeTensorAdapterBase::initCuTensorDescriptor(const CUDA::CuTensorHandle& handle,
                                                        const std::vector<int64_t>& shape,
                                                        cutensorTensorDescriptor_t& desc) const {
    std::vector<int64_t> strides;
    strides.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); i++) strides.push_back(ngraph::row_major_stride(shape, i));
    throwIfError(::cutensorInitTensorDescriptor(
        &handle.get(), &desc, shape.size(), shape.data(), strides.data(), element_type_, CUTENSOR_OP_IDENTITY));
}

void TransposeInputTensorAdapter::execute(const InferenceRequestContext& context,
                                          CUDA::DevicePointer<const void*> input,
                                          const std::vector<Workbuffers::mutable_buffer>& dst) const {
    TransposeTensorAdapterBase::execute(context, input.get(), workbuffer_.requiredPtr(dst));
}

void TransposeOutputTensorAdapter::execute(const InferenceRequestContext& context,
                                           const std::vector<Workbuffers::mutable_buffer>& src,
                                           CUDA::DevicePointer<void*> output) const {
    TransposeTensorAdapterBase::execute(context, workbuffer_.requiredPtr(src), output.get());
}

}  // namespace CUDAPlugin::RNN::Details
