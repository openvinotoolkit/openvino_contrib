// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "variadic_split.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/split.hpp>
#include <utility>
#include <vector>

#include "converters.hpp"
#include "cuda/runtime.hpp"
#include "cuda_op_buffers_extractor.hpp"
#include "ngraph/op/variadic_split.hpp"

namespace CUDAPlugin {

namespace {

template <typename T>
std::vector<int64_t> getSplitLengths(const ngraph::op::v0::Constant* node) {
    const auto split_lengths_size = OperationBuffersExtractor::GetTensorByteSize(node->output(0)) / sizeof(T);
    const std::vector<T> split_lengths{node->get_data_ptr<T>(), node->get_data_ptr<T>() + split_lengths_size};
    std::vector<int64_t> converted_split_lengths;
    converted_split_lengths.reserve(split_lengths.size());
    std::transform(
        split_lengths.begin(), split_lengths.end(), std::back_inserter(converted_split_lengths), [](const auto& s) {
            return static_cast<int64_t>(s);
        });
    return converted_split_lengths;
}

std::vector<int64_t> getSplitLengths(ngraph::op::v0::Constant* node) {
    switch (node->get_element_type()) {
        case ngraph::element::Type_t::i8:
            return getSplitLengths<int8_t>(node);
        case ngraph::element::Type_t::i16:
            return getSplitLengths<int16_t>(node);
        case ngraph::element::Type_t::i32:
            return getSplitLengths<int32_t>(node);
        case ngraph::element::Type_t::i64:
            return getSplitLengths<int64_t>(node);
        case ngraph::element::Type_t::u8:
            return getSplitLengths<uint8_t>(node);
        case ngraph::element::Type_t::u16:
            return getSplitLengths<uint16_t>(node);
        case ngraph::element::Type_t::u32:
            return getSplitLengths<uint32_t>(node);
        case ngraph::element::Type_t::u64:
            return getSplitLengths<uint64_t>(node);
        default: {
            throwIEException(
                fmt::format("split_lengths element type = {} is not supported by VariadicSplit operation "
                            "!!",
                            static_cast<ngraph::element::Type_t>(node->get_element_type())));
        }
    }
}

}  // namespace

VariadicSplitOp::VariadicSplitOp(const CreationContext& context,
                                 const ngraph::Node& node,
                                 IndexCollection&& inputIds,
                                 IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    auto variadic_split_node = dynamic_cast<const ngraph::op::v1::VariadicSplit*>(&node);
    Expects(variadic_split_node);
    auto input_element_type = variadic_split_node->get_input_element_type(0);
    auto axis_node = dynamic_cast<ngraph::op::v0::Constant*>(variadic_split_node->get_input_node_ptr(1));
    auto split_lengths_node = dynamic_cast<ngraph::op::v0::Constant*>(variadic_split_node->get_input_node_ptr(2));
    Expects(axis_node);
    Expects(split_lengths_node);
    auto output_element_type = variadic_split_node->get_output_element_type(0);
    Expects(variadic_split_node->get_input_size() == 3);
    Expects(input_element_type == output_element_type);
    switch (input_element_type) {
        case ngraph::element::Type_t::undefined:
        case ngraph::element::Type_t::dynamic:
        case ngraph::element::Type_t::u1:
            throwIEException(
                fmt::format("Input element type = {} is not supported by VariadicSplit operation "
                            "!!",
                            static_cast<ngraph::element::Type_t>(input_element_type)));
    }
    const auto element_type = input_element_type;

    const auto& data_shape = variadic_split_node->get_input_shape(0);
    const int64_t axis = *axis_node->get_data_ptr<int64_t>();
    Expects(axis >= 0 && axis < data_shape.size());
    const size_t orig_axis_size = data_shape[axis];

    const std::vector<int64_t> split_lengths = getSplitLengths(split_lengths_node);

    buildAxisHelpers(split_lengths, orig_axis_size);
    buildSplitIndexHelper(split_lengths, orig_axis_size);

    const size_t axis_split_step_size =
        std::accumulate(data_shape.begin() + axis + 1, data_shape.end(), 1, std::multiplies<size_t>());
    Ensures(axis_split_step_size != 0);
    const size_t num_split_elements =
        std::accumulate(data_shape.begin(), data_shape.end(), 1, std::multiplies<size_t>());
    const size_t num_all_chunks = num_split_elements / axis_split_step_size;
    Ensures(num_all_chunks != 0);
    const unsigned max_block_size = context.device().props().maxThreadsPerBlock;
    const unsigned num_blocks = (num_split_elements % max_block_size == 0) ? (num_split_elements / max_block_size)
                                                                           : (num_split_elements / max_block_size + 1);
    const unsigned threads_per_block = (num_blocks == 1) ? num_split_elements : max_block_size;

    variadic_split_kernel_ = kernel::VariadicSplit{convertDataType<CUDAPlugin::kernel::Type_t>(element_type),
                                                   num_all_chunks,
                                                   axis_split_step_size,
                                                   orig_axis_size,
                                                   num_blocks,
                                                   threads_per_block};
}

void VariadicSplitOp::buildAxisHelpers(const std::vector<int64_t>& split_lengths, const size_t orig_axis_size) {
    const auto num_of_remain_parts = std::count(split_lengths.begin(), split_lengths.end(), -1);
    Expects(num_of_remain_parts <= 1);
    const size_t total_split_size =
        std::accumulate(split_lengths.begin(), split_lengths.end(), 0) + num_of_remain_parts;

    size_t prev_total_split_offset_size = 0;
    for (const auto& split_size : split_lengths) {
        size_t updated_split_size = 0;
        if (split_size == -1) {
            updated_split_size = orig_axis_size - total_split_size;
        } else {
            updated_split_size = split_size;
        }
        axis_sizes_.push_back(updated_split_size);
        axis_offset_sizes_.push_back(prev_total_split_offset_size);
        prev_total_split_offset_size += updated_split_size;
    }
}

void VariadicSplitOp::buildSplitIndexHelper(const std::vector<int64_t>& split_lengths, const size_t orig_axis_size) {
    int64_t split_idx = 0;
    int64_t prev_total_split_size = split_lengths[0];
    split_idx_.reserve(orig_axis_size);
    for (int i = 0; i < orig_axis_size; ++i) {
        if (i >= prev_total_split_size) {
            prev_total_split_size += split_lengths[++split_idx];
        }
        split_idx_.push_back(split_idx);
    }
}

WorkbufferRequest VariadicSplitOp::GetWorkBufferRequest() const {
    std::vector<size_t> immutable_buffer_sizes(kNumberOfIWIdx);
    immutable_buffer_sizes.at(kSplitIdxIWBIdx) = sizeof(*split_idx_.data()) * split_idx_.size();
    immutable_buffer_sizes.at(kAxisSizesIWBIdx) = sizeof(*axis_sizes_.data()) * axis_sizes_.size();
    immutable_buffer_sizes.at(kAxisOffsetSizesIWBIdx) = sizeof(*axis_offset_sizes_.data()) * axis_offset_sizes_.size();
    return {immutable_buffer_sizes, {sizeof(void*) * axis_sizes_.size()}};
}

void VariadicSplitOp::InitSharedImmutableWorkbuffers(const IOperationExec::Buffers& buffers) {
    Expects(buffers.size() == 3);
    CUDA::DefaultStream::stream().upload(
        buffers.at(kSplitIdxIWBIdx), split_idx_.data(), sizeof(*split_idx_.data()) * split_idx_.size());
    CUDA::DefaultStream::stream().upload(
        buffers.at(kAxisSizesIWBIdx), axis_sizes_.data(), sizeof(*axis_sizes_.data()) * axis_sizes_.size());
    CUDA::DefaultStream::stream().upload(buffers.at(kAxisOffsetSizesIWBIdx),
                                         axis_offset_sizes_.data(),
                                         sizeof(*axis_offset_sizes_.data()) * axis_offset_sizes_.size());
}

void VariadicSplitOp::Execute(const InferenceRequestContext& context,
                              Inputs inputs,
                              Outputs outputs,
                              const Workbuffers& buffers) const {
    Expects(variadic_split_kernel_);
    Expects(inputs.size() == 3);
    Expects(outputs.size() == axis_sizes_.size());
    Expects(buffers.mutable_buffers.size() == 1);
    Expects(buffers.immutable_buffers.size() == 3);
    auto& stream = context.getThreadContext().stream();
    auto output_ptrs = buffers.mutable_buffers.at(kOutputPtrsMWBIdx);
    auto all_split_idxs = buffers.immutable_buffers.at(kSplitIdxIWBIdx);
    auto all_num_splits = buffers.immutable_buffers.at(kAxisSizesIWBIdx);
    auto axis_offset_sizes = buffers.immutable_buffers.at(kAxisOffsetSizesIWBIdx);
    stream.upload(output_ptrs, outputs.data(), sizeof(void*) * axis_sizes_.size());
    auto in = inputs[0];
    (*variadic_split_kernel_)(stream.get(),
                              static_cast<const void*>(in.get()),
                              static_cast<void**>(output_ptrs.get()),
                              static_cast<const void*>(all_split_idxs.get()),
                              static_cast<const void*>(all_num_splits.get()),
                              static_cast<const void*>(axis_offset_sizes.get()));
}

OPERATION_REGISTER(VariadicSplitOp, VariadicSplit);
}  // namespace CUDAPlugin
