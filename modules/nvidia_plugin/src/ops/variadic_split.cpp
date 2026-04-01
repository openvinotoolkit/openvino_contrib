// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "variadic_split.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/split.hpp>
#include <openvino/op/variadic_split.hpp>
#include <utility>
#include <vector>

#include "converters.hpp"
#include "cuda/runtime.hpp"
#include "cuda_op_buffers_extractor.hpp"

namespace ov {
namespace nvidia_gpu {

namespace {

template <typename T>
std::vector<int64_t> getSplitLengths(const ov::op::v0::Constant* node) {
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

std::vector<int64_t> getSplitLengths(ov::op::v0::Constant* node) {
    switch (node->get_element_type()) {
        case ov::element::Type_t::i8:
            return getSplitLengths<int8_t>(node);
        case ov::element::Type_t::i16:
            return getSplitLengths<int16_t>(node);
        case ov::element::Type_t::i32:
            return getSplitLengths<int32_t>(node);
        case ov::element::Type_t::i64:
            return getSplitLengths<int64_t>(node);
        case ov::element::Type_t::u8:
            return getSplitLengths<uint8_t>(node);
        case ov::element::Type_t::u16:
            return getSplitLengths<uint16_t>(node);
        case ov::element::Type_t::u32:
            return getSplitLengths<uint32_t>(node);
        case ov::element::Type_t::u64:
            return getSplitLengths<uint64_t>(node);
        default: {
            throw_ov_exception(
                fmt::format("split_lengths element type = {} is not supported by VariadicSplit operation "
                            "!!",
                            static_cast<ov::element::Type_t>(node->get_element_type())));
        }
    }
}

}  // namespace

VariadicSplitOp::VariadicSplitOp(const CreationContext& context,
                                 const ov::Node& node,
                                 IndexCollection&& inputIds,
                                 IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    auto variadic_split_node = dynamic_cast<const ov::op::v1::VariadicSplit*>(&node);
    OPENVINO_ASSERT(variadic_split_node, "Node name: ", GetName());
    auto input_element_type = variadic_split_node->get_input_element_type(0);
    auto axis_node = dynamic_cast<ov::op::v0::Constant*>(variadic_split_node->get_input_node_ptr(1));
    auto split_lengths_node = dynamic_cast<ov::op::v0::Constant*>(variadic_split_node->get_input_node_ptr(2));
    OPENVINO_ASSERT(axis_node, "Node name: ", GetName());
    OPENVINO_ASSERT(split_lengths_node, "Node name: ", GetName());
    auto output_element_type = variadic_split_node->get_output_element_type(0);
    OPENVINO_ASSERT(variadic_split_node->get_input_size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(input_element_type == output_element_type, "Node name: ", GetName());
    switch (input_element_type) {
        case ov::element::Type_t::dynamic:
        case ov::element::Type_t::u1:
            throw_ov_exception(
                fmt::format("Input element type = {} is not supported by VariadicSplit operation "
                            "!!",
                            static_cast<ov::element::Type_t>(input_element_type)));
    }
    const auto element_type = input_element_type;

    const auto& data_shape = variadic_split_node->get_input_shape(0);
    auto axis = axis_node->cast_vector<int64_t>()[0];
    if (axis < 0) {
        axis += static_cast<int64_t>(variadic_split_node->get_input_partial_shape(0).rank().get_length());
    }
    OPENVINO_ASSERT(axis >= 0 && axis < data_shape.size(), "Node name: ", GetName());
    const size_t orig_axis_size = data_shape[axis];

    const std::vector<int64_t> split_lengths = getSplitLengths(split_lengths_node);

    buildAxisHelpers(split_lengths, orig_axis_size);
    buildSplitIndexHelper(split_lengths, orig_axis_size);

    const size_t axis_split_step_size =
        std::accumulate(data_shape.begin() + axis + 1, data_shape.end(), 1, std::multiplies<size_t>());
    OPENVINO_ASSERT(axis_split_step_size != 0, "Node name: ", GetName());
    const size_t num_split_elements =
        std::accumulate(data_shape.begin(), data_shape.end(), 1, std::multiplies<size_t>());
    const size_t num_all_chunks = num_split_elements / axis_split_step_size;
    OPENVINO_ASSERT(num_all_chunks != 0, "Node name: ", GetName());
    const unsigned max_block_size = context.device().props().maxThreadsPerBlock;
    const unsigned num_blocks = (num_split_elements % max_block_size == 0) ? (num_split_elements / max_block_size)
                                                                           : (num_split_elements / max_block_size + 1);
    const unsigned threads_per_block = (num_blocks == 1) ? num_split_elements : max_block_size;

    variadic_split_kernel_ = kernel::VariadicSplit{convertDataType<ov::nvidia_gpu::kernel::Type_t>(element_type),
                                                   num_all_chunks,
                                                   axis_split_step_size,
                                                   orig_axis_size,
                                                   num_blocks,
                                                   threads_per_block};
}

void VariadicSplitOp::buildAxisHelpers(const std::vector<int64_t>& split_lengths, const size_t orig_axis_size) {
    const auto num_of_remain_parts = std::count(split_lengths.begin(), split_lengths.end(), -1);
    OPENVINO_ASSERT(num_of_remain_parts <= 1, "Node name: ", GetName());
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
    OPENVINO_ASSERT(buffers.size() == 3, "Node name: ", GetName());
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
    OPENVINO_ASSERT(variadic_split_kernel_, "Node name: ", GetName());
    OPENVINO_ASSERT(inputs.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == axis_sizes_.size(), "Node name: ", GetName());
    OPENVINO_ASSERT(buffers.mutable_buffers.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(buffers.immutable_buffers.size() == 3, "Node name: ", GetName());
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

CudaGraphCompatibility VariadicSplitOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::NONE; }

OPERATION_REGISTER(VariadicSplitOp, VariadicSplit);
}  // namespace nvidia_gpu
}  // namespace ov
