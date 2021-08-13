// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather.hpp"

#include <cuda_operation_registry.hpp>
#include "details/cuda_ngraph_import.hpp"

#include <fmt/format.h>

namespace CUDAPlugin {

template<typename IndexType>
static __global__ void
gather(unsigned data_length_bytes, size_t index_range_, unsigned els_per_thread,
       const uint8_t* src_dataDict, const IndexType* src_index, uint8_t* dst_data) {
    const auto indices_size_ = gridDim.y;
    const auto i = blockIdx.y;
    const auto j = blockIdx.x;
    const auto k = threadIdx.x * els_per_thread;
    const auto idx = src_index[i];
    if (idx >= index_range_) {
    // TODO: find a way to handle an error raised in a kernel (assertion or trap) properly
        __trap();
    }
    unsigned thread_offset;
    for (int l = 0; l < els_per_thread; ++l) {
    thread_offset = k + l;
    if (thread_offset >= data_length_bytes) {
        return;
    }
    dst_data[data_length_bytes * (i + j * indices_size_) + thread_offset]
        = src_dataDict[data_length_bytes * (idx + j * index_range_) + thread_offset];
    }
}

GatherOp::GatherOp(const CUDA::CreationContext& context,
                   const ngraph::Node& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
        : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    Expects(dynamic_cast<const ngraph::op::v1::Gather*>(&node) != nullptr);
    Expects(node.get_input_size() == 3);
    Expects(node.get_output_size() == 1);
    const auto element_type = node.get_input_element_type(0);
    switch (element_type) {
        case ngraph::element::Type_t::undefined:
        case ngraph::element::Type_t::dynamic:
        case ngraph::element::Type_t::u1:
            THROW_IE_EXCEPTION << fmt::format(
                                "Params element type = {} is not supported by Gather operation!",
                                static_cast<ngraph::element::Type_t>(element_type));
    }
    Expects(node.get_output_element_type(0) == element_type);

    element_size_ = element_type.size();

    const auto& params_shape = node.get_input_shape(0);
    const auto& params_shape_size = params_shape.size();
    const auto& indices_shape = node.get_input_shape(1);
    const auto& out_shape = node.get_output_shape(0);

    dict_size_ = ngraph::shape_size(params_shape);
    indices_size_ = ngraph::shape_size(indices_shape);
    out_size_ = ngraph::shape_size(out_shape);

    const auto axis_shape = node.get_input_shape(2);
    Expects(axis_shape.size() == 0 || axis_shape.size() == 1);
    const auto axis_node = dynamic_cast<ngraph::op::v0::Constant*>(node.get_input_node_ptr(2));
    Expects(axis_node);

    indices_type_ = node.get_input_element_type(1);
    int64_t raw_axis;
    switch (indices_type_) {
        case ngraph::element::Type_t::i64: raw_axis = *axis_node->get_data_ptr<int64_t>(); break;
        case ngraph::element::Type_t::i32: raw_axis = *axis_node->get_data_ptr<int32_t>(); break;
        default: THROW_IE_EXCEPTION << fmt::format(
                                    "Params index type = {} is not supported by Gather operation!",
                                    indices_type_);
    }
    const unsigned axis = raw_axis >= 0 ? raw_axis : raw_axis + params_shape_size;
    Expects(axis < params_shape_size);

    num_dicts_= 1;
    for (int i = 0; i < axis; i++) {
        num_dicts_ *= params_shape[i];
    }

    index_range_ = params_shape[axis];

    data_length_ = 1;
    for (int i = axis + 1; i < params_shape_size; ++i) {
        data_length_ *= params_shape[i];
    }

    if (data_length_ == 0) {
        THROW_IE_EXCEPTION << fmt::format(
        "data_length_ == 0: incorrect input parameters dimension!");
    }
}

void GatherOp::Execute(const InferenceRequestContext& context, Inputs inputs,Outputs outputs,
                       const Workbuffers&) {
    switch (indices_type_) {
        case ngraph::element::i64: return Execute<int64_t>(context, inputs, outputs);
        case ngraph::element::i32: return Execute<int32_t>(context, inputs, outputs);
        default: THROW_IE_EXCEPTION << fmt::format(
            "Index element type = {} is not supported by Gather operation !!",
            indices_type_);
    }
}

template <typename IndexType>
void GatherOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) {
    Expects(inputs.size() == 3);
    Expects(outputs.size() == 1);

    unsigned data_length_bytes = data_length_ * element_size_;

    const auto max_indices_index = indices_size_ - 1;
    const auto max_dict_index = num_dicts_ - 1;
    const auto out_size_bytes = out_size_ * element_size_;
    const bool boundary_ok = data_length_bytes <= out_size_bytes
        - (data_length_bytes * (max_indices_index + max_dict_index * indices_size_));
    Expects(boundary_ok);

    auto& threadContext = context.getThreadContext();
    const auto& device_props = threadContext.device().props();
    const int max_block_size = device_props.maxThreadsPerBlock;
    const unsigned els_per_thread = (data_length_bytes % max_block_size == 0) ?
                                        (data_length_bytes / max_block_size) :
                                        (data_length_bytes / max_block_size + 1);
    const unsigned threads_per_block = (els_per_thread == 1) ? data_length_bytes : max_block_size;

    Expects(num_dicts_ <= device_props.maxGridSize[0]);
    Expects(indices_size_ <= device_props.maxGridSize[1]);

    dim3 grid { num_dicts_, indices_size_ };
    gather<<<grid, threads_per_block, 0, threadContext.stream().get()>>>(
            data_length_bytes,
            index_range_,
            els_per_thread,
            static_cast<const uint8_t*>(inputs[0].get()),
            static_cast<const IndexType*>(inputs[1].get()),
            static_cast<uint8_t*>(outputs[0].get()));

    // TODO: find a way to handle an error raised in a kernel (assertion or trap) properly
    throwIfError(cudaGetLastError());
}

OPERATION_REGISTER(GatherOp, Gather);
} // namespace CUDAPlugin
