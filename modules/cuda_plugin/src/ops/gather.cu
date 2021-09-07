// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "details/cuda_ie_api_import_fix.hpp"
// ^^ must come before any other ie includes which use
// INFERENCE_ENGINE_DEPRECATED
#include "details/cuda_ngraph_import_fix.hpp"
// ^^ must come before any other ngraph includes which use
// NGRAPH_DEPRECATED
#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/gather.hpp>

#include "gather.hpp"

namespace CUDAPlugin {

namespace {

constexpr unsigned ELS_PER_THREAD_CHUNKS = 2;
constexpr unsigned ELS_PER_THREAD_DICTS = 1;

} // namespace

namespace kernel {

template<typename DataType, typename IndexType>
static inline __device__ void gather(
        unsigned data_length, size_t index_range, unsigned els_per_thread,
        const DataType* src_dict, const IndexType* src_index, DataType* dst_data,
        unsigned indices_size, unsigned indices_index, unsigned dict, unsigned chunk) {
    const auto dict_index = src_index[indices_index];
    if (dict_index >= index_range) {
        // TODO: find a way to handle an error raised in a kernel (assertion or trap) properly
        __trap();
    }
    unsigned thread_offset;
    for (int el = 0; el < els_per_thread; ++el) {
        thread_offset = chunk + el;
        if (thread_offset >= data_length) {
            return;
        }
        dst_data[data_length * (indices_index + dict * indices_size) + thread_offset]
            = src_dict[data_length * (dict_index + dict * index_range) + thread_offset];
    }
}

template<typename DataType, typename IndexType>
static __global__ void chunks_gather(
        unsigned data_length, size_t index_range,
        const DataType* src_dict, const IndexType* src_index, DataType* dst_data) {
    const auto indices_size = gridDim.y;
    const auto indices_index = blockIdx.y;
    const auto dict = blockIdx.x;
    const auto chunk = (blockIdx.z * blockDim.x + threadIdx.x) * ELS_PER_THREAD_CHUNKS;
    gather(data_length, index_range, ELS_PER_THREAD_CHUNKS,
           src_dict, src_index, dst_data,
           indices_size, indices_index, dict, chunk);
}

template<typename DataType, typename IndexType>
static __global__ void dicts_gather(
        unsigned data_length, size_t index_range, unsigned num_dicts,
        const DataType* src_dict, const IndexType* src_index, DataType* dst_data) {
    const auto indices_size = gridDim.y;
    const auto indices_index = blockIdx.y;
    const auto dict = blockIdx.z * blockDim.x + threadIdx.x;
    if (dict >= num_dicts) {
        return;
    }
    const auto chunk = blockIdx.x * ELS_PER_THREAD_DICTS;
    gather(data_length, index_range, ELS_PER_THREAD_DICTS,
           src_dict, src_index, dst_data,
           indices_size, indices_index, dict, chunk);
}

}  // namespace kernel

GatherOp::GatherOp(const CUDA::CreationContext& context,
                   const ngraph::Node& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
        : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    Expects(dynamic_cast<const ngraph::op::v1::Gather*>(&node) != nullptr);
    Expects(node.get_input_size() == 3);
    Expects(node.get_output_size() == 1);
    
    element_type_ = node.get_input_element_type(0);
    switch (element_type_) {
        case ngraph::element::Type_t::undefined:
        case ngraph::element::Type_t::dynamic:
        case ngraph::element::Type_t::u1:
            throwIEException(
                    fmt::format("Params element type = {} is not supported by Gather operation!",
                                static_cast<ngraph::element::Type_t>(element_type_)));
    }
    Expects(node.get_output_element_type(0) == element_type_);

    const auto& params_shape = node.get_input_shape(0);
    const auto& params_shape_size = params_shape.size();
    const auto& indices_shape = node.get_input_shape(1);
    const auto& out_shape = node.get_output_shape(0);
    const auto axis_shape = node.get_input_shape(2);

    Expects(axis_shape.size() == 0 || axis_shape.size() == 1);
    const auto axis_node = dynamic_cast<ngraph::op::v0::Constant*>(node.get_input_node_ptr(2));
    Expects(axis_node);

    indices_type_ = node.get_input_element_type(1);
    int64_t raw_axis;
    switch (indices_type_) {
        case ngraph::element::Type_t::i64: raw_axis = *axis_node->get_data_ptr<int64_t>(); break;
        case ngraph::element::Type_t::i32: raw_axis = *axis_node->get_data_ptr<int32_t>(); break;
        default: throwIEException(
                fmt::format("Params index type = {} is not supported by Gather operation!",
                            indices_type_));
    }
    const unsigned axis = raw_axis >= 0 ? raw_axis : raw_axis + params_shape_size;
    Expects(axis < params_shape_size);

    num_dicts_ = std::accumulate(params_shape.cbegin(), params_shape.cbegin() + axis, 1,
                                 std::multiplies<unsigned>());
    index_range_ = params_shape[axis];
    data_length_ = std::accumulate(params_shape.cbegin() + axis + 1, params_shape.end(), 1,
                                   std::multiplies<unsigned>());

    if (data_length_ == 0) {
        throwIEException("data_length_ == 0: incorrect input parameters dimension!");
    }

    indices_size_ = ngraph::shape_size(indices_shape);
    const auto out_size = ngraph::shape_size(out_shape);
    const auto max_indices_index = indices_size_ - 1;
    const auto max_dict_index = num_dicts_ - 1;

    const bool boundary_ok = data_length_ <= out_size
        - (data_length_ * (max_indices_index + max_dict_index * indices_size_));
    Expects(boundary_ok);

    const unsigned num_chunks = data_length_ % ELS_PER_THREAD_CHUNKS == 0
                                    ? data_length_ / ELS_PER_THREAD_CHUNKS
                                    : data_length_ / ELS_PER_THREAD_CHUNKS + 1;

    const auto& device_props = context.device().props();
    const auto max_block_size = device_props.maxThreadsPerBlock;
    const auto max_grid_size = device_props.maxGridSize;

    gather_chunks_ = std::max(num_chunks, num_dicts_) == num_chunks;

    if (gather_chunks_) {
        blocks_per_grid_ = num_chunks % max_block_size == 0
                                            ? num_chunks / max_block_size
                                            : num_chunks / max_block_size + 1;
        threads_per_block_ = blocks_per_grid_ == 1 ? num_chunks : max_block_size;

        Expects(num_dicts_ <= max_grid_size[0]);
        Expects(indices_size_ <= max_grid_size[1]);
        Expects(blocks_per_grid_ <= max_grid_size[2]);
    } else {
        blocks_per_grid_ = num_dicts_ % max_block_size == 0
                                            ? num_dicts_ / max_block_size
                                            : num_dicts_ / max_block_size + 1;
        threads_per_block_ = blocks_per_grid_ == 1 ? num_dicts_ : max_block_size;

        Expects(data_length_ <= max_grid_size[0]);
        Expects(indices_size_ <= max_grid_size[1]);
        Expects(blocks_per_grid_ <= max_grid_size[2]);
    }
}

void GatherOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs,
                       const Workbuffers&) {
    switch (indices_type_) {
        case ngraph::element::i64: return ExecuteByDataType<int64_t>(context, inputs, outputs);
        case ngraph::element::i32: return ExecuteByDataType<int32_t>(context, inputs, outputs);
        default: throwIEException(
                fmt::format("Index element type = {} is not supported by Gather operation !!",
                indices_type_));
    }
}

template<typename IndexType>
void GatherOp::ExecuteByDataType(const InferenceRequestContext& context, Inputs inputs,
                                 Outputs outputs) {
    switch (element_type_) {
        case ngraph::element::boolean: return ExecuteImpl<bool, IndexType>(context, inputs,
                                                                           outputs);
        case ngraph::element::bf16: return ExecuteImpl<__nv_bfloat16, IndexType>(context, inputs,
                                                                                 outputs);
        case ngraph::element::f16: return ExecuteImpl<__half, IndexType>(context, inputs,
                                                                         outputs);
        case ngraph::element::f32: return ExecuteImpl<float, IndexType>(context, inputs,
                                                                        outputs);
        case ngraph::element::f64: return ExecuteImpl<double, IndexType>(context, inputs,
                                                                         outputs);
        case ngraph::element::i8: return ExecuteImpl<int8_t, IndexType>(context, inputs,
                                                                        outputs);
        case ngraph::element::i16: return ExecuteImpl<int16_t, IndexType>(context, inputs,
                                                                          outputs);
        case ngraph::element::i32: return ExecuteImpl<int32_t, IndexType>(context, inputs,
                                                                          outputs);
        case ngraph::element::i64: return ExecuteImpl<int64_t, IndexType>(context, inputs,
                                                                          outputs);
        case ngraph::element::u8: return ExecuteImpl<uint8_t, IndexType>(context, inputs,
                                                                          outputs);
        case ngraph::element::u16: return ExecuteImpl<uint16_t, IndexType>(context, inputs,
                                                                           outputs);
        case ngraph::element::u32: return ExecuteImpl<uint32_t, IndexType>(context, inputs,
                                                                           outputs);
        case ngraph::element::u64: return ExecuteImpl<uint64_t, IndexType>(context, inputs,
                                                                           outputs);
        default: throwIEException(
                fmt::format("Index element type = {} is not supported by Gather operation !!",
                            indices_type_));
    }
}

template<typename DataType, typename IndexType>
void GatherOp::ExecuteImpl(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) {
    Expects(inputs.size() == 3);
    Expects(outputs.size() == 1);

    const auto& stream = context.getThreadContext().stream().get();

    const auto src_dict = static_cast<const DataType*>(inputs[0].get());
    const auto src_index = static_cast<const IndexType*>(inputs[1].get());
    auto dst_data = static_cast<DataType*>(outputs[0].get());

    if (gather_chunks_) {
        dim3 grid { num_dicts_, indices_size_, blocks_per_grid_ };
        kernel::chunks_gather<<<grid, threads_per_block_, 0, stream>>>(data_length_,
                                                                       index_range_,
                                                                       src_dict,
                                                                       src_index,
                                                                       dst_data);
    } else {
        dim3 grid { data_length_, indices_size_, blocks_per_grid_ };
        kernel::dicts_gather<<<grid, threads_per_block_, 0, stream>>>(data_length_,
                                                                      index_range_,
                                                                      num_dicts_,
                                                                      src_dict,
                                                                      src_index,
                                                                      dst_data);
                }
    // TODO: find a way to handle an error raised in a kernel (assertion or trap) properly in CUDA Plugin
    throwIfError(cudaGetLastError());
}

OPERATION_REGISTER(GatherOp, Gather);
} // namespace CUDAPlugin
