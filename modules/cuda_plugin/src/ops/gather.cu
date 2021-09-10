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

template <typename DataType, typename IndexType>
static inline __device__ void gather(unsigned data_length,
                                     size_t index_range,
                                     unsigned els_per_thread,
                                     unsigned indices_size,
                                     unsigned indices_index,
                                     unsigned dict,
                                     unsigned chunk,
                                     const DataType* src_dict,
                                     const IndexType* src_index,
                                     DataType* dst_data) {
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

template <typename DataType, typename IndexType>
static __global__ void chunks_gather(unsigned data_length,
                                     size_t index_range,
                                     unsigned num_dicts,
                                     unsigned dicts_batch_stride,
                                     unsigned indices_batch_stride,
                                     unsigned out_batch_stride,
                                     const DataType* src_dict,
                                     const IndexType* src_index,
                                     DataType* dst_data) {
    const auto indices_size = gridDim.y;
    const auto indices_index = blockIdx.y;
    const auto dict = blockIdx.x % num_dicts;
    const auto batch = blockIdx.x / num_dicts;
    const auto chunk = (blockIdx.z * blockDim.x + threadIdx.x) * ELS_PER_THREAD_CHUNKS;
    gather(data_length,
           index_range,
           ELS_PER_THREAD_CHUNKS,
           indices_size,
           indices_index,
           dict,
           chunk,
           src_dict + batch * dicts_batch_stride,
           src_index + batch * indices_batch_stride,
           dst_data + batch * out_batch_stride);
}

template<typename DataType, typename IndexType>
static __global__ void dicts_gather(
        unsigned data_length, size_t index_range, unsigned num_dicts,
        unsigned dicts_batch_stride, unsigned indices_batch_stride, unsigned out_batch_stride,
        const DataType* src_dict, const IndexType* src_index, DataType* dst_data) {
    const auto indices_size = gridDim.y;
    const auto indices_index = blockIdx.y;
    const auto dict = blockIdx.z * blockDim.x + threadIdx.x;
    if (dict >= num_dicts) {
        return;
    }
    const auto chunk = blockIdx.x % data_length * ELS_PER_THREAD_DICTS;
    const auto batch = blockIdx.x / data_length;
    gather(data_length,
           index_range,
           ELS_PER_THREAD_DICTS,
           indices_size,
           indices_index,
           dict,
           chunk,
           src_dict + batch * dicts_batch_stride,
           src_index + batch * indices_batch_stride,
           dst_data + batch * out_batch_stride);
}

}  // namespace kernel

GatherOp::GatherOp(const CUDA::CreationContext& context,
                   const ngraph::Node& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    Expects(node.get_input_size() == 3);
    Expects(node.get_output_size() == 1);
    const auto gather_v7 = dynamic_cast<const ngraph::op::v7::Gather*>(&node);
    const auto gather_v1 = dynamic_cast<const ngraph::op::v1::Gather*>(&node);
    Expects(gather_v7 || gather_v1);

    const auto gather_base = dynamic_cast<const ngraph::op::util::GatherBase*>(&node);
    Expects(gather_base);

    // For now CUDA operators support only static shapes
    Expects(node.get_input_partial_shape(0).rank().is_static() && node.get_input_partial_shape(1).rank().is_static() &&
            node.get_input_partial_shape(2).rank().is_static());

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

    const auto& dict_shape = node.get_input_shape(0);
    const auto& dict_shape_size = dict_shape.size();
    const auto& indices_shape = node.get_input_shape(1);
    const auto& out_shape = node.get_output_shape(0);
    const auto axis_shape = node.get_input_shape(2);

    indices_type_ = node.get_input_element_type(1);
    if (indices_type_ != ngraph::element::Type_t::i64 && indices_type_ != ngraph::element::Type_t::i32) {
        throwIEException(fmt::format("Params index type = {} is not supported by Gather operation!", indices_type_));
    }

    const auto axis = gather_base->get_axis();
    Expects(axis >= 0 && axis < dict_shape_size);

    int64_t batch_dims = 0;
    if (gather_v7) {
        batch_dims = gather_v7->get_batch_dims();
        Expects(batch_dims >= 0 && batch_dims < dict_shape_size && batch_dims < indices_shape.size() &&
                batch_dims <= axis);

        bool batch_check_ok = true;
        for (int i = 0; i < batch_dims; ++i) {
            if (dict_shape[i] != indices_shape[i]) {
                batch_check_ok = false;
                break;
            }
        }
        Expects(batch_check_ok);
    }

    num_dicts_ =
        std::accumulate(dict_shape.cbegin() + batch_dims, dict_shape.cbegin() + axis, 1, std::multiplies<unsigned>());
    index_range_ = dict_shape[axis];
    data_length_ = std::accumulate(dict_shape.cbegin() + axis + 1, dict_shape.cend(), 1, std::multiplies<unsigned>());

    if (data_length_ == 0) {
        throwIEException("data_length_ == 0: incorrect input parameters dimension!");
    }

    indices_size_ =
        std::accumulate(indices_shape.cbegin() + batch_dims, indices_shape.cend(), 1, std::multiplies<unsigned>());
    const auto out_size =
        std::accumulate(out_shape.cbegin() + batch_dims, out_shape.cend(), 1, std::multiplies<unsigned>());

    const auto batch_count =
        std::accumulate(dict_shape.cbegin(), dict_shape.cbegin() + batch_dims, 1, std::multiplies<unsigned>());
    dicts_batch_stride_ =
        std::accumulate(dict_shape.cbegin() + batch_dims, dict_shape.cend(), 1, std::multiplies<unsigned>());
    indices_batch_stride_ =
        std::accumulate(indices_shape.cbegin() + batch_dims, indices_shape.cend(), 1, std::multiplies<unsigned>());
    out_batch_stride_ =
        std::accumulate(out_shape.cbegin() + batch_dims, out_shape.cend(), 1, std::multiplies<unsigned>());

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
        grid_dim_x_ = num_dicts_ * batch_count;

        Expects(grid_dim_x_ <= max_grid_size[0]);
        Expects(indices_size_ <= max_grid_size[1]);
        Expects(blocks_per_grid_ <= max_grid_size[2]);
    } else {
        blocks_per_grid_ = num_dicts_ % max_block_size == 0
                                            ? num_dicts_ / max_block_size
                                            : num_dicts_ / max_block_size + 1;
        threads_per_block_ = blocks_per_grid_ == 1 ? num_dicts_ : max_block_size;
        grid_dim_x_ = data_length_ * batch_count;

        Expects(grid_dim_x_ <= max_grid_size[0]);
        Expects(indices_size_ <= max_grid_size[1]);
        Expects(blocks_per_grid_ <= max_grid_size[2]);
    }
}

void GatherOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers&) const {
    switch (indices_type_) {
        case ngraph::element::i64: return ExecuteByDataType<int64_t>(context, inputs, outputs);
        case ngraph::element::i32: return ExecuteByDataType<int32_t>(context, inputs, outputs);
        default: throwIEException(
                fmt::format("Index element type = {} is not supported by Gather operation !!",
                indices_type_));
    }
}

template <typename IndexType>
void GatherOp::ExecuteByDataType(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) const {
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

template <typename DataType, typename IndexType>
void GatherOp::ExecuteImpl(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) const {
    Expects(inputs.size() == 3);
    Expects(outputs.size() == 1);

    const auto& stream = context.getThreadContext().stream().get();

    const auto src_dict = static_cast<const DataType*>(inputs[0].get());
    const auto src_index = static_cast<const IndexType*>(inputs[1].get());
    auto dst_data = static_cast<DataType*>(outputs[0].get());

    if (gather_chunks_) {
        kernel::chunks_gather<<<{grid_dim_x_, indices_size_, blocks_per_grid_}, threads_per_block_, 0, stream>>>(
            data_length_,
            index_range_,
            num_dicts_,
            dicts_batch_stride_,
            indices_batch_stride_,
            out_batch_stride_,
            src_dict,
            src_index,
            dst_data);
    } else {
        kernel::dicts_gather<<<{grid_dim_x_, indices_size_, blocks_per_grid_}, threads_per_block_, 0, stream>>>(
            data_length_,
            index_range_,
            num_dicts_,
            dicts_batch_stride_,
            indices_batch_stride_,
            out_batch_stride_,
            src_dict,
            src_index,
            dst_data);
    }
    // TODO: find a way to handle an error raised in a kernel (assertion or trap) properly in CUDA Plugin
    throwIfError(cudaGetLastError());
}

OPERATION_REGISTER(GatherOp, Gather);
} // namespace CUDAPlugin
