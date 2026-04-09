// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <error.hpp>
#include <numeric>
#include <openvino/core/except.hpp>
#include <openvino/op/gather.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

namespace {

constexpr unsigned ELS_PER_THREAD_CHUNKS = 2;
constexpr unsigned ELS_PER_THREAD_DICTS = 1;

}  // namespace

GatherOp::GatherOp(const CreationContext& context,
                   const ov::Node& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(node.get_input_size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());
    const auto gather_v8 = dynamic_cast<const ov::op::v8::Gather*>(&node);
    const auto gather_v7 = dynamic_cast<const ov::op::v7::Gather*>(&node);
    const auto gather_v1 = dynamic_cast<const ov::op::v1::Gather*>(&node);
    OPENVINO_ASSERT(gather_v8 || gather_v7 || gather_v1, "Node name: ", GetName());

    const auto gather_base = dynamic_cast<const ov::op::util::GatherBase*>(&node);
    OPENVINO_ASSERT(gather_base, "Node name: ", GetName());

    // For now CUDA operators support only static shapes
    OPENVINO_ASSERT(node.get_input_partial_shape(0).rank().is_static() &&
                        node.get_input_partial_shape(1).rank().is_static() &&
                        node.get_input_partial_shape(2).rank().is_static(),
                    "Node name: ",
                    GetName());

    const ov::element::Type_t element_type = node.get_input_element_type(0);
    switch (element_type) {
        case ov::element::Type_t::dynamic:
        case ov::element::Type_t::u1:
            throw_ov_exception(fmt::format("Params element type = {} is not supported by Gather operation!",
                                           static_cast<ov::element::Type_t>(element_type)));
    }
    OPENVINO_ASSERT(node.get_output_element_type(0) == element_type, "Node name: ", GetName());

    const auto& dict_shape = node.get_input_shape(0);
    const auto& dict_shape_size = dict_shape.size();
    const auto& indices_shape = node.get_input_shape(1);
    const auto& out_shape = node.get_output_shape(0);
    const auto axis_shape = node.get_input_shape(2);

    const ov::element::Type_t indices_type = node.get_input_element_type(1);
    if (indices_type != ov::element::Type_t::i64 && indices_type != ov::element::Type_t::i32) {
        throw_ov_exception(fmt::format("Params index type = {} is not supported by Gather operation!", indices_type));
    }

    const auto axis = gather_base->get_axis();
    OPENVINO_ASSERT(axis >= 0 && axis < dict_shape_size, "Node name: ", GetName());

    int64_t batch_dims = 0;
    if (gather_v8 || gather_v7) {
        batch_dims = gather_v8 ? gather_v8->get_batch_dims() : gather_v7->get_batch_dims();
        OPENVINO_ASSERT(
            batch_dims >= 0 && batch_dims < dict_shape_size && batch_dims < indices_shape.size() && batch_dims <= axis,
            "Node name: ",
            GetName());

        bool batch_check_ok = true;
        for (int i = 0; i < batch_dims; ++i) {
            if (dict_shape[i] != indices_shape[i]) {
                batch_check_ok = false;
                break;
            }
        }
        OPENVINO_ASSERT(batch_check_ok, "Node name: ", GetName());
    }

    const unsigned num_dicts =
        std::accumulate(dict_shape.cbegin() + batch_dims, dict_shape.cbegin() + axis, 1, std::multiplies<unsigned>());
    const unsigned index_range = dict_shape[axis];
    const unsigned data_length =
        std::accumulate(dict_shape.cbegin() + axis + 1, dict_shape.cend(), 1, std::multiplies<unsigned>());

    if (data_length == 0) {
        throw_ov_exception("data_length == 0: incorrect input parameters dimension!");
    }

    const unsigned indices_size =
        std::accumulate(indices_shape.cbegin() + batch_dims, indices_shape.cend(), 1, std::multiplies<unsigned>());
    const auto out_size =
        std::accumulate(out_shape.cbegin() + batch_dims, out_shape.cend(), 1, std::multiplies<unsigned>());

    const auto batch_count =
        std::accumulate(dict_shape.cbegin(), dict_shape.cbegin() + batch_dims, 1, std::multiplies<unsigned>());
    const unsigned dicts_batch_stride =
        std::accumulate(dict_shape.cbegin() + batch_dims, dict_shape.cend(), 1, std::multiplies<unsigned>());
    const unsigned indices_batch_stride =
        std::accumulate(indices_shape.cbegin() + batch_dims, indices_shape.cend(), 1, std::multiplies<unsigned>());
    const unsigned out_batch_stride =
        std::accumulate(out_shape.cbegin() + batch_dims, out_shape.cend(), 1, std::multiplies<unsigned>());

    const auto max_indices_index = indices_size - 1;
    const auto max_dict_index = num_dicts - 1;

    const bool boundary_ok =
        data_length <= out_size - (data_length * (max_indices_index + max_dict_index * indices_size));
    OPENVINO_ASSERT(boundary_ok, "Node name: ", GetName());

    const unsigned num_chunks = data_length % ELS_PER_THREAD_CHUNKS == 0 ? data_length / ELS_PER_THREAD_CHUNKS
                                                                         : data_length / ELS_PER_THREAD_CHUNKS + 1;

    const auto& device_props = context.device().props();
    const auto max_block_size = device_props.maxThreadsPerBlock;
    const auto max_grid_size = device_props.maxGridSize;

    const bool gather_chunks = std::max(num_chunks, num_dicts) == num_chunks;

    unsigned blocks_per_grid{};
    unsigned threads_per_block{};
    unsigned grid_dim_x{};
    unsigned grid_dim_y{};

    if (gather_chunks) {
        blocks_per_grid =
            num_chunks % max_block_size == 0 ? num_chunks / max_block_size : num_chunks / max_block_size + 1;
        threads_per_block = blocks_per_grid == 1 ? num_chunks : max_block_size;
        grid_dim_x = indices_size * batch_count;
        grid_dim_y = num_dicts;
    } else {
        blocks_per_grid = num_dicts % max_block_size == 0 ? num_dicts / max_block_size : num_dicts / max_block_size + 1;
        threads_per_block = blocks_per_grid == 1 ? num_dicts : max_block_size;
        grid_dim_x = indices_size * batch_count;
        grid_dim_y = data_length;
    }

    OPENVINO_ASSERT(grid_dim_x <= max_grid_size[0], "Node name: ", GetName());
    OPENVINO_ASSERT(grid_dim_y <= max_grid_size[1], "Node name: ", GetName());
    OPENVINO_ASSERT(blocks_per_grid <= max_grid_size[2], "Node name: ", GetName());

    gather_kernel_ = kernel::Gather{convertDataType<ov::nvidia_gpu::kernel::Type_t>(element_type),
                                    convertDataType<ov::nvidia_gpu::kernel::Type_t>(indices_type),
                                    num_dicts,
                                    index_range,
                                    data_length,
                                    indices_size,
                                    gather_chunks,
                                    blocks_per_grid,
                                    threads_per_block,
                                    grid_dim_x,
                                    grid_dim_y,
                                    dicts_batch_stride,
                                    indices_batch_stride,
                                    out_batch_stride,
                                    ELS_PER_THREAD_CHUNKS,
                                    ELS_PER_THREAD_DICTS};
}

void GatherOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    (*gather_kernel_)(context.getThreadContext().stream().get(), inputs[0].get(), inputs[1].get(), outputs[0].get());
}

CudaGraphCompatibility GatherOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(GatherOp, Gather);
}  // namespace nvidia_gpu
}  // namespace ov
