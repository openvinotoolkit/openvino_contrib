// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "scatter_update.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <numeric>
#include <openvino/op/constant.hpp>
#include <openvino/op/scatter_update.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

ScatterUpdateOp::ScatterUpdateOp(const CreationContext& context,
                                 const ov::Node& node,
                                 IndexCollection&& inputIds,
                                 IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    OPENVINO_ASSERT(node.get_input_size() == 4, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    const ov::element::Type_t data_type = node.get_input_element_type(0);
    switch (data_type) {
        case ov::element::Type_t::dynamic:
        case ov::element::Type_t::u1:
            throw_ov_exception(
                fmt::format("Params element type = {} is not supported by ScatterUpdate operation!", data_type));
    }
    // updates and output types must match the data type
    OPENVINO_ASSERT(node.get_input_element_type(2) == data_type, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_element_type(0) == data_type, "Node name: ", GetName());

    const ov::element::Type_t indices_type = node.get_input_element_type(1);
    if (indices_type != ov::element::Type_t::i64 && indices_type != ov::element::Type_t::i32) {
        throw_ov_exception(
            fmt::format("Indices element type = {} is not supported by ScatterUpdate operation!", indices_type));
    }

    const auto& data_shape = node.get_input_shape(0);
    const auto& indices_shape = node.get_input_shape(1);
    const auto& output_shape = node.get_output_shape(0);
    OPENVINO_ASSERT(data_shape == output_shape, "Node name: ", GetName());

    // axis is the 4th input and must be a Constant.
    const auto axis_constant = ov::as_type_ptr<const ov::op::v0::Constant>(node.get_input_node_shared_ptr(3));
    OPENVINO_ASSERT(axis_constant, "Node name: ", GetName(), "; ScatterUpdate axis input must be a Constant");
    const auto rank = static_cast<int64_t>(data_shape.size());
    int64_t axis = axis_constant->cast_vector<int64_t>().at(0);
    if (axis < 0) axis += rank;
    OPENVINO_ASSERT(axis >= 0 && axis < rank, "Node name: ", GetName(), "; ScatterUpdate axis is out of range");

    const size_t axis_dim = data_shape[axis];
    const size_t inner_size =
        std::accumulate(data_shape.begin() + axis + 1, data_shape.end(), size_t{1}, std::multiplies<size_t>());
    const size_t indices_size =
        std::accumulate(indices_shape.begin(), indices_shape.end(), size_t{1}, std::multiplies<size_t>());
    const size_t num_input_elements =
        std::accumulate(data_shape.begin(), data_shape.end(), size_t{1}, std::multiplies<size_t>());
    const size_t outer_size = (axis_dim * inner_size) == 0 ? 0 : num_input_elements / (axis_dim * inner_size);
    const size_t num_update_elements = outer_size * indices_size * inner_size;
    // The kernel launches one thread per output column (outer x inner); each
    // thread iterates the indices sequentially to preserve last-write-wins.
    const size_t num_columns = outer_size * inner_size;

    const size_t max_block_size = context.device().props().maxThreadsPerBlock;
    const size_t num_threads = num_columns == 0 ? 1 : std::min(num_columns, max_block_size);
    const size_t num_blocks = num_columns == 0 ? 1 : (num_columns + num_threads - 1) / num_threads;

    kernel_ = kernel::ScatterUpdate{convertDataType<ov::nvidia_gpu::kernel::Type_t>(data_type),
                                    convertDataType<ov::nvidia_gpu::kernel::Type_t>(indices_type),
                                    num_input_elements,
                                    num_update_elements,
                                    indices_size,
                                    inner_size,
                                    axis_dim,
                                    num_blocks,
                                    num_threads};
}

void ScatterUpdateOp::Execute(const InferenceRequestContext& context,
                              Inputs inputs,
                              Outputs outputs,
                              const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 4, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    (*kernel_)(context.getThreadContext().stream().get(),
               inputs[0].get(),
               inputs[1].get(),
               inputs[2].get(),
               outputs[0].get());
}

CudaGraphCompatibility ScatterUpdateOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(ScatterUpdateOp, ScatterUpdate);

}  // namespace nvidia_gpu
}  // namespace ov
