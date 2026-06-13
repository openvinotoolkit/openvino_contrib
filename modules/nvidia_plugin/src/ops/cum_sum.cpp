// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cum_sum.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <numeric>
#include <openvino/op/constant.hpp>
#include <openvino/op/cum_sum.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

CumSumOp::CumSumOp(const CreationContext& context,
                   const ov::Node& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    const auto cum_sum = dynamic_cast<const ov::op::v0::CumSum*>(&node);
    OPENVINO_ASSERT(cum_sum, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_input_size() == 2, "Node name: ", GetName());  // data, axis
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    const ov::element::Type_t data_type = node.get_input_element_type(0);
    switch (data_type) {
        case ov::element::Type_t::dynamic:
        case ov::element::Type_t::u1:
            throw_ov_exception(fmt::format("Element type = {} is not supported by CumSum operation!", data_type));
    }

    const auto& data_shape = node.get_input_shape(0);
    const auto rank = static_cast<int64_t>(data_shape.size());
    OPENVINO_ASSERT(rank > 0, "Node name: ", GetName(), "; CumSum needs rank >= 1");

    // axis is the 2nd input and must be a Constant.
    const auto axis_constant = ov::as_type_ptr<const ov::op::v0::Constant>(node.get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(axis_constant, "Node name: ", GetName(), "; CumSum axis input must be a Constant");
    int64_t axis = axis_constant->cast_vector<int64_t>().at(0);
    if (axis < 0) {
        axis += rank;
    }
    OPENVINO_ASSERT(axis >= 0 && axis < rank, "Node name: ", GetName(), "; CumSum axis is out of range");

    const size_t outer_size =
        std::accumulate(data_shape.begin(), data_shape.begin() + axis, size_t{1}, std::multiplies<size_t>());
    const size_t axis_size = data_shape[axis];
    const size_t inner_size =
        std::accumulate(data_shape.begin() + axis + 1, data_shape.end(), size_t{1}, std::multiplies<size_t>());

    const size_t num_rows = outer_size * inner_size;
    const size_t max_block_size = context.device().props().maxThreadsPerBlock;
    const size_t num_threads = num_rows == 0 ? 1 : std::min(num_rows, max_block_size);
    const size_t num_blocks = num_rows == 0 ? 1 : (num_rows + num_threads - 1) / num_threads;

    kernel_ = kernel::CumSum{convertDataType<ov::nvidia_gpu::kernel::Type_t>(data_type),
                             outer_size,
                             axis_size,
                             inner_size,
                             cum_sum->is_exclusive(),
                             cum_sum->is_reverse(),
                             num_blocks,
                             num_threads};
}

void CumSumOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    (*kernel_)(context.getThreadContext().stream().get(), inputs[0].get(), outputs[0].get());
}

CudaGraphCompatibility CumSumOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(CumSumOp, CumSum);

}  // namespace nvidia_gpu
}  // namespace ov
