// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "one_hot.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <numeric>
#include <openvino/op/one_hot.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

OneHotOp::OneHotOp(const CreationContext& context,
                   const ov::Node& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    const auto one_hot = dynamic_cast<const ov::op::v1::OneHot*>(&node);
    OPENVINO_ASSERT(one_hot, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_input_size() == 4, "Node name: ", GetName());  // indices, depth, on, off
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    const ov::element::Type_t indices_type = node.get_input_element_type(0);
    if (indices_type != ov::element::Type_t::i32 && indices_type != ov::element::Type_t::i64) {
        throw_ov_exception(fmt::format("Index element type = {} is not supported by OneHot operation!", indices_type));
    }
    const ov::element::Type_t element_type = node.get_output_element_type(0);

    const auto& out_shape = node.get_output_shape(0);
    const auto rank = static_cast<int64_t>(out_shape.size());
    int64_t axis = one_hot->get_axis();
    if (axis < 0) {
        axis += rank;
    }
    OPENVINO_ASSERT(axis >= 0 && axis < rank, "Node name: ", GetName(), "; OneHot axis is out of range");

    const size_t outer_size =
        std::accumulate(out_shape.begin(), out_shape.begin() + axis, size_t{1}, std::multiplies<size_t>());
    const size_t depth = out_shape[axis];
    const size_t inner_size =
        std::accumulate(out_shape.begin() + axis + 1, out_shape.end(), size_t{1}, std::multiplies<size_t>());

    const size_t total = outer_size * depth * inner_size;
    const size_t max_block_size = context.device().props().maxThreadsPerBlock;
    const size_t num_threads = total == 0 ? 1 : std::min(total, max_block_size);
    const size_t num_blocks = total == 0 ? 1 : (total + num_threads - 1) / num_threads;

    kernel_ = kernel::OneHot{convertDataType<ov::nvidia_gpu::kernel::Type_t>(element_type),
                             convertDataType<ov::nvidia_gpu::kernel::Type_t>(indices_type),
                             outer_size,
                             depth,
                             inner_size,
                             num_blocks,
                             num_threads};
}

void OneHotOp::Execute(const InferenceRequestContext& context,
                       Inputs inputs,
                       Outputs outputs,
                       const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 4, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    // inputs: 0=indices, 1=depth (unused at runtime), 2=on_value, 3=off_value
    (*kernel_)(context.getThreadContext().stream().get(),
               inputs[0].get(),
               inputs[2].get(),
               inputs[3].get(),
               outputs[0].get());
}

CudaGraphCompatibility OneHotOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(OneHotOp, OneHot);

}  // namespace nvidia_gpu
}  // namespace ov
