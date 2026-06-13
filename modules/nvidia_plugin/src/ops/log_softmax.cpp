// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "log_softmax.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <numeric>
#include <openvino/op/log_softmax.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

LogSoftmaxOp::LogSoftmaxOp(const CreationContext& context,
                           const ov::Node& node,
                           IndexCollection&& inputIds,
                           IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    const auto log_softmax = dynamic_cast<const ov::op::v5::LogSoftmax*>(&node);
    OPENVINO_ASSERT(log_softmax, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_input_size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    const ov::element::Type_t data_type = node.get_input_element_type(0);
    switch (data_type) {
        case ov::element::Type_t::f16:
        case ov::element::Type_t::bf16:
        case ov::element::Type_t::f32:
        case ov::element::Type_t::f64:
            break;
        default:
            throw_ov_exception(fmt::format("Element type = {} is not supported by LogSoftmax operation!", data_type));
    }

    const auto& data_shape = node.get_input_shape(0);
    const auto rank = static_cast<int64_t>(data_shape.size());
    int64_t axis = log_softmax->get_axis();
    if (axis < 0) {
        axis += rank;
    }
    OPENVINO_ASSERT(axis >= 0 && axis < rank, "Node name: ", GetName(), "; LogSoftmax axis is out of range");

    const size_t outer_size =
        std::accumulate(data_shape.begin(), data_shape.begin() + axis, size_t{1}, std::multiplies<size_t>());
    const size_t axis_size = data_shape[axis];
    const size_t inner_size =
        std::accumulate(data_shape.begin() + axis + 1, data_shape.end(), size_t{1}, std::multiplies<size_t>());

    const size_t num_blocks = outer_size * inner_size;
    constexpr size_t kThreadsPerBlock = 256;

    kernel_ = kernel::LogSoftmax{convertDataType<ov::nvidia_gpu::kernel::Type_t>(data_type),
                                 outer_size,
                                 axis_size,
                                 inner_size,
                                 num_blocks,
                                 kThreadsPerBlock};
}

void LogSoftmaxOp::Execute(const InferenceRequestContext& context,
                           Inputs inputs,
                           Outputs outputs,
                           const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());
    (*kernel_)(context.getThreadContext().stream().get(), inputs[0].get(), outputs[0].get());
}

CudaGraphCompatibility LogSoftmaxOp::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(LogSoftmaxOp, LogSoftmax);

}  // namespace nvidia_gpu
}  // namespace ov
