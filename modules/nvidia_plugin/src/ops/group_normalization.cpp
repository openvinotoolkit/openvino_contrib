// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_normalization.hpp"

#include <fmt/format.h>

#include <cuda_operation_registry.hpp>
#include <numeric>
#include <openvino/op/group_normalization.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

GroupNormalizationOp::GroupNormalizationOp(const CreationContext& context,
                                           const ov::Node& node,
                                           IndexCollection&& inputIds,
                                           IndexCollection&& outputIds)
    : OperationBase(context, node, std::move(inputIds), std::move(outputIds)) {
    const auto group_norm = dynamic_cast<const ov::op::v12::GroupNormalization*>(&node);
    OPENVINO_ASSERT(group_norm, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_input_size() == 3, "Node name: ", GetName());  // data, scale, bias
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    const ov::element::Type_t data_type = node.get_input_element_type(0);
    switch (data_type) {
        case ov::element::Type_t::f16:
        case ov::element::Type_t::bf16:
        case ov::element::Type_t::f32:
        case ov::element::Type_t::f64:
            break;
        default:
            throw_ov_exception(
                fmt::format("Element type = {} is not supported by GroupNormalization operation!", data_type));
    }
    OPENVINO_ASSERT(node.get_input_element_type(1) == data_type, "Node name: ", GetName());  // scale
    OPENVINO_ASSERT(node.get_input_element_type(2) == data_type, "Node name: ", GetName());  // bias
    OPENVINO_ASSERT(node.get_output_element_type(0) == data_type, "Node name: ", GetName());

    const auto& data_shape = node.get_input_shape(0);
    OPENVINO_ASSERT(data_shape.size() >= 2, "Node name: ", GetName(), "; GroupNormalization needs rank >= 2");

    const size_t batch = data_shape[0];
    const size_t num_channels = data_shape[1];
    const auto num_groups = static_cast<size_t>(group_norm->get_num_groups());
    OPENVINO_ASSERT(num_groups != 0 && num_channels % num_groups == 0,
                    "Node name: ",
                    GetName(),
                    "; num_channels must be divisible by num_groups");

    const size_t spatial_size =
        std::accumulate(data_shape.begin() + 2, data_shape.end(), size_t{1}, std::multiplies<size_t>());

    const size_t num_blocks = batch * num_groups;
    // Power-of-two block so the shared-memory tree reduction is exact; threads
    // stride over the (typically large) per-group element range.
    constexpr size_t kThreadsPerBlock = 256;

    kernel_ = kernel::GroupNormalization{convertDataType<ov::nvidia_gpu::kernel::Type_t>(data_type),
                                         batch,
                                         num_channels,
                                         num_groups,
                                         spatial_size,
                                         group_norm->get_epsilon(),
                                         num_blocks,
                                         kThreadsPerBlock};
}

void GroupNormalizationOp::Execute(const InferenceRequestContext& context,
                                   Inputs inputs,
                                   Outputs outputs,
                                   const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 3, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    (*kernel_)(context.getThreadContext().stream().get(),
               inputs[0].get(),
               inputs[1].get(),
               inputs[2].get(),
               outputs[0].get());
}

CudaGraphCompatibility GroupNormalizationOp::GetCudaGraphCompatibilityImpl() const {
    return CudaGraphCompatibility::FULL;
}

OPERATION_REGISTER(GroupNormalizationOp, GroupNormalization);

}  // namespace nvidia_gpu
}  // namespace ov
