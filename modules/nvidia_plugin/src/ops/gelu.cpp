// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gelu.hpp"

#include "cuda_operation_registry.hpp"

namespace ov {
namespace nvidia_gpu {

static OperationBase::Ptr gelu_factory(const CreationContext& context,
                                       const std::shared_ptr<ov::Node>& in_node,
                                       OperationBase::IndexCollection&& inputIds,
                                       OperationBase::IndexCollection&& outputIds) {
    const OperationBase::IndexCollection inputs{inputIds};
    const OperationBase::IndexCollection outputs{outputIds};

    auto node_v7 = std::dynamic_pointer_cast<ov::op::v7::Gelu>(in_node);
    if (node_v7) {
        auto mode = node_v7->get_approximation_mode();
        if (ov::op::GeluApproximationMode::TANH == mode) {
            return std::make_shared<GeluTanhOp>(
                context, *node_v7, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
        }
        return std::make_shared<GeluErfOp>(
            context, *node_v7, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
    }
    auto node_v0 = std::dynamic_pointer_cast<ov::op::v0::Gelu>(in_node);
    OPENVINO_ASSERT(node_v0);
    return std::make_shared<GeluOp>(
        context, *node_v0, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
}

OPERATION_REGISTER_FACTORY(gelu_factory, Gelu)

}  // namespace nvidia_gpu
}  // namespace ov
