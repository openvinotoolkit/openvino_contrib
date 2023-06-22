// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <openvino/core/except.hpp>
#include <sstream>

#include "add_cuda.hpp"
#include "add_cudnn.hpp"
#include "cuda_operation_registry.hpp"

namespace ov {
namespace nvidia_gpu {

static OperationBase::Ptr addFactory(const CreationContext& context,
                                     const std::shared_ptr<ov::Node>& in_node,
                                     OperationBase::IndexCollection&& inputIds,
                                     OperationBase::IndexCollection&& outputIds) {
    auto node = std::dynamic_pointer_cast<ov::op::v1::Add>(in_node);
    OPENVINO_ASSERT(node);

    const OperationBase::IndexCollection inputs{inputIds};
    const OperationBase::IndexCollection outputs{outputIds};

    std::stringstream exception_msg;
    try {
        return std::make_shared<AddCuDnnOp>(
            context, node, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
    } catch (const std::exception& e) {
        exception_msg << "Failed to create AddCuDnn impl: " << e.what();
    }
    try {
        return std::make_shared<AddCudaOp>(
            context, *node, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
    } catch (const std::exception& e) {
        exception_msg << "\nFailed to create AddCuda impl: " << e.what();
    }
    throw_ov_exception(fmt::format("Add node is not supported:\n{}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(addFactory, Add)

}  // namespace nvidia_gpu
}  // namespace ov
