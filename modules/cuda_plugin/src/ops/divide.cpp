// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/op/divide.hpp>

#include "cuda_operation_registry.hpp"
#include "divide_cuda.hpp"

namespace CUDAPlugin {

static OperationBase::Ptr divideFactory(const CreationContext& context,
                                        const std::shared_ptr<ov::Node>& in_node,
                                        OperationBase::IndexCollection&& inputIds,
                                        OperationBase::IndexCollection&& outputIds) {
    auto node = std::dynamic_pointer_cast<ov::op::v1::Divide>(in_node);
    Expects(node);

    const OperationBase::IndexCollection inputs{inputIds};
    const OperationBase::IndexCollection outputs{outputIds};

    if (node->is_pythondiv()) {
        return std::make_shared<PythonDivideOp>(
            context, *node, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
    }
    return std::make_shared<DivideOp>(
        context, *node, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
}

OPERATION_REGISTER_FACTORY(divideFactory, Divide)

}  // namespace CUDAPlugin
