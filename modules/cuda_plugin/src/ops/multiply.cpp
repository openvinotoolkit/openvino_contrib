// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <gsl/gsl_assert>
#include <sstream>

#include "cuda_operation_registry.hpp"
#include "multiply_cuda.hpp"
#include "multiply_cudnn.hpp"

namespace CUDAPlugin {

static OperationBase::Ptr multiplyFactory(const CreationContext& context,
                                          const std::shared_ptr<ngraph::Node>& node,
                                          OperationBase::IndexCollection&& inputIds,
                                          OperationBase::IndexCollection&& outputIds) {
    const OperationBase::IndexCollection inputs{inputIds};
    const OperationBase::IndexCollection outputs{outputIds};

    std::stringstream exception_msg;
    try {
        return std::make_shared<MultiplyCuDnnOp>(
            context, node, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
    } catch (const std::exception& e) {
        exception_msg << "Failed to create MultiplyCuDnn impl: " << e.what();
    }
    try {
        return std::make_shared<MultiplyCudaOp>(
            context, *node, OperationBase::IndexCollection{inputs}, OperationBase::IndexCollection{outputs});
    } catch (const std::exception& e) {
        exception_msg << "\nFailed to create MultiplyCuda impl: " << e.what();
    }
    throwIEException(fmt::format("Multiply node is not supported:\n{}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(multiplyFactory, Multiply)

}  // namespace CUDAPlugin
