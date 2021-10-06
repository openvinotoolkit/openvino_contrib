// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "relu.hpp"

#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

ReluOp::ReluOp(const CreationContext& context,
               const std::shared_ptr<ngraph::Node>& node,
               IndexCollection&& inputIds,
               IndexCollection&& outputIds)
    : ActivationForwardCuDnnOpBase{
          std::make_unique<CUDA::ReluDescriptor>(), context, *node, move(inputIds), move(outputIds)} {}

OPERATION_REGISTER(ReluOp, Relu);
}  // namespace CUDAPlugin
