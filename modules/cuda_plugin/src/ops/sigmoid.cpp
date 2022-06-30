// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sigmoid.hpp"

#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

SigmoidOp::SigmoidOp(const CreationContext& context,
                     const std::shared_ptr<ov::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : ActivationForwardCuDnnOpBase{
          std::make_unique<CUDA::SigmoidDescriptor>(), context, *node, move(inputIds), move(outputIds)} {}

OPERATION_REGISTER(SigmoidOp, Sigmoid);
}  // namespace CUDAPlugin
