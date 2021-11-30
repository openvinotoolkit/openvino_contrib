// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cuda/dnn.hpp>
#include <cuda_operation_registry.hpp>

#include "activation_forward_cudnn_base.hpp"

namespace CUDAPlugin {
namespace {

class ReluOp : public ActivationForwardCuDnnOpBase {

public:
    ReluOp(const CreationContext& context,
           const std::shared_ptr<ngraph::Node>& node,
           IndexCollection&& inputIds,
           IndexCollection&& outputIds)
        : ActivationForwardCuDnnOpBase{
              std::make_unique<CUDA::ReluDescriptor>(), context, *node, move(inputIds), move(outputIds)} {}
};

}  // namespace

OPERATION_REGISTER(ReluOp, Relu);
}  // namespace CUDAPlugin
