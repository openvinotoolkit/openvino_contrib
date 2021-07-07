// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "add.hpp"

#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

AddOp::AddOp(const CUDA::Device& device, const std::shared_ptr<ngraph::Node>& node,
             IndexCollection&& inputIds, IndexCollection&& outputIds)
    : CuDnnTensorOpBase{device, node, move(inputIds), move(outputIds),
                         cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD} {}

OPERATION_REGISTER(AddOp, Add);

}  // namespace CUDAPlugin
