// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "greater.hpp"

#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

GreaterOp::GreaterOp(const CreationContext& context,
                     const std::shared_ptr<ngraph::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : CuDnnTensorOpBase{context, node, move(inputIds), move(outputIds), cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MAX} {}

OPERATION_REGISTER(GreaterOp, Greater);

}  // namespace CUDAPlugin
