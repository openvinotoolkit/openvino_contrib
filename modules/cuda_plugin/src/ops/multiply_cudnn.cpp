// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "multiply_cudnn.hpp"

namespace CUDAPlugin {

MultiplyCuDnnOp::MultiplyCuDnnOp(const CreationContext& context,
                                 const std::shared_ptr<ngraph::Node>& node,
                                 IndexCollection&& inputIds,
                                 IndexCollection&& outputIds)
    : CuDnnTensorOpBase{context, node, move(inputIds), move(outputIds), cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL} {}

}  // namespace CUDAPlugin
