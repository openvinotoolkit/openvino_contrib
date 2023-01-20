// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "multiply_cudnn.hpp"

namespace ov {
namespace nvidia_gpu {

MultiplyCuDnnOp::MultiplyCuDnnOp(const CreationContext& context,
                                 const std::shared_ptr<ov::Node>& node,
                                 IndexCollection&& inputIds,
                                 IndexCollection&& outputIds)
    : CuDnnTensorOpBase{context, node, move(inputIds), move(outputIds), cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL} {}

}  // namespace nvidia_gpu
}  // namespace ov
