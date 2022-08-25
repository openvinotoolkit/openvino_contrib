// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "add_cudnn.hpp"

#include <cuda_operation_registry.hpp>

namespace ov {
namespace nvidia_gpu {

AddCuDnnOp::AddCuDnnOp(const CreationContext& context,
                       const std::shared_ptr<ov::Node>& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds)
    : CuDnnTensorOpBase{context, node, move(inputIds), move(outputIds), cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD} {}

}  // namespace nvidia_gpu
}  // namespace ov
