// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_operation_registry.hpp"
#include "reduce_max.hpp"

namespace ov {
namespace nvidia_gpu {

ReduceMaxOp::ReduceMaxOp(const CreationContext& context,
                         const ov::Node& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : ReduceOp(context, node, move(inputIds), move(outputIds), CUDA::DnnReduceMaxDescriptor(reduceCompType(node))) {}

OPERATION_REGISTER(ReduceMaxOp, ReduceMax);

}  // namespace nvidia_gpu
}  // namespace ov
