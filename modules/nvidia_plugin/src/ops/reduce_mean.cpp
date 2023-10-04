// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_operation_registry.hpp"
#include "reduce_mean.hpp"

namespace ov {
namespace nvidia_gpu {

ReduceMeanOp::ReduceMeanOp(const CreationContext& context,
                         const ov::Node& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : ReduceOp(context, node, move(inputIds), move(outputIds), CUDA::DnnReduceAvgDescriptor(reduceCompType(node))) {}

OPERATION_REGISTER(ReduceMeanOp, ReduceMean);

}  // namespace nvidia_gpu
}  // namespace ov
