// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_operation_registry.hpp"
#include "reduce_min.hpp"

namespace ov {
namespace nvidia_gpu {

ReduceMinOp::ReduceMinOp(const CreationContext& context,
                         const ov::Node& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : ReduceOp(context, node, move(inputIds), move(outputIds), CUDA::DnnReduceMinDescriptor(reduceCompType(node))) {}

OPERATION_REGISTER(ReduceMinOp, ReduceMin);

}  // namespace nvidia_gpu
}  // namespace ov
