// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_operation_registry.hpp"
#include "reduce_prod.hpp"

namespace ov {
namespace nvidia_gpu {

ReduceProdOp::ReduceProdOp(const CreationContext& context,
                         const ov::Node& node,
                         IndexCollection&& inputIds,
                         IndexCollection&& outputIds)
    : ReduceOp(context, node, move(inputIds), move(outputIds), CUDA::DnnReduceMulDescriptor(reduceCompType(node))) {}

OPERATION_REGISTER(ReduceProdOp, ReduceProd);

}  // namespace nvidia_gpu
}  // namespace ov
