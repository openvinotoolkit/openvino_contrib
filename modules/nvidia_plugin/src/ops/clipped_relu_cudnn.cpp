// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "clipped_relu_cudnn.hpp"

#include <fmt/format.h>

#include <cuda/dnn.hpp>
#include <cuda_operation_registry.hpp>

namespace ov {
namespace nvidia_gpu {

ClippedReluCuDnnOp::ClippedReluCuDnnOp(const CreationContext& context,
                                       const NodeOp& node,
                                       IndexCollection&& inputIds,
                                       IndexCollection&& outputIds)
    : ActivationForwardCuDnnOpBase{std::make_unique<CUDA::ClippedReluDescriptor>(node.get_max()),
                                   context,
                                   node,
                                   move(inputIds),
                                   move(outputIds)} {
    const auto min = node.get_min();
    const auto max = node.get_max();
    if (min != 0.0) {
        throwIEException(fmt::format("ov::nvidia_gpu::ClippedReluCuDnnOp: Clamp min != 0.0, min = {}", min));
    }
    if (max < 0.0) {
        throwIEException(fmt::format("ov::nvidia_gpu::ClippedReluCuDnnOp: Clamp max < 0.0, max = {}", max));
    }
}

}  // namespace nvidia_gpu
}  // namespace ov
