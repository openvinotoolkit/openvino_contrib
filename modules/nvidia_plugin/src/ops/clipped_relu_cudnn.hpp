// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/clamp.hpp>

#include "activation_forward_cudnn_base.hpp"

namespace ov {
namespace nvidia_gpu {

class ClippedReluCuDnnOp : public ActivationForwardCuDnnOpBase {
public:
    using NodeOp = ov::op::v0::Clamp;

    ClippedReluCuDnnOp(const CreationContext& context,
                       const NodeOp& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds);
};

}  // namespace nvidia_gpu
}  // namespace ov
