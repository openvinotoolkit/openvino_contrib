// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/clamp.hpp>

#include "activation_forward_cudnn_base.hpp"

namespace CUDAPlugin {

class ClippedReluCuDnnOp : public ActivationForwardCuDnnOpBase {
public:
    using NodeOp = ngraph::op::Clamp;

    ClippedReluCuDnnOp(const CreationContext& context,
                       const NodeOp& node,
                       IndexCollection&& inputIds,
                       IndexCollection&& outputIds);
};

}  // namespace CUDAPlugin
