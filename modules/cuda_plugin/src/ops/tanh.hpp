// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "activation_forward_cudnn_base.hpp"

namespace CUDAPlugin {

class TanhOp : public ActivationForwardCuDnnOpBase {
public:
    TanhOp(const CreationContext& context,
           const std::shared_ptr<ov::Node>& node,
           IndexCollection&& inputIds,
           IndexCollection&& outputIds);
};

}  // namespace CUDAPlugin
