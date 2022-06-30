// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cudnn_tensor_op_base.hpp"

namespace CUDAPlugin {

class AddCuDnnOp : public CuDnnTensorOpBase {
public:
    AddCuDnnOp(const CreationContext& context,
               const std::shared_ptr<ov::Node>& node,
               IndexCollection&& inputIds,
               IndexCollection&& outputIds);
};

}  // namespace CUDAPlugin
