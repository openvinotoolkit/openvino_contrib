// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cudnn_tensor_op_base.hpp"

namespace CUDAPlugin {

class AddOp : public CuDnnTensorOpBase {
  public:
    AddOp(const CUDA::Device& device, const std::shared_ptr<ngraph::Node>& node,
          IndexCollection&& inputIds, IndexCollection&& outputIds);
};

}  // namespace CUDAPlugin
