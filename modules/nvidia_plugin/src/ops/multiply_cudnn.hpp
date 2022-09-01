// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cudnn_tensor_op_base.hpp"

namespace ov {
namespace nvidia_gpu {

class MultiplyCuDnnOp : public CuDnnTensorOpBase {
public:
    MultiplyCuDnnOp(const CreationContext& context,
                    const std::shared_ptr<ov::Node>& node,
                    IndexCollection&& inputIds,
                    IndexCollection&& outputIds);
};

}  // namespace nvidia_gpu
}  // namespace ov
