// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/subtract.hpp"

#include "elementwise_binary.hpp"
#include "kernels/subtract.hpp"

namespace ov {
namespace nvidia_gpu {

class SubtractOp : public ElementwiseBinaryOp<ov::op::v1::Subtract, kernel::Subtract> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
