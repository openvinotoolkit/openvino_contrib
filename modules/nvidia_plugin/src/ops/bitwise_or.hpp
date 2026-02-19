// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/bitwise_or.hpp"
#include "openvino/op/bitwise_or.hpp"

namespace ov {
namespace nvidia_gpu {

class BitwiseOrOp : public ElementwiseBinaryOp<ov::op::v13::BitwiseOr, kernel::BitwiseOr> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
