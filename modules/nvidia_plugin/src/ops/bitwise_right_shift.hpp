// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/bitwise_right_shift.hpp"
#include "openvino/op/bitwise_right_shift.hpp"

namespace ov {
namespace nvidia_gpu {

class BitwiseRightShiftOp : public ElementwiseBinaryOp<ov::op::v15::BitwiseRightShift, kernel::BitwiseRightShift> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
