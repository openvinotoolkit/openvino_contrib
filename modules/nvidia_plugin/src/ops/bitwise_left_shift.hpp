// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/bitwise_left_shift.hpp"
#include "openvino/op/bitwise_left_shift.hpp"

namespace ov {
namespace nvidia_gpu {

class BitwiseLeftShiftOp : public ElementwiseBinaryOp<ov::op::v15::BitwiseLeftShift, kernel::BitwiseLeftShift> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
