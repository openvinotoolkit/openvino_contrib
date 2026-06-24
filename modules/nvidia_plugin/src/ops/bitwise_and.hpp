// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/bitwise_and.hpp"
#include "openvino/op/bitwise_and.hpp"

namespace ov {
namespace nvidia_gpu {

class BitwiseAndOp : public ElementwiseBinaryOp<ov::op::v13::BitwiseAnd, kernel::BitwiseAnd> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
