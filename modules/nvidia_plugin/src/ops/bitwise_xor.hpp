// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/bitwise_xor.hpp"
#include "openvino/op/bitwise_xor.hpp"

namespace ov {
namespace nvidia_gpu {

class BitwiseXorOp : public ElementwiseBinaryOp<ov::op::v13::BitwiseXor, kernel::BitwiseXor> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
