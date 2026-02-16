// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/logical_or.hpp"
#include "openvino/op/logical_or.hpp"

namespace ov {
namespace nvidia_gpu {

class LogicalOrOp : public ElementwiseBinaryOp<ov::op::v1::LogicalOr, kernel::LogicalOr> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
