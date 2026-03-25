// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/logical_and.hpp"
#include "openvino/op/logical_and.hpp"

namespace ov {
namespace nvidia_gpu {

class LogicalAndOp : public ElementwiseBinaryOp<ov::op::v1::LogicalAnd, kernel::LogicalAnd> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
