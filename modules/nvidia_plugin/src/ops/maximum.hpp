// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/maximum.hpp>

#include "elementwise_binary.hpp"
#include "kernels/maximum.hpp"

namespace ov {
namespace nvidia_gpu {

class MaximumOp : public ElementwiseBinaryOp<ov::op::v1::Maximum, kernel::Maximum> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
