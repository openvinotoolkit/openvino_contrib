// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/power.hpp>

#include "elementwise_binary.hpp"
#include "kernels/power.hpp"

namespace CUDAPlugin {

class PowerOp : public ElementwiseBinaryOp<ov::op::v1::Power, kernel::Power> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace CUDAPlugin
