// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/squared_difference.hpp>

#include "elementwise_binary.hpp"
#include "kernels/squared_difference.hpp"

namespace CUDAPlugin {

class SquaredDifferenceOp : public ElementwiseBinaryOp<ov::op::v0::SquaredDifference, kernel::SquaredDifference> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace CUDAPlugin
