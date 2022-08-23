// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/minimum.hpp>

#include "elementwise_binary.hpp"
#include "kernels/minimum.hpp"

namespace CUDAPlugin {

class MinimumOp : public ElementwiseBinaryOp<ov::op::v1::Minimum, kernel::Minimum> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace CUDAPlugin
