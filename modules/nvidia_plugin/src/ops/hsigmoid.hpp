// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/hsigmoid.hpp"
#include "openvino/op/hsigmoid.hpp"

namespace ov {
namespace nvidia_gpu {

class HSigmoidOp : public ElementwiseUnaryOp<ov::op::v5::HSigmoid, kernel::HSigmoid> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
