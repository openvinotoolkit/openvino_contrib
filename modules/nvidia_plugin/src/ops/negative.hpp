// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/negative.hpp"
#include "openvino/op/negative.hpp"

namespace ov {
namespace nvidia_gpu {

class NegativeOp : public ElementwiseUnaryOp<ov::op::v0::Negative, kernel::Negative> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
