// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/sqrt.hpp"
#include "openvino/op/sqrt.hpp"

namespace ov {
namespace nvidia_gpu {

class SqrtOp : public ElementwiseUnaryOp<ov::op::v0::Sqrt, kernel::Sqrt> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
