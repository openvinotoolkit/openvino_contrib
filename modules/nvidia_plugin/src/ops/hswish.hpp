// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/hswish.hpp"
#include "openvino/op/hswish.hpp"

namespace ov {
namespace nvidia_gpu {

class HSwishOp : public ElementwiseUnaryOp<ov::op::v4::HSwish, kernel::HSwish> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
