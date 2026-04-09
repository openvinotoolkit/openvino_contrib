// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/softplus.hpp"
#include "openvino/op/softplus.hpp"

namespace ov {
namespace nvidia_gpu {

class SoftPlusOp : public ElementwiseUnaryOp<ov::op::v4::SoftPlus, kernel::SoftPlus> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
