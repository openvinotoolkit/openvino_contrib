// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/sinh.hpp"
#include "openvino/op/sinh.hpp"

namespace ov {
namespace nvidia_gpu {

class SinhOp : public ElementwiseUnaryOp<ov::op::v0::Sinh, kernel::Sinh> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
