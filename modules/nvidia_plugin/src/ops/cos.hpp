// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/cos.hpp"
#include "openvino/op/cos.hpp"

namespace ov {
namespace nvidia_gpu {

class CosOp : public ElementwiseUnaryOp<ov::op::v0::Cos, kernel::Cos> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
