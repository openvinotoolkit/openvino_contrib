// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include "openvino/op/cos.hpp"

#include "elementwise_unary.hpp"
#include "kernels/cos.hpp"

namespace ov {
namespace nvidia_gpu {

class CosOp : public ElementwiseUnaryOp<ov::op::v0::Cos, kernel::Cos> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
