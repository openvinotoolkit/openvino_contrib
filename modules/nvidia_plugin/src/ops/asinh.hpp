// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/asinh.hpp"
#include "openvino/op/asinh.hpp"

namespace ov {
namespace nvidia_gpu {

class AsinhOp : public ElementwiseUnaryOp<ov::op::v3::Asinh, kernel::Asinh> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
