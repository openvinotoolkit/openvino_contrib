// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include "openvino/op/mish.hpp"

#include "elementwise_unary.hpp"
#include "kernels/mish.hpp"

namespace ov {
namespace nvidia_gpu {

class MishOp : public ElementwiseUnaryOp<ov::op::v4::Mish, kernel::Mish> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
