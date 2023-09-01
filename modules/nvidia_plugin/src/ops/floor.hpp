// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include "openvino/op/floor.hpp"

#include "elementwise_unary.hpp"
#include "kernels/floor.hpp"

namespace ov {
namespace nvidia_gpu {

class FloorOp : public ElementwiseUnaryOp<ov::op::v0::Floor, kernel::Floor> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
