// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/cosh.hpp"
#include "openvino/op/cosh.hpp"

namespace ov {
namespace nvidia_gpu {

class CoshOp : public ElementwiseUnaryOp<ov::op::v0::Cosh, kernel::Cosh> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
