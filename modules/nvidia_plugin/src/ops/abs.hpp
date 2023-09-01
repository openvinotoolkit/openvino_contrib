// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include "openvino/op/abs.hpp"

#include "elementwise_unary.hpp"
#include "kernels/abs.hpp"

namespace ov {
namespace nvidia_gpu {

class AbsOp : public ElementwiseUnaryOp<ov::op::v0::Abs, kernel::Abs> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
