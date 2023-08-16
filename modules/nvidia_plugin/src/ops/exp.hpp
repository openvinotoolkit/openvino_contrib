// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include "openvino/op/exp.hpp"

#include "elementwise_unary.hpp"
#include "kernels/exp.hpp"

namespace ov {
namespace nvidia_gpu {

class ExpOp : public ElementwiseUnaryOp<ov::op::v0::Exp, kernel::Exp> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
