// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/soft_sign.hpp"
#include "openvino/op/softsign.hpp"

namespace ov {
namespace nvidia_gpu {

class SoftSignOp : public ElementwiseUnaryOp<ov::op::v9::SoftSign, kernel::SoftSign> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
