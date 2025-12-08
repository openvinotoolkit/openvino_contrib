// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/atan.hpp"
#include "openvino/op/atan.hpp"

namespace ov {
namespace nvidia_gpu {

class AtanOp : public ElementwiseUnaryOp<ov::op::v0::Atan, kernel::Atan> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov

