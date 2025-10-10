// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/atanh.hpp"
#include "openvino/op/atanh.hpp"

namespace ov {
namespace nvidia_gpu {

class AtanhOp : public ElementwiseUnaryOp<ov::op::v3::Atanh, kernel::Atanh> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov

