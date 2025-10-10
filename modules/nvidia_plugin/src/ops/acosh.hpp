// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/acosh.hpp"
#include "openvino/op/acosh.hpp"

namespace ov {
namespace nvidia_gpu {

class AcoshOp : public ElementwiseUnaryOp<ov::op::v3::Acosh, kernel::Acosh> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
