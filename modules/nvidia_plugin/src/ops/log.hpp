// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include "openvino/op/log.hpp"

#include "elementwise_unary.hpp"
#include "kernels/log.hpp"

namespace ov {
namespace nvidia_gpu {

class LogOp : public ElementwiseUnaryOp<ov::op::v0::Log, kernel::Log> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
