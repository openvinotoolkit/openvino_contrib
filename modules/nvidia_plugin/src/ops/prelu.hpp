// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include "openvino/op/prelu.hpp"

#include "elementwise_binary.hpp"
#include "kernels/prelu.hpp"

namespace ov {
namespace nvidia_gpu {

class PReluOp : public ElementwiseBinaryOp<ov::op::v0::PRelu, kernel::PRelu> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
