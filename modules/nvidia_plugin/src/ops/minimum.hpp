// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/minimum.hpp"
#include "openvino/op/minimum.hpp"

namespace ov {
namespace nvidia_gpu {

class MinimumOp : public ElementwiseBinaryOp<ov::op::v1::Minimum, kernel::Minimum> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
