// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/add.hpp>

#include "elementwise_binary.hpp"
#include "kernels/add.hpp"

namespace ov {
namespace nvidia_gpu {

class AddCudaOp : public ElementwiseBinaryOp<ov::op::v1::Add, kernel::Add> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
