// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <ngraph/op/prelu.hpp>

#include "elementwise_binary.hpp"
#include "kernels/prelu.hpp"

namespace CUDAPlugin {

class PReluOp : public ElementwiseBinaryOp<ov::op::v0::PRelu, kernel::PRelu> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace CUDAPlugin
