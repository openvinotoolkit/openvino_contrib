// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <ngraph/op/prelu.hpp>

#include "elementwise_binary.hpp"
#include "kernels/prelu.hpp"

namespace CUDAPlugin {

using PReluOp = ElementwiseBinaryOp<ngraph::op::v0::PRelu, kernel::PRelu>;

}  // namespace CUDAPlugin
