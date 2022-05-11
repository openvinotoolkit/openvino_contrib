// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/add.hpp>

#include "elementwise_binary.hpp"
#include "kernels/add.hpp"

namespace CUDAPlugin {

using AddCudaOp = ElementwiseBinaryOp<ngraph::op::v1::Add, kernel::Add>;

}  // namespace CUDAPlugin
