// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/maximum.hpp>

#include "elementwise_binary.hpp"
#include "kernels/maximum.hpp"

namespace CUDAPlugin {

using MaximumOp = ElementwiseBinaryOp<ngraph::op::v1::Maximum, kernel::Maximum>;

}  // namespace CUDAPlugin
