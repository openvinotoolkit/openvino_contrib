// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/divide.hpp>

#include "elementwise_binary.hpp"
#include "kernels/divide.hpp"

namespace CUDAPlugin {

using DivideOp = ElementwiseBinaryOp<ngraph::op::v1::Divide, kernel::Divide>;
using PythonDivideOp = ElementwiseBinaryOp<ngraph::op::v1::Divide, kernel::PythonDivide>;

}  // namespace CUDAPlugin
