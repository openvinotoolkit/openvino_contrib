// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/divide.hpp>

#include "elementwise_binary.hpp"
#include "kernels/divide.hpp"

namespace CUDAPlugin {

class DivideOp : public ElementwiseBinaryOp<ngraph::op::v1::Divide, kernel::Divide> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};
class PythonDivideOp : public ElementwiseBinaryOp<ngraph::op::v1::Divide, kernel::PythonDivide> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace CUDAPlugin
