// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/multiply.hpp>

#include "elementwise_binary.hpp"
#include "kernels/multiply.hpp"

namespace CUDAPlugin {

using MultiplyCudaOpBase = ElementwiseBinaryOp<ngraph::op::v1::Multiply, kernel::Multiply>;
class MultiplyCudaOp : public MultiplyCudaOpBase {
public:
    using NodeOp = ngraph::op::v1::Multiply;
    MultiplyCudaOp(const CreationContext& context,
                   const NodeOp& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds);
};

}  // namespace CUDAPlugin
