// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/asin.hpp"
#include "openvino/op/asin.hpp"

namespace ov {
namespace nvidia_gpu {

class AsinOp : public ElementwiseUnaryOp<ov::op::v0::Asin, kernel::Asin> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
