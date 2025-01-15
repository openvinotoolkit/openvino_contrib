// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>

#include "elementwise_unary.hpp"
#include "kernels/acos.hpp"
#include "openvino/op/acos.hpp"

namespace ov {
namespace nvidia_gpu {

class AcosOp : public ElementwiseUnaryOp<ov::op::v0::Acos, kernel::Acos> {
public:
    using ElementwiseUnaryOp::ElementwiseUnaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov

