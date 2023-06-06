// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/multiply.hpp"

#include "elementwise_binary.hpp"
#include "kernels/multiply.hpp"

namespace ov {
namespace nvidia_gpu {

using MultiplyCudaOpBase = ElementwiseBinaryOp<ov::op::v1::Multiply, kernel::Multiply>;
class MultiplyCudaOp : public MultiplyCudaOpBase {
public:
    using NodeOp = ov::op::v1::Multiply;
    MultiplyCudaOp(const CreationContext& context,
                   const NodeOp& node,
                   IndexCollection&& inputIds,
                   IndexCollection&& outputIds);
};

}  // namespace nvidia_gpu
}  // namespace ov
