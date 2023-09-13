// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.hpp"
#include "kernels/mod.hpp"
#include "openvino/op/mod.hpp"

namespace ov {
namespace nvidia_gpu {

class ModOp : public ElementwiseBinaryOp<ov::op::v1::Mod, kernel::Mod> {
public:
    using ElementwiseBinaryOp::ElementwiseBinaryOp;
};

}  // namespace nvidia_gpu
}  // namespace ov
