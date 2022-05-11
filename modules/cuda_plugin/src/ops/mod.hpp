// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/mod.hpp>

#include "elementwise_binary.hpp"
#include "kernels/mod.hpp"

namespace CUDAPlugin {

using ModOp = ElementwiseBinaryOp<ngraph::op::v1::Mod, kernel::Mod>;

}  // namespace CUDAPlugin
