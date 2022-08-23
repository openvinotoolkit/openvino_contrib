// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subtract.hpp"

#include <cuda_operation_registry.hpp>

namespace ov {
namespace nvidia_gpu {

OPERATION_REGISTER(SubtractOp, Subtract)

}  // namespace nvidia_gpu
}  // namespace ov
