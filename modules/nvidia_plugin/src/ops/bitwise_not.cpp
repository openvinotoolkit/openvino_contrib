// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bitwise_not.hpp"

#include <cuda_operation_registry.hpp>

namespace ov {
namespace nvidia_gpu {

OPERATION_REGISTER(BitwiseNotOp, BitwiseNot);

}  // namespace nvidia_gpu
}  // namespace ov
