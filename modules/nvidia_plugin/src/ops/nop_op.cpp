// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nop_op.hpp"

#include <cuda_operation_registry.hpp>

namespace ov {
namespace nvidia_gpu {

OPERATION_REGISTER(NopOp, Constant);
OPERATION_REGISTER(NopOp, Reshape);
OPERATION_REGISTER(NopOp, Squeeze);
OPERATION_REGISTER(NopOp, Unsqueeze);
OPERATION_REGISTER(NopOp, ConcatOptimized);

}  // namespace nvidia_gpu
}  // namespace ov
