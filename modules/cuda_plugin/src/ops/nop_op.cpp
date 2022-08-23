// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nop_op.hpp"

#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

OPERATION_REGISTER(NopOp, Constant);
OPERATION_REGISTER(NopOp, Reshape);
OPERATION_REGISTER(NopOp, Squeeze);
OPERATION_REGISTER(NopOp, Unsqueeze);
OPERATION_REGISTER(NopOp, ConcatOptimized);

}  // namespace CUDAPlugin
