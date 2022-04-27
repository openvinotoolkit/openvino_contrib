// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "maximum.hpp"

#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

OPERATION_REGISTER(MaximumOp, Maximum)

}  // namespace CUDAPlugin
