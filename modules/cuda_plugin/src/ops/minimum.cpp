// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "minimum.hpp"

#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

OPERATION_REGISTER(MinimumOp, Minimum)

}  // namespace CUDAPlugin
