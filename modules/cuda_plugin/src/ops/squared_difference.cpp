// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "squared_difference.hpp"

#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

OPERATION_REGISTER(SquaredDifferenceOp, SquaredDifference)

}  // namespace CUDAPlugin
