// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "floor_mod.hpp"

#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

OPERATION_REGISTER(FloorModOp, FloorMod)

}  // namespace CUDAPlugin
