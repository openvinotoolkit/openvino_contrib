// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "minimum.hpp"

#include <cuda_operation_registry.hpp>

namespace ov {
namespace nvidia_gpu {

OPERATION_REGISTER(MinimumOp, Minimum)

}  // namespace nvidia_gpu
}  // namespace ov
