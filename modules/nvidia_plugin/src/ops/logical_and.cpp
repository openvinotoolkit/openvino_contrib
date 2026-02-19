// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "logical_and.hpp"

#include <cuda_operation_registry.hpp>

namespace ov {
namespace nvidia_gpu {

OPERATION_REGISTER(LogicalAndOp, LogicalAnd);

}  // namespace nvidia_gpu
}  // namespace ov
