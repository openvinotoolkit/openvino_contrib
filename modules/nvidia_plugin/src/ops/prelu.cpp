// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prelu.hpp"

#include "cuda_operation_registry.hpp"

namespace ov {
namespace nvidia_gpu {

OPERATION_REGISTER(PReluOp, PRelu)

}  // namespace nvidia_gpu
}  // namespace ov
