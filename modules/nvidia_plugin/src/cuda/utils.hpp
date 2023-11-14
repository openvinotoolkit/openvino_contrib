// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime_api.h>

namespace CUDA {

inline bool operator==(dim3 rhs, dim3 lhs) { return rhs.x == lhs.x && rhs.y == lhs.y && rhs.z == lhs.z; }

inline bool operator!=(dim3 rhs, dim3 lhs) { return !(rhs == lhs); }

}  // namespace CUDA
