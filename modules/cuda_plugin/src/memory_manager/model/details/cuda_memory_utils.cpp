// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/runtime.hpp>

#include "memory_manager/model/cuda_memory_model.hpp"

namespace CUDAPlugin {

size_t applyAllignment(size_t value) {
    constexpr size_t allignment = CUDA::memoryAlignment;
    return (value % allignment) == 0 ? value : value - (value % allignment) + allignment;
}

}  // namespace CUDAPlugin
