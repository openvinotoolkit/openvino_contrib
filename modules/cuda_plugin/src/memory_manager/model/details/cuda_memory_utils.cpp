// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_manager/model/cuda_memory_model.hpp"
#include "cuda/device.hpp"

namespace CUDAPlugin {

size_t applyAllignment(size_t value) {
    const size_t allignment = CudaDevice::GetMemoryAllignment();
    return (value % allignment) == 0 ? value : value - (value % allignment) + allignment;
}

}  // namespace CUDAPlugin
