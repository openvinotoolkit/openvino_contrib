// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime_api.h>
#include <vector>

namespace CUDA {

class CudaDevice {
 public:
    static int GetNumDevices();
    static std::vector<cudaDeviceProp> GetAllDevicesProp();
    static size_t GetDeviceConcurrentKernels(const cudaDeviceProp& devProp);
    static unsigned GetMaxGridBlockSizeParams(unsigned deviceId);
    static size_t GetMemoryAllignment();
};

} // namespace CUDAPlugin
