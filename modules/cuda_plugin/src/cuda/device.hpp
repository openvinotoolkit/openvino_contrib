// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime_api.h>

namespace CUDAPlugin {

class CudaDevice {
 public:
    static int GetNumDevices();
    static std::vector<cudaDeviceProp> GetAllDevicesProp();
    static size_t GetDeviceConcurrentKernels(const cudaDeviceProp& devProp);
};

}
