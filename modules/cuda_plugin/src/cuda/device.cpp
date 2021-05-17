// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <details/ie_exception.hpp>
#include <fmt/format.h>

#include "device.hpp"
#include "props.hpp"

namespace CUDAPlugin {

int CudaDevice::GetNumDevices() {
    int count;
    auto err = cudaGetDeviceCount(&count);
    if (cudaSuccess != err) {
        THROW_IE_EXCEPTION << "Cuda error: " << cudaGetErrorString(err);
    }
    return count;
}

std::vector<cudaDeviceProp>
CudaDevice::GetAllDevicesProp() {
    const int devCount = GetNumDevices();
    std::vector<cudaDeviceProp> devices(devCount);
    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp prop;
        auto err = cudaGetDeviceProperties(&prop, i);
        if (cudaSuccess != err) {
            THROW_IE_EXCEPTION << "Cuda error: " << cudaGetErrorString(err);
        }
        devices[i] = prop;
    }
    return devices;
}

size_t
CudaDevice::GetDeviceConcurrentKernels(const cudaDeviceProp& devProp) {
    size_t concurrentKernels = 1;
    const bool canConcurrentKernels = devProp.concurrentKernels;
    if (canConcurrentKernels) {
        const int majorComputeCompatibility = devProp.major;
        const int minorComputeCompatibility = devProp.minor;
        const std::string computeCompatibility =
            std::to_string(majorComputeCompatibility) + "." +
                std::to_string(minorComputeCompatibility);
        concurrentKernels = cudaConcurrentKernels.at(computeCompatibility);
    }
    return concurrentKernels;
}

unsigned
CudaDevice::GetMaxGridBlockSizeParams(unsigned deviceId) {
    std::vector<cudaDeviceProp> devices = GetAllDevicesProp();
    if (deviceId >= devices.size()) {
        THROW_IE_EXCEPTION << fmt::format("deviceId {} is out of range");
    }
    return devices[deviceId].maxThreadsPerBlock;
}

} // namespace CUDAPlugin
