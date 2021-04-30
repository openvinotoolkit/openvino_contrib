// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda.h>
#include "cuda_plugin.hpp"

namespace CUDAPlugin {
template <>
std::string Plugin::getCudaAttribute<Plugin::cuda_attribute::name, std::string>() const {
    char buffer[1024];
    CUresult error = cuInit(0);
    if ( error != CUDA_SUCCESS )
        THROW_IE_EXCEPTION << "CUDA INIT ERROR " << error;
    error = cuDeviceGetName(buffer, sizeof(buffer)-1, cudaDeviceID());
    if ( error != CUDA_SUCCESS )
        THROW_IE_EXCEPTION << "CUDA ERROR " << error;
    return { buffer, std::strlen(buffer) };
}
} // namespace CUDAPlugin
