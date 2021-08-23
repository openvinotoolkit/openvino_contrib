// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runtime.hpp"
#include <cutensor.h>

inline void throwIfError(cutensorStatus_t err,
                         const std::experimental::source_location& location =
                             std::experimental::source_location::current()) {
    if (err != CUTENSOR_STATUS_SUCCESS)
        CUDAPlugin::throwIEException(cutensorGetErrorString(err), location);
}

inline void logIfError(cutensorStatus_t err,
                       const std::experimental::source_location& location =
                           std::experimental::source_location::current()) {
    if (err != CUTENSOR_STATUS_SUCCESS)
        CUDAPlugin::logError(cutensorGetErrorString(err), location);
}

namespace CUDA {
namespace cutensor::details {
constexpr inline cudaError_t dummyDestroy(cutensorHandle_t) { return cudaSuccess; }
}

class CuTensorHandle : public UniqueBase<cutensorInit, cutensor::details::dummyDestroy> {
};

}  // namespace CUDA
