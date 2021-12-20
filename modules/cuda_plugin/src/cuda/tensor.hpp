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

class CuTensorHandle : public Handle<cutensorHandle_t> {
public:
    CuTensorHandle() : Handle(cutensorInit, nullptr) {}
};

}  // namespace CUDA
