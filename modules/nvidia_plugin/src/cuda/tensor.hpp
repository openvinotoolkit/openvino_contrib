// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cutensor.h>

#include "runtime.hpp"

inline void throwIfError(
    cutensorStatus_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != CUTENSOR_STATUS_SUCCESS) ov::nvidia_gpu::throw_ov_exception(cutensorGetErrorString(err), location);
}

inline void logIfError(
    cutensorStatus_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != CUTENSOR_STATUS_SUCCESS) ov::nvidia_gpu::logError(cutensorGetErrorString(err), location);
}

namespace CUDA {

class CuTensorHandle : public Handle<cutensorHandle_t> {
public:
#if defined(CUTENSOR_VERSION) && CUTENSOR_VERSION >= 20000
    CuTensorHandle() : Handle(cutensorCreate, cutensorDestroy) {}
#else
    CuTensorHandle() : Handle(cutensorInit, nullptr) {}
#endif
};

}  // namespace CUDA
