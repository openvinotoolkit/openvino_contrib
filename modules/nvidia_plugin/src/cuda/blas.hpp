// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cublas_v2.h>

#include "runtime.hpp"

inline std::string cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CuBlas Status Success";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CuBlas Status Not Initialized";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CuBlas Status Allocation Failed";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CuBlas Status Invalid Value";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CuBlas Status Architecture Mismatched";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CuBlas Status mapping Error";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CuBlas Status Execution Failed";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CuBlas Status Internal Error";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CuBlas Status Not Supported";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CuBlas Status License Error";
        default:
            return "CuBlas Unknown Status";
    }
}

inline void throwIfError(
    cublasStatus_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != CUBLAS_STATUS_SUCCESS) ov::nvidia_gpu::throw_ov_exception(cublasGetErrorString(err), location);
}

inline void logIfError(
    cublasStatus_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != CUBLAS_STATUS_SUCCESS) ov::nvidia_gpu::logError(cublasGetErrorString(err), location);
}

namespace CUDA {

class CuBlasHandle : public Handle<cublasHandle_t> {
public:
    CuBlasHandle() : Handle((cublasCreate), cublasDestroy) {}
    void setStream(Stream& stream) { throwIfError(cublasSetStream(get(), stream.get())); }
};

}  // namespace CUDA
