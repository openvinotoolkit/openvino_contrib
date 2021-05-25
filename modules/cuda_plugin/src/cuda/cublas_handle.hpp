// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fmt/format.h>
#include <details/ie_no_copy.hpp>
#include <cublas_v2.h>
#include <cublas_api.h>
#include <fmt/format.h>
#include <ie_extension.h>
#include "runtime.hpp"

namespace CUDA {

class CuBlasHandle : public UniqueBase<cublasCreate, cublasDestroy> {
 public:
    CuBlasHandle() = default;
    explicit CuBlasHandle(Stream& stream) {
        call(cublasSetStream, stream.get());
    }

    template <typename ... TArgs>
    void call(cublasStatus_t(*cublasApiFunc)(cublasHandle_t, TArgs...), TArgs... args) const {
        throwIfError(cublasApiFunc(get(), args...));
    }
};

} // namespace CUDA
