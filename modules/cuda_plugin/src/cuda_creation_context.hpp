// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_config.hpp>

#include "cuda/blas.hpp"
#include "cuda/dnn.hpp"
#include "cuda/tensor.hpp"

namespace CUDAPlugin {

class CreationContext {
    CUDA::Device device_;
    CUDA::DnnHandle dnn_handle_;
    bool optimize_option_;

public:
    explicit CreationContext(CUDA::Device d, bool optimizeOption)
        : device_{d.setCurrent()}, optimize_option_{optimizeOption} {}
    CUDA::Device device() const { return device_; }
    const CUDA::DnnHandle& dnnHandle() const { return dnn_handle_; }
    bool optimizeOption() const noexcept { return optimize_option_; }
};

}  // namespace CUDAPlugin
