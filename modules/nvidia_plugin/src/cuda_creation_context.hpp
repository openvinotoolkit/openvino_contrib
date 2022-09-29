// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_config.hpp>

#include "cuda/blas.hpp"
#include "cuda/dnn.hpp"
#include "cuda/tensor.hpp"

namespace ov {
namespace nvidia_gpu {

class CreationContext {
    CUDA::Device device_;
    CUDA::DnnHandle dnn_handle_;
    bool op_bench_option_;

public:
    explicit CreationContext(CUDA::Device d, bool opBenchOption)
        : device_{d.setCurrent()}, op_bench_option_{opBenchOption} {}
    CUDA::Device device() const { return device_; }
    const CUDA::DnnHandle& dnnHandle() const { return dnn_handle_; }
    bool opBenchOption() const noexcept { return op_bench_option_; }
};

}  // namespace nvidia_gpu
}  // namespace ov
