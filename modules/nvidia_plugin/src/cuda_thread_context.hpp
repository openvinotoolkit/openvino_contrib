// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda/blas.hpp"
#include "cuda/dnn.hpp"
#include "cuda/tensor.hpp"

namespace ov {
namespace nvidia_gpu {

class ThreadContext {
    CUDA::Device device_;
    CUDA::Stream stream_;
    CUDA::DnnHandle dnnHandle_;
    CUDA::CuBlasHandle cuBlasHandle_;
    CUDA::CuTensorHandle cuTensorHandle_;

public:
    explicit ThreadContext(CUDA::Device d) : device_{d.setCurrent()} {
        dnnHandle_.setStream(stream_);
        cuBlasHandle_.setStream(stream_);
    }
    CUDA::Device device() const { return device_; }
    const CUDA::Stream& stream() const noexcept { return stream_; }
    const CUDA::DnnHandle& dnnHandle() const noexcept { return dnnHandle_; }
    const CUDA::CuBlasHandle& cuBlasHandle() const noexcept { return cuBlasHandle_; }
    const CUDA::CuTensorHandle& cuTensorHandle() const noexcept { return cuTensorHandle_; }
};

}  // namespace nvidia_gpu
}  // namespace ov
