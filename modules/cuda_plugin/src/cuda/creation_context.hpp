// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_config.hpp>

#include "dnn.hpp"

namespace CUDA {

class CreationContext {
    CUDA::Device device_;
    CUDA::DnnHandle dnnHandle_;
    bool optimize_option_;

public:
    explicit CreationContext(CUDA::Device d, bool optimizeOption)
        : device_{d.setCurrent()}, optimize_option_{optimizeOption} {}
    CUDA::Device device() const { return device_; }
    auto& dnnHandle() const { return dnnHandle_; }
    bool optimizeOption() const noexcept { return optimize_option_; }
};

}  // namespace CUDA
