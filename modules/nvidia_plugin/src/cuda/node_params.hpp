// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime_api.h>

#include <vector>

namespace CUDA {

struct NodeParams {
    NodeParams(void* kernel, dim3 gridDim, dim3 blockDim) {
        knp_.func = kernel;
        knp_.gridDim = gridDim;
        knp_.blockDim = blockDim;
        knp_.sharedMemBytes = 0;
        knp_.kernelParams = nullptr;
        knp_.extra = nullptr;
        ptrs_.reserve(20);
    }

    template <typename T>
    void add_args(const T& value) {
        ptrs_.emplace_back(const_cast<T*>(&value));
    }

    template <typename T, typename... Args>
    void add_args(const T& arg, Args&&... args) {
        add_args(std::forward<const T&>(arg));
        add_args(std::forward<Args>(args)...);
    };

    const cudaKernelNodeParams& get_knp() {
        knp_.kernelParams = ptrs_.data();
        return knp_;
    }

    void reset_args() { ptrs_.clear(); }

private:
    std::vector<void*> ptrs_;
    cudaKernelNodeParams knp_;
};

}  // namespace CUDA
