// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime_api.h>

#include <cuda/utils.hpp>
#include <vector>

namespace CUDA {

struct NodeParams {
    NodeParams(void* kernel, dim3 gridDim, dim3 blockDim) : knp_{kernel, gridDim, blockDim, 0u, nullptr, nullptr} {
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

    friend bool operator==(const NodeParams& lhs, const NodeParams& rhs);

private:
    std::vector<void*> ptrs_;
    cudaKernelNodeParams knp_;
};

inline bool operator==(const NodeParams& lhs, const NodeParams& rhs) {
    return lhs.ptrs_ == rhs.ptrs_ && rhs.knp_.func == lhs.knp_.func && rhs.knp_.gridDim == lhs.knp_.gridDim &&
           rhs.knp_.blockDim == lhs.knp_.blockDim && rhs.knp_.sharedMemBytes == lhs.knp_.sharedMemBytes &&
           rhs.knp_.extra == lhs.knp_.extra;
}

}  // namespace CUDA
