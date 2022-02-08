// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda.h>

namespace CUDA {

template <typename T, unsigned N>
class Array {
public:
    __device__ explicit Array() = default;

    template <typename... TArgs>
    __device__ explicit Array(TArgs... args) : data_{args...} {
        static_assert(N >= sizeof...(TArgs), "More init values that can store data");
    }

    __device__ T& operator[](size_t i) { return data_[i]; }

    __device__ const T& operator[](size_t i) const { return data_[i]; }

    __device__ const T& at(size_t i) const {
#ifndef NDEBUG
        if (i >= size()) {
            __trap();
        }
#endif
        return data_[i];
    }

    __device__ constexpr size_t size() const { return N; }

private:
    T data_[N];
};

}  // namespace CUDA
