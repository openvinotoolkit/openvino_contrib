// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/float16.hpp>
#include <cuda/math.cuh>
#include <type_traits>

#include "divide.hpp"

namespace CUDAPlugin {
namespace kernel {

namespace {

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
__device__ static inline T div(T x, T y) {
    // NOTE: as by default CUDA math returns -1 for integer division by 0
    // this behavior was implemented to be more similar to floating point division:
    // inf ->limit_max, -inf ->limit_min, nan -> limit_min
    if (y == 0) {
        return x > 0 ? CUDA::math::limit_max<T>() : CUDA::math::limit_min<T>();
    }
    return x / y;
}

template <typename T, std::enable_if_t<!std::is_integral<T>::value, bool> = true>
__device__ static inline T div(T x, T y) {
    return x / y;
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
__device__ static inline T python_div(T x, T y) {
    // NOTE: as by default CUDA math returns -1 for integer division by 0
    // this behavior was implemented to be more similar to floating point division:
    // inf ->limit_max, -inf ->limit_min, nan -> limit_min
    if (y == 0) {
        return x > 0 ? CUDA::math::limit_max<T>() : CUDA::math::limit_min<T>();
    }
    const T div = x / y;
    const T mod = x % y;
    if (mod != 0 && (x < 0) != (y < 0)) {
        return div - 1;
    }
    return div;
}

template <typename T, std::enable_if_t<!std::is_integral<T>::value, bool> = true>
__device__ static inline T python_div(T x, T y) {
    return div(x, y);
}

}  // namespace

template <typename T>
struct DivideOpImpl {
    __device__ static inline T op(T in0, T in1) { return div(in0, in1); }
};

template <typename T>
struct PythonDivideOpImpl {
    __device__ static inline T op(T in0, T in1) { return python_div(in0, in1); }
};

Divide::Divide(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
    : impl_{element_type, out_num_elements, max_threads_per_block} {}

void Divide::operator()(cudaStream_t stream,
                        const void* in0,
                        const NumpyBroadcastMapper& in0_mapper,
                        const void* in1,
                        const NumpyBroadcastMapper& in1_mapper,
                        void* out) const {
    impl_(stream, in0, in0_mapper, in1, in1_mapper, out);
}

PythonDivide::PythonDivide(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
    : impl_{element_type, out_num_elements, max_threads_per_block} {}

void PythonDivide::operator()(cudaStream_t stream,
                              const void* in0,
                              const NumpyBroadcastMapper& in0_mapper,
                              const void* in1,
                              const NumpyBroadcastMapper& in1_mapper,
                              void* out) const {
    impl_(stream, in0, in0_mapper, in1, in1_mapper, out);
}

}  // namespace kernel
}  // namespace CUDAPlugin
