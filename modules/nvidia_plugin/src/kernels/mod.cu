// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/float16.hpp>
#include <cuda/math.cuh>
#include <type_traits>

#include "mod.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace {

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
__device__ static inline T mod(T x, T y) {
    // NOTE: for y = 0 and any x returning int value similar to nan
    if (y == 0) {
        return CUDA::math::limit_min<T>();
    }
    return x % y;
}

template <typename T, std::enable_if_t<!std::is_integral<T>::value, bool> = true>
__device__ static inline T mod(T x, T y) {
    return x - y * CUDA::math::trunc(x / y);
}

}  // namespace

template <typename T>
struct ModOpImpl {
    __device__ static inline T op(T in0, T in1) { return mod(in0, in1); }
};

Mod::Mod(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
    : impl_{element_type, out_num_elements, max_threads_per_block} {}

void Mod::operator()(cudaStream_t stream,
                     const void* in0,
                     const NumpyBroadcastMapper& in0_mapper,
                     const void* in1,
                     const NumpyBroadcastMapper& in1_mapper,
                     void* out) const {
    impl_(stream, in0, in0_mapper, in1, in1_mapper, out);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
