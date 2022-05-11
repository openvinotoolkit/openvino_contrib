// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/float16.hpp>
#include <cuda/math.cuh>
#include <type_traits>

#include "floor_mod.hpp"

namespace CUDAPlugin {
namespace kernel {

namespace {

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
__device__ static inline T floor_mod(T x, T y) {
    // NOTE: return value according to reference implementation in
    // openvino/ngraph/core/reference/include/ngraph/runtime/reference/floor_mod.hpp
    // const double divisor = static_cast<double>(y);
    // return x - y * std::floor(x / divisor);
    // for y = 0 it is static_cast<T>(-nan) which is minimal limit for integers
    if (y == 0) {
        return CUDA::math::limit_min<T>();
    }
    const T mod = x % y;
    if (mod == 0) {
        return mod;
    }
    T div = x / y;
    if ((x < 0) != (y < 0)) {
        --div;
    }
    return x - y * div;
}

template <typename T, std::enable_if_t<!std::is_integral<T>::value, bool> = true>
__device__ static inline T floor_mod(T x, T y) {
    return x - y * CUDA::math::floor(x / y);
}

}  // namespace

template <typename T>
struct FloorModOpImpl {
    __device__ static inline T op(T in0, T in1) { return floor_mod(in0, in1); }
};

FloorMod::FloorMod(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
    : impl_{element_type, out_num_elements, max_threads_per_block} {}

void FloorMod::operator()(cudaStream_t stream,
                          const void* in0,
                          const NumpyBroadcastMapper& in0_mapper,
                          const void* in1,
                          const NumpyBroadcastMapper& in1_mapper,
                          void* out) const {
    impl_(stream, in0, in0_mapper, in1, in1_mapper, out);
}

}  // namespace kernel
}  // namespace CUDAPlugin
