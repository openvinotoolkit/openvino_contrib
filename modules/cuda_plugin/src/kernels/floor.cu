// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/float16.hpp>

#include "floor.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
struct FloorOpImpl {
    __device__ static inline T op(T x) { return ::floor(x); }
};

template <>
struct FloorOpImpl<float> {
    __device__ static inline float op(float x) { return ::floorf(x); }
};

template <>
struct FloorOpImpl<__half> {
    __device__ static inline __half op(__half x) {
#ifdef CUDA_HAS_HALF_MATH
        return ::hfloor(x);
#else
        return FloorOpImpl<float>::op(static_cast<float>(x));
#endif  // CUDA_HAS_HALF_MATH
    }
};

#ifdef CUDA_HAS_BF16_TYPE
template <>
struct FloorOpImpl<__nv_bfloat16> {
    __device__ static inline __nv_bfloat16 op(__nv_bfloat16 x) {
#ifdef CUDA_HAS_BF16_MATH
        return ::hfloor(x);
#else
        return FloorOpImpl<float>::op(static_cast<float>(x));
#endif  // CUDA_HAS_BF16_MATH
    }
};
#endif  // CUDA_HAS_BF16_TYPE

Floor::Floor(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
    : ewu_{element_type, max_threads_per_block, num_elements} {}

void Floor::operator()(cudaStream_t stream, const void* in, void* out) const { ewu_(stream, in, out); }

}  // namespace kernel
}  // namespace CUDAPlugin
