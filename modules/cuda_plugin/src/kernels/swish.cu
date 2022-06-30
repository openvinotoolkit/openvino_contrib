// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/float16.hpp>

#include "swish.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
struct SwishOpImpl {
    __device__ static inline T op(T x, double beta) {
        return x / (static_cast<T>(1.0f) + ::exp(-x * static_cast<T>(beta)));
    }
};

template <>
struct SwishOpImpl<float> {
    __device__ static inline float op(float x, double beta) {
        // TODO: Possible optimization: use ::__expf(float) instead at the cost of precision
        return x / (1.0f + ::expf(-x * static_cast<float>(beta)));
    }
};

template <>
struct SwishOpImpl<__half> {
    __device__ static inline __half op(__half x, double beta) {
#ifdef CUDA_HAS_HALF_MATH
        return x / (static_cast<__half>(1.0f) + ::hexp(-x * static_cast<__half>(beta)));
#else
        return SwishOpImpl<float>::op(static_cast<float>(x), beta);
#endif  // CUDA_HAS_HALF_MATH
    }
};

#ifdef CUDA_HAS_BF16_TYPE
template <>
struct SwishOpImpl<__nv_bfloat16> {
    __device__ static inline __nv_bfloat16 op(__nv_bfloat16 x, double beta) {
#ifdef CUDA_HAS_BF16_MATH
        return x / (static_cast<__nv_bfloat16>(1.0f) + ::hexp(-x * static_cast<__nv_bfloat16>(beta)));
#else
        return SwishOpImpl<float>::op(static_cast<float>(x), beta);
#endif  // CUDA_HAS_BF16_MATH
    }
};
#endif  // CUDA_HAS_BF16_TYPE

Swish::Swish(Type_t element_type, size_t max_threads_per_block, size_t num_elements, double beta)
    : ewu_{element_type, max_threads_per_block, num_elements}, beta_{beta} {}

void Swish::operator()(cudaStream_t stream, const void* in, void* out) const { ewu_(stream, in, out, beta_); }

}  // namespace kernel
}  // namespace CUDAPlugin
