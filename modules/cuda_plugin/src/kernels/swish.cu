// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "elementwise.cuh"
#include "swish.hpp"
#include "tensor_helpers.hpp"

#ifdef __CUDACC__
#include <cuda_fp16.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
#endif  // __CUDACC__

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
#if __CUDA_ARCH__ >= 530
        return x / (static_cast<__half>(1.0f) + ::hexp(-x * static_cast<__half>(beta)));
#else
        return SwishOpImpl<float>::op(static_cast<float>(x), beta);
#endif
    }
};

#if CUDA_VERSION >= 11000
template <>
struct SwishOpImpl<__nv_bfloat16> {
    __device__ static inline __nv_bfloat16 op(__nv_bfloat16 x, double beta) {
#if __CUDA_ARCH__ >= 800
        return x / (static_cast<__nv_bfloat16>(1.0f) + ::hexp(-x * static_cast<__nv_bfloat16>(beta)));
#else
        return SwishOpImpl<float>::op(static_cast<float>(x), beta);
#endif  // __CUDA_ARCH__ >= 800
    }
};
#endif  // CUDA_VERSION >= 11000

Swish::Swish(Type_t element_type, size_t max_threads_per_block)
    : element_type_{element_type}, max_threads_per_block_{max_threads_per_block} {}

void Swish::operator()(cudaStream_t stream, const void* in, size_t num_elements, void* out, double beta) const {
    using SupportedElementTypes = ElementTypesSwitch<Type_t::f16,
                                                     Type_t::f32,
                                                     Type_t::f64,
#if CUDA_VERSION >= 11000
                                                     Type_t::bf16
#endif  // CUDA_VERSION >= 11000
                                                     >;
    using Switcher = ElementwiseUnary<SupportedElementTypes, SwishOpImpl>;
    Switcher switcher{element_type_, max_threads_per_block_};
    switcher(stream, in, num_elements, out, beta);
}

}  // namespace kernel
}  // namespace CUDAPlugin
