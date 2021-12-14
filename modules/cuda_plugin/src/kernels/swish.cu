// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "elementwise.cuh"
#include "swish.hpp"
#include "tensor_helpers.hpp"

#if defined __CUDACC__
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#endif

namespace CUDAPlugin {
namespace kernel {

template <typename T>
struct SwishOpImpl {
    __device__ static inline T op(T x, T beta) { return x / (static_cast<T>(1) + std::exp(-x * beta)); }
};

template <>
struct SwishOpImpl<__half> {
#if __CUDA_ARCH__ >= 530
    __device__ static inline __half op(__half x, __half beta) { return x / (__half(1.0f) + ::hexp(-x * beta)); }
#else
    __device__ static inline __half op(float x, float beta) { return x / (1.0f + ::expf(-x * beta)); }
#endif
};

#if CUDA_VERSION >= 11000
template <>
struct SwishOpImpl<__nv_bfloat16> {
#if __CUDA_ARCH__ >= 800
    __device__ static inline __nv_bfloat16 op(__nv_bfloat16 x, __nv_bfloat16 beta) {
        return x / (__nv_bfloat16(1.0f) + ::hexp(-x * beta));
    }
#else
    __device__ static inline __nv_bfloat16 op(float x, float beta) { return x / (1.0f + ::expf(-x * beta)); }
#endif
};
#endif

Swish::Swish(Type_t element_type, size_t max_threads_per_block)
    : element_type_{element_type}, max_threads_per_block_{max_threads_per_block} {}

void Swish::operator()(cudaStream_t stream, const void* in, void* out, size_t num_elements, double beta) const {
#if CUDA_VERSION >= 11000
    using SupportedElementTypes = ElementTypesSwitch<Type_t::f16, Type_t::f32, Type_t::f64, Type_t::bf16>;
#else
    using SupportedElementTypes = ElementTypesSwitch<Type_t::f16, Type_t::f32, Type_t::f64>;
#endif
    using Helper = ElementwiseHelper<SupportedElementTypes, SwishOpImpl>;
    Helper helper{element_type_, max_threads_per_block_};
    helper.unaryOperator(stream, in, out, num_elements, beta);
}

}  // namespace kernel
}  // namespace CUDAPlugin
