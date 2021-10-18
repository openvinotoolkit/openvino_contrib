// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>
#include <fmt/format.h>

#include <error.hpp>
#include <gsl/gsl_assert>

#include "greater.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
static __global__ void greater(size_t max_size,
                               const T* left_src,
                               const T* right_src,
                               const size_t* left_brcst_offsets,
                               const size_t* right_brcst_offsets,
                               const size_t* output_sizes,
                               bool* dst) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_size) {
        return;
    }

    // calc N, C, D, H, W indexes
    enum { N, C, D, H, W };
    const unsigned n = idx / output_sizes[N];
    const unsigned n_size = n * output_sizes[N];
    const unsigned c = (idx - n_size) / output_sizes[C];
    const unsigned c_size = c * output_sizes[C];
    const unsigned d = (idx - n_size - c_size) / output_sizes[D];
    const unsigned d_size = d * output_sizes[D];
    const unsigned h = (idx - n_size - c_size - d_size) / output_sizes[H];
    const unsigned h_size = h * output_sizes[H];
    const unsigned w = (idx - n_size - c_size - d_size - h_size) / output_sizes[W];

    const unsigned left_idx = left_brcst_offsets[N] * n + left_brcst_offsets[C] * c + left_brcst_offsets[D] * d +
                              left_brcst_offsets[H] * h + left_brcst_offsets[W] * w;
    const unsigned right_idx = right_brcst_offsets[N] * n + right_brcst_offsets[C] * c + right_brcst_offsets[D] * d +
                               right_brcst_offsets[H] * h + right_brcst_offsets[W] * w;

    dst[idx] = left_src[left_idx] > right_src[right_idx];
}

Greater::Greater(Type_t element_type, size_t max_size, size_t num_blocks, size_t threads_per_block)
    : element_type_{element_type},
      max_size_{max_size},
      num_blocks_{num_blocks},
      threads_per_block_{threads_per_block} {}

void Greater::operator()(const cudaStream_t stream,
                         const void* left_src,
                         const void* right_src,
                         const size_t* left_brcst_offsets,
                         const size_t* right_brcst_offsets,
                         const size_t* output_sizes,
                         void* dst) const {
    switch (element_type_) {
        case Type_t::boolean:
            return Call<bool>(stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
#if CUDA_VERSION >= 11000
        case Type_t::bf16:
            return Call<__nv_bfloat16>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::f16:
            return Call<__nv_bfloat16>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
#endif
        case Type_t::f32:
            return Call<float>(stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::f64:
            return Call<double>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::i8:
            return Call<int8_t>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::i16:
            return Call<int16_t>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::i32:
            return Call<int32_t>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::i64:
            return Call<int64_t>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::u8:
            return Call<uint8_t>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::u16:
            return Call<uint16_t>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::u32:
            return Call<uint32_t>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::u64:
            return Call<uint64_t>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        default:
            throwIEException(fmt::format("Input element type = {} is not supported by Split operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

template <typename T>
void Greater::Call(const cudaStream_t stream,
                   const void* left_src,
                   const void* right_src,
                   const size_t* left_brcst_offsets,
                   const size_t* right_brcst_offsets,
                   const size_t* output_sizes,
                   void* dst) const {
    greater<T><<<num_blocks_, threads_per_block_, 0, stream>>>(max_size_,
                                                               static_cast<const T*>(left_src),
                                                               static_cast<const T*>(right_src),
                                                               left_brcst_offsets,
                                                               right_brcst_offsets,
                                                               output_sizes,
                                                               static_cast<bool*>(dst));
}

}  // namespace kernel
}  // namespace CUDAPlugin
