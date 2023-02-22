// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <cuda/float16.hpp>

#include "comparison.hpp"
#include "details/error.hpp"
#include "details/type_validator.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T, typename OP>
static __global__ void comparison(size_t max_size,
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

    dst[idx] = OP::op(left_src[left_idx], right_src[right_idx]);
}

template <typename T, Comparison::Op_t>
struct OpImpl;
template <typename T>
struct OpImpl<T, Comparison::Op_t::GREATER> {
    __device__ static inline bool op(T left, T right) { return left > right; }
};
template <typename T>
struct OpImpl<T, Comparison::Op_t::LESS> {
    __device__ static inline bool op(T left, T right) { return left < right; }
};

Comparison::Comparison(Op_t op_type, Type_t element_type, size_t max_size, size_t num_blocks, size_t threads_per_block)
    : op_type_{op_type},
      element_type_{element_type},
      max_size_{max_size},
      num_blocks_{num_blocks},
      threads_per_block_{threads_per_block} {
    TypeValidator<AllElementTypesSwitch>::check(element_type_);
}

void Comparison::operator()(const cudaStream_t stream,
                            const void* left_src,
                            const void* right_src,
                            const size_t* left_brcst_offsets,
                            const size_t* right_brcst_offsets,
                            const size_t* output_sizes,
                            void* dst) const {
    switch (element_type_) {
        case Type_t::boolean:
            return Call<bool>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return Call<__nv_bfloat16>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::f16:
            return Call<__nv_bfloat16>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
#endif
        case Type_t::f32:
            return Call<float>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::f64:
            return Call<double>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::i8:
            return Call<int8_t>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::i16:
            return Call<int16_t>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::i32:
            return Call<int32_t>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::i64:
            return Call<int64_t>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::u8:
            return Call<uint8_t>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::u16:
            return Call<uint16_t>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::u32:
            return Call<uint32_t>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        case Type_t::u64:
            return Call<uint64_t>(
                op_type_, stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
        default:
            throwIEException(fmt::format("Input element type = {} is not supported by Comparison operation !!",
                                         static_cast<Type_t>(element_type_)));
    }
}

template <typename T>
void Comparison::Call(Comparison::Op_t type,
                      const cudaStream_t stream,
                      const void* left_src,
                      const void* right_src,
                      const size_t* left_brcst_offsets,
                      const size_t* right_brcst_offsets,
                      const size_t* output_sizes,
                      void* dst) const {
    switch (type) {
        case Comparison::Op_t::GREATER:
            Call<T, Comparison::Op_t::GREATER>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
            break;
        case Comparison::Op_t::LESS:
            Call<T, Comparison::Op_t::LESS>(
                stream, left_src, right_src, left_brcst_offsets, right_brcst_offsets, output_sizes, dst);
            break;
        default:
            throwIEException(fmt::format("Input operation = {} is not supported by Comparison operation !!",
                                         static_cast<Type_t>(type)));
    }
}

template <typename T, Comparison::Op_t OP>
void Comparison::Call(const cudaStream_t stream,
                      const void* left_src,
                      const void* right_src,
                      const size_t* left_brcst_offsets,
                      const size_t* right_brcst_offsets,
                      const size_t* output_sizes,
                      void* dst) const {
    comparison<T, OpImpl<T, OP>><<<num_blocks_, threads_per_block_, 0, stream>>>(max_size_,
                                                                                 static_cast<const T*>(left_src),
                                                                                 static_cast<const T*>(right_src),
                                                                                 left_brcst_offsets,
                                                                                 right_brcst_offsets,
                                                                                 output_sizes,
                                                                                 static_cast<bool*>(dst));
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
