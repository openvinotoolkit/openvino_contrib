// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cuda_fp16.h>
#include <fmt/format.h>

#include <cstdint>

#include "error.hpp"
#include "pad.cuh"

namespace CUDAPlugin {
namespace kernel {

// TODO: Would be optimized adding template specializations. Left without this optimization in sake of readability.
template <int PayloadRank>
static inline __device__ bool is_padding(const long ndim_src_indices[PayloadRank],
                                         const std::size_t src_shape[PayloadRank]) {
#pragma unroll PayloadRank
    for (int i = 0; i < PayloadRank; i++) {
        const auto src_dim = ndim_src_indices[i];
        if (src_dim < 0 || src_dim >= src_shape[i]) return true;
    }
    return false;
}

template <int PayloadRank>
static inline __device__ void ndim_dst_indices2ndim_src_indices(const long ndim_dst_indices[PayloadRank],
                                                                const std::size_t pad_begin[PayloadRank],
                                                                long ndim_src_indices[PayloadRank]) {
#pragma unroll PayloadRank
    for (int i = 0; i < PayloadRank; i++) ndim_src_indices[i] = ndim_dst_indices[i] - pad_begin[i];
}

template <int PayloadRank>
static inline __device__ void extract_ndim_dst_indices(long ndim_dst_indices[PayloadRank],
                                                       const std::size_t dst_shape[PayloadRank]) {
#pragma unroll PayloadRank
    for (int i = 0; i < PayloadRank; i++)
        ndim_dst_indices[i] = index_in_dim<PayloadRank>(i, dst_shape, ConstModePad::kElementsPerThread);
}

// TODO: Strides would be precalculated on the host side. Left without this optimization in sake of readability.
template <int PayloadRank>
static inline __device__ std::size_t ndim_indices2index(const long ndim_indices[PayloadRank],
                                                        const std::size_t shape[PayloadRank]) {
    std::size_t stride = 1;
    auto result = ndim_indices[PayloadRank - 1];
#pragma unroll PayloadRank
    for (int i = PayloadRank - 2; i >= 0; i--) {
        stride *= shape[i + 1];
        result += stride * ndim_indices[i];
    }
    return result;
}

template <typename T, int PayloadRank>
static inline __global__ void pad_const_mode(const T* src,
                                             T* dst,
                                             const std::size_t pad_begin[PayloadRank],
                                             const std::size_t src_shape[PayloadRank],
                                             const std::size_t dst_shape[PayloadRank],
                                             const T* pad_value) {
    auto lastDim = PayloadRank - 1;
    auto index_in_last_dim = index_in_dim<PayloadRank>(lastDim, dst_shape, ConstModePad::kElementsPerThread);
    if (index_in_last_dim < dst_shape[lastDim]) {
        const auto dst_index = index<PayloadRank>(dst_shape, ConstModePad::kElementsPerThread);
        long ndim_dst_indices[PayloadRank];
        extract_ndim_dst_indices<PayloadRank>(ndim_dst_indices, dst_shape);
        long ndim_src_indices[PayloadRank];
        ndim_dst_indices2ndim_src_indices<PayloadRank>(ndim_dst_indices, pad_begin, ndim_src_indices);
        if (is_padding<PayloadRank>(ndim_src_indices, src_shape))
            dst[dst_index] = *pad_value;
        else
            dst[dst_index] = src[ndim_indices2index<PayloadRank>(ndim_src_indices, src_shape)];
    }
}

ConstModePad::ConstModePad(eltwise::KernelExecAttrs&& kernelExecAttrs,
                           ngraph::element::Type_t dtype,
                           std::size_t outputRank)
    : kernel_exec_attrs_{std::move(kernelExecAttrs)}, dtype_{dtype}, output_rank_{outputRank} {}

void ConstModePad::operator()(cudaStream_t stream,
                              const void* src,
                              void* dst,
                              const void* begin,
                              const std::size_t* srcShape,
                              const std::size_t* dstShape,
                              const void* padValue) const {
    /*
     * Since Pad is a data movement operation which doesn't change values,
     * it's type agnostic for types of the same width.
     * In sake of reducing code duplication and binary size, types of the same width are processed
     * by unsigned integer template instantiation version of appropriate width.
     * */
    using Type_t = ngraph::element::Type_t;
    switch (dtype_) {
        case Type_t::f32:
        case Type_t::i32:
        case Type_t::u32:
            callKernel<std::uint32_t>(stream, src, dst, begin, srcShape, dstShape, padValue);
            break;
        case Type_t::f16:
        case Type_t::i16:
        case Type_t::u16:
            callKernel<std::uint16_t>(stream, src, dst, begin, srcShape, dstShape, padValue);
            break;
        case Type_t::u8:
        case Type_t::i8:
        case Type_t::boolean:
            callKernel<std::uint8_t>(stream, src, dst, begin, srcShape, dstShape, padValue);
            break;
        default:
            throwIEException(fmt::format("Index element type = {} is not supported by Pad operation !", dtype_));
    }
}

template <typename T>
void ConstModePad::callKernel(cudaStream_t stream,
                              const void* src,
                              void* dst,
                              const void* begin,
                              const std::size_t* srcShape,
                              const std::size_t* dstShape,
                              const void* padValue) const {
    switch (output_rank_) {
        case 1:
            callKernel<T, 1>(stream, src, dst, begin, srcShape, dstShape, padValue);
            break;
        case 2:
            callKernel<T, 2>(stream, src, dst, begin, srcShape, dstShape, padValue);
            break;
        case 3:
            callKernel<T, 3>(stream, src, dst, begin, srcShape, dstShape, padValue);
            break;
        case 4:
            callKernel<T, 4>(stream, src, dst, begin, srcShape, dstShape, padValue);
            break;
        case 5:
            callKernel<T, 5>(stream, src, dst, begin, srcShape, dstShape, padValue);
            break;
    }
}

template <typename T, int PayloadRank>
void ConstModePad::callKernel(cudaStream_t stream,
                              const void* src,
                              void* dst,
                              const void* begin,
                              const std::size_t* srcShape,
                              const std::size_t* dstShape,
                              const void* padValue) const {
    pad_const_mode<T, PayloadRank>
        <<<kernel_exec_attrs_.grid, kernel_exec_attrs_.block, 0, stream>>>(static_cast<const T*>(src),
                                                                           static_cast<T*>(dst),
                                                                           static_cast<const size_t*>(begin),
                                                                           srcShape,
                                                                           dstShape,
                                                                           static_cast<const T*>(padValue));
}

}  // namespace kernel
}  // namespace CUDAPlugin
