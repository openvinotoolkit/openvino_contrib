// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <algorithm>
#include <cmath>

#include "interpolate_details.cuh"
#include "interpolate_nearest.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

static inline __device__ float calc_output_index(const InterpolateNearest::CoordinateTransformMode mode,
                                                 const unsigned index,
                                                 const float scale) {
    using CoordinateTransformMode = InterpolateNearest::CoordinateTransformMode;
    float output_index = {};
    switch (mode) {
        case CoordinateTransformMode::asymmetric:
            output_index = static_cast<float>(index) * scale;
            break;
        case CoordinateTransformMode::tf_half_pixel_for_nn:
            output_index = static_cast<float>(index) * scale;
            break;
        default:
            assert(false);
    }
    return output_index;
}

static inline __device__ unsigned round_index(const InterpolateNearest::NearestMode mode,
                                              const float index,
                                              const float scale) {
    using NearestMode = InterpolateNearest::NearestMode;
    switch (mode) {
        case NearestMode::simple:
            if (scale < 1.0)
                return static_cast<unsigned>(std::ceil(index));
            else
                return static_cast<unsigned>(index);
        case NearestMode::floor:
            return static_cast<unsigned>(std::floor(index));
        case NearestMode::ceil:
            return static_cast<unsigned>(std::ceil(index));
        case NearestMode::round_prefer_ceil:
            return static_cast<unsigned>(std::round(index));
        case NearestMode::round_prefer_floor:
            if (index == static_cast<unsigned>(index) + 0.5f)
                return static_cast<unsigned>(std::floor(index));
            else
                return static_cast<unsigned>(std::round(index));
        default:
            assert(false);
            return 0;
    }
}

static inline __device__ unsigned input_index(const InterpolateNearest::NearestMode nearest_mode,
                                              const InterpolateNearest::CoordinateTransformMode transform_mode,
                                              const unsigned index,
                                              const float scale,
                                              const unsigned output_dimension_size,
                                              const unsigned input_dimension_size) {
    if (scale == 1.0f) return index;

    using details = InterpolateNearest::details;
    const float in_coord = details::get_original_coordinate<float>(
        transform_mode, index, scale, output_dimension_size, input_dimension_size);
    return round_index(nearest_mode, in_coord, scale);
}

static inline __device__ unsigned min_position(const unsigned idx, const size_t max_length) {
    return idx < max_length ? idx : static_cast<unsigned>(max_length) - 1;
}

template <typename T = float>
static __global__ void interpolate(const InterpolateNearest::NearestMode nearest_mode,
                                   const InterpolateNearest::CoordinateTransformMode transform_mode,
                                   const T* src,
                                   const size_t input_strides[4],
                                   const size_t output_strides[4],
                                   const float scales[4],
                                   const size_t input_shape[4],
                                   const size_t output_shape[4],
                                   T* dst) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_strides[0] * output_shape[0]) return;

    enum { N, C, H, W };
    // calc N, C, H, W indexes
    const unsigned n_out = idx / output_strides[N];
    const unsigned n_out_size = n_out * output_strides[N];
    const unsigned c_out = (idx - n_out_size) / output_strides[C];
    const unsigned c_out_size = c_out * output_strides[C];
    const unsigned h_out = (idx - n_out_size - c_out_size) / output_strides[H];
    const unsigned h_out_size = h_out * output_strides[H];
    const unsigned w_out = (idx - n_out_size - c_out_size - h_out_size) / output_strides[W];

    unsigned n_in = input_index(nearest_mode, transform_mode, n_out, scales[N], output_shape[N], input_shape[N]);
    n_in = min_position(n_in, input_shape[N]);
    unsigned c_in = input_index(nearest_mode, transform_mode, c_out, scales[C], output_shape[C], input_shape[C]);
    c_in = min_position(c_in, input_shape[C]);
    unsigned h_in = input_index(nearest_mode, transform_mode, h_out, scales[H], output_shape[H], input_shape[H]);
    h_in = min_position(h_in, input_shape[H]);
    unsigned w_in = input_index(nearest_mode, transform_mode, w_out, scales[W], output_shape[W], input_shape[W]);
    w_in = min_position(w_in, input_shape[W]);

    const unsigned dst_idx = n_out * output_strides[N] + c_out * output_strides[C] + h_out * output_strides[H] + w_out;
    const unsigned src_idx = n_in * input_strides[N] + c_in * input_strides[C] + h_in * input_strides[H] + w_in;

    dst[dst_idx] = src[src_idx];
}

static inline __device__ void output_indexes(const InterpolateNearest::NearestMode nearest_mode,
                                             const InterpolateNearest::CoordinateTransformMode transform_mode,
                                             const unsigned input_idx,
                                             const float scale,
                                             unsigned& from,
                                             unsigned& to) {
    from = round_index(nearest_mode, calc_output_index(transform_mode, input_idx, scale), scale);
    to = round_index(nearest_mode, calc_output_index(transform_mode, (input_idx + 1), scale), scale);
    if (to > from) --to;
}

template <typename T = float>
static __global__ void upscale_interpolate(const InterpolateNearest::NearestMode nearest_mode,
                                           const InterpolateNearest::CoordinateTransformMode transform_mode,
                                           const T* src,
                                           const size_t input_strides[4],
                                           const size_t output_strides[4],
                                           const float scales[4],
                                           const size_t input_shape[4],
                                           const size_t output_shape[4],
                                           T* dst) {
    enum { N, C, H, W };
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_strides[0] * input_shape[0]) {
        return;
    }

    const unsigned n_in = idx / input_strides[N];
    const unsigned n_in_size = n_in * input_strides[N];
    const unsigned c_in = (idx - n_in_size) / input_strides[C];
    const unsigned c_in_size = c_in * input_strides[C];
    const unsigned h_in = (idx - n_in_size - c_in_size) / input_strides[H];
    const unsigned h_in_size = h_in * input_strides[H];
    const unsigned w_in = (idx - n_in_size - c_in_size - h_in_size) / input_strides[W];
    unsigned src_idx = n_in * input_strides[N] + c_in * input_strides[C] + h_in * input_strides[H] + w_in;
    src_idx = src_idx >= input_strides[0] ? input_strides[0] - 1 : src_idx;
    T src_val = src[src_idx];

    unsigned n_out_from, n_out_to, c_out_from, c_out_to, h_out_from, h_out_to, w_out_from, w_out_to;
    output_indexes(nearest_mode, transform_mode, n_in, scales[N], n_out_from, n_out_to);
    n_out_from = min_position(n_out_from, output_shape[N]);
    n_out_to = min_position(n_out_to, output_shape[N]);
    output_indexes(nearest_mode, transform_mode, c_in, scales[C], c_out_from, c_out_to);
    c_out_from = min_position(c_out_from, output_shape[C]);
    c_out_to = min_position(c_out_to, output_shape[C]);
    output_indexes(nearest_mode, transform_mode, h_in, scales[H], h_out_from, h_out_to);
    h_out_from = min_position(h_out_from, output_shape[H]);
    h_out_to = min_position(h_out_to, output_shape[H]);
    output_indexes(nearest_mode, transform_mode, w_in, scales[W], w_out_from, w_out_to);
    w_out_from = min_position(w_out_from, output_shape[W]);
    w_out_to = min_position(w_out_to, output_shape[W]);

    for (unsigned n = n_out_from; n <= n_out_to; ++n) {
        const unsigned n_idx = n * output_strides[N];
        for (unsigned c = c_out_from; c <= c_out_to; ++c) {
            const unsigned c_idx = c * output_strides[C];
            for (unsigned h = h_out_from; h <= h_out_to; ++h) {
                const unsigned h_idx = h * output_strides[H];
                for (unsigned w = w_out_from; w <= w_out_to; ++w) {
                    unsigned dst_idx = n_idx + c_idx + h_idx + w;
                    dst_idx = dst_idx >= output_strides[0] ? output_strides[0] - 1 : dst_idx;
                    dst[dst_idx] = src_val;
                }
            }
        }
    }
}

InterpolateNearest::InterpolateNearest(size_t num_blocks,
                                       size_t threads_per_block,
                                       ov::nvidia_gpu::kernel::Type_t element_type,
                                       bool use_optimized_kernel,
                                       NearestMode nearest_mode,
                                       CoordinateTransformMode transform_mode)
    : num_blocks_{num_blocks},
      threads_per_block_{threads_per_block},
      element_type_{element_type},
      use_optimized_kernel_{use_optimized_kernel},
      nearest_mode_{nearest_mode},
      transform_mode_{transform_mode} {}

void InterpolateNearest::operator()(const cudaStream_t stream,
                                    const void* src,
                                    const size_t* input_strides,
                                    const size_t* output_strides,
                                    const float* scales,
                                    const size_t* input_shape,
                                    const size_t* output_shape,
                                    void* dst) const {
    switch (element_type_) {
        case Type_t::f32:
            return callKernel<float>(
                stream, src, input_strides, output_strides, scales, input_shape, output_shape, dst);
        case Type_t::f16:
            return callKernel<__half>(
                stream, src, input_strides, output_strides, scales, input_shape, output_shape, dst);
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return callKernel<__nv_bfloat16>(
                stream, src, input_strides, output_strides, scales, input_shape, output_shape, dst);
#endif
        case Type_t::i8:
            return callKernel<int8_t>(
                stream, src, input_strides, output_strides, scales, input_shape, output_shape, dst);
        case Type_t::u8:
            return callKernel<uint8_t>(
                stream, src, input_strides, output_strides, scales, input_shape, output_shape, dst);
        default:
            throwIEException(
                fmt::format("Element type = {} is not supported by InterpolateNearest operation !!", element_type_));
    }
}

template <typename T>
void InterpolateNearest::callKernel(const cudaStream_t stream,
                                    const void* src,
                                    const size_t* input_strides,
                                    const size_t* output_strides,
                                    const float* scales,
                                    const size_t* input_shape,
                                    const size_t* output_shape,
                                    void* dst) const {
    if (use_optimized_kernel_)
        kernel::upscale_interpolate<T><<<num_blocks_, threads_per_block_, 0, stream>>>(nearest_mode_,
                                                                                       transform_mode_,
                                                                                       static_cast<const T*>(src),
                                                                                       input_strides,
                                                                                       output_strides,
                                                                                       scales,
                                                                                       input_shape,
                                                                                       output_shape,
                                                                                       static_cast<T*>(dst));
    else
        kernel::interpolate<T><<<num_blocks_, threads_per_block_, 0, stream>>>(nearest_mode_,
                                                                               transform_mode_,
                                                                               static_cast<const T*>(src),
                                                                               input_strides,
                                                                               output_strides,
                                                                               scales,
                                                                               input_shape,
                                                                               output_shape,
                                                                               static_cast<T*>(dst));
}

}  // namespace kernel

}  // namespace nvidia_gpu
}  // namespace ov
