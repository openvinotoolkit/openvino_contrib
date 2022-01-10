// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>
#include <printf.h>

#include <cmath>

#include "interpolate.hpp"

namespace CUDAPlugin {
namespace kernel {

static __device__ unsigned input_idx(const unsigned output_idx, const float scale) {
    if (scale == 1.0f) return output_idx;

    float res_idx = output_idx / scale;
    if (scale < 1.0)
        return static_cast<unsigned>(std::ceil(res_idx));
    else
        return static_cast<unsigned>(res_idx);
}

template <typename T = float>
static __global__ void interpolate(
    const T* src, const size_t input_strides[4], const size_t output_strides[4], const float scales[4], T* dst) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_strides[0]) return;

    enum { N, C, H, W };
    // calc N, C, H, W indexes
    const unsigned n_out = idx / output_strides[N];
    const unsigned n_out_size = n_out * output_strides[N];
    const unsigned c_out = (idx - n_out_size) / output_strides[C];
    const unsigned c_out_size = c_out * output_strides[C];
    const unsigned h_out = (idx - n_out_size - c_out_size) / output_strides[H];
    const unsigned h_out_size = h_out * output_strides[H];
    const unsigned w_out = (idx - n_out_size - c_out_size - h_out_size) / output_strides[W];

    const unsigned n_in = input_idx(n_out, scales[N]);
    const unsigned c_in = input_idx(c_out, scales[C]);
    const unsigned h_in = input_idx(h_out, scales[H]);
    const unsigned w_in = input_idx(w_out, scales[W]);

    const unsigned dst_idx = n_out * output_strides[N] + c_out * output_strides[C] + h_out * output_strides[H] + w_out;
    const unsigned src_idx = n_in * input_strides[N] + c_in * input_strides[C] + h_in * input_strides[H] + w_in;

    dst[dst_idx] = src[src_idx];
}

static __device__ void output_idx(const unsigned input_idx, const float scale, unsigned& from, unsigned& to) {
    if (scale == 1.0f) {
        from = input_idx;
        to = input_idx;
        return;
    }

    from = std::ceil(input_idx * scale);
    to = std::ceil((input_idx + 1) * scale) - 1;
}

template <typename T = float>
static __global__ void upscale_interpolate(
    const T* src, const size_t input_strides[4], const size_t output_strides[4], const float scales[4], T* dst) {
    enum { N, C, H, W };
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= input_strides[0]) {
        return;
    }

    const unsigned n_in = idx / input_strides[N];
    const unsigned n_in_size = n_in * input_strides[N];
    const unsigned c_in = (idx - n_in_size) / input_strides[C];
    const unsigned c_in_size = c_in * input_strides[C];
    const unsigned h_in = (idx - n_in_size - c_in_size) / input_strides[H];
    const unsigned h_in_size = h_in * input_strides[H];
    const unsigned w_in = (idx - n_in_size - c_in_size - h_in_size) / input_strides[W];
    const unsigned src_idx = n_in * input_strides[N] + c_in * input_strides[C] + h_in * input_strides[H] + w_in;
    T src_val = src[src_idx];

    unsigned n_out_from, n_out_to, c_out_from, c_out_to, h_out_from, h_out_to, w_out_from, w_out_to;
    output_idx(n_in, scales[N], n_out_from, n_out_to);
    output_idx(c_in, scales[C], c_out_from, c_out_to);
    output_idx(h_in, scales[H], h_out_from, h_out_to);
    output_idx(w_in, scales[W], w_out_from, w_out_to);

    for (unsigned n = n_out_from; n <= n_out_to; ++n) {
        const unsigned n_idx = n * output_strides[N];
        for (unsigned c = c_out_from; c <= c_out_to; ++c) {
            const unsigned c_idx = c * output_strides[C];
            for (unsigned h = h_out_from; h <= h_out_to; ++h) {
                const unsigned h_idx = h * output_strides[H];
                for (unsigned w = w_out_from; w <= w_out_to; ++w) {
                    const unsigned dst_idx = n_idx + c_idx + h_idx + w;
                    dst[dst_idx] = src_val;
                }
            }
        }
    }
}

Interpolate::Interpolate(size_t num_blocks,
                         size_t threads_per_block,
                         CUDAPlugin::kernel::Type_t element_type,
                         bool upscale)
    : num_blocks_{num_blocks}, threads_per_block_{threads_per_block}, element_type_{element_type}, upscale_{upscale} {}

void Interpolate::operator()(const cudaStream_t stream,
                             const void* src,
                             const size_t* input_strides,
                             const size_t* output_strides,
                             const float* scales,
                             void* dst) const {
    switch (element_type_) {
        case Type_t::f32:
            return callKernel<float>(stream, src, input_strides, output_strides, scales, dst);
        case Type_t::f16:
            return callKernel<__half>(stream, src, input_strides, output_strides, scales, dst);
        default:
            throwIEException(
                fmt::format("Index element type = {} is not supported by Gather operation !!", element_type_));
    }
}

template <typename T>
void Interpolate::callKernel(const cudaStream_t stream,
                             const void* src,
                             const size_t* input_strides,
                             const size_t* output_strides,
                             const float* scales,
                             void* dst) const {
    if (upscale_)
        kernel::upscale_interpolate<T><<<num_blocks_, threads_per_block_, 0, stream>>>(
            static_cast<const T*>(src), input_strides, output_strides, scales, static_cast<T*>(dst));
    else
        kernel::interpolate<T><<<num_blocks_, threads_per_block_, 0, stream>>>(
            static_cast<const T*>(src), input_strides, output_strides, scales, static_cast<T*>(dst));
}

}  // namespace kernel

}  // namespace CUDAPlugin
