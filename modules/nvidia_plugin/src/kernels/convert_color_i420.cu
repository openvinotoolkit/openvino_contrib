// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <cuda/float16.hpp>
#include <cuda/math.cuh>
#include <type_traits>

#include "convert_color_i420.hpp"
#include "details/error.hpp"
#include "details/tensor_helpers.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <ColorConversion ColorFormat, typename T>
__global__ void color_convert_i420(const T* arg_y,
                                   const T* arg_u,
                                   const T* arg_v,
                                   T* out_ptr,
                                   const size_t batch_size,
                                   const size_t image_h,
                                   const size_t image_w,
                                   const size_t stride_y,
                                   const size_t stride_uv) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (batch_size * image_h * image_w)) {
        return;
    }

    const unsigned w = idx % image_w;
    const unsigned w_left = idx / image_w;
    const unsigned h = w_left % image_h;
    const unsigned h_left = w_left / image_h;
    const unsigned batch = h_left % batch_size;

    T* out = out_ptr + batch * image_w * image_h * 3;
    const T* y_ptr = arg_y + batch * stride_y;
    const T* u_ptr = arg_u + batch * stride_uv;
    const T* v_ptr = arg_v + batch * stride_uv;
    const size_t y_index = h * image_w + w;
    const auto y_val = static_cast<float>(y_ptr[y_index]);
    const size_t uv_index = (h / 2) * (image_w / 2) + (w / 2);
    const auto u_val = static_cast<float>(u_ptr[uv_index]);
    const auto v_val = static_cast<float>(v_ptr[uv_index]);

    T r, g, b;
    yuv_pixel_to_rgb(y_val, u_val, v_val, r, g, b);
    if (ColorFormat == ColorConversion::RGB) {
        out[y_index * 3] = r;
        out[y_index * 3 + 1] = g;
        out[y_index * 3 + 2] = b;
    } else if (ColorFormat == ColorConversion::BGR) {
        out[y_index * 3] = b;
        out[y_index * 3 + 1] = g;
        out[y_index * 3 + 2] = r;
    }
}

template <ColorConversion Conversion>
I420ColorConvert<Conversion>::I420ColorConvert(const Type_t element_type,
                                               const size_t max_threads_per_block,
                                               const size_t batch_size,
                                               const size_t image_h,
                                               const size_t image_w,
                                               const size_t stride_y,
                                               const size_t stride_uv)
    : element_type_{element_type},
      batch_size_{batch_size},
      image_h_{image_h},
      image_w_{image_w},
      stride_y_{stride_y},
      stride_uv_{stride_uv} {
    std::tie(num_blocks_, threads_per_block_) =
        calculateElementwiseGrid(batch_size * image_h * image_w, max_threads_per_block);
}

template <ColorConversion Conversion>
void I420ColorConvert<Conversion>::operator()(cudaStream_t stream, const void* in, void* out) const {
    Switcher::switch_(element_type_, *this, stream, in, out);
}

template <ColorConversion Conversion>
void I420ColorConvert<Conversion>::operator()(
    cudaStream_t stream, const void* in0, const void* in1, const void* in2, void* out) const {
    Switcher::switch_(element_type_, *this, stream, in0, in1, in2, out);
}

template <ColorConversion Conversion>
template <typename T, typename... Args>
constexpr void I420ColorConvert<Conversion>::case_(cudaStream_t stream, Args&&... args) const noexcept {
    callKernel<T>(stream, std::forward<Args>(args)...);
}

template <ColorConversion Conversion>
template <typename T, typename... Args>
void I420ColorConvert<Conversion>::default_(T t, cudaStream_t, Args&&...) const noexcept {
    throwIEException(fmt::format("Element type = {} is not supported by I420ColorConvert operation.", t));
}

template <ColorConversion Conversion>
template <typename T>
void I420ColorConvert<Conversion>::callKernel(const cudaStream_t stream, const void* in, void* out) const {
    return color_convert_i420<Conversion>
        <<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const T*>(in),
                                                         static_cast<const T*>(in) + image_w_ * image_h_,
                                                         static_cast<const T*>(in) + 5 * image_w_ * image_h_ / 4,
                                                         static_cast<T*>(out),
                                                         batch_size_,
                                                         image_h_,
                                                         image_w_,
                                                         stride_y_,
                                                         stride_uv_);
}

template <ColorConversion Conversion>
template <typename T>
void I420ColorConvert<Conversion>::callKernel(
    const cudaStream_t stream, const void* in0, const void* in1, const void* in2, void* out) const {
    return color_convert_i420<Conversion><<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const T*>(in0),
                                                                                          static_cast<const T*>(in1),
                                                                                          static_cast<const T*>(in2),
                                                                                          static_cast<T*>(out),
                                                                                          batch_size_,
                                                                                          image_h_,
                                                                                          image_w_,
                                                                                          stride_y_,
                                                                                          stride_uv_);
}

template class I420ColorConvert<ColorConversion::RGB>;
template class I420ColorConvert<ColorConversion::BGR>;

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
