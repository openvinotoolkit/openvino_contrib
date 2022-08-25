// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#ifdef __CUDACC__
#include <cuda/math.cuh>
#endif
#include <cuda/float16.hpp>

namespace ov {
namespace nvidia_gpu {
namespace kernel {

enum class ColorConversion { RGB, BGR };

#ifdef __CUDACC__
template <typename T>
__device__ void yuv_pixel_to_rgb(const float y_val, const float u_val, const float v_val, T &r, T &g, T &b) {
    const float c = y_val - 16.f;
    const float d = u_val - 128.f;
    const float e = v_val - 128.f;
    constexpr float lo = 0.f;
    constexpr float hi = 255.f;
    auto clip = [lo, hi](const float a) -> T {
        if (std::is_integral<T>::value) {
            return static_cast<T>(CUDA::math::min(CUDA::math::max(CUDA::math::round(a), lo), hi));
        } else {
            return static_cast<T>(CUDA::math::min(CUDA::math::max(a, lo), hi));
        }
    };
    b = clip(1.164f * c + 2.018f * d);
    g = clip(1.164f * c - (0.391f) * d - (0.813f) * e);
    r = clip(1.164f * c + (1.596f) * e);
}
#endif

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
