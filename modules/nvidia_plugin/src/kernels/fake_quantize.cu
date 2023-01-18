// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <cuda/float16.hpp>
#include <cuda/math.cuh>

#include "error.hpp"
#include "fake_quantize.hpp"
#include "tensor_helpers.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
static __global__ void fake_quantize(size_t max_size,
                                     T levels_1,
                                     const T* arg,
                                     const T* in_low,
                                     const T* in_high,
                                     const T* out_low,
                                     const T* out_high,
                                     NumpyBroadcastMapper in_low_mapper,
                                     NumpyBroadcastMapper in_high_mapper,
                                     NumpyBroadcastMapper out_low_mapper,
                                     NumpyBroadcastMapper out_high_mapper,
                                     T* out) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_size) {
        return;
    }

    const T in_low_val = in_low[in_low_mapper.srcIndex(idx)];
    const T in_high_val = in_high[in_high_mapper.srcIndex(idx)];
    const T out_low_val = out_low[out_low_mapper.srcIndex(idx)];
    const T out_high_val = out_high[out_high_mapper.srcIndex(idx)];

    const T value = arg[idx];

    if (value <= CUDA::math::min(in_low_val, in_high_val)) {
        out[idx] = out_low_val;
    } else if (value > CUDA::math::max(in_low_val, in_high_val)) {
        out[idx] = out_high_val;
    } else {
        out[idx] = CUDA::math::round((value - in_low_val) / (in_high_val - in_low_val) * levels_1) / levels_1 *
                       (out_high_val - out_low_val) +
                   out_low_val;
    }
}

FakeQuantize::FakeQuantize(Type_t element_type, std::size_t max_size, std::size_t threads_per_block, std::size_t levels)
    : element_type_{element_type}, max_size_{max_size}, threads_per_block_{threads_per_block}, levels_{levels} {
    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(max_size_, threads_per_block_);
}

void FakeQuantize::operator()(const cudaStream_t stream,
                              const void* arg,
                              const void* in_low,
                              const void* in_high,
                              const void* out_low,
                              const void* out_high,
                              const NumpyBroadcastMapper& in_low_mapper,
                              const NumpyBroadcastMapper& in_high_mapper,
                              const NumpyBroadcastMapper& out_low_mapper,
                              const NumpyBroadcastMapper& out_high_mapper,
                              void* out) const {
    const std::size_t levels_1 = levels_ - 1;
    switch (element_type_) {
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return Call<__nv_bfloat16>(stream,
                                       arg,
                                       in_low,
                                       in_high,
                                       out_low,
                                       out_high,
                                       in_low_mapper,
                                       in_high_mapper,
                                       out_low_mapper,
                                       out_high_mapper,
                                       static_cast<float>(levels_1),
                                       out);
#endif
        case Type_t::f16:
            return Call<__half>(stream,
                                arg,
                                in_low,
                                in_high,
                                out_low,
                                out_high,
                                in_low_mapper,
                                in_high_mapper,
                                out_low_mapper,
                                out_high_mapper,
                                static_cast<float>(levels_1),
                                out);
        case Type_t::f32:
            return Call<float>(stream,
                               arg,
                               in_low,
                               in_high,
                               out_low,
                               out_high,
                               in_low_mapper,
                               in_high_mapper,
                               out_low_mapper,
                               out_high_mapper,
                               levels_1,
                               out);
        case Type_t::f64:
            return Call<double>(stream,
                                arg,
                                in_low,
                                in_high,
                                out_low,
                                out_high,
                                in_low_mapper,
                                in_high_mapper,
                                out_low_mapper,
                                out_high_mapper,
                                levels_1,
                                out);
        default:
            throwIEException(
                fmt::format("Input element type = {} is not supported by FakeQuatizer operation !!", element_type_));
    }
}

template <typename T>
void FakeQuantize::Call(const cudaStream_t stream,
                        const void* arg,
                        const void* in_low,
                        const void* in_high,
                        const void* out_low,
                        const void* out_high,
                        const NumpyBroadcastMapper& in_low_mapper,
                        const NumpyBroadcastMapper& in_high_mapper,
                        const NumpyBroadcastMapper& out_low_mapper,
                        const NumpyBroadcastMapper& out_high_mapper,
                        T levels_1,
                        void* out) const {
    kernel::fake_quantize<T><<<num_blocks_, threads_per_block_, 0, stream>>>(max_size_,
                                                                             levels_1,
                                                                             static_cast<const T*>(arg),
                                                                             static_cast<const T*>(in_low),
                                                                             static_cast<const T*>(in_high),
                                                                             static_cast<const T*>(out_low),
                                                                             static_cast<const T*>(out_high),
                                                                             in_low_mapper,
                                                                             in_high_mapper,
                                                                             out_low_mapper,
                                                                             out_high_mapper,
                                                                             static_cast<T*>(out));
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
