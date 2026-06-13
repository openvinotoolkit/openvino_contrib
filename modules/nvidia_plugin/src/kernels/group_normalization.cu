// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <cuda/float16.hpp>

#include "details/type_validator.hpp"
#include "group_normalization.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace {
// Accumulate reductions in float for the half/float cases and in double for f64
// to avoid catastrophic cancellation when computing the variance.
template <typename T>
struct Accumulator {
    using type = float;
};
template <>
struct Accumulator<double> {
    using type = double;
};

template <typename Acc>
__device__ Acc blockReduceSum(Acc value, Acc* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared[0];
}
}  // namespace

// One block per (N * num_groups) group; blockDim.x is a power of two.
template <typename T, typename Acc>
static __global__ void group_normalization_kernel(const T* x,
                                                  const T* scale,
                                                  const T* bias,
                                                  T* y,
                                                  const unsigned num_groups,
                                                  const unsigned channels_per_group,
                                                  const std::size_t spatial_size,
                                                  const std::size_t group_size,
                                                  const Acc epsilon) {
    extern __shared__ __align__(sizeof(Acc)) unsigned char shared_raw[];
    Acc* shared = reinterpret_cast<Acc*>(shared_raw);

    const unsigned group = blockIdx.x;                 // 0 .. N * num_groups - 1
    const unsigned group_in_batch = group % num_groups;
    const std::size_t group_base = static_cast<std::size_t>(group) * group_size;
    const unsigned channel_base = group_in_batch * channels_per_group;

    // 1. mean
    Acc sum = 0;
    for (std::size_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        sum += static_cast<Acc>(x[group_base + i]);
    }
    const Acc mean = blockReduceSum<Acc>(sum, shared) / static_cast<Acc>(group_size);
    __syncthreads();

    // 2. variance
    Acc sq_sum = 0;
    for (std::size_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        const Acc diff = static_cast<Acc>(x[group_base + i]) - mean;
        sq_sum += diff * diff;
    }
    const Acc variance = blockReduceSum<Acc>(sq_sum, shared) / static_cast<Acc>(group_size);
    const Acc inv_std = static_cast<Acc>(1) / sqrt(variance + epsilon);

    // 3. normalize + per-channel affine
    for (std::size_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        const unsigned channel = channel_base + static_cast<unsigned>(i / spatial_size);
        const Acc norm = (static_cast<Acc>(x[group_base + i]) - mean) * inv_std;
        y[group_base + i] = static_cast<T>(norm * static_cast<Acc>(scale[channel]) + static_cast<Acc>(bias[channel]));
    }
}

GroupNormalization::GroupNormalization(Type_t data_type,
                                       std::size_t batch,
                                       std::size_t num_channels,
                                       std::size_t num_groups,
                                       std::size_t spatial_size,
                                       double epsilon,
                                       std::size_t num_blocks,
                                       std::size_t threads_per_block)
    : data_type_{data_type},
      batch_{batch},
      num_channels_{num_channels},
      num_groups_{num_groups},
      spatial_size_{spatial_size},
      epsilon_{epsilon},
      num_blocks_{num_blocks},
      threads_per_block_{threads_per_block} {
    TypeValidator<ElementTypesSwitch<Type_t::f16, Type_t::bf16, Type_t::f32, Type_t::f64>>::check(data_type_);
}

void GroupNormalization::operator()(cudaStream_t stream,
                                    const void* data,
                                    const void* scale,
                                    const void* bias,
                                    void* output) const {
    switch (data_type_) {
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return call<__nv_bfloat16>(stream, data, scale, bias, output);
#endif
        case Type_t::f16:
            return call<__half>(stream, data, scale, bias, output);
        case Type_t::f32:
            return call<float>(stream, data, scale, bias, output);
        case Type_t::f64:
            return call<double>(stream, data, scale, bias, output);
        default:
            throw_ov_exception(
                fmt::format("Element type = {} is not supported by GroupNormalization operation !!", data_type_));
    }
}

template <typename T>
void GroupNormalization::call(cudaStream_t stream,
                              const void* data,
                              const void* scale,
                              const void* bias,
                              void* output) const {
    using Acc = typename Accumulator<T>::type;
    if (num_blocks_ == 0) {
        return;
    }
    const unsigned channels_per_group = static_cast<unsigned>(num_channels_ / num_groups_);
    const std::size_t group_size = static_cast<std::size_t>(channels_per_group) * spatial_size_;
    const std::size_t shared_bytes = threads_per_block_ * sizeof(Acc);

    group_normalization_kernel<T, Acc>
        <<<num_blocks_, threads_per_block_, shared_bytes, stream>>>(static_cast<const T*>(data),
                                                                    static_cast<const T*>(scale),
                                                                    static_cast<const T*>(bias),
                                                                    static_cast<T*>(output),
                                                                    static_cast<unsigned>(num_groups_),
                                                                    channels_per_group,
                                                                    spatial_size_,
                                                                    group_size,
                                                                    static_cast<Acc>(epsilon_));
    throwIfError(cudaPeekAtLastError());
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
