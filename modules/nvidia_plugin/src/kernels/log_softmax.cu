// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>
#include <math_constants.h>

#include <cuda/float16.hpp>

#include "details/type_validator.hpp"
#include "log_softmax.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace {
template <typename T>
struct Accumulator {
    using type = float;
};
template <>
struct Accumulator<double> {
    using type = double;
};

template <typename Acc, bool IsMax>
__device__ Acc blockReduce(Acc value, Acc* shared) {
    shared[threadIdx.x] = value;
    __syncthreads();
    for (unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const Acc other = shared[threadIdx.x + stride];
            if (IsMax) {
                shared[threadIdx.x] = other > shared[threadIdx.x] ? other : shared[threadIdx.x];
            } else {
                shared[threadIdx.x] += other;
            }
        }
        __syncthreads();
    }
    return shared[0];
}
}  // namespace

// One block per (outer, inner) row; blockDim.x is a power of two.
template <typename T, typename Acc>
static __global__ void log_softmax_kernel(const T* x,
                                          T* y,
                                          const std::size_t axis_size,
                                          const std::size_t inner_size) {
    extern __shared__ __align__(sizeof(Acc)) unsigned char shared_raw[];
    Acc* shared = reinterpret_cast<Acc*>(shared_raw);

    const std::size_t row = blockIdx.x;
    const std::size_t outer = row / inner_size;
    const std::size_t inner = row % inner_size;
    const std::size_t base = outer * axis_size * inner_size + inner;

    // 1. max over the axis
    Acc thread_max = -CUDART_INF_F;
    for (std::size_t j = threadIdx.x; j < axis_size; j += blockDim.x) {
        const Acc v = static_cast<Acc>(x[base + j * inner_size]);
        thread_max = v > thread_max ? v : thread_max;
    }
    const Acc row_max = blockReduce<Acc, true>(thread_max, shared);
    __syncthreads();

    // 2. sum(exp(x - max))
    Acc thread_sum = 0;
    for (std::size_t j = threadIdx.x; j < axis_size; j += blockDim.x) {
        thread_sum += exp(static_cast<Acc>(x[base + j * inner_size]) - row_max);
    }
    const Acc log_sum = log(blockReduce<Acc, false>(thread_sum, shared));

    // 3. y = (x - max) - log_sum
    for (std::size_t j = threadIdx.x; j < axis_size; j += blockDim.x) {
        const std::size_t offset = base + j * inner_size;
        y[offset] = static_cast<T>((static_cast<Acc>(x[offset]) - row_max) - log_sum);
    }
}

LogSoftmax::LogSoftmax(Type_t data_type,
                       std::size_t outer_size,
                       std::size_t axis_size,
                       std::size_t inner_size,
                       std::size_t num_blocks,
                       std::size_t threads_per_block)
    : data_type_{data_type},
      outer_size_{outer_size},
      axis_size_{axis_size},
      inner_size_{inner_size},
      num_blocks_{num_blocks},
      threads_per_block_{threads_per_block} {
    TypeValidator<ElementTypesSwitch<Type_t::f16, Type_t::bf16, Type_t::f32, Type_t::f64>>::check(data_type_);
}

void LogSoftmax::operator()(cudaStream_t stream, const void* input, void* output) const {
    switch (data_type_) {
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return call<__nv_bfloat16>(stream, input, output);
#endif
        case Type_t::f16:
            return call<__half>(stream, input, output);
        case Type_t::f32:
            return call<float>(stream, input, output);
        case Type_t::f64:
            return call<double>(stream, input, output);
        default:
            throw_ov_exception(fmt::format("Element type = {} is not supported by LogSoftmax operation !!", data_type_));
    }
}

template <typename T>
void LogSoftmax::call(cudaStream_t stream, const void* input, void* output) const {
    using Acc = typename Accumulator<T>::type;
    if (num_blocks_ == 0 || axis_size_ == 0) {
        return;
    }
    const std::size_t shared_bytes = threads_per_block_ * sizeof(Acc);
    log_softmax_kernel<T, Acc><<<num_blocks_, threads_per_block_, shared_bytes, stream>>>(
        static_cast<const T*>(input), static_cast<T*>(output), axis_size_, inner_size_);
    throwIfError(cudaPeekAtLastError());
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
