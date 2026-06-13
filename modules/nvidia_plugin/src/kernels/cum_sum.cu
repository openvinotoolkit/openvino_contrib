// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <cuda/float16.hpp>

#include "cum_sum.hpp"
#include "details/type_validator.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace {
// Accumulate floating point in float/double; integers in their own type so the
// result matches the reference's wrap-around.
template <typename T>
struct Accumulator {
    using type = T;
};
template <>
struct Accumulator<__half> {
    using type = float;
};
#ifdef CUDA_HAS_BF16_TYPE
template <>
struct Accumulator<__nv_bfloat16> {
    using type = float;
};
#endif
}  // namespace

template <typename T, typename Acc>
static __global__ void cum_sum_kernel(const T* x,
                                      T* y,
                                      const std::size_t axis_size,
                                      const std::size_t inner_size,
                                      const std::size_t num_rows,
                                      const bool exclusive,
                                      const bool reverse) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= num_rows) {
        return;
    }
    const std::size_t outer = row / inner_size;
    const std::size_t inner = row % inner_size;
    const std::size_t base = outer * axis_size * inner_size + inner;

    Acc acc = 0;
    for (std::size_t step = 0; step < axis_size; ++step) {
        const std::size_t j = reverse ? (axis_size - 1 - step) : step;
        const std::size_t offset = base + j * inner_size;
        if (exclusive) {
            y[offset] = static_cast<T>(acc);
            acc += static_cast<Acc>(x[offset]);
        } else {
            acc += static_cast<Acc>(x[offset]);
            y[offset] = static_cast<T>(acc);
        }
    }
}

CumSum::CumSum(Type_t data_type,
               std::size_t outer_size,
               std::size_t axis_size,
               std::size_t inner_size,
               bool exclusive,
               bool reverse,
               std::size_t num_blocks,
               std::size_t threads_per_block)
    : data_type_{data_type},
      outer_size_{outer_size},
      axis_size_{axis_size},
      inner_size_{inner_size},
      exclusive_{exclusive},
      reverse_{reverse},
      num_blocks_{num_blocks},
      threads_per_block_{threads_per_block} {
    TypeValidator<AllElementTypesSwitch>::check(data_type_);
}

void CumSum::operator()(cudaStream_t stream, const void* input, void* output) const {
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
        case Type_t::i8:
            return call<int8_t>(stream, input, output);
        case Type_t::i16:
            return call<int16_t>(stream, input, output);
        case Type_t::i32:
            return call<int32_t>(stream, input, output);
        case Type_t::i64:
            return call<int64_t>(stream, input, output);
        case Type_t::u8:
            return call<uint8_t>(stream, input, output);
        case Type_t::u16:
            return call<uint16_t>(stream, input, output);
        case Type_t::u32:
            return call<uint32_t>(stream, input, output);
        case Type_t::u64:
            return call<uint64_t>(stream, input, output);
        default:
            throw_ov_exception(fmt::format("Element type = {} is not supported by CumSum operation !!", data_type_));
    }
}

template <typename T>
void CumSum::call(cudaStream_t stream, const void* input, void* output) const {
    using Acc = typename Accumulator<T>::type;
    const std::size_t num_rows = outer_size_ * inner_size_;
    if (num_rows == 0 || axis_size_ == 0) {
        return;
    }
    cum_sum_kernel<T, Acc><<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const T*>(input),
                                                                          static_cast<T*>(output),
                                                                          axis_size_,
                                                                          inner_size_,
                                                                          num_rows,
                                                                          exclusive_,
                                                                          reverse_);
    throwIfError(cudaPeekAtLastError());
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
