// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <cuda/float16.hpp>

#include "details/error.hpp"
#include "details/type_validator.hpp"
#include "split.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
static __global__ void split(
    const size_t numSplitChunks, const size_t splitStepSize, const size_t numSplits, const T *x, T **y) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned chunkIdx = i / splitStepSize;
    const unsigned chunkOffset = i % splitStepSize;
    if (chunkIdx < numSplitChunks) {
        const unsigned splitIdx = chunkIdx % numSplits;
        const unsigned splitStepIdx = chunkIdx / numSplits;
        auto src = &x[chunkIdx * splitStepSize];
        auto dest = &y[splitIdx][splitStepIdx * splitStepSize];
        dest[chunkOffset] = src[chunkOffset];
    }
}

Split::Split(Type_t element_type,
             size_t num_splits,
             size_t num_split_chunks,
             size_t split_step_size,
             unsigned num_blocks,
             unsigned threads_per_block)
    : element_type_{element_type},
      num_splits_{num_splits},
      num_split_chunks_{num_split_chunks},
      split_step_size_{split_step_size},
      num_blocks_{num_blocks},
      threads_per_block_{threads_per_block} {
    TypeValidator<AllElementTypesSwitch>::check(element_type_);
}

void Split::operator()(cudaStream_t stream, const void *src, void **dst) const {
    switch (element_type_) {
        case Type_t::boolean:
            return Call<bool>(stream, src, dst);
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return Call<__nv_bfloat16>(stream, src, dst);
#endif
        case Type_t::f16:
            return Call<__half>(stream, src, dst);
        case Type_t::f32:
            return Call<float>(stream, src, dst);
        case Type_t::f64:
            return Call<double>(stream, src, dst);
        case Type_t::i8:
            return Call<int8_t>(stream, src, dst);
        case Type_t::i16:
            return Call<int16_t>(stream, src, dst);
        case Type_t::i32:
            return Call<int32_t>(stream, src, dst);
        case Type_t::i64:
            return Call<int64_t>(stream, src, dst);
        case Type_t::u8:
            return Call<uint8_t>(stream, src, dst);
        case Type_t::u16:
            return Call<uint16_t>(stream, src, dst);
        case Type_t::u32:
            return Call<uint32_t>(stream, src, dst);
        case Type_t::u64:
            return Call<uint64_t>(stream, src, dst);
        default:
            throw_ov_exception(
                fmt::format("Input element type = {} is not supported by Split operation "
                            "!!",
                            static_cast<Type_t>(element_type_)));
    }
}

template <typename T>
void Split::Call(cudaStream_t stream, const void *src, void **dst) const {
    split<T><<<num_blocks_, threads_per_block_, 0, stream>>>(
        num_split_chunks_, split_step_size_, num_splits_, static_cast<const T *>(src), reinterpret_cast<T **>(dst));
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
