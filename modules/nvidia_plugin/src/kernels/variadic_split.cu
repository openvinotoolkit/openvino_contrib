// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cooperative_groups.h>
#include <fmt/format.h>

#include <cuda/float16.hpp>

#include "details/error.hpp"
#include "details/type_validator.hpp"
#include "variadic_split.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
static __global__ void variadic_split(const size_t numAllChunks,
                                      const size_t axisSplitStepSize,
                                      const size_t origAxisSize,
                                      const T *x,
                                      T **y,
                                      const size_t *splitIdxs,
                                      const size_t *axisSizes,
                                      const size_t *axisOffsetSizes) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned chunkIdx = i / axisSplitStepSize;
    const unsigned chunkOffset = i % axisSplitStepSize;
    if (chunkIdx < numAllChunks) {
        const unsigned splitStepIdx = chunkIdx / origAxisSize;
        const unsigned splitStepIdxOffset = chunkIdx % origAxisSize;
        const unsigned splitIdx = splitIdxs[splitStepIdxOffset];
        const unsigned axisSize = axisSizes[splitIdx];
        const unsigned axisOffsetSize = axisOffsetSizes[splitIdx];
        auto src = &x[chunkIdx * axisSplitStepSize];
        auto dest = &y[splitIdx][splitStepIdx * axisSize * axisSplitStepSize +
                                 (splitStepIdxOffset - axisOffsetSize) * axisSplitStepSize];
        dest[chunkOffset] = src[chunkOffset];
    }
}

VariadicSplit::VariadicSplit(Type_t element_type,
                             size_t num_all_chunks,
                             size_t axis_split_step_size,
                             size_t orig_axis_size,
                             unsigned num_blocks,
                             unsigned threads_per_block)
    : element_type_{element_type},
      num_all_chunks_{num_all_chunks},
      axis_split_step_size_{axis_split_step_size},
      orig_axis_size_{orig_axis_size},
      num_blocks_{num_blocks},
      threads_per_block_{threads_per_block} {
    TypeValidator<AllElementTypesSwitch>::check(element_type_);
}

void VariadicSplit::operator()(cudaStream_t stream,
                               const void *src,
                               void **dst,
                               const void *splitIdxs,
                               const void *axisSizes,
                               const void *axisOffsetSizes) const {
    switch (element_type_) {
        case Type_t::boolean:
            return call<bool>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return call<__nv_bfloat16>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
#endif
        case Type_t::f16:
            return call<__half>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
        case Type_t::f32:
            return call<float>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
        case Type_t::f64:
            return call<double>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
        case Type_t::i8:
            return call<int8_t>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
        case Type_t::i16:
            return call<int16_t>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
        case Type_t::i32:
            return call<int32_t>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
        case Type_t::i64:
            return call<int64_t>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
        case Type_t::u8:
            return call<uint8_t>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
        case Type_t::u16:
            return call<uint16_t>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
        case Type_t::u32:
            return call<uint32_t>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
        case Type_t::u64:
            return call<uint64_t>(stream, src, dst, splitIdxs, axisSizes, axisOffsetSizes);
        default:
            throw_ov_exception(
                fmt::format("Input element type = {} is not supported by Split operation "
                            "!!",
                            static_cast<Type_t>(element_type_)));
    }
}

template <typename T>
void VariadicSplit::call(cudaStream_t stream,
                         const void *src,
                         void **dst,
                         const void *splitIdxs,
                         const void *axisSizes,
                         const void *axisOffsetSizes) const {
    variadic_split<T><<<num_blocks_, threads_per_block_, 0, stream>>>(num_all_chunks_,
                                                                      axis_split_step_size_,
                                                                      orig_axis_size_,
                                                                      static_cast<const T *>(src),
                                                                      reinterpret_cast<T **>(dst),
                                                                      static_cast<const size_t *>(splitIdxs),
                                                                      static_cast<const size_t *>(axisSizes),
                                                                      static_cast<const size_t *>(axisOffsetSizes));
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
