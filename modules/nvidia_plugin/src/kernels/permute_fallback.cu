// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include "details/error.hpp"
#include "permute_fallback.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
static __global__ void permute_fallback_kernel(const PermuteFallbackParams params, const T* src, T* dst) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= params.num_elements) {
        return;
    }
    size_t rem = idx;
    size_t in_offset = 0;
    size_t out_offset = 0;
    for (int k = 0; k < params.rank; ++k) {
        const size_t coord = rem / static_cast<size_t>(params.out_strides[k]);
        rem -= coord * static_cast<size_t>(params.out_strides[k]);
        in_offset += coord * static_cast<size_t>(params.in_strides_permuted[k]);
        out_offset += coord * static_cast<size_t>(params.out_strides[k]);
    }
    dst[out_offset] = src[in_offset];
}

void permute_fallback(cudaStream_t stream,
                      const PermuteFallbackParams& params,
                      size_t element_size,
                      const void* src,
                      void* dst) {
    if (params.num_elements == 0) {
        return;
    }
    constexpr unsigned kBlock = 256;
    const unsigned blocks = static_cast<unsigned>((params.num_elements + kBlock - 1) / kBlock);
    switch (element_size) {
        case 1:
            permute_fallback_kernel<uint8_t>
                <<<blocks, kBlock, 0, stream>>>(params, static_cast<const uint8_t*>(src), static_cast<uint8_t*>(dst));
            break;
        case 2:
            permute_fallback_kernel<uint16_t><<<blocks, kBlock, 0, stream>>>(
                params, static_cast<const uint16_t*>(src), static_cast<uint16_t*>(dst));
            break;
        case 4:
            permute_fallback_kernel<uint32_t><<<blocks, kBlock, 0, stream>>>(
                params, static_cast<const uint32_t*>(src), static_cast<uint32_t*>(dst));
            break;
        case 8:
            permute_fallback_kernel<uint64_t><<<blocks, kBlock, 0, stream>>>(
                params, static_cast<const uint64_t*>(src), static_cast<uint64_t*>(dst));
            break;
        default:
            throw_ov_exception(fmt::format("permute_fallback: unsupported element size {}", element_size));
    }
    throwIfError(cudaPeekAtLastError());
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
