// Copyright (C) 2021-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ov {
namespace nvidia_gpu {
namespace kernel {

/**
 * @brief Generic tensor permutation kernel, used as a fallback when the
 *        cuTENSOR permutation path is unavailable on the current device
 *        (e.g. cuTENSOR built without SASS/PTX for this GPU architecture).
 *
 * One thread per output element: the linear output index is decomposed into
 * output coordinates, and the input offset is computed via the permuted input
 * strides. Supports tensors up to kMaxRank dims and 1/2/4/8-byte elements.
 */
struct PermuteFallbackParams {
    static constexpr size_t kMaxRank = 8;
    int rank;
    size_t num_elements;
    int64_t out_extents[kMaxRank];
    int64_t out_strides[kMaxRank];
    // Input stride for the dimension that maps to output dim k.
    int64_t in_strides_permuted[kMaxRank];
};

void permute_fallback(cudaStream_t stream,
                      const PermuteFallbackParams& params,
                      size_t element_size,
                      const void* src,
                      void* dst);

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
