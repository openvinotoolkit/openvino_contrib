// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ov {
namespace nvidia_gpu {
namespace kernel {

/**
 * @brief CUDA kernel wrapper for opset8 GatherND.
 *
 * Flattens the problem to [batch_size, num_index_tuples, slice_size]: for each
 * batch and index tuple, the L last-axis index values select a starting offset
 * inside the data batch block, and a contiguous slice of `slice_size` elements
 * is copied. One thread per output element.
 */
struct GatherNDParams {
    static constexpr int kMaxIndexLen = 8;
    size_t batch_size;
    size_t num_index_tuples;
    size_t slice_size;
    size_t batch_block_size;       // product of data dims from batch_dims onward
    int index_len;                 // L = indices.shape[-1]
    size_t num_output_elements;
    size_t indexed_stride[kMaxIndexLen];  // data row-major stride of each indexed dim
    long long indexed_dim[kMaxIndexLen];  // extent of each indexed dim (negative-index wrap)
};

void gather_nd(cudaStream_t stream,
               const GatherNDParams& params,
               size_t element_size,
               size_t index_size,
               const void* data,
               const void* indices,
               void* output);

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
