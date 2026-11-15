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
 * @brief CUDA kernel wrapper for opset6 GatherElements.
 *
 * data and indices share the same rank; the output has the indices' shape. For
 * each output coordinate c, output[c] = data[c with c[axis] replaced by
 * indices[c]]. One thread per output element: it decomposes the output index
 * with the indices' row-major strides, substitutes the gathered axis
 * coordinate, and reads the data element via the data's row-major strides.
 */
struct GatherElementsParams {
    static constexpr int kMaxRank = 8;
    int rank;
    int axis;
    size_t num_output_elements;
    long long data_axis_dim;            // data.shape[axis], for negative-index wrap
    size_t indices_strides[kMaxRank];   // row-major strides of the indices/output shape
    size_t data_strides[kMaxRank];      // row-major strides of the data shape
};

void gather_elements(cudaStream_t stream,
                     const GatherElementsParams& params,
                     size_t element_size,
                     size_t index_size,
                     const void* data,
                     const void* indices,
                     void* output);

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
