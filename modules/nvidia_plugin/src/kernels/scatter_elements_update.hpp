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
 * @brief CUDA kernel wrapper for ScatterElementsUpdate (reduction = none).
 *
 * output starts as a copy of `data`; then for each element c of indices/updates
 * (same shape), output[c with c[axis] replaced by indices[c]] = updates[c].
 *
 * One thread per "row" — the non-axis coordinates of the indices/updates shape.
 * Each thread walks the axis sequentially, so duplicate target indices resolve
 * to last-write-wins with no inter-thread race (distinct rows never target the
 * same output element).
 */
struct ScatterElementsUpdateParams {
    static constexpr int kMaxRank = 8;
    int rank;
    int axis;
    size_t num_rows;            // product of the updates' non-axis extents
    size_t updates_axis_size;   // updates.shape[axis]
    long long data_axis_dim;    // data.shape[axis], for negative-index wrap
    size_t num_data_elements;   // for the initial data -> output copy
    size_t updates_shape[kMaxRank];    // indices/updates extents
    size_t updates_strides[kMaxRank];  // row-major strides of the indices/updates shape
    size_t data_strides[kMaxRank];     // row-major strides of the data/output shape
};

void scatter_elements_update(cudaStream_t stream,
                             const ScatterElementsUpdateParams& params,
                             size_t element_size,
                             size_t index_size,
                             const void* data,
                             const void* indices,
                             const void* updates,
                             void* output);

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
