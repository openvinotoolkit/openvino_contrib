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
 * @brief CUDA kernel wrapper for opset1 Tile.
 *
 * Repeats `data` along each axis according to `repeats`. One thread per output
 * element: the output linear index is decomposed into output coordinates, each
 * is taken modulo the (rank-aligned) input extent, and the resulting input
 * offset is copied. Element-type agnostic — dispatched by element byte size.
 */
struct TileParams {
    static constexpr size_t kMaxRank = 8;
    int rank;
    size_t num_output_elements;
    size_t input_shape[kMaxRank];     // input extents, left-padded with 1s to the output rank
    size_t input_strides[kMaxRank];   // row-major strides of the padded input shape
    size_t output_strides[kMaxRank];  // row-major strides of the output shape
};

void tile(cudaStream_t stream,
          const TileParams& params,
          size_t element_size,
          const void* src,
          void* dst);

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
