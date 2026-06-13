// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

/**
 * @brief CUDA kernel wrapper for opset3 CumSum.
 *
 * The tensor is viewed as [outer, axis, inner]; one thread per (outer, inner)
 * row sequentially scans the axis. `exclusive` writes the running sum before
 * adding the current element; `reverse` scans the axis from the end. Float
 * inputs accumulate in float/double; integers accumulate in their own type
 * (matching the reference wrap-around).
 */
class CumSum {
public:
    CumSum(Type_t data_type,
           std::size_t outer_size,
           std::size_t axis_size,
           std::size_t inner_size,
           bool exclusive,
           bool reverse,
           std::size_t num_blocks,
           std::size_t threads_per_block);

    void operator()(cudaStream_t stream, const void* input, void* output) const;

    template <typename T>
    void call(cudaStream_t stream, const void* input, void* output) const;

private:
    Type_t data_type_;
    std::size_t outer_size_;
    std::size_t axis_size_;
    std::size_t inner_size_;
    bool exclusive_;
    bool reverse_;
    std::size_t num_blocks_;
    std::size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
