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
 * @brief CUDA kernel wrapper for opset5 LogSoftmax.
 *
 * Computes, along `axis`, the numerically stable
 *   y = (x - max) - log(sum(exp(x - max))).
 *
 * The tensor is viewed as [outer, axis, inner] (the axis to reduce in the
 * middle). One block per (outer, inner) row reduces the axis elements (strided
 * by inner_size) for the max and the exp-sum via shared memory, accumulating in
 * float (double for f64).
 */
class LogSoftmax {
public:
    LogSoftmax(Type_t data_type,
               std::size_t outer_size,
               std::size_t axis_size,
               std::size_t inner_size,
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
    std::size_t num_blocks_;
    std::size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
