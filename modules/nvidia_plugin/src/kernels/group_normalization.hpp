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
 * @brief CUDA kernel wrapper for opset12 GroupNormalization.
 *
 * data is [N, C, *spatial], scale and bias are [C]. The C channels are split
 * into num_groups contiguous groups; for each (n, group) the mean and variance
 * are computed over all the group's elements (channels_per_group * spatial),
 * the data is normalized with (x - mean) / sqrt(var + epsilon), and then scaled
 * and shifted per channel: y = norm * scale[c] + bias[c].
 *
 * Launched as one block per (N * num_groups) group; threads of a block stride
 * over the group's elements and cooperate via shared memory for the two
 * reductions (mean, variance).
 */
class GroupNormalization {
public:
    GroupNormalization(Type_t data_type,
                       std::size_t batch,
                       std::size_t num_channels,
                       std::size_t num_groups,
                       std::size_t spatial_size,
                       double epsilon,
                       std::size_t num_blocks,
                       std::size_t threads_per_block);

    void operator()(cudaStream_t stream,
                    const void* data,
                    const void* scale,
                    const void* bias,
                    void* output) const;

    template <typename T>
    void call(cudaStream_t stream, const void* data, const void* scale, const void* bias, void* output) const;

private:
    Type_t data_type_;
    std::size_t batch_;
    std::size_t num_channels_;
    std::size_t num_groups_;
    std::size_t spatial_size_;
    double epsilon_;
    std::size_t num_blocks_;
    std::size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
