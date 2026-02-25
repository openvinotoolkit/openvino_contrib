// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class VariadicSplit {
public:
    VariadicSplit(Type_t element_type,
                  size_t num_all_chunks,
                  size_t axis_split_step_size,
                  size_t orig_axis_size,
                  unsigned num_blocks,
                  unsigned threads_per_block);
    VariadicSplit(VariadicSplit &&) = default;
    VariadicSplit &operator=(VariadicSplit &&) = default;

    void operator()(cudaStream_t stream,
                    const void *src,
                    void **dst,
                    const void *splitIdxs,
                    const void *axisSizes,
                    const void *axisOffsetSizes) const;

private:
    template <typename T>
    void call(cudaStream_t stream,
              const void *src,
              void **dst,
              const void *splitIdxs,
              const void *axisSizes,
              const void *axisOffsetSizes) const;

    Type_t element_type_{};
    size_t num_all_chunks_{};
    size_t axis_split_step_size_{};
    size_t orig_axis_size_{};
    unsigned num_blocks_{};
    unsigned threads_per_block_{};
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
