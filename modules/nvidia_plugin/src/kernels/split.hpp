// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class Split {
public:
    Split(Type_t element_type,
          size_t num_splits,
          size_t num_split_chunks,
          size_t split_step_size,
          unsigned num_blocks,
          unsigned threads_per_block);
    Split(Split&&) = default;
    Split& operator=(Split&&) = default;

    void operator()(cudaStream_t stream, const void* src, void** dst) const;

    [[nodiscard]] size_t mutableWbSize() const { return sizeof(float*) * num_splits_; }

private:
    template <typename T>
    void Call(cudaStream_t stream, const void* src, void** dst) const;

    Type_t element_type_{};
    size_t num_splits_{};
    size_t num_split_chunks_{};
    size_t split_step_size_{};
    unsigned num_blocks_{};
    unsigned threads_per_block_{};
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
