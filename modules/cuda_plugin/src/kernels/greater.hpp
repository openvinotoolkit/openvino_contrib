// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "cuda_type_traits.hpp"

namespace CUDAPlugin {
namespace kernel {

class Greater {
public:
    Greater(Type_t element_type, size_t max_size, size_t numBlocks, size_t threadsPerBlock);
    Greater(Greater&&) = default;
    Greater& operator=(Greater&&) = default;

    void operator()(const cudaStream_t stream,
                    const void* left_src,
                    const void* right_src,
                    const size_t* left_brcst_offsets,
                    const size_t* right_brcst_offsets,
                    const size_t* output_sizes,
                    void* dst) const;

private:
    template <typename T>
    void Call(const cudaStream_t stream,
              const void* left_src,
              const void* right_src,
              const size_t* left_brcst_offsets,
              const size_t* right_brcst_offsets,
              const size_t* output_sizes,
              void* dst) const;

    Type_t element_type_{};
    size_t max_size_{};
    size_t num_blocks_{};
    size_t threads_per_block_{};
};

}  // namespace kernel
}  // namespace CUDAPlugin
