// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_type_traits.hpp"
#include "error.hpp"

namespace CUDAPlugin {
namespace kernel {

class Comparison {
public:
    enum class Op_t { GREATER, LESS };

    Comparison(Op_t op, Type_t element_type, size_t max_size, size_t num_blocks, size_t threads_per_block);
    Comparison(Comparison&&) = default;
    Comparison& operator=(Comparison&&) = default;

    void operator()(const cudaStream_t stream,
                    const void* left_src,
                    const void* right_src,
                    const size_t* left_brcst_offsets,
                    const size_t* right_brcst_offsets,
                    const size_t* output_sizes,
                    void* dst) const;

private:
    template <typename T>
    void Call(Comparison::Op_t type,
              const cudaStream_t stream,
              const void* left_src,
              const void* right_src,
              const size_t* left_brcst_offsets,
              const size_t* right_brcst_offsets,
              const size_t* output_sizes,
              void* dst) const;

    template <typename T, Op_t OP>
    void Call(const cudaStream_t stream,
              const void* left_src,
              const void* right_src,
              const size_t* left_brcst_offsets,
              const size_t* right_brcst_offsets,
              const size_t* output_sizes,
              void* dst) const;
    Op_t op_type_{};
    Type_t element_type_{};
    size_t max_size_{};
    size_t num_blocks_{};
    size_t threads_per_block_{};
};

}  // namespace kernel
}  // namespace CUDAPlugin
