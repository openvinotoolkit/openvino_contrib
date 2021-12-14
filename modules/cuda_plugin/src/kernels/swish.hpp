// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_type_traits.hpp"
#include "error.hpp"

namespace CUDAPlugin {
namespace kernel {

/**
 * Elementwise multiplication for tensors of integers.
 */
class Swish {
public:
    Swish(Type_t element_type, size_t max_threads_per_block);
    Swish(Swish&&) = default;
    Swish& operator=(Swish&&) = default;

    /**
     * @param out   Output buffer. Is expected to be large enough to fit max(in0_num_elements, in1_num_elements)
     * elements.
     */
    void operator()(cudaStream_t stream, const void* in, void* out, size_t num_elements, double beta) const;

private:
    Type_t element_type_;
    size_t max_threads_per_block_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
