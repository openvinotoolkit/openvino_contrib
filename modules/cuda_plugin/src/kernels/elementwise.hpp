// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_type_traits.hpp"

namespace CUDAPlugin {
namespace kernel {

class Elementwise {
public:
    enum class Op_t { add, mul };

    Elementwise(Op_t op_type, Type_t element_type, size_t max_threads_per_block);
    Elementwise(Elementwise&&) = default;
    Elementwise& operator=(Elementwise&&) = default;

    /**
     * @param out   Output buffer. Is expected to be large enough to fit max(in0_num_elements, in1_num_elements)
     * elements.
     */
    void operator()(cudaStream_t stream,
                    const void* in0,
                    size_t in0_num_elements,
                    const void* in1,
                    size_t in1_num_elements,
                    void* out) const;

private:
    Op_t op_type_;
    Type_t element_type_;
    size_t max_threads_per_block_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
