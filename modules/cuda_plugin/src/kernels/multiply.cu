// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multiply.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
struct MultiplyOpImpl {
    __device__ static inline T op(T in0, T in1) { return in0 * in1; }
};

Multiply::Multiply(Type_t element_type, size_t max_threads_per_block, size_t in0_num_elements, size_t in1_num_elements)
    : ewb_{element_type, max_threads_per_block, in0_num_elements, in1_num_elements} {}

void Multiply::operator()(cudaStream_t stream, const void* in0, const void* in1, void* out) const {
    ewb_(stream, in0, in1, out);
}

}  // namespace kernel
}  // namespace CUDAPlugin
