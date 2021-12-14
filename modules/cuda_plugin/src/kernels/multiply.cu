// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "elementwise.cuh"
#include "multiply.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
struct MultiplyOpImpl {
    __device__ static inline T op(T in0, T in1) { return in0 * in1; }
};

Multiply::Multiply(Type_t element_type, size_t max_threads_per_block)
    : element_type_{element_type}, max_threads_per_block_{max_threads_per_block} {}

void Multiply::operator()(cudaStream_t stream,
                          const void* in0,
                          size_t in0_num_elements,
                          const void* in1,
                          size_t in1_num_elements,
                          void* out) const {
    using SupportedElementTypes =
        ElementTypesSwitch<Type_t::i16, Type_t::i32, Type_t::i64, Type_t::u8, Type_t::u16, Type_t::u32, Type_t::u64>;
    using Helper = ElementwiseHelper<SupportedElementTypes, MultiplyOpImpl>;
    Helper helper{element_type_, max_threads_per_block_};
    helper.binaryOperator(stream, in0, in0_num_elements, in1, in1_num_elements, out);
}

}  // namespace kernel
}  // namespace CUDAPlugin
