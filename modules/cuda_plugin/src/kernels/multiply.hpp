// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_type_traits.hpp"
#include "elementwise_binary.cuh"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
struct MultiplyOpImpl;

/**
 * Elementwise multiplication for tensors of integers.
 */
class Multiply {
public:
    Multiply(Type_t element_type, size_t max_threads_per_block, size_t in0_num_elements, size_t in1_num_elements);

    /**
     * @param out   Output buffer. Is expected to be large enough to fit max(in0_num_elements, in1_num_elements)
     * elements.
     */
    void operator()(cudaStream_t stream, const void* in0, const void* in1, void* out) const;

private:
    using SupportedElementTypes =
        ElementTypesSwitch<Type_t::i16, Type_t::i32, Type_t::i64, Type_t::u8, Type_t::u16, Type_t::u32, Type_t::u64>;
    ElementwiseBinary<SupportedElementTypes, MultiplyOpImpl> ewb_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
