// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "details/cuda_type_traits.hpp"
#include "details/elementwise_unary.cuh"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct BitwiseNotOpImpl;

class BitwiseNot {
public:
    BitwiseNot(Type_t element_type, size_t max_threads_per_block, size_t num_elements);
    BitwiseNot(BitwiseNot&&) = default;
    BitwiseNot& operator=(BitwiseNot&&) = default;

    void operator()(cudaStream_t stream, const void* in, void* out) const;

private:
    ElementwiseUnary<IntegerAndBooleanElementTypesSwitch, BitwiseNotOpImpl> ewu_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
