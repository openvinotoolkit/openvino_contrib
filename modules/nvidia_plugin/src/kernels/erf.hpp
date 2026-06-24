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
struct ErfOpImpl;

class Erf {
public:
    Erf(Type_t element_type, size_t max_threads_per_block, size_t num_elements);
    Erf(Erf&&) = default;
    Erf& operator=(Erf&&) = default;

    void operator()(cudaStream_t stream, const void* in, void* out) const;

private:
    ElementwiseUnary<FloatElementTypesSwitch, ErfOpImpl> ewu_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
