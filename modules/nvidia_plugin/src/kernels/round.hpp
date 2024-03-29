// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "details/cuda_type_traits.hpp"
#include "details/elementwise_unary.cuh"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct RoundOpImpl;

class Round {
public:
    Round(Type_t element_type, size_t max_threads_per_block, size_t num_elements);
    Round(Round&&) = default;
    Round& operator=(Round&&) = default;

    void operator()(cudaStream_t stream, const void* in, void* out) const;

private:
    ElementwiseUnary<FloatElementTypesSwitch, RoundOpImpl> ewu_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
