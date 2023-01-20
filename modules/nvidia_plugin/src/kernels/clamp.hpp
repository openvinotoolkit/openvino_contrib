// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_type_traits.hpp"
#include "elementwise_unary.cuh"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct ClampOpImpl;

using EWClamp = ElementwiseUnary<AllElementTypesSwitch, ClampOpImpl>;

class Clamp {
public:
    Clamp(Type_t element_type, size_t max_threads_per_block, size_t num_elements, double min, double max);
    Clamp(Clamp&&) = default;
    Clamp& operator=(Clamp&&) = default;

    void operator()(cudaStream_t stream, const void* in, void* out) const;

private:
    Type_t element_type_;
    EWClamp ew_clamp_;
    double min_;
    double max_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
