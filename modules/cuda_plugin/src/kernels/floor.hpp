// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_type_traits.hpp"
#include "elementwise_unary.cuh"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
struct FloorOpImpl;

class Floor {
public:
    Floor(Type_t element_type, size_t max_threads_per_block, size_t num_elements);
    Floor(Floor&&) = default;
    Floor& operator=(Floor&&) = default;

    void operator()(cudaStream_t stream, const void* in, void* out) const;

private:
    using SupportedElementTypes = ElementTypesSwitch<Type_t::f16,
                                                     Type_t::f32,
                                                     Type_t::f64
#if CUDA_VERSION >= 11000
                                                     ,
                                                     Type_t::bf16
#endif  // CUDA_VERSION >= 11000
                                                     >;
private:
    ElementwiseUnary<SupportedElementTypes, FloorOpImpl> ewu_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
