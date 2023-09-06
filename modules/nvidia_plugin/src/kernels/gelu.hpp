// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "details/cuda_type_traits.hpp"
#include "details/elementwise_unary.cuh"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct GeluErfOpImpl;

template <typename T>
struct GeluTanhOpImpl;

/**
 * Elementwise Gelu operation implementastion with Erf approximation
 */
class GeluErf {
public:
    GeluErf(Type_t element_type, size_t max_threads_per_block, size_t num_elements);

    void operator()(cudaStream_t stream, const void* in0, void* out) const;

private:
    ElementwiseUnary<FloatElementTypesSwitch, GeluErfOpImpl> impl_;
};

/**
 * Elementwise Gelu operation implementastion with Tanh approximation
 */
class GeluTanh {
public:
    GeluTanh(Type_t element_type, size_t max_threads_per_block, size_t num_elements);

    void operator()(cudaStream_t stream, const void* in0, void* out) const;

private:
    ElementwiseUnary<FloatElementTypesSwitch, GeluTanhOpImpl> impl_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
