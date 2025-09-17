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
struct AcosOpImpl;
/**
 * Elementwise Acos operation
 */
class Acos {
public:
    Acos(Type_t element_type, size_t max_threads_per_block, size_t num_elements);

    void operator()(cudaStream_t stream, const void* in0, void* out) const;

private:
    ElementwiseUnary<AllElementTypesSwitch, AcosOpImpl> impl_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
