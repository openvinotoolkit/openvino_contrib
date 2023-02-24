// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "details/cuda_type_traits.hpp"
#include "details/elementwise_binary.cuh"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct DivideOpImpl;

template <typename T>
struct PythonDivideOpImpl;

/**
 * Performs element-wise Divide operation with two given tensors applying numpy
 * broadcasting if needed.
 */
class Divide {
public:
    Divide(Type_t element_type, size_t max_threads_per_block, size_t out_num_elements);

    void operator()(cudaStream_t stream,
                    const void* in0,
                    const NumpyBroadcastMapper& in0_mapper,
                    const void* in1,
                    const NumpyBroadcastMapper& in1_mapper,
                    void* out) const;

private:
    ElementwiseBinary<AllElementTypesSwitch, DivideOpImpl> impl_;
};

/**
 * Performs element-wise PythonDivide (floor division) operation with two given tensors applying numpy
 * broadcasting if needed.
 */
class PythonDivide {
public:
    PythonDivide(Type_t element_type, size_t max_threads_per_block, size_t out_num_elements);

    void operator()(cudaStream_t stream,
                    const void* in0,
                    const NumpyBroadcastMapper& in0_mapper,
                    const void* in1,
                    const NumpyBroadcastMapper& in1_mapper,
                    void* out) const;

private:
    ElementwiseBinary<AllElementTypesSwitch, PythonDivideOpImpl> impl_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
