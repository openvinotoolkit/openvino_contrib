// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "details/cuda_type_traits.hpp"
#include "details/elementwise_binary.cuh"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct LogicalXorOpImpl;

/**
 * Performs element-wise LogicalXor operation with two given tensors applying numpy
 * broadcasting if needed.
 */
class LogicalXor {
public:
    LogicalXor(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block);

    void operator()(cudaStream_t stream,
                    const void* in0,
                    const NumpyBroadcastMapper& in0_mapper,
                    const void* in1,
                    const NumpyBroadcastMapper& in1_mapper,
                    void* out) const;

private:
    ElementwiseBinary<ElementTypesSwitch<Type_t::boolean>, LogicalXorOpImpl> impl_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
