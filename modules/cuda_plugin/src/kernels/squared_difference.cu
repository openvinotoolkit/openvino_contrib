// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/float16.hpp>
#include <type_traits>

#include "squared_difference.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
struct SquaredDifferenceOpImpl {
    __device__ static inline T op(T in0, T in1) {
        const T diff = in0 - in1;
        return diff * diff;
    }
};

SquaredDifference::SquaredDifference(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
    : impl_{element_type, out_num_elements, max_threads_per_block} {}

void SquaredDifference::operator()(cudaStream_t stream,
                                   const void* in0,
                                   const NumpyBroadcastMapper& in0_mapper,
                                   const void* in1,
                                   const NumpyBroadcastMapper& in1_mapper,
                                   void* out) const {
    impl_(stream, in0, in0_mapper, in1, in1_mapper, out);
}

}  // namespace kernel
}  // namespace CUDAPlugin
