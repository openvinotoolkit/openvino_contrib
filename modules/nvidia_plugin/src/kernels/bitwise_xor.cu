// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bitwise_xor.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct BitwiseXorOpImpl {
    __device__ static inline T op(T in0, T in1) { return in0 ^ in1; }
};

BitwiseXor::BitwiseXor(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
    : impl_{element_type, out_num_elements, max_threads_per_block} {}

void BitwiseXor::operator()(cudaStream_t stream,
                              const void* in0,
                              const NumpyBroadcastMapper& in0_mapper,
                              const void* in1,
                              const NumpyBroadcastMapper& in1_mapper,
                              void* out) const {
    impl_(stream, in0, in0_mapper, in1, in1_mapper, out);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
