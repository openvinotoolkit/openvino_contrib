// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bitwise_not.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct BitwiseNotOpImpl {
    __device__ static inline T op(T x) { return ~x; }
};

template <>
struct BitwiseNotOpImpl<bool> {
    __device__ static inline bool op(bool x) { return !x; }
};

BitwiseNot::BitwiseNot(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
    : ewu_{element_type, max_threads_per_block, num_elements} {}

void BitwiseNot::operator()(cudaStream_t stream, const void* in, void* out) const { ewu_(stream, in, out); }

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
