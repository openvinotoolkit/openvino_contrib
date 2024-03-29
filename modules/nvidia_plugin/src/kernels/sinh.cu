// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sinh.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace cumath = CUDA::math;

template <typename T>
struct SinhOpImpl {
    __device__ static inline T op(T x) {
        return cumath::sinh(x);
    }
};

Sinh::Sinh(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
    : impl_{element_type, max_threads_per_block, num_elements} {}

void Sinh::operator()(cudaStream_t stream, const void* in0, void* out) const {
    impl_(stream, in0, out);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
