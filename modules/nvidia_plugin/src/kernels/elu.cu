// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda/float16.hpp>

#include "details/tensor_helpers.hpp"
#include "elu.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

namespace cumath = CUDA::math;
template <typename T>
struct EluOpImpl {
    __device__ static inline T op(T x, float alpha) {
        return x >= static_cast<T>(0.0) ? x : static_cast<T>(alpha) * (cumath::exp(x) - static_cast<T>(1.0));
    }
};

Elu::Elu(Type_t element_type, size_t max_threads_per_block, size_t num_elements, float alpha)
    : impl_{element_type, max_threads_per_block, num_elements}, alpha_{alpha} {}

void Elu::operator()(cudaStream_t stream, const void* in, void* out) const { impl_(stream, in, out, alpha_); }

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
