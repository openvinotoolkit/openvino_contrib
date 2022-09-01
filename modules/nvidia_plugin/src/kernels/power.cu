// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert.cuh"
#include "power.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct PowerOpImpl {
    __device__ static inline T op(T in0, T in1) { return pow(in0, in1); }
};

template <>
struct PowerOpImpl<__half> {
    __device__ static inline __half op(__half in0, __half in1) {
        return cast<__half>(powf(cast<float>(in0), cast<float>(in1)));
    }
};

Power::Power(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
    : impl_{element_type, out_num_elements, max_threads_per_block} {}

void Power::operator()(cudaStream_t stream,
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
