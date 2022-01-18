// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert.cuh"
#include "elementwise_binary.cuh"
#include "power.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
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

Power::Power(Type_t element_type, size_t max_threads_per_block)
    : element_type_{element_type}, max_threads_per_block_{max_threads_per_block} {}

void Power::operator()(cudaStream_t stream,
                       const void* in0,
                       size_t in0_num_elements,
                       const void* in1,
                       size_t in1_num_elements,
                       void* out) const {
    using SupportedElementTypes = ElementTypesSwitch<
#ifdef CUDA_HAS_BF16_TYPE
        Type_t::bf16,
#endif
        Type_t::f16,
        Type_t::f32,
        Type_t::f64,
        Type_t::i8,
        Type_t::i16,
        Type_t::i32,
        Type_t::i64,
        Type_t::u8,
        Type_t::u16,
        Type_t::u32,
        Type_t::u64>;
    using Switcher = ElementwiseBinary<SupportedElementTypes, PowerOpImpl>;
    Switcher switcher{element_type_, max_threads_per_block_, in0_num_elements, in1_num_elements};
    switcher(stream, in0, in1, out);
}

}  // namespace kernel
}  // namespace CUDAPlugin
