// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "elementwise_binary.cuh"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T>
struct PowerOpImpl;

/**
 * Elementwise power operation.
 */
class Power {
public:
    Power(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block);

    void operator()(cudaStream_t stream,
                    const void* in0,
                    const NumpyBroadcastMapper& in0_mapper,
                    const void* in1,
                    const NumpyBroadcastMapper& in1_mapper,
                    void* out) const;

private:
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

    ElementwiseBinary<SupportedElementTypes, PowerOpImpl> impl_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
