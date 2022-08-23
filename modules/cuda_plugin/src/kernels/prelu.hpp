// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_type_traits.hpp"
#include "elementwise_binary.cuh"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
struct PReluOpImpl;

/**
 * Elementwise PRelu operation.
 */
class PRelu {
public:
    PRelu(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block);

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
        Type_t::f64>;

    ElementwiseBinary<SupportedElementTypes, PReluOpImpl> impl_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
