// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/float16.hpp>

#include "cuda_type_traits.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

#ifdef CUDA_HAS_BF16_TYPE
template <typename TOutput, typename TInput>
__device__
    typename std::enable_if<std::is_same<TInput, __half>::value || std::is_same<TInput, __nv_bfloat16>::value ||
                                std::is_same<TOutput, __half>::value || std::is_same<TOutput, __nv_bfloat16>::value,
                            TOutput>::type
#else
template <typename TOutput, typename TInput>
__device__
    typename std::enable_if<std::is_same<TInput, __half>::value || std::is_same<TOutput, __half>::value, TOutput>::type
#endif
    cast(TInput in) {
    return static_cast<TOutput>(static_cast<float>(in));
}

#ifdef CUDA_HAS_BF16_TYPE
template <typename TOutput, typename TInput>
__device__
    typename std::enable_if<!(std::is_same<TInput, __half>::value || std::is_same<TInput, __nv_bfloat16>::value ||
                              std::is_same<TOutput, __half>::value || std::is_same<TOutput, __nv_bfloat16>::value),
                            TOutput>::type
#else
template <typename TOutput, typename TInput>
__device__ typename std::enable_if<!(std::is_same<TInput, __half>::value || std::is_same<TOutput, __half>::value),
                                   TOutput>::type
#endif
    cast(TInput in) {
    return static_cast<TOutput>(in);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
