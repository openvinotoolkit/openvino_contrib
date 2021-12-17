// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "cuda_type_traits.hpp"

namespace CUDAPlugin {
namespace kernel {

#if CUDA_VERSION >= 11000
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

#if CUDA_VERSION >= 11000
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
}  // namespace CUDAPlugin
