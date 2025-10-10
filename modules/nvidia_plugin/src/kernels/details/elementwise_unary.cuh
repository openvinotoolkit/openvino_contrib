// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fmt/format.h>

#include <tuple>
#include <utility>

#include "cuda_type_traits.hpp"
#include "element_types_switch.hpp"
#include "tensor_helpers.hpp"
#include "type_validator.hpp"
#ifdef __CUDACC__
#include "cuda/math.cuh"
#endif  // __CUDACC__

namespace ov {
namespace nvidia_gpu {
namespace kernel {

#ifdef __CUDACC__

template <typename T, typename OP, typename... Args>
__global__ void elementwise_unary(const T* in, size_t num_elements, T* out, Args... args) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        out[i] = OP::op(in[i], args...);
    }
}

#endif  // __CUDACC__

template <typename ElementTypes, template <typename... TArgs> typename OP>
class ElementwiseUnary {
public:
    ElementwiseUnary(Type_t element_type, size_t max_threads_per_block, size_t num_elements)
        : element_type_{element_type}, num_elements_{num_elements} {
        TypeValidator<ElementTypes>::check(element_type);
        std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(num_elements_, max_threads_per_block);
    }

    /**
     * @param out   Output buffer. Is expected to be of the same size as input (num_elements * sizeof(<one_element>))
     */
    template <typename... Args>
    void operator()(cudaStream_t stream, const void* in, void* out, Args&&... args) const {
        ElementTypes::switch_(element_type_, *this, stream, in, out, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    constexpr void callKernel(cudaStream_t stream, const void* in, void* out, Args&&... args) const noexcept {
#ifdef __CUDACC__
        elementwise_unary<T, OP<T>><<<num_blocks_, threads_per_block_, 0, stream>>>(
            static_cast<const T*>(in), num_elements_, static_cast<T*>(out), std::forward<Args>(args)...);
#endif  // __CUDACC__
    }

    template <typename T, typename... Args>
    constexpr void case_(cudaStream_t stream, const void* in, void* out, Args&&... args) const noexcept {
        callKernel<T>(stream, in, out, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    void default_(T t, cudaStream_t, const void*, void*, Args...) const noexcept {
        throwTypeNotSupported(t);
    }

private:
    Type_t element_type_;
    size_t num_elements_;
    size_t num_blocks_;
    size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
