// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fmt/format.h>

#include <tuple>
#include <utility>

#include "cuda_type_traits.hpp"
#include "element_types_switch.hpp"
#include "numpy_broadcast_mapper.cuh"
#include "tensor_helpers.hpp"
#include "type_validator.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

#ifdef __CUDACC__

template <typename T, typename OP, typename... Args>
__global__ void elementwise_binary(const T* in0, const T* in1, T* out, size_t out_num_elements, Args... args) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_num_elements) {
        out[i] = OP::op(in0[i], in1[i], args...);
    }
}

template <typename T, typename OP, typename... Args>
__global__ void elementwise_binary_broadcasting(const T* in0,
                                                NumpyBroadcastMapper in0_mapper,
                                                const T* in1,
                                                NumpyBroadcastMapper in1_mapper,
                                                T* out,
                                                size_t out_num_elements,
                                                Args... args) {
    const unsigned out_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_index < out_num_elements) {
        const unsigned in0_index = in0_mapper.srcIndex(out_index);
        const unsigned in1_index = in1_mapper.srcIndex(out_index);
        out[out_index] = OP::op(in0[in0_index], in1[in1_index], args...);
    }
}

#endif  // __CUDACC__

template <typename ElementTypes, template <typename... TArgs> typename OP>
class ElementwiseBinary {
public:
    ElementwiseBinary(Type_t element_type, size_t out_num_elements, size_t max_threads_per_block)
        : num_blocks_{}, threads_per_block_{}, element_type_{element_type}, out_num_elements_{out_num_elements} {
        TypeValidator<ElementTypes>::check(element_type);
        std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(out_num_elements, max_threads_per_block);
    }

    template <typename... Args>
    void operator()(cudaStream_t stream,
                    const void* in0,
                    const NumpyBroadcastMapper& in0_mapper,
                    const void* in1,
                    const NumpyBroadcastMapper& in1_mapper,
                    void* out,
                    Args&&... args) const {
        if (in0_mapper.identity() && in1_mapper.identity()) {
            (*this)(stream, in0, in1, out, std::forward<Args>(args)...);
        } else {
            ElementTypes::switch_(
                element_type_, *this, stream, in0, in0_mapper, in1, in1_mapper, out, std::forward<Args>(args)...);
        }
    }

    /**
     * Simple variant of elementwise invocation for the case when all input and output shapes are the same.
     * It is expected to be more quick then generic variant which supports broadcasting.
     */
    template <typename... Args>
    void operator()(cudaStream_t stream, const void* in0, const void* in1, void* out, Args&&... args) const {
        ElementTypes::switch_(element_type_, *this, stream, in0, in1, out, std::forward<Args>(args)...);
    }

private:
    friend ElementTypes;

    template <typename T, typename... Args>
    constexpr void case_(cudaStream_t stream,
                         const void* in0,
                         const NumpyBroadcastMapper& in0_mapper,
                         const void* in1,
                         const NumpyBroadcastMapper& in1_mapper,
                         void* out,
                         Args&&... args) const noexcept {
#ifdef __CUDACC__
        elementwise_binary_broadcasting<T, OP<T>>
            <<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const T*>(in0),
                                                             in0_mapper,
                                                             static_cast<const T*>(in1),
                                                             in1_mapper,
                                                             static_cast<T*>(out),
                                                             out_num_elements_,
                                                             std::forward<Args>(args)...);
#endif  // __CUDACC__
    }

    template <typename T, typename... Args>
    void default_(T t,
                  cudaStream_t,
                  const void*,
                  const NumpyBroadcastMapper&,
                  const void*,
                  const NumpyBroadcastMapper&,
                  void*,
                  Args...) const noexcept {
        throwTypeNotSupported(t);
    }

    template <typename T, typename... Args>
    constexpr void case_(cudaStream_t stream, const void* in0, const void* in1, void* out, Args&&... args) const
        noexcept {
#ifdef __CUDACC__
        elementwise_binary<T, OP<T>><<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const T*>(in0),
                                                                                     static_cast<const T*>(in1),
                                                                                     static_cast<T*>(out),
                                                                                     out_num_elements_,
                                                                                     std::forward<Args>(args)...);
#endif  // __CUDACC__
    }

    template <typename T, typename... Args>
    void default_(T t, cudaStream_t, const void*, const void*, void*, Args...) const noexcept {
        throwTypeNotSupported(t);
    }

private:
    size_t num_blocks_;
    size_t threads_per_block_;
    Type_t element_type_;
    size_t out_num_elements_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
