// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fmt/format.h>

#include <gsl/gsl_assert>
#include <tuple>
#include <utility>

#include "cuda_type_traits.hpp"
#include "elementtypeswitch.hpp"
#include "error.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

#ifdef __CUDACC__

template <typename T, typename OP, typename... Args>
__global__ void elementwise_unary(const T* in, size_t num_elements, T* out, Args... args) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        out[i] = OP::op(in[i], args...);
    }
}

template <typename T, typename OP, typename... Args>
__global__ void elementwise_binary(const T* in0, const T* in1, size_t num_elements, T* out, Args... args) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        out[i] = OP::op(in0[i], in1[i], args...);
    }
}

template <typename T, typename OP, typename... Args>
__global__ void elementwise_binary_broadcasting(
    const T* in, size_t in_num_elements, const T* broadcasted, size_t broadcasted_num_elements, T* out, Args... args) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < in_num_elements) {
        out[i] = OP::op(in[i], broadcasted[i % broadcasted_num_elements], args...);
    }
}

#endif  // __CUDACC__

template <typename ElementTypes, template <typename> typename OP>
class ElementwiseUnary {
public:
    ElementwiseUnary(Type_t element_type, size_t max_threads_per_block)
        : element_type_{element_type}, max_threads_per_block_{max_threads_per_block} {}

    /**
     * @param out   Output buffer. Is expected to be of the same size as input (num_elements * sizeof(<one_element>))
     */
    template <typename... Args>
    void operator()(cudaStream_t stream, const void* in, size_t num_elements, void* out, Args&&... args) const {
        ElementTypes::switch_(element_type_, *this, stream, in, num_elements, out, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    constexpr void callKernel(
        cudaStream_t stream, const void* in, size_t num_elements, void* out, Args&&... args) const noexcept {
#ifdef __CUDACC__
        size_t num_blocks{}, threads_per_block{};
        std::tie(num_blocks, threads_per_block) = calculateElementwiseGrid(num_elements, max_threads_per_block_);
        elementwise_unary<T, OP<T>><<<num_blocks, threads_per_block, 0, stream>>>(
            static_cast<const T*>(in), num_elements, static_cast<T*>(out), std::forward<Args>(args)...);
#endif  // __CUDACC__
    }

    template <typename T, typename... Args>
    constexpr void case_(
        cudaStream_t stream, const void* in, size_t num_elements, void* out, Args&&... args) const noexcept {
        callKernel<T>(stream, in, num_elements, out, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    void default_(T t, cudaStream_t, const void*, size_t, void*, size_t, Args...) const noexcept {
        throwIEException(fmt::format("Element type = {} is not supported.", t));
    }

private:
    Type_t element_type_;
    size_t max_threads_per_block_;
};

template <typename ElementTypes, template <typename> typename OP>
class ElementwiseBinary {
public:
    ElementwiseBinary(Type_t element_type, size_t max_threads_per_block)
        : element_type_{element_type}, max_threads_per_block_{max_threads_per_block} {}

    /**
     * @param out   Output buffer. Is expected to be large enough to fit
     * max(in0_num_elements, in1_num_elements) elements.
     */
    template <typename... Args>
    void operator()(cudaStream_t stream,
                    const void* in0,
                    size_t in0_num_elements,
                    const void* in1,
                    size_t in1_num_elements,
                    void* out,
                    Args&&... args) const {
        ElementTypes::switch_(element_type_,
                              *this,
                              stream,
                              in0,
                              in0_num_elements,
                              in1,
                              in1_num_elements,
                              out,
                              std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    constexpr void callKernel(cudaStream_t stream,
                              const void* in0,
                              size_t in0_num_elements,
                              const void* in1,
                              size_t in1_num_elements,
                              void* out,
                              Args&&... args) const noexcept {
#ifdef __CUDACC__
        size_t num_blocks{}, threads_per_block{};
        if (in0_num_elements == in1_num_elements) {
            std::tie(num_blocks, threads_per_block) =
                calculateElementwiseGrid(in0_num_elements, max_threads_per_block_);
            elementwise_binary<T, OP<T>><<<num_blocks, threads_per_block, 0, stream>>>(static_cast<const T*>(in0),
                                                                                       static_cast<const T*>(in1),
                                                                                       in0_num_elements,
                                                                                       static_cast<T*>(out),
                                                                                       std::forward<Args>(args)...);
        } else if (in0_num_elements < in1_num_elements) {
            Expects(in1_num_elements % in0_num_elements == 0);
            std::tie(num_blocks, threads_per_block) =
                calculateElementwiseGrid(in1_num_elements, max_threads_per_block_);
            elementwise_binary_broadcasting<T, OP<T>>
                <<<num_blocks, threads_per_block, 0, stream>>>(static_cast<const T*>(in1),
                                                               in1_num_elements,
                                                               static_cast<const T*>(in0),
                                                               in0_num_elements,
                                                               static_cast<T*>(out),
                                                               std::forward<Args>(args)...);
        } else {
            Expects(in0_num_elements % in1_num_elements == 0);
            std::tie(num_blocks, threads_per_block) =
                calculateElementwiseGrid(in0_num_elements, max_threads_per_block_);
            elementwise_binary_broadcasting<T, OP<T>>
                <<<num_blocks, threads_per_block, 0, stream>>>(static_cast<const T*>(in0),
                                                               in0_num_elements,
                                                               static_cast<const T*>(in1),
                                                               in1_num_elements,
                                                               static_cast<T*>(out),
                                                               std::forward<Args>(args)...);
        }
#endif  // __CUDACC__
    }

    template <typename T, typename... Args>
    constexpr void case_(cudaStream_t stream,
                         const void* in0,
                         size_t in0_num_elements,
                         const void* in1,
                         size_t in1_num_elements,
                         void* out,
                         Args&&... args) const noexcept {
        callKernel<T>(stream, in0, in0_num_elements, in1, in1_num_elements, out, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    void default_(T t, cudaStream_t, const void*, size_t, const void*, size_t, void*, Args...) const noexcept {
        throwIEException(fmt::format("Element type = {} is not supported.", t));
    }

private:
    Type_t element_type_;
    size_t max_threads_per_block_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
