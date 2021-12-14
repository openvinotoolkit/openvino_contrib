// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fmt/format.h>

#include <gsl/gsl_assert>
#include <tuple>

#include "cuda_type_traits.hpp"
#include "elementtypeswitch.hpp"
#include "error.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T, typename OP, typename... Args>
static __global__ void elementwise_unary(const T* in, T* out, size_t num_elements, Args... extraArgs) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        out[i] = OP::op(in[i], extraArgs...);
    }
}

template <typename T, typename OP, typename... Args>
static __global__ void elementwise_binary(const T* in0, const T* in1, T* out, size_t num_elements, Args... extraArgs) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        out[i] = OP::op(in0[i], in1[i], extraArgs...);
    }
}

template <typename T, typename OP, typename... Args>
static __global__ void elementwise_binary_broadcasting(const T* in,
                                                       size_t in_num_elements,
                                                       const T* broadcasted,
                                                       size_t broadcasted_num_elements,
                                                       T* out,
                                                       Args... extraArgs) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < in_num_elements) {
        out[i] = OP::op(in[i], broadcasted[i % broadcasted_num_elements], extraArgs...);
    }
}

template <typename ElementTypes, template <typename> typename OP>
class ElementwiseHelper {
public:
    ElementwiseHelper(Type_t element_type, size_t max_threads_per_block)
        : element_type_{element_type}, max_threads_per_block_{max_threads_per_block} {}
    ElementwiseHelper(ElementwiseHelper&&) = default;
    ElementwiseHelper& operator=(ElementwiseHelper&&) = default;

    template <typename... Args>
    void unaryOperator(cudaStream_t stream, const void* in, void* out, size_t num_elements, Args... extraArgs) const {
        ElementTypes::switch_(element_type_, *this, stream, in, out, num_elements, extraArgs...);
    }

    /**
     * @param out   Output buffer. Is expected to be large enough to fit
     * max(in0_num_elements, in1_num_elements) elements.
     */
    template <typename... Args>
    void binaryOperator(cudaStream_t stream,
                        const void* in0,
                        size_t in0_num_elements,
                        const void* in1,
                        size_t in1_num_elements,
                        void* out,
                        Args... extraArgs) const {
        ElementTypes::switch_(
            element_type_, *this, stream, in0, in0_num_elements, in1, in1_num_elements, out, extraArgs...);
    }

public:  // Type switch for unary operators
    template <typename T, typename... Args>
    constexpr void case_(
        cudaStream_t stream, const void* in, void* out, size_t num_elements, Args... extraArgs) const noexcept {
        size_t num_blocks{}, threads_per_block{};
        std::tie(num_blocks, threads_per_block) = calculateElementwiseGrid(num_elements, max_threads_per_block_);
        elementwise_unary<T, OP<T>><<<num_blocks, threads_per_block, 0, stream>>>(
            static_cast<const T*>(in), static_cast<T*>(out), num_elements, extraArgs...);
    }

    template <typename T, typename... Args>
    void default_(T t, cudaStream_t, const void*, void*, size_t, Args...) const noexcept {
        throwIEException(fmt::format("Element type = {} is not supported.", t));
    }

public:  // Type switch for binary operators
    template <typename T, typename... Args>
    constexpr void case_(cudaStream_t stream,
                         const void* in0,
                         size_t in0_num_elements,
                         const void* in1,
                         size_t in1_num_elements,
                         void* out,
                         Args... extraArgs) const noexcept {
        if (in0_num_elements == in1_num_elements) {
            size_t num_blocks{}, threads_per_block{};
            std::tie(num_blocks, threads_per_block) =
                calculateElementwiseGrid(in0_num_elements, max_threads_per_block_);
            elementwise_binary<T, OP<T>><<<num_blocks, threads_per_block, 0, stream>>>(static_cast<const T*>(in0),
                                                                                       static_cast<const T*>(in1),
                                                                                       static_cast<T*>(out),
                                                                                       in0_num_elements,
                                                                                       extraArgs...);
        } else if (in0_num_elements < in1_num_elements) {
            Expects(in1_num_elements % in0_num_elements == 0);
            size_t num_blocks{}, threads_per_block{};
            std::tie(num_blocks, threads_per_block) =
                calculateElementwiseGrid(in1_num_elements, max_threads_per_block_);
            elementwise_binary_broadcasting<T, OP<T>>
                <<<num_blocks, threads_per_block, 0, stream>>>(static_cast<const T*>(in1),
                                                               in1_num_elements,
                                                               static_cast<const T*>(in0),
                                                               in0_num_elements,
                                                               static_cast<T*>(out),
                                                               extraArgs...);
        } else {
            Expects(in0_num_elements % in1_num_elements == 0);
            size_t num_blocks{}, threads_per_block{};
            std::tie(num_blocks, threads_per_block) =
                calculateElementwiseGrid(in0_num_elements, max_threads_per_block_);
            elementwise_binary_broadcasting<T, OP<T>>
                <<<num_blocks, threads_per_block, 0, stream>>>(static_cast<const T*>(in0),
                                                               in0_num_elements,
                                                               static_cast<const T*>(in1),
                                                               in1_num_elements,
                                                               static_cast<T*>(out),
                                                               extraArgs...);
        }
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
