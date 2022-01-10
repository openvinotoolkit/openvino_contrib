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
class ElementwiseBinary {
public:
    ElementwiseBinary(Type_t element_type,
                      size_t max_threads_per_block,
                      size_t in0_num_elements,
                      size_t in1_num_elements)
        : element_type_{element_type}, in0_num_elements_{in0_num_elements}, in1_num_elements_{in1_num_elements} {
        size_t num_elements{};
        if (in0_num_elements_ == in1_num_elements_) {
            mode_ = BroadcastMode::NONE;
            num_elements = in0_num_elements_;
        } else if (in0_num_elements_ < in1_num_elements_) {
            Expects(in1_num_elements_ % in0_num_elements_ == 0);
            mode_ = BroadcastMode::IN0;
            num_elements = in1_num_elements_;
        } else {
            Expects(in0_num_elements_ % in1_num_elements_ == 0);
            mode_ = BroadcastMode::IN1;
            num_elements = in0_num_elements_;
        }
        std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(num_elements, max_threads_per_block);
    }

    /**
     * @param out   Output buffer. Is expected to be large enough to fit
     * max(in0_num_elements, in1_num_elements) elements.
     */
    template <typename... Args>
    void operator()(cudaStream_t stream, const void* in0, const void* in1, void* out, Args&&... args) const {
        ElementTypes::switch_(element_type_, *this, stream, in0, in1, out, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    constexpr void callKernel(
        cudaStream_t stream, const void* in0, const void* in1, void* out, Args&&... args) const noexcept {
#ifdef __CUDACC__
        switch (mode_) {
            case BroadcastMode::NONE:
                elementwise_binary<T, OP<T>>
                    <<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const T*>(in0),
                                                                     static_cast<const T*>(in1),
                                                                     in0_num_elements_,
                                                                     static_cast<T*>(out),
                                                                     std::forward<Args>(args)...);
                break;
            case BroadcastMode::IN0:
                elementwise_binary_broadcasting<T, OP<T>>
                    <<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const T*>(in1),
                                                                     in1_num_elements_,
                                                                     static_cast<const T*>(in0),
                                                                     in0_num_elements_,
                                                                     static_cast<T*>(out),
                                                                     std::forward<Args>(args)...);
                break;
            case BroadcastMode::IN1:
                elementwise_binary_broadcasting<T, OP<T>>
                    <<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const T*>(in0),
                                                                     in0_num_elements_,
                                                                     static_cast<const T*>(in1),
                                                                     in1_num_elements_,
                                                                     static_cast<T*>(out),
                                                                     std::forward<Args>(args)...);
                break;
            default:
                Ensures(false);
        }
#endif  // __CUDACC__
    }

    template <typename T, typename... Args>
    constexpr void case_(
        cudaStream_t stream, const void* in0, const void* in1, void* out, Args&&... args) const noexcept {
        callKernel<T>(stream, in0, in1, out, std::forward<Args>(args)...);
    }

    template <typename T, typename... Args>
    void default_(T t, cudaStream_t, const void*, const void*, void*, Args...) const noexcept {
        throwIEException(fmt::format("Element type = {} is not supported.", t));
    }

private:
    enum class BroadcastMode { NONE, IN0, IN1 };

    Type_t element_type_;
    size_t in0_num_elements_;
    size_t in1_num_elements_;
    BroadcastMode mode_;
    size_t num_blocks_;
    size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
