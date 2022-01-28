// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "cuda_type_traits.hpp"
#include "error.hpp"

namespace CUDAPlugin {
namespace kernel {

class Broadcast {
public:
    Broadcast(CUDAPlugin::kernel::Type_t element_type,
              size_t shape_rank,
              size_t dst_num_elements,
              size_t max_threads_per_block);

    /**
     * @param broadcast_dims for each dimension, indicates whether a dimension is broadcasted.
     *                       0 - broadcasted, 1 - not broadcasted. Passing other values causes
     *                       undefined behavior.
     * @param src_strides Source tensor strides. Source tensor shape is extended with 1 and has
     *                    the same shape rank as output tensor.
     */
    void operator()(const cudaStream_t stream,
                    const void* src,
                    void* dst,
                    const size_t* broadcast_dims,
                    const size_t* src_strides,
                    const size_t* dst_strides) const;

    template <typename T, typename... Args>
    constexpr void case_(cudaStream_t stream, Args&&... args) const noexcept;

    template <typename T, typename... Args>
    void default_(T t, cudaStream_t, const void*, void*, Args...) const noexcept;

private:
    template <typename T>
    void callKernel(const cudaStream_t stream,
                    const void* src,
                    void* dst,
                    const size_t* broadcast_dims,
                    const size_t* src_strides,
                    const size_t* dst_strides) const;

private:
    Type_t element_type_;
    size_t shape_rank_;
    size_t dst_num_elements_;
    size_t num_blocks_;
    size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
