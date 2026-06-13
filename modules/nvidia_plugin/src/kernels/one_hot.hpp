// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

/**
 * @brief CUDA kernel wrapper for opset1 OneHot.
 *
 * Inserts a new axis of size `depth`; the tensor is viewed as
 * [outer, depth, inner]. For output element (o, p, i) the value is on_value when
 * indices[o, i] == p and off_value otherwise (indices outside [0, depth) yield
 * off_value everywhere). One thread per output element.
 */
class OneHot {
public:
    OneHot(Type_t element_type,
           Type_t indices_type,
           std::size_t outer_size,
           std::size_t depth,
           std::size_t inner_size,
           std::size_t num_blocks,
           std::size_t threads_per_block);

    void operator()(cudaStream_t stream,
                    const void* indices,
                    const void* on_value,
                    const void* off_value,
                    void* output) const;

    template <typename T>
    void callByIndexType(cudaStream_t stream,
                         const void* indices,
                         const void* on_value,
                         const void* off_value,
                         void* output) const;

    template <typename T, typename Idx>
    void call(cudaStream_t stream,
              const void* indices,
              const void* on_value,
              const void* off_value,
              void* output) const;

private:
    Type_t element_type_;
    Type_t indices_type_;
    std::size_t outer_size_;
    std::size_t depth_;
    std::size_t inner_size_;
    std::size_t num_blocks_;
    std::size_t threads_per_block_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
