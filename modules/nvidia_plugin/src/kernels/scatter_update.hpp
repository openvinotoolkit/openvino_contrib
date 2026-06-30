// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

/**
 * @brief CUDA kernel wrapper for opset3 ScatterUpdate.
 *
 * Semantics: output is a copy of `data`; then along `axis` each slice selected
 * by `indices` is overwritten by the corresponding slice of `updates`.
 *
 * Layout of `updates` is data.shape[:axis] + indices.shape + data.shape[axis+1:].
 * For a flat update element u:
 *   outer = u / (indices_size * inner_size)
 *   rem   = u % (indices_size * inner_size)
 *   idx   = rem / inner_size            // flat index into `indices`
 *   inner = rem % inner_size
 *   out   = (outer * axis_dim + indices[idx]) * inner_size + inner
 */
class ScatterUpdate {
public:
    ScatterUpdate(Type_t data_type,
                  Type_t indices_type,
                  size_t num_input_elements,
                  size_t num_update_elements,
                  size_t indices_size,
                  size_t inner_size,
                  size_t axis_dim,
                  size_t num_blocks,
                  size_t num_threads);

    void operator()(const cudaStream_t stream,
                    const void* input,
                    const void* indices,
                    const void* updates,
                    void* output) const;

    template <typename DataType, typename IndexType>
    void Call(const cudaStream_t stream,
              const void* input,
              const void* indices,
              const void* updates,
              void* output) const;

    template <typename IndexType>
    void CallByDataType(const cudaStream_t stream,
                        const void* input,
                        const void* indices,
                        const void* updates,
                        void* output) const;

private:
    Type_t data_type_;
    Type_t indices_type_;
    size_t num_input_elements_;
    size_t num_update_elements_;
    size_t indices_size_;
    size_t inner_size_;
    size_t axis_dim_;
    size_t num_blocks_;
    size_t num_threads_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
