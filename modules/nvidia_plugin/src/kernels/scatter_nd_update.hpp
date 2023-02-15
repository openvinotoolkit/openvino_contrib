// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class ScatterNDUpdate {
public:
    ScatterNDUpdate(Type_t data_type,
                    Type_t indices_type,
                    size_t indices_last_dim,
                    size_t num_of_update_elements,
                    size_t num_of_elements,
                    size_t num_of_update_chunks,
                    size_t num_of_blocks,
                    size_t num_of_threads,
                    bool thread_per_element);

    void operator()(const cudaStream_t stream,
                    const void* input,
                    const void* indices,
                    const void* updates,
                    const size_t* input_data_dim_pading,
                    void* output) const;

    template <typename DataType, typename IndexType>
    void Call(const cudaStream_t stream,
              const void* input,
              const void* indices,
              const void* updates,
              const size_t* input_data_dim_pading,
              void* output) const;

    template <typename IndexType>
    void CallByDataType(const cudaStream_t stream,
                        const void* input,
                        const void* indices,
                        const void* updates,
                        const size_t* input_data_dim_pading,
                        void* output) const;

private:
    Type_t data_type_;
    Type_t indices_type_;
    size_t indices_last_dim_;
    size_t num_of_update_elements_;
    size_t num_of_input_elements_;
    size_t num_of_update_chunks_;
    size_t num_of_blocks_;
    size_t num_of_threads_;
    bool thread_per_element_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
