// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <driver_types.h>

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class Gather {
public:
    Gather(Type_t element_type,
           Type_t indices_type,
           unsigned num_dicts,
           unsigned index_range,
           unsigned data_length,
           unsigned indices_size,
           bool gather_chunks,
           unsigned blocks_per_grid,
           unsigned threads_per_block,
           unsigned grid_dim_x,
           unsigned grid_dim_y,
           unsigned dicts_batch_stride,
           unsigned indices_batch_stride,
           unsigned out_batch_stride,
           unsigned els_per_thread_chunks,
           unsigned els_per_thread_dicts);

    void operator()(const cudaStream_t stream, const void* src_dict, const void* src_index, void* dst_data) const;

private:
    template <typename IndexType>
    void CallByDataType(const cudaStream_t stream, const void* src_dict, const void* src_index, void* dst_data) const;

    template <typename DataType, typename IndexType>
    void Call(const cudaStream_t stream, const void* src_dict, const void* src_index, void* dst_data) const;

    Type_t element_type_;
    Type_t indices_type_;
    unsigned num_dicts_;
    unsigned index_range_;
    unsigned data_length_;
    unsigned indices_size_;
    bool gather_chunks_;
    unsigned blocks_per_grid_;
    unsigned threads_per_block_;
    unsigned grid_dim_x_;
    unsigned grid_dim_y_;
    unsigned dicts_batch_stride_;
    unsigned indices_batch_stride_;
    unsigned out_batch_stride_;
    unsigned els_per_thread_chunks_;
    unsigned els_per_thread_dicts_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
