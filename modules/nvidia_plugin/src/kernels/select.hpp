// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime.h>

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class SelectKernelOp {
public:
    using BrcstOffsetType = size_t;

public:
    SelectKernelOp(const size_t max_size,
                   const unsigned blocks_number,
                   const unsigned threads_per_block,
                   const Type_t operation_type);

    void operator()(const cudaStream_t stream,
                    const bool* condition,
                    const void* then_node,
                    const void* else_node,
                    const BrcstOffsetType* cond_brcst_offsets,
                    const BrcstOffsetType* then_brcst_offsets,
                    const BrcstOffsetType* else_brcst_offsets,
                    const BrcstOffsetType* output_sizes,
                    void* buffer) const;

private:
    template <typename T>
    void callKernel(const cudaStream_t stream,
                    const bool* condition,
                    const void* then_node,
                    const void* else_node,
                    const BrcstOffsetType* cond_brcst_offsets,
                    const BrcstOffsetType* then_brcst_offsets,
                    const BrcstOffsetType* else_brcst_offsets,
                    const BrcstOffsetType* output_sizes,
                    void* buffer) const;

private:
    size_t max_size_;
    unsigned blocks_number_;
    unsigned threads_per_block_;
    Type_t operation_type_;
};

}  // namespace kernel

}  // namespace nvidia_gpu
}  // namespace ov
