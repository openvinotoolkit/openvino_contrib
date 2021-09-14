// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime.h>

#include "ngraph/type/element_type.hpp"

namespace CUDAPlugin {
namespace kernel {

class SelectKernelOp {
public:
    using BrcstOffsetType = size_t;

public:
    SelectKernelOp(const size_t max_size,
                   const unsigned blocks_number,
                   const unsigned threads_per_block,
                   const ngraph::element::Type_t operation_type);

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
    ngraph::element::Type_t operation_type_;
};

}  // namespace kernel

}  // namespace CUDAPlugin
