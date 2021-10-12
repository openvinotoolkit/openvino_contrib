// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ngraph/type/element_type.hpp>
#include <set>

#include "error.hpp"

namespace CUDAPlugin {
namespace kernel {
class StridedSliceKernelOp {
public:
    StridedSliceKernelOp(const std::vector<int64_t> src_matrix_sizes,
                         const std::vector<int64_t> dst_matrix_sizes,
                         const std::set<size_t> reverse_axes,
                         const unsigned max_threads_per_block,
                         const unsigned blocks_number,
                         const unsigned threads_per_block,
                         const ngraph::element::Type_t element_type);

    void operator()(const cudaStream_t stream,
                    const int64_t* src_matrix_sizes,
                    const void* src,
                    const int64_t* begin,
                    const int64_t* end,
                    const int64_t* stride,
                    const int64_t* dst_matrix_sizes,
                    void* dst) const;

private:
    template <typename T>
    void callKernels(const cudaStream_t stream,
                     const int64_t* src_matrix_sizes,
                     const void* src,
                     const int64_t* begin,
                     const int64_t* end,
                     const int64_t* stride,
                     const int64_t* dst_matrix_sizes,
                     void* dst) const;
    template <typename T>
    void callStridedSliceKernel(const cudaStream_t stream,
                                const int64_t* src_matrix_sizes,
                                const void* src,
                                const int64_t* begin,
                                const int64_t* end,
                                const int64_t* stride,
                                const int64_t* dst_matrix_sizes,
                                void* dst) const;
    template <typename T>
    void callReverseAxesKernel(const cudaStream_t stream, void* dst) const;

private:
    std::vector<int64_t> src_matrix_sizes_;
    std::vector<int64_t> dst_matrix_sizes_;
    std::set<size_t> reverse_axes_;
    unsigned max_threads_per_block_;
    unsigned blocks_number_;
    unsigned threads_per_block_;
    ngraph::element::Type_t element_type_;
};

}  // namespace kernel
}  // namespace CUDAPlugin
