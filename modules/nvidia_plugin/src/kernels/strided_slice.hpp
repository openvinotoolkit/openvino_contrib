// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <set>
#include <vector>

#include "details/cuda_type_traits.hpp"
#include "details/error.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {
template <typename T_INT>
class StridedSliceKernelOp {
public:
    StridedSliceKernelOp(const std::vector<T_INT> src_matrix_sizes,
                         const std::vector<T_INT> dst_matrix_sizes,
                         const std::set<size_t> reverse_axes,
                         const unsigned max_threads_per_block,
                         const unsigned blocks_number,
                         const unsigned threads_per_block,
                         const Type_t element_type,
                         const Type_t element_type_integer);

    void operator()(const cudaStream_t stream,
                    const T_INT* src_matrix_sizes,
                    const void* src,
                    const T_INT* begin,
                    const T_INT* end,
                    const T_INT* stride,
                    const T_INT* dst_matrix_sizes,
                    void* dst) const;

private:
    template <typename T>
    void callKernels(const cudaStream_t stream,
                     const T_INT* src_matrix_sizes,
                     const void* src,
                     const T_INT* begin,
                     const T_INT* end,
                     const T_INT* stride,
                     const T_INT* dst_matrix_sizes,
                     void* dst) const;
    template <typename T>
    void callStridedSliceKernel(const cudaStream_t stream,
                                const T_INT* src_matrix_sizes,
                                const void* src,
                                const T_INT* begin,
                                const T_INT* end,
                                const T_INT* stride,
                                const T_INT* dst_matrix_sizes,
                                void* dst) const;
    template <typename T>
    void callReverseAxesKernel(const cudaStream_t stream, void* dst) const;

private:
    std::vector<T_INT> src_matrix_sizes_;
    std::vector<T_INT> dst_matrix_sizes_;
    std::set<size_t> reverse_axes_;
    unsigned max_threads_per_block_;
    unsigned blocks_number_;
    unsigned threads_per_block_;
    Type_t element_type_;
    Type_t element_type_integer_;
};

template class StridedSliceKernelOp<int32_t>;
template class StridedSliceKernelOp<int64_t>;

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
