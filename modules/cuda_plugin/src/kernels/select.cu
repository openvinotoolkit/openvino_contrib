// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_fp16.h>
#include <fmt/format.h>

#include "select.hpp"

namespace CUDAPlugin {

namespace kernel {
template <typename T>
static __global__ void select(const bool* condition,
                              const T* then_node,
                              const T* else_node,
                              const SelectKernelOp::BrcstOffsetType* cond_brcst_offsets,
                              const SelectKernelOp::BrcstOffsetType* then_brcst_offsets,
                              const SelectKernelOp::BrcstOffsetType* else_brcst_offsets,
                              const SelectKernelOp::BrcstOffsetType* output_sizes,
                              const size_t max_size,
                              T* buffer) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_size) return;
    // calc N, C, D, H, W indexes
    enum { N, C, D, H, W };
    const unsigned n = idx / output_sizes[N];
    const unsigned n_size = n * output_sizes[N];
    const unsigned c = (idx - n_size) / output_sizes[C];
    const unsigned c_size = c * output_sizes[C];
    const unsigned d = (idx - n_size - c_size) / output_sizes[D];
    const unsigned d_size = d * output_sizes[D];
    const unsigned h = (idx - n_size - c_size - d_size) / output_sizes[H];
    const unsigned h_size = h * output_sizes[H];
    const unsigned w = (idx - n_size - c_size - d_size - h_size) / output_sizes[W];
    const unsigned cond_idx = cond_brcst_offsets[N] * n + cond_brcst_offsets[C] * c + cond_brcst_offsets[D] * d +
                              cond_brcst_offsets[H] * h + cond_brcst_offsets[W] * w;
    const unsigned then_idx = then_brcst_offsets[N] * n + then_brcst_offsets[C] * c + then_brcst_offsets[D] * d +
                              then_brcst_offsets[H] * h + then_brcst_offsets[W] * w;
    const unsigned else_idx = else_brcst_offsets[N] * n + else_brcst_offsets[C] * c + else_brcst_offsets[D] * d +
                              else_brcst_offsets[H] * h + else_brcst_offsets[W] * w;
    buffer[idx] = condition[cond_idx] ? then_node[then_idx] : else_node[else_idx];
}

SelectKernelOp::SelectKernelOp(const size_t max_size,
                               const unsigned blocks_number,
                               const unsigned threads_per_block,
                               const ov::element::Type_t operation_type)
    : max_size_{max_size},
      blocks_number_{blocks_number},
      threads_per_block_{threads_per_block},
      operation_type_{operation_type} {}

void SelectKernelOp::operator()(const cudaStream_t stream,
                                const bool* condition,
                                const void* then_node,
                                const void* else_node,
                                const BrcstOffsetType* cond_brcst_offsets,
                                const BrcstOffsetType* then_brcst_offsets,
                                const BrcstOffsetType* else_brcst_offsets,
                                const BrcstOffsetType* output_sizes,
                                void* buffer) const {
    switch (operation_type_) {
        case ov::element::u8:
            return callKernel<uint8_t>(stream,
                                       condition,
                                       then_node,
                                       else_node,
                                       cond_brcst_offsets,
                                       then_brcst_offsets,
                                       else_brcst_offsets,
                                       output_sizes,
                                       buffer);
        case ov::element::i16:
            return callKernel<int16_t>(stream,
                                       condition,
                                       then_node,
                                       else_node,
                                       cond_brcst_offsets,
                                       then_brcst_offsets,
                                       else_brcst_offsets,
                                       output_sizes,
                                       buffer);
        case ov::element::f16:
            return callKernel<__half>(stream,
                                      condition,
                                      then_node,
                                      else_node,
                                      cond_brcst_offsets,
                                      then_brcst_offsets,
                                      else_brcst_offsets,
                                      output_sizes,
                                      buffer);
        case ov::element::f32:
            return callKernel<float>(stream,
                                     condition,
                                     then_node,
                                     else_node,
                                     cond_brcst_offsets,
                                     then_brcst_offsets,
                                     else_brcst_offsets,
                                     output_sizes,
                                     buffer);
        default:
            throwIEException(
                fmt::format("Index element type = {} is not supported by Gather operation !!", operation_type_));
    }
}

template <typename T>
void SelectKernelOp::callKernel(const cudaStream_t stream,
                                const bool* condition,
                                const void* then_node,
                                const void* else_node,
                                const BrcstOffsetType* cond_brcst_offsets,
                                const BrcstOffsetType* then_brcst_offsets,
                                const BrcstOffsetType* else_brcst_offsets,
                                const BrcstOffsetType* output_sizes,
                                void* buffer) const {
    kernel::select<T><<<blocks_number_, threads_per_block_, 0, stream>>>(condition,
                                                                         static_cast<const T*>(then_node),
                                                                         static_cast<const T*>(else_node),
                                                                         cond_brcst_offsets,
                                                                         then_brcst_offsets,
                                                                         else_brcst_offsets,
                                                                         output_sizes,
                                                                         max_size_,
                                                                         static_cast<T*>(buffer));
}

}  // namespace kernel

}  // namespace CUDAPlugin
