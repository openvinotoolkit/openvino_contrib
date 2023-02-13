// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_type_traits.hpp"
#include "elementtypeswitch.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

class TopK {
public:
    enum class SortType {
        None,
        SortIndices,
        SortValues,
    };

    enum class ComputeType {
        Min,
        Max,
    };

    static constexpr size_t kNumKernelParamDim = 5;
    struct KernelParam {
        size_t input_shape_axis[kNumKernelParamDim]{};
        size_t output_shape_axis[kNumKernelParamDim]{};
        size_t input_strides[kNumKernelParamDim]{};
        size_t output_strides[kNumKernelParamDim]{};
    };

    TopK(Type_t element_type,
         Type_t index_element_type,
         TopK::ComputeType compute_type,
         TopK::SortType sort_type,
         std::size_t num_input_element,
         std::size_t num_output_element,
         std::size_t k,
         std::size_t workspace_chunk_size,
         size_t max_threads_per_block);
    TopK(TopK&&) = default;
    TopK& operator=(TopK&&) = default;

    void operator()(cudaStream_t stream,
                    const void* in,
                    void* out_value,
                    void* out_index,
                    void* workspace,
                    const void* kernel_param) const;

private:
    template <typename TElementType>
    void callKernelByElementType(cudaStream_t stream,
                                 const void* in,
                                 void* out_value,
                                 void* out_index,
                                 void* workspace,
                                 const void* kernel_param) const;
    template <typename TElementType, typename TIndexElementType>
    void callKernelByIndexElementType(cudaStream_t stream,
                                      const void* in,
                                      void* out_value,
                                      void* out_index,
                                      void* workspace,
                                      const void* kernel_param) const;
    template <typename TElementType, typename TIndexElementType, ComputeType ComputeType>
    void callKernelByComputeType(cudaStream_t stream,
                                 const void* in,
                                 void* out_value,
                                 void* out_index,
                                 void* workspace,
                                 const void* kernel_param) const;
    template <typename TElementType, typename TIndexElementType, ComputeType ComputeType, SortType SortType>
    void callKernelBySortType(cudaStream_t stream,
                              const void* in,
                              void* out_value,
                              void* out_index,
                              void* workspace,
                              const void* kernel_param) const;

    struct KernelGridParam {
        size_t num_blocks_;
        size_t threads_per_block_;
    };

    Type_t element_type_;
    Type_t index_element_type_;
    TopK::ComputeType compute_type_;
    TopK::SortType sort_type_;
    size_t num_input_element_;
    size_t num_output_element_;
    size_t k_;
    size_t input_iterations_;
    size_t workspace_chunks_;
    size_t workspace_chunk_size_;
    KernelGridParam preprocess_;
    KernelGridParam sort_;
    KernelGridParam store_;
};

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
