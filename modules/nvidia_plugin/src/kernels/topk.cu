// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cassert>
#include <cstdio>
#include <cuda/float16.hpp>

#include "cuda/stl/algorithms/sort.cuh"
#include "error.hpp"
#include "fmt/format.h"
#include "tensor_helpers.hpp"
#include "topk.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

using TopKShape = Shape<size_t, TopK::kNumKernelParamDim>;

template <typename T, typename U>
__device__ inline bool compare_min(const CUDA::Pair<T, U>& a, const CUDA::Pair<T, U>& b) {
    if (a.first < b.first) return true;
    if (b.first < a.first) return false;
    return a.second < b.second;
}

template <typename T, typename U>
__device__ inline bool compare_max(const CUDA::Pair<T, U>& a, const CUDA::Pair<T, U>& b) {
    if (a.first == b.first) {
        // TODO: Probably bug in reference implementation
        return a.second < b.second;
    }
    return compare_min(b, a);
}

template <typename T, typename U>
__device__ inline bool sort_indices_ascending(const CUDA::Pair<T, U>& a, const CUDA::Pair<T, U>& b) {
    return a.second < b.second;
}

template <typename T, typename TIdx>
__global__ void topk_preprocess(const T* in,
                                CUDA::Pair<T, TIdx>* workspace,
                                const std::size_t num_input_element,
                                const std::size_t workspace_chunk_size,
                                const TopK::KernelParam* kernel_param) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_input_element) {
        return;
    }

    TopKShape indexes{};
    calculate_indexes_by_flat_address(indexes, i, kernel_param->input_shape_axis);
    const size_t input_index = flat_address_by_strides(kernel_param->input_strides, indexes);

    workspace[i].first = in[input_index];
    workspace[i].second = i % workspace_chunk_size;
}

template <TopK::ComputeType Compute, TopK::SortType Sort, typename T, typename TIdx>
__global__ void topk_sort(CUDA::Pair<T, TIdx>* workspace,
                          const std::size_t workspace_chunks,
                          const std::size_t workspace_chunk_size,
                          const std::size_t k) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= workspace_chunks) {
        return;
    }

    CUDA::Pair<T, TIdx>* begin = &workspace[i * workspace_chunk_size];
    CUDA::Pair<T, TIdx>* end = &workspace[(i + 1) * workspace_chunk_size];

    if (Compute == TopK::ComputeType::Max) {
        CUDA::algorithms::partial_quick_sort_iterative(begin, end, k, compare_max<T, TIdx>);
    } else {
        CUDA::algorithms::partial_quick_sort_iterative(begin, end, k, compare_min<T, TIdx>);
    }
    switch (Sort) {
        case TopK::SortType::SortIndices: {
            CUDA::algorithms::quick_sort_iterative(begin, begin + k, sort_indices_ascending<T, TIdx>);
        } break;
        default:
            break;
    }
}

template <typename TValue, typename UIndex>
__global__ void topk_store(TValue* out_val,
                           UIndex* out_idx,
                           const CUDA::Pair<TValue, UIndex>* workspace,
                           const std::size_t k,
                           const std::size_t num_output_element,
                           const std::size_t workspace_chunk_size,
                           const TopK::KernelParam* kernel_param) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_output_element) {
        return;
    }

    TopKShape indexes{};
    calculate_indexes_by_flat_address(indexes, i, kernel_param->output_shape_axis);
    const size_t output_index = flat_address_by_strides(kernel_param->output_strides, indexes);
    const size_t chunk_id = i / k;
    const size_t sub_id = i % k;
    const size_t workspace_index = chunk_id * workspace_chunk_size + sub_id;

    out_val[output_index] = workspace[workspace_index].first;
    out_idx[output_index] = workspace[workspace_index].second;
}

TopK::TopK(const Type_t element_type,
           const Type_t index_element_type,
           const TopK::ComputeType compute_type,
           const TopK::SortType sort_type,
           const std::size_t num_input_element,
           const std::size_t num_output_element,
           const std::size_t k,
           const std::size_t workspace_chunk_size,
           const std::size_t max_threads_per_block)
    : element_type_{element_type},
      index_element_type_{index_element_type},
      compute_type_{compute_type},
      sort_type_{sort_type},
      num_input_element_{num_input_element},
      num_output_element_{num_output_element},
      k_{k},
      workspace_chunks_{num_input_element / workspace_chunk_size},
      workspace_chunk_size_{workspace_chunk_size} {
    preprocess_.num_blocks_ = (num_input_element + max_threads_per_block - 1) / max_threads_per_block;
    preprocess_.threads_per_block_ = (preprocess_.num_blocks_ == 1) ? num_input_element : max_threads_per_block;

    sort_.num_blocks_ = (workspace_chunks_ + max_threads_per_block - 1) / max_threads_per_block;
    sort_.threads_per_block_ = (sort_.num_blocks_ == 1) ? workspace_chunks_ : max_threads_per_block;

    store_.num_blocks_ = (num_output_element + max_threads_per_block - 1) / max_threads_per_block;
    store_.threads_per_block_ = (store_.num_blocks_ == 1) ? num_output_element : max_threads_per_block;
}

template <typename TElementType>
void TopK::callKernelByElementType(cudaStream_t stream,
                                   const void* in,
                                   void* out_value,
                                   void* out_index,
                                   void* workspace,
                                   const void* kernel_param) const {
    switch (index_element_type_) {
        case Type_t::i32: {
            callKernelByIndexElementType<TElementType, std::int32_t>(
                stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case Type_t::i64: {
            callKernelByIndexElementType<TElementType, std::int64_t>(
                stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        default: {
            throwIEException(fmt::format("Index element type = {} is not supported by TopK operation !!",
                                         static_cast<Type_t>(index_element_type_)));
        }
    }
}

template <typename TElementType, typename TIndexElementType>
void TopK::callKernelByIndexElementType(cudaStream_t stream,
                                        const void* in,
                                        void* out_value,
                                        void* out_index,
                                        void* workspace,
                                        const void* kernel_param) const {
    switch (compute_type_) {
        case ComputeType::Min: {
            callKernelByComputeType<TElementType, TIndexElementType, ComputeType::Min>(
                stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case ComputeType::Max: {
            callKernelByComputeType<TElementType, TIndexElementType, ComputeType::Max>(
                stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        default: {
            throwIEException(
                fmt::format("Unknown compute type = {} for TopK operation !!", static_cast<Type_t>(compute_type_)));
        }
    }
}

template <typename TElementType, typename TIndexElementType, TopK::ComputeType ComputeType>
void TopK::callKernelByComputeType(cudaStream_t stream,
                                   const void* in,
                                   void* out_value,
                                   void* out_index,
                                   void* workspace,
                                   const void* kernel_param) const {
    switch (sort_type_) {
        case SortType::None: {
            callKernelBySortType<TElementType, TIndexElementType, ComputeType, SortType::None>(
                stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case SortType::SortIndices: {
            callKernelBySortType<TElementType, TIndexElementType, ComputeType, SortType::SortIndices>(
                stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case SortType::SortValues: {
            callKernelBySortType<TElementType, TIndexElementType, ComputeType, SortType::SortValues>(
                stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        default: {
            throwIEException(
                fmt::format("Unknown sort type = {} for TopK operation !!", static_cast<Type_t>(sort_type_)));
        }
    }
}

template <typename TElementType, typename TIndexElementType, TopK::ComputeType ComputeType, TopK::SortType SortType>
void TopK::callKernelBySortType(cudaStream_t stream,
                                const void* in,
                                void* out_value,
                                void* out_index,
                                void* workspace,
                                const void* kernel_param) const {
    const KernelParam* kernel_param_ptr = static_cast<const KernelParam*>(kernel_param);
    topk_preprocess<<<preprocess_.num_blocks_, preprocess_.threads_per_block_, 0, stream>>>(
        static_cast<const TElementType*>(in),
        static_cast<CUDA::Pair<TElementType, TIndexElementType>*>(workspace),
        num_input_element_,
        workspace_chunk_size_,
        kernel_param_ptr);
    topk_sort<ComputeType, SortType><<<sort_.num_blocks_, sort_.threads_per_block_, 0, stream>>>(
        static_cast<CUDA::Pair<TElementType, TIndexElementType>*>(workspace),
        workspace_chunks_,
        workspace_chunk_size_,
        k_);
    topk_store<<<store_.num_blocks_, store_.threads_per_block_, 0, stream>>>(
        static_cast<TElementType*>(out_value),
        static_cast<TIndexElementType*>(out_index),
        static_cast<CUDA::Pair<TElementType, TIndexElementType>*>(workspace),
        k_,
        num_output_element_,
        workspace_chunk_size_,
        kernel_param_ptr);
}

void TopK::operator()(cudaStream_t stream,
                      const void* in,
                      void* out_value,
                      void* out_index,
                      void* workspace,
                      const void* kernel_param) const {
    switch (element_type_) {
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16: {
            callKernelByElementType<__nv_bfloat16>(stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
#endif
        case Type_t::f16: {
            callKernelByElementType<__half>(stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case Type_t::f32: {
            callKernelByElementType<float>(stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case Type_t::i8: {
            callKernelByElementType<std::int8_t>(stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case Type_t::i16: {
            callKernelByElementType<std::int16_t>(stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case Type_t::i32: {
            callKernelByElementType<std::int32_t>(stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case Type_t::i64: {
            callKernelByElementType<std::int64_t>(stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case Type_t::u8: {
            callKernelByElementType<std::uint8_t>(stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case Type_t::u16: {
            callKernelByElementType<std::uint16_t>(stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case Type_t::u32: {
            callKernelByElementType<std::uint32_t>(stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        case Type_t::u64: {
            callKernelByElementType<std::uint64_t>(stream, in, out_value, out_index, workspace, kernel_param);
            break;
        }
        default: {
            throwIEException(
                fmt::format("Input element type = {} is not supported by TopK operation "
                            "!!",
                            static_cast<Type_t>(element_type_)));
        }
    }
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
