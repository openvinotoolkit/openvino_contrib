// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <fmt/format.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include <cuda_fp16.h>

#include <error.hpp>

#include "gather.hpp"

namespace CUDAPlugin {

namespace kernel {

template <typename DataType, typename IndexType>
static inline __device__ void gather(unsigned data_length,
                                     size_t index_range,
                                     unsigned els_per_thread,
                                     unsigned indices_size,
                                     unsigned indices_index,
                                     unsigned dict,
                                     unsigned chunk,
                                     const DataType* src_dict,
                                     const IndexType* src_index,
                                     DataType* dst_data) {
    const auto dict_index = src_index[indices_index];
    if (dict_index >= index_range) {
        // TODO: find a way to handle an error raised in a kernel (assertion or trap) properly
        __trap();
    }
    unsigned thread_offset;
    for (int el = 0; el < els_per_thread; ++el) {
        thread_offset = chunk + el;
        if (thread_offset >= data_length) {
            return;
        }
        dst_data[data_length * (indices_index + dict * indices_size) + thread_offset] =
            src_dict[data_length * (dict_index + dict * index_range) + thread_offset];
    }
}

template <typename DataType, typename IndexType>
static __global__ void chunks_gather(unsigned data_length,
                                     size_t index_range,
                                     unsigned num_dicts,
                                     unsigned dicts_batch_stride,
                                     unsigned indices_batch_stride,
                                     unsigned out_batch_stride,
                                     unsigned els_per_thread,
                                     const DataType* src_dict,
                                     const IndexType* src_index,
                                     DataType* dst_data) {
    const auto indices_size = gridDim.y;
    const auto indices_index = blockIdx.y;
    const auto dict = blockIdx.x % num_dicts;
    const auto batch = blockIdx.x / num_dicts;
    const auto chunk = (blockIdx.z * blockDim.x + threadIdx.x) * els_per_thread;
    gather(data_length,
           index_range,
           els_per_thread,
           indices_size,
           indices_index,
           dict,
           chunk,
           src_dict + batch * dicts_batch_stride,
           src_index + batch * indices_batch_stride,
           dst_data + batch * out_batch_stride);
}

template <typename DataType, typename IndexType>
static __global__ void dicts_gather(unsigned data_length,
                                    size_t index_range,
                                    unsigned num_dicts,
                                    unsigned dicts_batch_stride,
                                    unsigned indices_batch_stride,
                                    unsigned out_batch_stride,
                                    unsigned els_per_thread,
                                    const DataType* src_dict,
                                    const IndexType* src_index,
                                    DataType* dst_data) {
    const auto indices_size = gridDim.y;
    const auto indices_index = blockIdx.y;
    const auto dict = blockIdx.z * blockDim.x + threadIdx.x;
    if (dict >= num_dicts) {
        return;
    }
    const auto chunk = blockIdx.x % data_length * els_per_thread;
    const auto batch = blockIdx.x / data_length;
    gather(data_length,
           index_range,
           els_per_thread,
           indices_size,
           indices_index,
           dict,
           chunk,
           src_dict + batch * dicts_batch_stride,
           src_index + batch * indices_batch_stride,
           dst_data + batch * out_batch_stride);
}

Gather::Gather(Type_t element_type,
               Type_t indices_type,
               unsigned num_dicts,
               unsigned index_range,
               unsigned data_length,
               unsigned indices_size,
               bool gather_chunks,
               unsigned blocks_per_grid,
               unsigned threads_per_block,
               unsigned grid_dim_x,
               unsigned dicts_batch_stride,
               unsigned indices_batch_stride,
               unsigned out_batch_stride,
               unsigned els_per_thread_chunks,
               unsigned els_per_thread_dicts)
    : element_type_(element_type),
      indices_type_(indices_type),
      num_dicts_(num_dicts),
      index_range_(index_range),
      data_length_(data_length),
      indices_size_(indices_size),
      gather_chunks_(gather_chunks),
      blocks_per_grid_(blocks_per_grid),
      threads_per_block_(threads_per_block),
      grid_dim_x_(grid_dim_x),
      dicts_batch_stride_(dicts_batch_stride),
      indices_batch_stride_(indices_batch_stride),
      out_batch_stride_(out_batch_stride),
      els_per_thread_chunks_(els_per_thread_chunks),
      els_per_thread_dicts_(els_per_thread_dicts) {}

void Gather::operator()(const cudaStream_t stream, const void* src_dict, const void* src_index, void* dst_data) const {
    switch (indices_type_) {
        case Type_t::i64:
            return CallByDataType<int64_t>(stream, src_dict, src_index, dst_data);
        case Type_t::i32:
            return CallByDataType<int32_t>(stream, src_dict, src_index, dst_data);
        default:
            throwIEException(
                fmt::format("Index element type = {} is not supported by Gather operation !!", indices_type_));
    }
}

template <typename IndexType>
void Gather::CallByDataType(const cudaStream_t stream,
                            const void* src_dict,
                            const void* src_index,
                            void* dst_data) const {
    switch (element_type_) {
        case Type_t::boolean:
            return Call<bool, IndexType>(stream, src_dict, src_index, dst_data);
#if CUDA_VERSION >= 11000
        case Type_t::bf16:
            return Call<__nv_bfloat16, IndexType>(stream, src_dict, src_index, dst_data);
#endif
        case Type_t::f16:
            return Call<__half, IndexType>(stream, src_dict, src_index, dst_data);
        case Type_t::f32:
            return Call<float, IndexType>(stream, src_dict, src_index, dst_data);
        case Type_t::f64:
            return Call<double, IndexType>(stream, src_dict, src_index, dst_data);
        case Type_t::i8:
            return Call<int8_t, IndexType>(stream, src_dict, src_index, dst_data);
        case Type_t::i16:
            return Call<int16_t, IndexType>(stream, src_dict, src_index, dst_data);
        case Type_t::i32:
            return Call<int32_t, IndexType>(stream, src_dict, src_index, dst_data);
        case Type_t::i64:
            return Call<int64_t, IndexType>(stream, src_dict, src_index, dst_data);
        case Type_t::u8:
            return Call<uint8_t, IndexType>(stream, src_dict, src_index, dst_data);
        case Type_t::u16:
            return Call<uint16_t, IndexType>(stream, src_dict, src_index, dst_data);
        case Type_t::u32:
            return Call<uint32_t, IndexType>(stream, src_dict, src_index, dst_data);
        case Type_t::u64:
            return Call<uint64_t, IndexType>(stream, src_dict, src_index, dst_data);
        default:
            throwIEException(
                fmt::format("Index element type = {} is not supported by Gather operation !!", indices_type_));
    }
}

template <typename DataType, typename IndexType>
void Gather::Call(const cudaStream_t stream, const void* src_dict, const void* src_index, void* dst_data) const {
    dim3 grid{grid_dim_x_, indices_size_, blocks_per_grid_};

    const auto src_dict_typed = static_cast<const DataType*>(src_dict);
    const auto src_index_typed = static_cast<const IndexType*>(src_index);
    auto dst_data_typed = static_cast<DataType*>(dst_data);

    if (gather_chunks_) {
        kernel::chunks_gather<<<grid, threads_per_block_, 0, stream>>>(data_length_,
                                                                       index_range_,
                                                                       num_dicts_,
                                                                       dicts_batch_stride_,
                                                                       indices_batch_stride_,
                                                                       out_batch_stride_,
                                                                       els_per_thread_chunks_,
                                                                       src_dict_typed,
                                                                       src_index_typed,
                                                                       dst_data_typed);
    } else {
        kernel::dicts_gather<<<grid, threads_per_block_, 0, stream>>>(data_length_,
                                                                      index_range_,
                                                                      num_dicts_,
                                                                      dicts_batch_stride_,
                                                                      indices_batch_stride_,
                                                                      els_per_thread_dicts_,
                                                                      out_batch_stride_,
                                                                      src_dict_typed,
                                                                      src_index_typed,
                                                                      dst_data_typed);
    }
    // TODO: find a way to handle an error raised in a kernel (assertion or trap) properly in CUDA Plugin
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throwIEException(cudaGetErrorString(err));
    }
}

}  // namespace kernel
}  // namespace CUDAPlugin
