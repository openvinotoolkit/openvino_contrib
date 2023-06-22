// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_device_runtime_api.h>
#include <fmt/format.h>
#include <stdio.h>

#include <cuda/float16.hpp>
#include <numeric>

#include "details/type_validator.hpp"
#include "scatter_nd_update.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename DataType, typename IndexType>
static inline __device__ void scatter_nd_update(const size_t indices_last_dim,
                                                const size_t num_of_update_elements,
                                                const size_t* input_data_dim_pading,
                                                const size_t update_element,
                                                const size_t update_chunk,
                                                const IndexType* indices,
                                                const DataType* updates,
                                                DataType* output) {
    const auto begin = update_chunk * indices_last_dim;
    const auto end = begin + indices_last_dim;
    size_t out_index{};
    for (size_t j{begin}, k{}; j < end; ++j, ++k) out_index += indices[j] * input_data_dim_pading[k];

    output[out_index + update_element] = updates[update_chunk * num_of_update_elements + update_element];
}

template <typename DataType, typename IndexType>
static inline __global__ void update_thread_per_element(const size_t indices_last_dim,
                                                        const size_t num_of_update_elements,
                                                        const size_t* input_data_dim_pading,
                                                        const IndexType* indices,
                                                        const DataType* updates,
                                                        DataType* output) {
    const auto update_element = blockIdx.y * blockDim.x + threadIdx.x;
    const auto update_chunk = blockIdx.x;

    if (update_element >= num_of_update_elements) return;

    scatter_nd_update(indices_last_dim,
                      num_of_update_elements,
                      input_data_dim_pading,
                      update_element,
                      update_chunk,
                      indices,
                      updates,
                      output);
}

template <typename DataType, typename IndexType>
static inline __global__ void update_thread_per_chunk(const size_t indices_last_dim,
                                                      const size_t num_of_update_elements,
                                                      const size_t num_of_update_chunks,
                                                      const size_t* input_data_dim_pading,
                                                      const IndexType* indices,
                                                      const DataType* updates,
                                                      DataType* output) {
    const auto update_chunk = blockIdx.y * blockDim.x + threadIdx.x;
    const auto update_element = blockIdx.x;

    if (update_chunk >= num_of_update_chunks) return;

    scatter_nd_update(indices_last_dim,
                      num_of_update_elements,
                      input_data_dim_pading,
                      update_element,
                      update_chunk,
                      indices,
                      updates,
                      output);
}

ScatterNDUpdate::ScatterNDUpdate(Type_t data_type,
                                 Type_t indices_type,
                                 size_t indices_last_dim,
                                 size_t num_of_update_elements,
                                 size_t num_of_input_elements,
                                 size_t num_of_update_chunks,
                                 size_t num_of_blocks,
                                 size_t num_of_threads,
                                 bool thread_per_element)
    : data_type_(data_type),
      indices_type_(indices_type),
      indices_last_dim_(indices_last_dim),
      num_of_update_elements_(num_of_update_elements),
      num_of_input_elements_(num_of_input_elements),
      num_of_update_chunks_(num_of_update_chunks),
      num_of_blocks_(num_of_blocks),
      num_of_threads_(num_of_threads),
      thread_per_element_(thread_per_element) {
    TypeValidator<AllElementTypesSwitch>::check(data_type_);
    TypeValidator<ElementTypesSwitch<Type_t::i64, Type_t::i32>>::check(indices_type_);
}

void ScatterNDUpdate::operator()(const cudaStream_t stream,
                                 const void* input,
                                 const void* indices,
                                 const void* updates,
                                 const size_t* input_data_dim_pading,
                                 void* output) const {
    switch (indices_type_) {
        case Type_t::i64:
            return CallByDataType<int64_t>(stream, input, indices, updates, input_data_dim_pading, output);
        case Type_t::i32:
            return CallByDataType<int32_t>(stream, input, indices, updates, input_data_dim_pading, output);
        default:
            throw_ov_exception(
                fmt::format("Index element type = {} is not supported by ScatterNDUpdate operation !!", indices_type_));
    }
}

template <typename IndexType>
void ScatterNDUpdate::CallByDataType(const cudaStream_t stream,
                                     const void* input,
                                     const void* indices,
                                     const void* updates,
                                     const size_t* input_data_dim_pading,
                                     void* output) const {
    switch (data_type_) {
        case Type_t::boolean:
            return Call<bool, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return Call<__nv_bfloat16, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
#endif
        case Type_t::f16:
            return Call<__half, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
        case Type_t::f32:
            return Call<float, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
        case Type_t::f64:
            return Call<double, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
        case Type_t::i8:
            return Call<int8_t, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
        case Type_t::i16:
            return Call<int16_t, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
        case Type_t::i32:
            return Call<int32_t, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
        case Type_t::i64:
            return Call<int64_t, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
        case Type_t::u8:
            return Call<uint8_t, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
        case Type_t::u16:
            return Call<uint16_t, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
        case Type_t::u32:
            return Call<uint32_t, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
        case Type_t::u64:
            return Call<uint64_t, IndexType>(stream, input, indices, updates, input_data_dim_pading, output);
        default:
            throw_ov_exception(
                fmt::format("Index element type = {} is not supported by ScatterNDUpdate operation !!", indices_type_));
    }
}

template <typename DataType, typename IndexType>
void ScatterNDUpdate::Call(const cudaStream_t stream,
                           const void* input,
                           const void* indices,
                           const void* updates,
                           const size_t* input_data_dim_pading,
                           void* output) const {
    // at the beginning we need the output to be the same as the input
    throwIfError(
        cudaMemcpyAsync(output, input, num_of_input_elements_ * sizeof(DataType), cudaMemcpyDeviceToDevice, stream));

    const auto indices_typed = static_cast<const IndexType*>(indices);
    const auto updates_typed = static_cast<const DataType*>(updates);
    auto output_typed = static_cast<DataType*>(output);

    if (thread_per_element_) {
        dim3 grid{static_cast<unsigned int>(num_of_update_chunks_), static_cast<unsigned int>(num_of_blocks_)};
        kernel::update_thread_per_element<<<grid, num_of_threads_, 0, stream>>>(indices_last_dim_,
                                                                                num_of_update_elements_,
                                                                                input_data_dim_pading,
                                                                                indices_typed,
                                                                                updates_typed,
                                                                                output_typed);
    } else {
        dim3 grid{static_cast<unsigned int>(num_of_update_elements_), static_cast<unsigned int>(num_of_blocks_)};
        kernel::update_thread_per_chunk<<<grid, num_of_threads_, 0, stream>>>(indices_last_dim_,
                                                                              num_of_update_elements_,
                                                                              num_of_update_chunks_,
                                                                              input_data_dim_pading,
                                                                              indices_typed,
                                                                              updates_typed,
                                                                              output_typed);
    }
}
}  // namespace kernel

}  // namespace nvidia_gpu
}  // namespace ov
