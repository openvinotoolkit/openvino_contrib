// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_device_runtime_api.h>
#include <fmt/format.h>

#include <cuda/float16.hpp>

#include "details/type_validator.hpp"
#include "scatter_update.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

// One thread per output "column" (fixed outer/inner coordinates). The thread
// iterates the indices in row-major order and writes sequentially, so when
// `indices` contains duplicates the last write wins — matching the reference
// (sequential) ScatterUpdate semantics, with no inter-thread races.
template <typename DataType, typename IndexType>
static inline __global__ void scatter_update_kernel(const size_t num_columns,
                                                    const size_t indices_size,
                                                    const size_t inner_size,
                                                    const size_t axis_dim,
                                                    const IndexType* indices,
                                                    const DataType* updates,
                                                    DataType* output) {
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= num_columns) return;

    const size_t outer = col / inner_size;
    const size_t inner = col % inner_size;

    for (size_t idx = 0; idx < indices_size; ++idx) {
        IndexType axis_coord = indices[idx];
        if (axis_coord < 0) axis_coord += static_cast<IndexType>(axis_dim);
        // Skip out-of-range indices rather than writing past the output buffer.
        // ScatterUpdate indices are expected to be valid; a malformed index must
        // not corrupt device memory.
        if (axis_coord < 0 || static_cast<size_t>(axis_coord) >= axis_dim) {
            continue;
        }
        const size_t out = (outer * axis_dim + static_cast<size_t>(axis_coord)) * inner_size + inner;
        output[out] = updates[(outer * indices_size + idx) * inner_size + inner];
    }
}

ScatterUpdate::ScatterUpdate(Type_t data_type,
                             Type_t indices_type,
                             size_t num_input_elements,
                             size_t num_update_elements,
                             size_t indices_size,
                             size_t inner_size,
                             size_t axis_dim,
                             size_t num_blocks,
                             size_t num_threads)
    : data_type_(data_type),
      indices_type_(indices_type),
      num_input_elements_(num_input_elements),
      num_update_elements_(num_update_elements),
      indices_size_(indices_size),
      inner_size_(inner_size),
      axis_dim_(axis_dim),
      num_blocks_(num_blocks),
      num_threads_(num_threads) {
    TypeValidator<AllElementTypesSwitch>::check(data_type_);
    TypeValidator<ElementTypesSwitch<Type_t::i64, Type_t::i32>>::check(indices_type_);
}

void ScatterUpdate::operator()(const cudaStream_t stream,
                               const void* input,
                               const void* indices,
                               const void* updates,
                               void* output) const {
    switch (indices_type_) {
        case Type_t::i64:
            return CallByDataType<int64_t>(stream, input, indices, updates, output);
        case Type_t::i32:
            return CallByDataType<int32_t>(stream, input, indices, updates, output);
        default:
            throw_ov_exception(
                fmt::format("Index element type = {} is not supported by ScatterUpdate operation !!", indices_type_));
    }
}

template <typename IndexType>
void ScatterUpdate::CallByDataType(const cudaStream_t stream,
                                   const void* input,
                                   const void* indices,
                                   const void* updates,
                                   void* output) const {
    switch (data_type_) {
        case Type_t::boolean:
            return Call<bool, IndexType>(stream, input, indices, updates, output);
#ifdef CUDA_HAS_BF16_TYPE
        case Type_t::bf16:
            return Call<__nv_bfloat16, IndexType>(stream, input, indices, updates, output);
#endif
        case Type_t::f16:
            return Call<__half, IndexType>(stream, input, indices, updates, output);
        case Type_t::f32:
            return Call<float, IndexType>(stream, input, indices, updates, output);
        case Type_t::f64:
            return Call<double, IndexType>(stream, input, indices, updates, output);
        case Type_t::i8:
            return Call<int8_t, IndexType>(stream, input, indices, updates, output);
        case Type_t::i16:
            return Call<int16_t, IndexType>(stream, input, indices, updates, output);
        case Type_t::i32:
            return Call<int32_t, IndexType>(stream, input, indices, updates, output);
        case Type_t::i64:
            return Call<int64_t, IndexType>(stream, input, indices, updates, output);
        case Type_t::u8:
            return Call<uint8_t, IndexType>(stream, input, indices, updates, output);
        case Type_t::u16:
            return Call<uint16_t, IndexType>(stream, input, indices, updates, output);
        case Type_t::u32:
            return Call<uint32_t, IndexType>(stream, input, indices, updates, output);
        case Type_t::u64:
            return Call<uint64_t, IndexType>(stream, input, indices, updates, output);
        default:
            throw_ov_exception(
                fmt::format("Data element type = {} is not supported by ScatterUpdate operation !!", data_type_));
    }
}

template <typename DataType, typename IndexType>
void ScatterUpdate::Call(const cudaStream_t stream,
                         const void* input,
                         const void* indices,
                         const void* updates,
                         void* output) const {
    // Output starts as a copy of the input data.
    throwIfError(
        cudaMemcpyAsync(output, input, num_input_elements_ * sizeof(DataType), cudaMemcpyDeviceToDevice, stream));

    if (num_update_elements_ == 0 || indices_size_ == 0) return;

    const size_t num_columns = num_update_elements_ / indices_size_;
    const auto indices_typed = static_cast<const IndexType*>(indices);
    const auto updates_typed = static_cast<const DataType*>(updates);
    auto output_typed = static_cast<DataType*>(output);

    scatter_update_kernel<<<num_blocks_, num_threads_, 0, stream>>>(num_columns,
                                                                    indices_size_,
                                                                    inner_size_,
                                                                    axis_dim_,
                                                                    indices_typed,
                                                                    updates_typed,
                                                                    output_typed);
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
