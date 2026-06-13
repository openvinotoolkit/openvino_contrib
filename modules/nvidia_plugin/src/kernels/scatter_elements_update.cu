// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include "details/error.hpp"
#include "scatter_elements_update.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T, typename Idx>
static __global__ void scatter_elements_update_kernel(const ScatterElementsUpdateParams params,
                                                      const Idx* indices,
                                                      const T* updates,
                                                      T* output) {
    const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= params.num_rows) {
        return;
    }
    // Decompose the row into the non-axis coordinates, accumulating the base
    // offsets into the updates/indices buffer and the data/output buffer.
    size_t rem = row;
    size_t updates_base = 0;
    size_t data_base = 0;
    for (int k = params.rank - 1; k >= 0; --k) {
        if (k == params.axis) {
            continue;
        }
        const size_t coord = rem % params.updates_shape[k];
        rem /= params.updates_shape[k];
        updates_base += coord * params.updates_strides[k];
        data_base += coord * params.data_strides[k];
    }

    const size_t updates_axis_stride = params.updates_strides[params.axis];
    const size_t data_axis_stride = params.data_strides[params.axis];
    for (size_t j = 0; j < params.updates_axis_size; ++j) {
        const size_t updates_offset = updates_base + j * updates_axis_stride;
        long long target = static_cast<long long>(indices[updates_offset]);
        if (target < 0) {
            target += params.data_axis_dim;
        }
        output[data_base + static_cast<size_t>(target) * data_axis_stride] = updates[updates_offset];
    }
}

namespace {
template <typename Idx>
void launchByElementSize(cudaStream_t stream,
                         const ScatterElementsUpdateParams& params,
                         size_t element_size,
                         const void* indices,
                         const void* updates,
                         void* output) {
    if (params.num_rows == 0) {
        return;
    }
    constexpr unsigned kBlock = 256;
    const unsigned blocks = static_cast<unsigned>((params.num_rows + kBlock - 1) / kBlock);
    const auto* idx = static_cast<const Idx*>(indices);
    switch (element_size) {
        case 1:
            scatter_elements_update_kernel<uint8_t, Idx><<<blocks, kBlock, 0, stream>>>(
                params, idx, static_cast<const uint8_t*>(updates), static_cast<uint8_t*>(output));
            break;
        case 2:
            scatter_elements_update_kernel<uint16_t, Idx><<<blocks, kBlock, 0, stream>>>(
                params, idx, static_cast<const uint16_t*>(updates), static_cast<uint16_t*>(output));
            break;
        case 4:
            scatter_elements_update_kernel<uint32_t, Idx><<<blocks, kBlock, 0, stream>>>(
                params, idx, static_cast<const uint32_t*>(updates), static_cast<uint32_t*>(output));
            break;
        case 8:
            scatter_elements_update_kernel<uint64_t, Idx><<<blocks, kBlock, 0, stream>>>(
                params, idx, static_cast<const uint64_t*>(updates), static_cast<uint64_t*>(output));
            break;
        default:
            throw_ov_exception(fmt::format("scatter_elements_update: unsupported element size {}", element_size));
    }
}
}  // namespace

void scatter_elements_update(cudaStream_t stream,
                             const ScatterElementsUpdateParams& params,
                             size_t element_size,
                             size_t index_size,
                             const void* data,
                             const void* indices,
                             const void* updates,
                             void* output) {
    // output begins as a copy of data; the kernel then overwrites the scattered
    // positions.
    throwIfError(cudaMemcpyAsync(output, data, params.num_data_elements * element_size, cudaMemcpyDeviceToDevice,
                                 stream));
    switch (index_size) {
        case 4:
            launchByElementSize<int32_t>(stream, params, element_size, indices, updates, output);
            break;
        case 8:
            launchByElementSize<int64_t>(stream, params, element_size, indices, updates, output);
            break;
        default:
            throw_ov_exception(fmt::format("scatter_elements_update: unsupported index size {}", index_size));
    }
    throwIfError(cudaPeekAtLastError());
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
