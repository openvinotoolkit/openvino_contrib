// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include "details/error.hpp"
#include "gather_elements.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T, typename Idx>
static __global__ void gather_elements_kernel(const GatherElementsParams params,
                                              const T* data,
                                              const Idx* indices,
                                              T* output) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= params.num_output_elements) {
        return;
    }
    long long axis_coord = static_cast<long long>(indices[idx]);
    if (axis_coord < 0) {
        axis_coord += params.data_axis_dim;
    }
    size_t rem = idx;
    size_t data_offset = 0;
    for (int k = 0; k < params.rank; ++k) {
        const size_t coord = rem / params.indices_strides[k];
        rem -= coord * params.indices_strides[k];
        const size_t use = (k == params.axis) ? static_cast<size_t>(axis_coord) : coord;
        data_offset += use * params.data_strides[k];
    }
    output[idx] = data[data_offset];
}

namespace {
template <typename Idx>
void launchByElementSize(cudaStream_t stream,
                         const GatherElementsParams& params,
                         size_t element_size,
                         const void* data,
                         const void* indices,
                         void* output) {
    constexpr unsigned kBlock = 256;
    const unsigned blocks = static_cast<unsigned>((params.num_output_elements + kBlock - 1) / kBlock);
    const auto* idx = static_cast<const Idx*>(indices);
    switch (element_size) {
        case 1:
            gather_elements_kernel<uint8_t, Idx><<<blocks, kBlock, 0, stream>>>(
                params, static_cast<const uint8_t*>(data), idx, static_cast<uint8_t*>(output));
            break;
        case 2:
            gather_elements_kernel<uint16_t, Idx><<<blocks, kBlock, 0, stream>>>(
                params, static_cast<const uint16_t*>(data), idx, static_cast<uint16_t*>(output));
            break;
        case 4:
            gather_elements_kernel<uint32_t, Idx><<<blocks, kBlock, 0, stream>>>(
                params, static_cast<const uint32_t*>(data), idx, static_cast<uint32_t*>(output));
            break;
        case 8:
            gather_elements_kernel<uint64_t, Idx><<<blocks, kBlock, 0, stream>>>(
                params, static_cast<const uint64_t*>(data), idx, static_cast<uint64_t*>(output));
            break;
        default:
            throw_ov_exception(fmt::format("gather_elements: unsupported element size {}", element_size));
    }
}
}  // namespace

void gather_elements(cudaStream_t stream,
                     const GatherElementsParams& params,
                     size_t element_size,
                     size_t index_size,
                     const void* data,
                     const void* indices,
                     void* output) {
    if (params.num_output_elements == 0) {
        return;
    }
    switch (index_size) {
        case 4:
            launchByElementSize<int32_t>(stream, params, element_size, data, indices, output);
            break;
        case 8:
            launchByElementSize<int64_t>(stream, params, element_size, data, indices, output);
            break;
        default:
            throw_ov_exception(fmt::format("gather_elements: unsupported index size {}", index_size));
    }
    throwIfError(cudaPeekAtLastError());
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
