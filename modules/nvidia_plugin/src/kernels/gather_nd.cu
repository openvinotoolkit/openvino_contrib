// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include "details/error.hpp"
#include "gather_nd.hpp"

namespace ov {
namespace nvidia_gpu {
namespace kernel {

template <typename T, typename Idx>
static __global__ void gather_nd_kernel(const GatherNDParams params,
                                        const T* data,
                                        const Idx* indices,
                                        T* output) {
    const size_t out_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out_idx >= params.num_output_elements) {
        return;
    }
    const size_t slice = out_idx % params.slice_size;
    const size_t bt = out_idx / params.slice_size;
    const size_t tuple = bt % params.num_index_tuples;
    const size_t batch = bt / params.num_index_tuples;

    const size_t index_base = (batch * params.num_index_tuples + tuple) * static_cast<size_t>(params.index_len);
    size_t data_offset = batch * params.batch_block_size + slice;
    for (int l = 0; l < params.index_len; ++l) {
        long long coord = static_cast<long long>(indices[index_base + l]);
        if (coord < 0) {
            coord += params.indexed_dim[l];
        }
        data_offset += static_cast<size_t>(coord) * params.indexed_stride[l];
    }
    output[out_idx] = data[data_offset];
}

namespace {
template <typename Idx>
void launchByElementSize(cudaStream_t stream,
                         const GatherNDParams& params,
                         size_t element_size,
                         const void* data,
                         const void* indices,
                         void* output) {
    constexpr unsigned kBlock = 256;
    const unsigned blocks = static_cast<unsigned>((params.num_output_elements + kBlock - 1) / kBlock);
    const auto* idx = static_cast<const Idx*>(indices);
    switch (element_size) {
        case 1:
            gather_nd_kernel<uint8_t, Idx><<<blocks, kBlock, 0, stream>>>(
                params, static_cast<const uint8_t*>(data), idx, static_cast<uint8_t*>(output));
            break;
        case 2:
            gather_nd_kernel<uint16_t, Idx><<<blocks, kBlock, 0, stream>>>(
                params, static_cast<const uint16_t*>(data), idx, static_cast<uint16_t*>(output));
            break;
        case 4:
            gather_nd_kernel<uint32_t, Idx><<<blocks, kBlock, 0, stream>>>(
                params, static_cast<const uint32_t*>(data), idx, static_cast<uint32_t*>(output));
            break;
        case 8:
            gather_nd_kernel<uint64_t, Idx><<<blocks, kBlock, 0, stream>>>(
                params, static_cast<const uint64_t*>(data), idx, static_cast<uint64_t*>(output));
            break;
        default:
            throw_ov_exception(fmt::format("gather_nd: unsupported element size {}", element_size));
    }
}
}  // namespace

void gather_nd(cudaStream_t stream,
               const GatherNDParams& params,
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
            throw_ov_exception(fmt::format("gather_nd: unsupported index size {}", index_size));
    }
    throwIfError(cudaPeekAtLastError());
}

}  // namespace kernel
}  // namespace nvidia_gpu
}  // namespace ov
