// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include "broadcast.hpp"
#include "elementtypeswitch.hpp"
#include "tensor_helpers.hpp"

namespace CUDAPlugin {
namespace kernel {

template <typename T>
static __global__ void broadcast(const T* src,
                                 T* dst,
                                 size_t rank,
                                 const size_t* broadcast_dims,
                                 const size_t* src_strides,
                                 const size_t* dst_strides,
                                 size_t dst_num_elements) {
    const unsigned dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_idx >= dst_num_elements) {
        return;
    }

    unsigned src_idx = 0;
    unsigned i = dst_idx;
    for (unsigned r = 0; r < rank; r++) {
        unsigned dst_stride = dst_strides[r];
        unsigned dst_coord = i / dst_stride;
        i = i % dst_stride;
        unsigned src_coord = broadcast_dims[r] * dst_coord;
        src_idx += src_coord * src_strides[r];
    }

    dst[dst_idx] = src[src_idx];
}

Broadcast::Broadcast(CUDAPlugin::kernel::Type_t element_type,
                     size_t shape_rank,
                     size_t dst_num_elements,
                     size_t max_threads_per_block)
    : element_type_{element_type}, shape_rank_{shape_rank}, dst_num_elements_{dst_num_elements} {
    std::tie(num_blocks_, threads_per_block_) = calculateElementwiseGrid(dst_num_elements_, max_threads_per_block);
}

void Broadcast::operator()(const cudaStream_t stream,
                           const void* src,
                           void* dst,
                           const size_t* broadcast_dims,
                           const size_t* src_strides,
                           const size_t* dst_strides) const {
    AllElementTypesSwitch::switch_(element_type_, *this, stream, src, dst, broadcast_dims, src_strides, dst_strides);
}

template <typename T, typename... Args>
constexpr void Broadcast::case_(cudaStream_t stream, Args&&... args) const noexcept {
    callKernel<T>(stream, std::forward<Args>(args)...);
}

template <typename T, typename... Args>
void Broadcast::default_(T t, cudaStream_t, const void*, void*, Args...) const noexcept {
    throwIEException(fmt::format("Element type = {} is not supported by Broadcast operation.", t));
}

template <typename T>
void Broadcast::callKernel(const cudaStream_t stream,
                           const void* src,
                           void* dst,
                           const size_t* broadcast_dims,
                           const size_t* src_strides,
                           const size_t* dst_strides) const {
    broadcast<T><<<num_blocks_, threads_per_block_, 0, stream>>>(static_cast<const T*>(src),
                                                                 static_cast<T*>(dst),
                                                                 shape_rank_,
                                                                 broadcast_dims,
                                                                 src_strides,
                                                                 dst_strides,
                                                                 dst_num_elements_);
}

}  // namespace kernel
}  // namespace CUDAPlugin
