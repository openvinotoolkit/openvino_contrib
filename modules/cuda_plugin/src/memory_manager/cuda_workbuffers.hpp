// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstddef>
#include <cuda/device_pointers.hpp>
#include <vector>

namespace CUDAPlugin {

/**
 * @brief WorkbufferRequest - a POD structure describing operator's memory demands
 */
struct WorkbufferRequest {
  using size_in_bytes_t = size_t;
  std::vector<size_in_bytes_t> immutable_sizes;
  std::vector<size_in_bytes_t> mutable_sizes;
};

/**
 * @brief Workbuffers - structure holding preallocated memory buffers
 */
struct Workbuffers {
    using immutable_buffer = CUDA::DevicePointer<const void*>;
    using mutable_buffer = CUDA::DevicePointer<void*>;
    std::vector<immutable_buffer> immutable_buffers;
    std::vector<mutable_buffer> mutable_buffers;
#ifndef __NVCC__  // no std::byte in c++14. TODO: drop guard after splitting kernels
    template <int N>
    CUDA::DeviceBuffer<std::byte> mutableSpan(size_t workspaceSize) const {
        if (!workspaceSize) return {};
        return {mutable_buffers.at(N).cast<std::byte*>().get(), workspaceSize};
    }
#endif
};

/**
 * @brief WorkbufferIds - structure holding the memory buffers' indices
 */
struct WorkbufferIds {
  using vector_of_ids = std::vector<BufferID>;
  vector_of_ids immutableIds;
  vector_of_ids mutableIds;
};

}
