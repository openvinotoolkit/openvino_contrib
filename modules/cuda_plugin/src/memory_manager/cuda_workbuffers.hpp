// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <cstddef>
#include <vector>

#include <gpu/device_pointers.hpp>

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
  using immutable_buffer = InferenceEngine::gpu::DevicePointer<const void*>;
  using mutable_buffer = InferenceEngine::gpu::DevicePointer<void*>;
  std::vector<immutable_buffer> immutable_buffers;
  std::vector<mutable_buffer> mutable_buffers;
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
