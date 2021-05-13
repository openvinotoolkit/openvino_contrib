// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_immutable_memory_block_builder.hpp"

#include <details/ie_exception.hpp>

#include "memory_manager/cuda_device_mem_block.hpp"

#include <iostream>
#include <cuda_runtime_api.h>

namespace CUDAPlugin {

void
ImmutableMemoryBlockBuilder::addAllocation(TensorID id, const void* data, size_t bsize) {
  IE_ASSERT(data != nullptr);
  model_builder_.addAllocation(id, bsize);
  allocations_.emplace_back(AllocRecord {id, data, bsize});
}

std::shared_ptr<DeviceMemBlock>
ImmutableMemoryBlockBuilder::build() {
  auto memory_block = std::make_shared<DeviceMemBlock>(model_builder_.build());
  for (const auto& allocation : allocations_) {
    void* device_ptr = memory_block->deviceTensorPtr(allocation.id);
    IE_ASSERT(device_ptr != nullptr);
    auto err = ::cudaMemcpy(device_ptr, allocation.data, allocation.bsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
      THROW_IE_EXCEPTION
        << "::cudaMemcpy() failed: "
        << "code " << err
        << ", description: " << cudaGetErrorString(err);
  }
  return memory_block;
}

}  // namespace CUDAPlugin
