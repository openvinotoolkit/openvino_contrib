// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_device_mem_block.hpp"

#include <details/ie_exception.hpp>

#include <iostream>
#include <cuda_runtime_api.h>

namespace CUDAPlugin {

DeviceMemBlock::DeviceMemBlock(MemoryModel::Ptr model)
  : model_{ model }, device_mem_ptr_{ nullptr }
{
  auto err = ::cudaMalloc(&device_mem_ptr_, model_->deviceMemoryBlockSize());
  if (err != cudaSuccess)
    THROW_IE_EXCEPTION
      << "::cudaMalloc() failed: "
      << "code " << err
      << ", description: " << cudaGetErrorString(err);
}

DeviceMemBlock::~DeviceMemBlock() {
  auto err = ::cudaFree(device_mem_ptr_);
  if (err != cudaSuccess) {
    std::cerr << "::cudaFree() failed: "
              << "code " << err
              << ", description: " << cudaGetErrorString(err)
              << std::endl;
  }
  assert(err == cudaSuccess);
}

void* DeviceMemBlock::deviceTensorPtr(MemoryModel::TensorID id) {
  ptrdiff_t offset = 0;
  if (device_mem_ptr_ && model_->offsetForTensor(id, offset))
    return reinterpret_cast<uint8_t*>(device_mem_ptr_) + offset;
  else
    return nullptr;
}

}  // namespace CUDAPlugin
