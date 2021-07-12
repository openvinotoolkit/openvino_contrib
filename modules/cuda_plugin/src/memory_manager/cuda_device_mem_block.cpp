// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_device_mem_block.hpp"

#include <details/ie_exception.hpp>

#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda/runtime.hpp>

namespace CUDAPlugin {

DeviceMemBlock::DeviceMemBlock(MemoryModel::Ptr model) : model_{move(model)} {}

void* DeviceMemBlock::deviceTensorPtr(MemoryModel::TensorID id) {
  if (ptrdiff_t offset = 0; model_->offsetForTensor(id, offset))
    return reinterpret_cast<uint8_t*>(device_mem_ptr_.get()) + offset;
  return nullptr;
}

}  // namespace CUDAPlugin
