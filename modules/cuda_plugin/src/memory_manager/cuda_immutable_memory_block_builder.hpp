// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include "memory_manager/model/cuda_memory_model.hpp"
#include "memory_manager/model/cuda_immutable_memory_model_builder.hpp"

namespace CUDAPlugin {

class DeviceMemBlock;

/**
 * Builds `DeviceMemBlock` which stores immutable constant tensor data in
 * device memory. These tensors can be shared between multiple inference
 * requests.
 */
class ImmutableMemoryBlockBuilder {
public:
  using TensorID = MemoryModel::TensorID;

  /**
   * Adds a tensor allocation.
   * @param [in] id Tensor identifier. Will be used to obtain corresponding
   * device side tensor pointer.
   * @param [in] data Constant tensor data. Caller should guarantee that this
   * pointer is still valid when `ImmutableMemoryBlockBuilder::build()` method
   * is invoked.
   * @param [in] bsize Tensor memory size in bytes.
   * @throws InferenceEngineException if tensor is already added.
   */
  void addAllocation(TensorID id, const void* data, size_t bsize);

  /**
   * @brief Creates and initializes DeviceMemBlock object.
   *
   * This method allocates continuous memory block on device and initializes
   * it with tensor data from host.
   */
  std::shared_ptr<DeviceMemBlock> build();

private:
  ImmutableMemoryModelBuilder model_builder_;
  struct AllocRecord {
    TensorID id;
    const void* data;
    size_t bsize;
  };
  std::vector<AllocRecord> allocations_;
};

}  // namespace CUDAPlugin
