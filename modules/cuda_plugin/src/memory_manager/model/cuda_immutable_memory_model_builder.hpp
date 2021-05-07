// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_manager/model/cuda_memory_model.hpp"

namespace CUDAPlugin {

/**
 * Builds MemoryModel for immutable memory blob wich contains constant
 * tensors. Such memory blob can be used by multiple infer requests at
 * the same time.
 */
class ImmutableMemoryModelBuilder {
public:
  using TensorID = MemoryModel::TensorID;

  ImmutableMemoryModelBuilder();

  /**
   * Defines a single tensor allocation.
   * @param [in] id Tensor identifier. Will be used to obtain device side
   * tensor pointer.
   * @param [in] bsize Tensor memory size in bytes.
   * @throws InferenceEngineException if tensor is already added.
   */
  void addAllocation(TensorID id, size_t bsize);

  /**
   * Creates and initializes MemoryModel object.
   */
  MemoryModel::Ptr build();

private:
  ptrdiff_t end_offset_;
  std::unordered_map<TensorID, ptrdiff_t> offsets_;
};

}  // namespace CUDAPlugin
