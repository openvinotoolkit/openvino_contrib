// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_manager/model/cuda_memory_model.hpp"
#include "memory_manager/model/details/cuda_memory_solver.hpp"

namespace CUDAPlugin {

/**
 * Builds MemoryModel for mutable memory blob wich contains input,
 * output and intermediate tensors.
 */
class MemoryModelBuilder {
public:
  using TensorID = MemoryModel::TensorID;

  /**
   * Defines a single tensor allocation.
   *
   * @param [in] id Tensor identifier. Will be used to obtain device side
   * tensor pointer.
   * @param [in] start Execution order index of first use. The data is
   * produced here.
   * @param [in] finish The execution order index of last use. After that
   * data will be released. -1 is a reserved value for "till to end".
   * @param [in] bsize Tensor memory size in bytes.
   * @throws InferenceEngineException if tensor is already added.
   */
  void addAllocation(TensorID id, int start, int finish, size_t bsize);

  /**
   * Creates and initializes MemoryModel object.
   */
  MemoryModel::Ptr build();

private:
  std::vector<MemorySolver::Box> boxes_;
  std::unordered_map<TensorID, ptrdiff_t> offsets_;
};

}  // namespace CUDAPlugin
