// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include <gsl/span>

#include "gpu/device_pointers.hpp"
#include "memory_manager/model/cuda_memory_model.hpp"
#include "cuda_device_mem_block.hpp"
#include "cuda_workbuffers.hpp"

namespace CUDAPlugin {

class MemoryModel;
class IOperationMeta;
class IOperationExec;

/**
 * @brief MemoryManager provides device side tensor pointers by
 * combining together tensor locations from shared immutable memory
 * blob and infer request specific mutable memory blob.
 *
 * Shared immutable tensors are allocated when executable network is loaded
 * and then shared between multiple infer requests. Additionally, MemoryManager
 * allocates mutable memory blob which is used by only one infer request at a time.
 */
class MemoryManager {
public:
  using TensorID = MemoryModel::TensorID;

  using InputTensors = std::vector<InferenceEngine::gpu::DevicePointer<const void*>>;
  using OutputTensors = std::vector<InferenceEngine::gpu::DevicePointer<void*>>;

  /**
   * @param[in] immutableTensors Immutable memory blob which stores constant tensors
   * which are used by multiple infer requests at the same time.
   * @param[in] mutableMemoryModel Infer request specific mutable memory model. It is
   * used to allocate a memory which is used by a single infer request at a time.
   * @param[in] immutableWorkbufferMemory Blob for immutable workbuffers
   */
  MemoryManager(std::shared_ptr<DeviceMemBlock> immutableTensors,
                MemoryModel::Ptr mutableMemoryModel,
                std::shared_ptr<DeviceMemBlock> immutableWorkbufferMemory = nullptr);

  /**
   * Maps input tensor identifiers into device side tensor pointers.
   * @param[in] operation An operation which defines input tensors.
   * @returns An array of corresponding input tensor pointers.
   * @throws InferenceEngineException if any of tensor pointers is not found
   */
  InputTensors inputTensorPointers(const IOperationMeta& operation);

  /**
   * Maps output tensor identifiers into device side tensor pointers.
   * @param[in] operation An operation which defines output tensors.
   * @returns An array of corresponding output tensor pointers.
   * @throws InferenceEngineException if any of tensor pointers is not found
   */
  OutputTensors outputTensorPointers(const IOperationMeta& operation);

  /**
   * Maps operation onto device side work work buffer pointers.
   * @param[in] operation An operation
   * @returns Work buffer pointers
   * @throws InferenceEngineException if any of tensor pointers is not found
   */
  Workbuffers workBuffers(const IOperationExec& operation) const;

private:
  std::shared_ptr<DeviceMemBlock> immutable_tensors_;
  std::unique_ptr<DeviceMemBlock> mutable_tensors_;
  std::shared_ptr<DeviceMemBlock> immutable_workbuffers_;
};

}  // namespace CUDAPlugin
