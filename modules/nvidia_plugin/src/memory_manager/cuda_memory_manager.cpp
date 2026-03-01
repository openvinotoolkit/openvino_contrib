// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_memory_manager.hpp"

#include "cuda_dynamic_buffer_context.hpp"
#include "cuda_operation_base.hpp"

namespace ov {
namespace nvidia_gpu {

MemoryManager::MemoryManager(DeviceMemBlock::Ptr immutableTensors,
                             MemoryModel::Ptr mutableMemoryModel,
                             DeviceMemBlock::Ptr immutableWorkbufferMemory)
    : immutable_tensors_{immutableTensors},
      mutable_tensors_model_{mutableMemoryModel},
      immutable_workbuffers_{immutableWorkbufferMemory} {}

MemoryManager::InputTensors MemoryManager::inputTensorPointers(const IOperationMeta& operation,
                                                               CUDA::DevicePointer<void*> mutableBufferPtr,
                                                               const DynamicBufferContext& dynBufCtx) const {
    InputTensors result;
    for (auto id : operation.GetInputIds()) {
        auto dynBuf = dynBufCtx.getDynamicBuffer(id.GetBuffer().GetId());
        if (dynBuf) {
            result.emplace_back(dynBuf->get());
            continue;
        }
        const void* ptr = immutable_tensors_->deviceTensorPtr(id);
        if (ptr == nullptr) ptr = mutable_tensors_model_->deviceTensorPtr(mutableBufferPtr.cast<uint8_t*>(), id);
        OPENVINO_ASSERT(ptr != nullptr, "Tensor not found. ID is " + to_string(id));
        result.emplace_back(ptr);
    }
    return result;
}

MemoryManager::OutputTensors MemoryManager::outputTensorPointers(const IOperationMeta& operation,
                                                                 CUDA::DevicePointer<void*> mutableBufferPtr,
                                                                 const DynamicBufferContext& dynBufCtx) const {
    OutputTensors result;
    for (auto id : operation.GetOutputIds()) {
        auto dynBuf = dynBufCtx.getDynamicBuffer(id.GetBuffer().GetId());
        if (dynBuf) {
            result.emplace_back(dynBuf->get());
            continue;
        }
        void* ptr = mutable_tensors_model_->deviceTensorPtr(mutableBufferPtr.cast<uint8_t*>(), id);
        OPENVINO_ASSERT(ptr != nullptr, "Tensor not found. ID is " + to_string(id));
        result.emplace_back(ptr);
    }
    return result;
}

Workbuffers MemoryManager::workBuffers(const IOperationExec& operation,
                                       CUDA::DevicePointer<void*> mutableBufferPtr) const {
    Workbuffers result{};
    const auto& indices = operation.GetWorkbufferIds();
    for (const auto immutable_id : indices.immutableIds) {
        void* ptr = immutable_workbuffers_->deviceBufferPtr(immutable_id);
        OPENVINO_ASSERT(ptr != nullptr, "Workbuffer not found. ID is " + std::to_string(immutable_id));
        result.immutable_buffers.emplace_back(ptr);
    }
    for (const auto mutable_id : indices.mutableIds) {
        void* ptr = mutable_tensors_model_->deviceBufferPtr(mutableBufferPtr.cast<uint8_t*>(), mutable_id);
        OPENVINO_ASSERT(ptr != nullptr, "Workbuffer not found. ID is " + std::to_string(mutable_id));
        result.mutable_buffers.emplace_back(ptr);
    }
    return result;
}

}  // namespace nvidia_gpu
}  // namespace ov
