// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_manager/model/cuda_memory_model.hpp"

#include <gsl/pointers>

namespace ov {
namespace nvidia_gpu {

MemoryModel::MemoryModel(size_t bsize,
                         const std::unordered_map<BufferID, ptrdiff_t>& offsets,
                         const std::unordered_map<TensorID, TensorID>& virtual_tensors)
    : bsize_{bsize}, offsets_{offsets}, virtual_tensors_{virtual_tensors} {
    std::transform(
        offsets.begin(), offsets.end(), std::back_inserter(buffer_ids_), [](const auto& b) { return b.first; });
    std::sort(buffer_ids_.begin(), buffer_ids_.end());
}

MemoryModel::MemoryModel(size_t bsize, const std::unordered_map<BufferID, ptrdiff_t>& offsets)
    : bsize_{bsize}, offsets_{offsets} {
    std::transform(
        offsets.begin(), offsets.end(), std::back_inserter(buffer_ids_), [](const auto& b) { return b.first; });
    std::sort(buffer_ids_.begin(), buffer_ids_.end());
}

size_t MemoryModel::deviceMemoryBlockSize() const { return bsize_; }

void* MemoryModel::deviceBufferPtr(CUDA::DevicePointer<uint8_t*> devPtr, const BufferID& id) {
    if (ptrdiff_t offset = 0; offsetForBuffer(id, offset)) return devPtr.get() + offset;
    return nullptr;
}

void* MemoryModel::deviceTensorPtr(CUDA::DevicePointer<uint8_t*> devPtr, const TensorID& id) {
    auto buffer_id_iter = virtual_tensors_.find(id);
    BufferID buffer_id{};
    if (buffer_id_iter != virtual_tensors_.end()) {
        buffer_id = buffer_id_iter->second.GetId();
    } else {
        buffer_id = id.GetBuffer().GetId();
    }
    if (auto bufferPtr = deviceBufferPtr(devPtr, buffer_id); bufferPtr) {
        return static_cast<uint8_t*>(bufferPtr) + id.GetOffset();
    }
    return nullptr;
}

bool MemoryModel::offsetForBuffer(BufferID id, ptrdiff_t& offset) const {
    auto it = offsets_.find(id);
    if (it == offsets_.end()) return false;
    offset = it->second;
    return true;
}

}  // namespace nvidia_gpu
}  // namespace ov
