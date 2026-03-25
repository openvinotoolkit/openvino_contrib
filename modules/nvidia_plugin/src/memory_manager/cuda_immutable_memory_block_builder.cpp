// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_immutable_memory_block_builder.hpp"

#include "openvino/core/except.hpp"

namespace ov {
namespace nvidia_gpu {

void ImmutableMemoryBlockBuilder::addAllocation(BufferID id, const void* data, size_t bsize) {
    OPENVINO_ASSERT(data != nullptr);
    model_builder_.addAllocation(id, bsize);
    allocations_.emplace_back(AllocRecord{id, data, bsize});
}

size_t ImmutableMemoryBlockBuilder::deviceMemoryBlockSize() const { return model_builder_.deviceMemoryBlockSize(); }

std::pair<DeviceMemBlock::Ptr, MemoryModel::Ptr> ImmutableMemoryBlockBuilder::build() {
    auto memory_model = model_builder_.build();
    auto memory_block = std::make_shared<DeviceMemBlock>(memory_model);
    for (const auto& allocation : allocations_) {
        void* device_ptr = memory_block->deviceBufferPtr(allocation.id);
        OPENVINO_ASSERT(device_ptr != nullptr);
        throwIfError(::cudaMemcpy(device_ptr, allocation.data, allocation.bsize, cudaMemcpyHostToDevice));
    }
    return {memory_block, memory_model};
}

}  // namespace nvidia_gpu
}  // namespace ov
