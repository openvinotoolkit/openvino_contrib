// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "memory_manager/cuda_device_mem_block.hpp"
#include "memory_manager/model/cuda_immutable_memory_model_builder.hpp"
#include "memory_manager/model/cuda_memory_model.hpp"

namespace ov {
namespace nvidia_gpu {

class DeviceMemBlock;

/**
 * Builds `DeviceMemBlock` which stores immutable constant tensor data in
 * device memory. These tensors can be shared between multiple inference
 * requests.
 */
class ImmutableMemoryBlockBuilder {
public:
    /**
     * Adds a tensor allocation.
     * @param [in] id Buffer identifier. Will be used to obtain corresponding
     * device side tensor pointer.
     * @param [in] data Constant tensor data. Caller should guarantee that this
     * pointer is still valid when `ImmutableMemoryBlockBuilder::build()` method
     * is invoked.
     * @param [in] bsize Tensor memory size in bytes.
     * @throws ov::Exception if
     *  - allocation size is zero
     *  - tensor with specified id is already added
     *  - data pointer is nullptr
     */
    void addAllocation(BufferID id, const void* data, size_t bsize);

    /**
     * @brief Creates and initializes DeviceMemBlock object.
     *
     * This method allocates continuous memory block on device and initializes
     * it with tensor data from host.
     */
    std::pair<DeviceMemBlock::Ptr, MemoryModel::Ptr> build();

    size_t deviceMemoryBlockSize() const;

private:
    ImmutableMemoryModelBuilder model_builder_;
    struct AllocRecord {
        BufferID id;
        const void* data;
        size_t bsize;
    };
    std::vector<AllocRecord> allocations_;
};

}  // namespace nvidia_gpu
}  // namespace ov
