// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_manager/model/cuda_memory_model.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * Builds MemoryModel for immutable memory blob wich contains constant
 * tensors. Such memory blob can be used by multiple infer requests at
 * the same time.
 */
class ImmutableMemoryModelBuilder {
public:
    ImmutableMemoryModelBuilder();

    /**
     * Defines a single tensor allocation.
     * @param [in] id Buffer identifier. Will be used to obtain device side
     * tensor pointer.
     * @param [in] bsize Tensor memory size in bytes.
     * @throws ov::Exception if allocation size is zero or tensor
     * with specified id is already added.
     */
    void addAllocation(BufferID id, size_t bsize);

    /**
     * @returns The size of memory block
     */
    size_t deviceMemoryBlockSize() const;

    /**
     * Creates and initializes MemoryModel object.
     */
    MemoryModel::Ptr build() const;

private:
    ptrdiff_t end_offset_;
    std::unordered_map<BufferID, ptrdiff_t> offsets_;
};

}  // namespace nvidia_gpu
}  // namespace ov
