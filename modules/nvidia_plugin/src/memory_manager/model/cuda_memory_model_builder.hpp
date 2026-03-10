// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "memory_manager/model/cuda_memory_model.hpp"
#include "openvino/runtime/memory_solver.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * Builds MemoryModel for mutable memory blob wich contains input,
 * output and intermediate tensors.
 */
class MemoryModelBuilder {
public:
    /**
     * Defines a single tensor allocation.
     *
     * @param [in] id Buffer identifier. Will be used to obtain device side
     * tensor pointer.
     * @param [in] producerIndex Execution order index of first use. The data is
     * produced here.
     * @param [in] lastConsumerIndex The execution order index of last use. After that
     * data will be released. -1 is a reserved value for "till to end".
     * @param [in] bsize Tensor memory size in bytes.
     * @throws ov::Exception if allocation size is zero or tensor
     * with specified id is already added.
     */
    void addAllocation(BufferID id, int producerIndex, int lastConsumerIndex, size_t bsize);

    /**
     * Creates and initializes MemoryModel object.
     */
    MemoryModel::Ptr build();

private:
    std::vector<ov::MemorySolver::Box> boxes_;
    std::unordered_map<BufferID, ptrdiff_t> offsets_;
};

}  // namespace nvidia_gpu
}  // namespace ov
