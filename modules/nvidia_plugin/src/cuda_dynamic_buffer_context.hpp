// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <unordered_map>

#include "cuda/runtime.hpp"
#include "memory_manager/tensor_types.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Per-inference-request registry of dynamically allocated output buffers.
 *
 * When a DynamicOperation allocates GPU memory for its outputs, it registers
 * the device pointer here. Downstream operations (via MemoryManager) check
 * this context before falling back to static memory offsets.
 *
 * Allocations are stored as CUDA::Allocation (stream-ordered, ref-counted).
 * They are released either:
 *  - Early, via releaseDynamicBuffer() when the last consumer finishes
 *    (driven by DynamicOperation's release list).
 *  - At end of inference, when DynamicBufferContext is destroyed.
 */
class DynamicBufferContext {
public:
    DynamicBufferContext() = default;
    DynamicBufferContext(DynamicBufferContext&&) = default;
    DynamicBufferContext& operator=(DynamicBufferContext&&) = default;
    DynamicBufferContext(const DynamicBufferContext&) = delete;
    DynamicBufferContext& operator=(const DynamicBufferContext&) = delete;

    void registerDynamicOutput(BufferID id, CUDA::Allocation allocation) {
        allocations_.insert_or_assign(id, std::move(allocation));
    }

    void releaseDynamicBuffer(BufferID id) {
        allocations_.erase(id);
    }

    std::optional<CUDA::DevicePointer<void*>> getDynamicOutput(BufferID id) const {
        auto it = allocations_.find(id);
        if (it != allocations_.end()) {
            return CUDA::DevicePointer<void*>{it->second.get()};
        }
        return std::nullopt;
    }

    bool hasDynamicOutput(BufferID id) const {
        return allocations_.count(id) > 0;
    }

private:
    std::unordered_map<BufferID, CUDA::Allocation> allocations_;
};

}  // namespace nvidia_gpu
}  // namespace ov
