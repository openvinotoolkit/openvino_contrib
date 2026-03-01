// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <unordered_map>

#include "cuda/runtime.hpp"
#include "memory_manager/tensor_types.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Per-inference-request context for dynamically shaped tensors.
 *
 * Stores both GPU allocations and runtime shapes for dynamic outputs.
 * When a DynamicOperation allocates GPU memory for its outputs, it registers
 * the device pointer and shape here. Downstream operations query this context
 * for both pointer resolution (via MemoryManager) and shape information.
 *
 * Allocations are released either:
 *  - Early, via releaseDynamicBuffer() when the last consumer finishes.
 *  - At end of inference, when DynamicBufferContext is destroyed.
 * Shapes persist for the entire inference (not erased on buffer release).
 */
class DynamicBufferContext {
public:
    DynamicBufferContext() = default;
    DynamicBufferContext(DynamicBufferContext&&) = default;
    DynamicBufferContext& operator=(DynamicBufferContext&&) = default;
    DynamicBufferContext(const DynamicBufferContext&) = delete;
    DynamicBufferContext& operator=(const DynamicBufferContext&) = delete;

    void registerDynamicBuffer(BufferID id, CUDA::Allocation allocation, ov::Shape shape) {
        allocations_.insert_or_assign(id, std::move(allocation));
        shapes_[id] = std::move(shape);
    }

    void releaseDynamicBuffer(BufferID id) {
        allocations_.erase(id);
    }

    std::optional<CUDA::DevicePointer<void*>> getDynamicBuffer(BufferID id) const {
        auto it = allocations_.find(id);
        if (it != allocations_.end()) {
            return CUDA::DevicePointer<void*>{it->second.get()};
        }
        return std::nullopt;
    }

    bool hasDynamicBuffer(BufferID id) const {
        return allocations_.count(id) > 0;
    }

    const ov::Shape& getShape(BufferID id) const {
        auto it = shapes_.find(id);
        OPENVINO_ASSERT(it != shapes_.end(), "Shape not found for BufferID: ", id);
        return it->second;
    }

    bool hasShape(BufferID id) const {
        return shapes_.count(id) > 0;
    }

private:
    std::unordered_map<BufferID, CUDA::Allocation> allocations_;
    std::unordered_map<BufferID, ov::Shape> shapes_;
};

}  // namespace nvidia_gpu
}  // namespace ov
