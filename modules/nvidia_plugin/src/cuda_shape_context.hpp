// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_map>

#include "memory_manager/tensor_types.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace nvidia_gpu {

/**
 * @brief Per-inference-request map of TensorID → actual runtime Shape.
 *
 * Populated by:
 *  - Static operations: pre-populated at inference start (shapes known at compile time).
 *  - DynamicOperation: registers output shapes after shape inference + Execute.
 *
 * Consumed by downstream DynamicOperations to determine actual input shapes.
 */
class ShapeContext {
public:
    void setShape(BufferID id, ov::Shape shape) {
        shapes_[id] = std::move(shape);
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
    std::unordered_map<BufferID, ov::Shape> shapes_;
};

}  // namespace nvidia_gpu
}  // namespace ov
