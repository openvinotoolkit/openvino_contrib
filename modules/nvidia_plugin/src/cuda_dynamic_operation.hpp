// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>

#include "cuda_creation_context.hpp"
#include "cuda_operation_base.hpp"
#include "utils/lru_cache.hpp"

namespace ov {
namespace op {
namespace v0 {
class Parameter;
class Result;
}  // namespace v0
}  // namespace op
namespace nvidia_gpu {

class DynamicBufferContext;

/**
 * @brief Key for caching static operations by input shapes.
 */
struct ShapeKey {
    std::vector<ov::Shape> input_shapes;

    bool operator==(const ShapeKey& other) const {
        return input_shapes == other.input_shapes;
    }
};

struct ShapeKeyHash {
    size_t operator()(const ShapeKey& key) const {
        size_t seed = 0;
        for (const auto& shape : key.input_shapes) {
            for (auto dim : shape) {
                seed ^= std::hash<size_t>{}(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            seed ^= std::hash<size_t>{}(shape.size()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

/**
 * @brief Cached static operation variant for a specific input shape combination.
 */
struct CachedOperation {
    OperationBase::Ptr operation;
    std::vector<ov::Shape> output_shapes;
    std::vector<size_t> output_sizes;
    WorkbufferRequest workbuffer_request;
    std::vector<CUDA::DefaultAllocation> immutable_wb_allocations;
    std::vector<CUDA::DevicePointer<void*>> immutable_wb_ptrs;
};

/**
 * @brief Delegator operation for nodes with dynamic shapes.
 *
 * Sits in exec_sequence_ like a normal operation. On Execute():
 * 1. Reads actual input shapes from DynamicBufferContext
 * 2. Looks up (or creates) a cached static operation for those shapes
 * 3. Allocates dynamic GPU memory for outputs and workbuffers
 * 4. Delegates Execute to the cached static operation
 * 5. Registers output pointers and shapes in DynamicBufferContext
 *
 * CUDA Graphs are incompatible with dynamic shapes.
 */
class DynamicOperation : public OperationBase {
public:
    DynamicOperation(const CreationContext& context,
                     const std::shared_ptr<ov::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds);

    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

    CudaGraphCompatibility GetCudaGraphCompatibility() const override {
        return CudaGraphCompatibility::NONE;
    }

    WorkbufferRequest GetWorkBufferRequest() const override {
        return {};
    }

    void setBuffersToRelease(std::vector<BufferID> ids) {
        release_ids_ = std::move(ids);
    }

private:
    void executeParameter(const ov::op::v0::Parameter& paramNode,
                          const InferenceRequestContext& context,
                          const CUDA::Stream& stream,
                          DynamicBufferContext& dynBufCtx) const;

    void executeResult(const ov::op::v0::Result& resultNode,
                       const InferenceRequestContext& context,
                       const CUDA::Stream& stream,
                       DynamicBufferContext& dynBufCtx) const;

    std::shared_ptr<CachedOperation> createCachedOperation(const ShapeKey& key) const;

    std::shared_ptr<ov::Node> original_node_;
    CreationContext creation_context_;

    // Real output IDs stored separately. OperationBase gets empty output IDs
    // so that MemoryManager::outputTensorPointers() doesn't try to resolve
    // dynamic output buffers that haven't been allocated yet.
    IndexCollection dynamic_output_ids_;

    // BufferIDs to release from DynamicBufferContext after Execute().
    // Set by SubGraph::initExecuteSequence() based on buffer lifespan analysis.
    std::vector<BufferID> release_ids_;

    static constexpr size_t kCacheCapacity = 16;
    mutable std::mutex cache_mutex_;
    mutable LruCache<ShapeKey, std::shared_ptr<CachedOperation>, ShapeKeyHash> shape_cache_{kCacheCapacity};
};

}  // namespace nvidia_gpu
}  // namespace ov
