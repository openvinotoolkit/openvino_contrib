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
namespace util {
class ReadValueBase;
class AssignBase;
}  // namespace util
}  // namespace op
namespace nvidia_gpu {

class DynamicBufferContext;

/**
 * @brief Key for caching static operations by input shapes (and, for shape-
 *        dependent ops, by input values).
 *
 * Most operations' output shape is determined by input shapes alone. But ops
 * like Broadcast/Reshape derive their output shape from input VALUES (the
 * target-shape tensor), so for those the downloaded small integer values are
 * folded into the key. input_values stays empty for ordinary ops, leaving the
 * key purely shape-based.
 */
struct ShapeKey {
    std::vector<ov::Shape> input_shapes;
    std::vector<std::vector<int64_t>> input_values;

    bool operator==(const ShapeKey& other) const {
        return input_shapes == other.input_shapes && input_values == other.input_values;
    }
};

struct ShapeKeyHash {
    size_t operator()(const ShapeKey& key) const {
        size_t seed = 0;
        auto mix = [&seed](size_t v) { seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2); };
        for (const auto& shape : key.input_shapes) {
            for (auto dim : shape) {
                mix(std::hash<size_t>{}(dim));
            }
            mix(std::hash<size_t>{}(shape.size()));
        }
        for (const auto& vals : key.input_values) {
            for (auto v : vals) {
                mix(std::hash<int64_t>{}(v));
            }
            mix(std::hash<size_t>{}(vals.size()) ^ 0x1234);
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
 * @brief Global LRU cache for dynamic shape operations.
 *
 * Shared across all DynamicOperation instances within a CompiledModel, so the
 * plugin holds a single bounded cache for the whole model instead of one cache
 * per operation. Uses a composite key (operation identity + input shapes/values)
 * and double-checked locking in getOrCreate(). Capacity 300 matches the Intel
 * GPU plugin (~1-2 entries per op for a typical model).
 */
class DynamicOperationCache {
public:
    static constexpr size_t kDefaultCapacity = 300;

    explicit DynamicOperationCache(size_t capacity = kDefaultCapacity) : cache_{capacity} {}

    template <typename Factory>
    std::shared_ptr<CachedOperation> getOrCreate(const void* op_id, const ShapeKey& key, Factory factory) {
        OperationShapeKey gkey{op_id, key};
        {
            std::lock_guard<std::mutex> lock{mutex_};
            auto* found = cache_.find(gkey);
            if (found) {
                return *found;
            }
        }
        auto op = factory();
        {
            std::lock_guard<std::mutex> lock{mutex_};
            auto* found = cache_.find(gkey);
            if (found) {
                return *found;
            }
            return cache_.insert(gkey, std::move(op));
        }
    }

private:
    struct OperationShapeKey {
        const void* op_id;
        ShapeKey shape_key;

        bool operator==(const OperationShapeKey& other) const {
            return op_id == other.op_id && shape_key == other.shape_key;
        }
    };

    struct OperationShapeKeyHash {
        size_t operator()(const OperationShapeKey& key) const {
            size_t seed = std::hash<const void*>{}(key.op_id);
            ShapeKeyHash shape_hash;
            seed ^= shape_hash(key.shape_key) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    std::mutex mutex_;
    LruCache<OperationShapeKey, std::shared_ptr<CachedOperation>, OperationShapeKeyHash> cache_;
};

/**
 * @brief Delegator operation for nodes with dynamic shapes.
 *
 * Sits in exec_sequence_ like a normal operation. On Execute() it reads the
 * actual input shapes from DynamicBufferContext, looks up (or creates) a cached
 * static operation for those shapes in the model-global cache, allocates
 * dynamic GPU memory for the outputs, delegates execution, and registers the
 * output buffers/shapes. Parameter/Result/ReadValue/Assign are handled as
 * boundary conditions. CUDA Graphs are incompatible with dynamic shapes.
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

    CudaGraphCompatibility GetCudaGraphCompatibilityImpl() const override {
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

    void executeReadValue(const ov::op::util::ReadValueBase& readValueNode,
                          const InferenceRequestContext& context,
                          const CUDA::Stream& stream,
                          Inputs inputTensors,
                          DynamicBufferContext& dynBufCtx) const;

    void executeAssign(const ov::op::util::AssignBase& assignNode,
                       const InferenceRequestContext& context,
                       const CUDA::Stream& stream,
                       Inputs inputTensors,
                       DynamicBufferContext& dynBufCtx) const;

    // The prepared state of the cached path: the cached static operation, the
    // resolved input pointers, and the freshly allocated output buffers. Built
    // by prepareCachedOperationContext(), then consumed by executeCachedOperation()
    // and finalizeOutputs().
    struct CachedOperationContext {
        std::shared_ptr<CachedOperation> cached;
        std::vector<CUDA::DevicePointer<const void*>> input_ptrs;
        std::vector<CUDA::Allocation> output_allocs;
    };

    // Collect input shapes/values, resolve input pointers, look up or create the
    // cached static op, and allocate outputs.
    CachedOperationContext prepareCachedOperationContext(const InferenceRequestContext& context,
                                                         const CUDA::Stream& stream,
                                                         Inputs inputTensors,
                                                         DynamicBufferContext& dynBufCtx) const;

    // Produce this op's output: a zero-copy input -> output copy for
    // Reshape/Squeeze/Unsqueeze (modeled by the registry as a NopOp), otherwise
    // the cached static kernel (skipped for a zero-element output, modeled by a
    // null operation).
    void executeCachedOperation(const InferenceRequestContext& context,
                                const CUDA::Stream& stream,
                                const CachedOperationContext& ctx) const;

    // Register output buffers/shapes and release buffers whose last consumer is
    // this operation.
    void finalizeOutputs(DynamicBufferContext& dynBufCtx, CachedOperationContext& ctx) const;

    std::shared_ptr<CachedOperation> createCachedOperation(
        const ShapeKey& key,
        const std::vector<CUDA::DevicePointer<const void*>>& input_ptrs,
        const CUDA::Stream& stream) const;

    std::shared_ptr<ov::Node> original_node_;
    CreationContext creation_context_;

    // Real output IDs stored separately. OperationBase gets empty output IDs
    // so that MemoryManager::outputTensorPointers() doesn't try to resolve
    // dynamic output buffers that haven't been allocated yet.
    IndexCollection dynamic_output_ids_;

    // BufferIDs to release from DynamicBufferContext after Execute().
    // Set by SubGraph::initExecuteSequence() based on buffer lifespan analysis.
    std::vector<BufferID> release_ids_;

    // Set when createCachedOperation() discovers this op's output shape stays
    // dynamic even with concrete input shapes, i.e. it depends on input VALUES
    // (e.g. Broadcast/Reshape target shape). Subsequent Execute() calls then
    // fold those values into the cache key. mutable: updated lazily during the
    // const Execute() path.
    mutable bool has_dynamic_output_ = false;
};

}  // namespace nvidia_gpu
}  // namespace ov
