// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cancellation_token.hpp>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

#include "cuda/runtime.hpp"
#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/model/cuda_memory_model.hpp"

class MemoryPoolTest;

namespace ov {
namespace nvidia_gpu {

/**
 * @brief A single dynamic GPU sub-allocation.
 *
 * Holds a shared reference to the underlying cudaMalloc allocation
 * plus an offset/size describing this consumer's portion.
 * Copyable (shared_ptr inside DefaultAllocation), but typically moved.
 */
struct DynamicChunk {
    CUDA::DefaultAllocation allocation;  ///< shared_ptr-based, ref-counted GPU memory
    size_t total_size;                   ///< total bytes of the underlying cudaMalloc
    size_t offset;                       ///< this consumer's start offset within allocation
    size_t usable_size;                  ///< this consumer's usable byte count

    /** Returns device pointer adjusted by offset. */
    void* get() const noexcept {
        return static_cast<uint8_t*>(allocation.get()) + offset;
    }
};

/**
 * @brief MemoryPool provides currently available DeviceMemBlock.
 *
 * This class is an owner of bunch of DeviceMemBlock-s and provides on request
 * WaitAndGet currently available DeviceMemBlock from pool.
 *
 * Additionally supports dynamic GPU memory allocation for operations that
 * need memory with sizes unknown at compile time (dynamic shapes).
 */
class MemoryPool : public std::enable_shared_from_this<MemoryPool> {
public:
    /**
     * @brief RAII handle for a dynamic GPU allocation.
     *
     * On destruction, returns the chunk back to the MemoryPool so it
     * can be recycled to pending requests or freed.
     * Move-only.
     */
    class DynamicHandle {
    public:
        DynamicHandle() = default;
        DynamicHandle(DynamicHandle&&) = default;
        DynamicHandle& operator=(DynamicHandle&&) = default;
        DynamicHandle(const DynamicHandle&) = delete;
        DynamicHandle& operator=(const DynamicHandle&) = delete;

        ~DynamicHandle() {
            if (pool_) pool_->ReleaseDynamicChunk(std::move(chunk_.value()));
        }

        /** Device pointer to the allocated region. */
        void* get() const noexcept { return chunk_->get(); }

        /** Usable size in bytes. */
        size_t size() const noexcept { return chunk_->usable_size; }

        explicit operator bool() const noexcept { return chunk_.has_value(); }

    private:
        friend class MemoryPool;
        DynamicHandle(DynamicChunk chunk, std::shared_ptr<MemoryPool> pool)
            : chunk_{std::move(chunk)}, pool_{std::move(pool)} {}

        std::optional<DynamicChunk> chunk_;
        std::shared_ptr<MemoryPool> pool_;
    };

    /**
     * @brief Proxy provides currently available DeviceMemBlock.
     *
     * Proxy provides access to DeviceMemBlock without given direct
     * access to callee
     */
    class Proxy {
    public:
        Proxy(Proxy&&) = default;
        Proxy& operator=(Proxy&&) = default;

        /**
         * Returns DeviceMemBlock to MemoryPool
         */
        ~Proxy() {
            if (pool_) pool_->PushBack(move(memory_block_));
        }

        /**
         * Provides DeviceMemBlock
         * @return DeviceMemBlock
         */
        DeviceMemBlock& Get() { return *memory_block_; }

        /**
         * Dynamically allocate GPU memory through the pool.
         * @param bytes Number of bytes to allocate
         * @param cancellationToken Token for cancellation support
         * @return RAII handle owning the allocation
         */
        DynamicHandle AllocateDynamic(size_t bytes, CancellationToken& cancellationToken) {
            return pool_->AllocateDynamic(bytes, cancellationToken);
        }

        /**
         * Initialize Proxy with MemoryPool and DeviceMemBlock.
         * MemoryPool is needed for returning back DeviceMemBlock
         * @param pool MemoryPool that is an owner of DeviceMemBlock
         * @param memManager DeviceMemBlock that will be temporary used
         */
        Proxy(std::shared_ptr<MemoryPool> pool, std::unique_ptr<DeviceMemBlock>&& memoryBlock)
            : pool_{move(pool)}, memory_block_{move(memoryBlock)} {}

    private:
        std::unique_ptr<DeviceMemBlock> memory_block_;
        std::shared_ptr<MemoryPool> pool_;
    };

    /**
     * Creates MemoryPool that owns @num DeviceMemBlock-s
     * @param num Number of DeviceMemBlock-s in pool
     * @param sharedConstantsBlob Blob with constants
     * @param memoryModel MemoryModel that is used by each DeviceMemBlock as a layout of memory blob
     *                    containing mutable/intermediate tensors".
     * @param immutableWorkbufferMemory Blob for immutable workbuffers
     */
    MemoryPool(size_t num, std::shared_ptr<MemoryModel> memoryModel);

    /**
     * Interrupt waiting of DeviceMemBlock Proxy object and dynamic allocations
     */
    void Interrupt();
    /**
     * Wait and return Proxy object
     * @return Proxy object through which we can access DeviceMemBlock
     */
    Proxy WaitAndGet(CancellationToken& cancellationToken);

    size_t Size() const;
    void Resize(size_t count);

private:
    friend class ::MemoryPoolTest;

    /**
     * @brief Internal pending request in the dynamic allocation queue.
     */
    struct PendingRequest {
        size_t requested_size;
        uint64_t request_id;
        bool done = false;
        std::optional<DynamicChunk> chunk;
    };

    /**
     * Move DeviceMemBlock back to pool
     * @param memManager DeviceMemBlock
     */
    void PushBack(std::unique_ptr<DeviceMemBlock> memManager);

    /**
     * Dynamically allocate GPU memory.
     *
     * Fast path: tries cudaMalloc directly.
     * Slow path: if cudaMalloc fails, queues a PendingRequest and waits
     * for another thread to release a suitable chunk.
     *
     * @param bytes Number of bytes to allocate (will be aligned)
     * @param cancellationToken Token for cancellation support
     * @return RAII handle owning the allocation
     * @throws ov::Exception on interruption or cancellation
     */
    DynamicHandle AllocateDynamic(size_t bytes, CancellationToken& cancellationToken);

    /**
     * Return a dynamic chunk back to the pool.
     * If there are pending requests that fit, recycles the memory.
     * Otherwise frees the GPU memory.
     */
    void ReleaseDynamicChunk(DynamicChunk chunk);

    // --- Static memory (existing) ---
    std::mutex mtx_;
    std::condition_variable cond_var_;
    std::vector<std::unique_ptr<DeviceMemBlock>> memory_blocks_;

    // --- Dynamic memory ---
    std::mutex dyn_mtx_;
    std::condition_variable dyn_cond_var_;
    std::deque<PendingRequest> pending_requests_;
    uint64_t next_request_id_{0};
    bool interrupted_{false};
};

}  // namespace nvidia_gpu
}  // namespace ov
