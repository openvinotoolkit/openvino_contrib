// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cancellation_token.hpp>
#include <condition_variable>
#include <mutex>

#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/model/cuda_memory_model.hpp"

class MemoryPoolTest;

namespace ov {
namespace nvidia_gpu {

/**
 * @brief MemoryPool provides currently available DeviceMemBlock.
 *
 * This class is an owner of bunch of DeviceMemBlock-s and provides on request
 * WaitAndGet currently available DeviceMemBlock from pool
 */
class MemoryPool : public std::enable_shared_from_this<MemoryPool> {
public:
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
     * Interrupt waiting of DeviceMemBlock Proxy object
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
     * Move DeviceMemBlock back to pool
     * @param memManager DeviceMemBlock
     */
    void PushBack(std::unique_ptr<DeviceMemBlock> memManager);

    std::mutex mtx_;
    std::condition_variable cond_var_;
    std::vector<std::unique_ptr<DeviceMemBlock>> memory_blocks_;
};

}  // namespace nvidia_gpu
}  // namespace ov
