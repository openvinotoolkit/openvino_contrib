// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <condition_variable>
#include <mutex>
#include <cancellation_token.hpp>

#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/model/cuda_memory_model.hpp"

class MemoryManagerPoolTest;

namespace CUDAPlugin {

/**
 * @brief MemoryManagerPool provides currently available MemoryManager.
 *
 * This class is an owner of bunch of MemoryManager-s and provides on request
 * WaitAndGet currently available MemoryManager from pool
 */
class MemoryManagerPool
    : public std::enable_shared_from_this<MemoryManagerPool> {
 public:
    /**
     * @brief Proxy provides currently available MemoryManager.
     *
     * Proxy provides access to MemoryManager without given direct
     * access to callee
     */
  class Proxy {
   public:
    Proxy(Proxy&&) = default;
    Proxy& operator=(Proxy&&) = default;

        /**
         * Returns MemoryManager to MemoryManagerPool
         */
    ~Proxy() {
      if (pool_) pool_->PushBack(move(memory_manager_));
    }

        /**
         * Provides MemoryManager
         * @return MemoryManager
         */
    MemoryManager& Get() { return *memory_manager_; }

        /**
         * Initialize Proxy with MemoryManagerPool and MemoryManager.
         * MemoryManagerPool is needed for returning back MemoryManager
         * @param pool MemoryManagerPool that is an owner of MemoryManager
         * @param memManager MemoryManager that will be temporary used
         */
    Proxy(std::shared_ptr<MemoryManagerPool> pool,
          std::unique_ptr<MemoryManager>&& memManager)
        : pool_{move(pool)}, memory_manager_{move(memManager)} {}

   private:
    std::unique_ptr<MemoryManager> memory_manager_;
    std::shared_ptr<MemoryManagerPool> pool_;
  };

    /**
     * Creates MemoryManagerPool that owns @num MemoryManager-s
     * @param num Number of MemoryManager-s in pool
     * @param sharedConstantsBlob Blob with constants
     * @param memoryModel MemoryModel that is used by each MemoryManager as a layout of memory blob
     *                    containing mutable/intermediate tensors".
     * @param immutableWorkbufferMemory Blob for immutable workbuffers
     */
  MemoryManagerPool(size_t num,
                    std::shared_ptr<DeviceMemBlock> sharedConstantsBlob,
                    std::shared_ptr<MemoryModel> memoryModel,
                    std::shared_ptr<DeviceMemBlock> immutableWorkbufferMemory = nullptr);

  /**
   * Interrupt waiting of MemoryManager Proxy object
   */
  void Interrupt();
  /**
   * Wait and return Proxy object
   * @return Proxy object through which we can access MemoryManager
   */
  Proxy WaitAndGet(CancellationToken& cancellationToken);

  size_t Size() const;

 private:
  friend class ::MemoryManagerPoolTest;

    /**
     * Move MemoryManager back to pool
     * @param memManager MemoryManager
     */
  void PushBack(std::unique_ptr<MemoryManager> memManager);

  std::mutex mtx_;
  std::condition_variable cond_var_;
  std::vector<std::unique_ptr<MemoryManager>> memory_managers_;
  size_t number_memory_managers_{};
};

}  // namespace CUDAPlugin
