// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mutex>
#include <condition_variable>
#include <deque>

#include "memory_manager/cuda_memory_manager.hpp"
#include "memory_manager/model/cuda_memory_model.hpp"

class MemoryManagerPoolTest;

namespace CUDAPlugin {

class MemoryManagerPool final : public std::enable_shared_from_this<MemoryManagerPool> {
 public:
    class Proxy final {
     public:
        Proxy(const Proxy&) = delete;
        Proxy& operator=(const Proxy&) = delete;

        Proxy(Proxy&&) = default;
        Proxy& operator=(Proxy&&) = default;

        explicit operator bool() const {
            return static_cast<bool>(pool_);
        }

        ~Proxy() {
            if (pool_) {
                pool_->PushBack(std::move(memory_manager_));
            }
        }

        MemoryManager& Get() {
          return *memory_manager_;
        }

     private:
        friend class MemoryManagerPool;

        explicit Proxy(std::shared_ptr<MemoryManagerPool> pool, std::unique_ptr<MemoryManager> memManager)
            : pool_{std::move(pool)}
            , memory_manager_{std::move(memManager)} {
        }

        std::unique_ptr<MemoryManager> memory_manager_;
        std::shared_ptr<MemoryManagerPool> pool_;
    };

    MemoryManagerPool(size_t num,
                      std::shared_ptr<DeviceMemBlock> sharedConstantsBlob,
                      std::shared_ptr<MemoryModel> memoryModel);
    Proxy WaitAndGet();

 private:
    friend class ::MemoryManagerPoolTest;

    void PushBack(std::unique_ptr<MemoryManager> memManager);

    std::mutex mtx_;
    std::condition_variable cond_var_;
    std::deque<std::unique_ptr<MemoryManager>> memory_managers_;
};

} // namespace CUDAPlugin
