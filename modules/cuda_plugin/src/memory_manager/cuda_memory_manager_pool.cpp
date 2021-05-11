// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_memory_manager_pool.hpp"
#include "model/cuda_memory_model.hpp"

namespace CUDAPlugin {

MemoryManagerPool::MemoryManagerPool(
        size_t num,
        std::shared_ptr<DeviceMemBlock> sharedConstantsBlob,
        std::shared_ptr<MemoryModel> memoryModel) {
    for (int i = 0; i < num; ++i) {
        memory_managers_.push_back(std::make_unique<MemoryManager>(sharedConstantsBlob, memoryModel));
    }
}

MemoryManagerPool::Proxy
MemoryManagerPool::WaitAndGet() {
    std::unique_lock<std::mutex> lock{mtx_};
    cond_var_.wait(lock, [this]() {
        return !memory_managers_.empty();
    });
    auto memoryManagerProxy = Proxy(shared_from_this(), std::move(memory_managers_.front()));
    memory_managers_.pop_front();
    return memoryManagerProxy;
}

void MemoryManagerPool::PushBack(std::unique_ptr<MemoryManager> memManager) {
    {
        std::lock_guard<std::mutex> lock{mtx_};
        memory_managers_.push_back(std::move(memManager));
    }
    cond_var_.notify_one();
}

} // namespace CUDAPlugin
