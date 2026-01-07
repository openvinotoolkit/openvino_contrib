// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_memory_pool.hpp"

#include <fmt/printf.h>

#include "model/cuda_memory_model.hpp"

namespace ov {
namespace nvidia_gpu {

MemoryPool::MemoryPool(const size_t num, std::shared_ptr<MemoryModel> memoryModel) {
    memory_blocks_.reserve(num);
    try {
        for (int i = 0; i < num; ++i) {
            memory_blocks_.push_back(std::make_unique<DeviceMemBlock>(memoryModel));
        }
    } catch (const std::exception& ex) {
        // TODO: Added log message when logging mechanism will be supported
        /**
         * NOTE: It is not possible to allocate all memory of GPU that is why
         *       we allocate as much as possible
         */
        if (memory_blocks_.empty()) {
            throw;
        }
    }
}

void MemoryPool::Interrupt() { cond_var_.notify_all(); }

MemoryPool::Proxy MemoryPool::WaitAndGet(CancellationToken& cancellationToken) {
    std::unique_lock<std::mutex> lock{mtx_};
    cond_var_.wait(lock, [this, &cancellationToken] {
        return !memory_blocks_.empty();
    });
    Proxy memoryManagerProxy{shared_from_this(), move(memory_blocks_.back())};
    memory_blocks_.pop_back();
    return memoryManagerProxy;
}

size_t MemoryPool::Size() const { return memory_blocks_.size(); }

void MemoryPool::Resize(size_t count) {
    const auto memoryManagersCount = memory_blocks_.size();
    if (count > memoryManagersCount) {
        throw_ov_exception(
            fmt::format("Cannot resize MemoryPool with {} value > than it was {}", count, memoryManagersCount));
    }
    memory_blocks_.resize(count);
}

void MemoryPool::PushBack(std::unique_ptr<DeviceMemBlock> memManager) {
    {
        std::lock_guard<std::mutex> lock{mtx_};
        memory_blocks_.push_back(std::move(memManager));
    }
    cond_var_.notify_one();
}

}  // namespace nvidia_gpu
}  // namespace ov
