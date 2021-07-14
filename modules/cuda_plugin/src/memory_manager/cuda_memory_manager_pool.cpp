// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_memory_manager_pool.hpp"
#include "model/cuda_memory_model.hpp"
#include <fmt/printf.h>

namespace CUDAPlugin {

MemoryManagerPool::MemoryManagerPool(
    const size_t num, std::shared_ptr<DeviceMemBlock> sharedConstantsBlob,
    std::shared_ptr<MemoryModel> memoryModel,
    std::shared_ptr<DeviceMemBlock> immutableWorkbufferMemory) {
  memory_managers_.reserve(num);
  try {
      for (int i = 0; i < num; ++i) {
          memory_managers_.push_back(
              std::make_unique<MemoryManager>(sharedConstantsBlob, memoryModel, immutableWorkbufferMemory));
      }
  } catch(const std::exception& ex) {
    // TODO: Added log message when logging mechanism will be supported
    /**
     * NOTE: It is not possible to allocate all memory of GPU that is why
     *       we allocate as much as possible
     */
    if (memory_managers_.empty()) {
        throw;
    }
  }
}

void MemoryManagerPool::Interrupt() {
    cond_var_.notify_all();
}

MemoryManagerPool::Proxy
MemoryManagerPool::WaitAndGet(CancellationToken& cancellationToken) {
    std::unique_lock<std::mutex> lock{mtx_};
    cond_var_.wait(lock, [this, &cancellationToken] {
        cancellationToken.Check();
        return !memory_managers_.empty();
    });
    Proxy memoryManagerProxy{shared_from_this(),
                             move(memory_managers_.back())};
    memory_managers_.pop_back();
    return memoryManagerProxy;
}

size_t MemoryManagerPool::Size() const {
    return memory_managers_.size();
}

void MemoryManagerPool::Resize(size_t count) {
  const auto memoryManagersCount = memory_managers_.size();
  if (count > memoryManagersCount) {
    THROW_IE_EXCEPTION << fmt::format(
        "Cannot resize MemoryManagerPool with {} value > than it was {}",
        count, memoryManagersCount);
  }
  memory_managers_.resize(count);
}

void MemoryManagerPool::PushBack(std::unique_ptr<MemoryManager> memManager) {
  {
    std::lock_guard<std::mutex> lock{mtx_};
    memory_managers_.push_back(std::move(memManager));
  }
  cond_var_.notify_one();
}

}  // namespace CUDAPlugin
