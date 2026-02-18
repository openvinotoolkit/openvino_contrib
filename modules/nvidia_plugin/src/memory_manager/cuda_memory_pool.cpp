// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_memory_pool.hpp"

#include <fmt/printf.h>

#include "memory_manager/model/details/cuda_memory_utils.hpp"
#include "model/cuda_memory_model.hpp"
#include "openvino/core/except.hpp"

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

void MemoryPool::Interrupt() {
    cond_var_.notify_all();
    {
        std::lock_guard<std::mutex> lock{dyn_mtx_};
        interrupted_ = true;
    }
    dyn_cond_var_.notify_all();
}

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

MemoryPool::DynamicHandle 
MemoryPool::AllocateDynamic(size_t bytes, CancellationToken& cancellationToken) {
    OPENVINO_ASSERT(bytes > 0, "Dynamic allocation size must be > 0");
    const size_t aligned_size = applyAllignment(bytes);

    // Fast path: try cudaMalloc directly
    try {
        auto allocation = CUDA::DefaultStream::stream().malloc(aligned_size);
        DynamicChunk chunk{std::move(allocation), aligned_size, 0, aligned_size};
        return DynamicHandle{std::move(chunk), shared_from_this()};
    } catch (...) {
        // cudaMalloc failed — fall through to slow path
    }

    // Slow path: queue a pending request and wait for a released chunk
    std::unique_lock<std::mutex> lock{dyn_mtx_};

    OPENVINO_ASSERT(!interrupted_, "MemoryPool was interrupted before dynamic allocation could be queued");

    const uint64_t my_id = next_request_id_++;
    pending_requests_.push_back(PendingRequest{aligned_size, my_id, false, std::nullopt});

    auto cur_req = std::prev(pending_requests_.end());

    dyn_cond_var_.wait(lock, [this, &cur_req, &cancellationToken] {
        return cur_req->done || interrupted_;
    });

    if (interrupted_ && !cur_req->done) {
        pending_requests_.erase(cur_req);
        OPENVINO_THROW("MemoryPool interrupted while waiting for dynamic allocation");
    }

    DynamicChunk result_chunk = std::move(cur_req->chunk.value());
    pending_requests_.erase(cur_req);

    return DynamicHandle{std::move(result_chunk), shared_from_this()};
}

void MemoryPool::ReleaseDynamicChunk(DynamicChunk chunk) {
    std::lock_guard<std::mutex> lock{dyn_mtx_};

    auto head = std::find_if(pending_requests_.begin(), pending_requests_.end(),
                             [](const PendingRequest& r) { return !r.done; });
    if (head == pending_requests_.end()) {
        return;
    }

    const size_t available = chunk.usable_size;
    if (head->requested_size <= available) {
        head->chunk = DynamicChunk{chunk.allocation, chunk.total_size, chunk.offset, head->requested_size};
        head->done = true;

        // Try to sub-allocate remaining space for next requests
        size_t current_offset = chunk.offset + head->requested_size;
        const size_t end_offset = chunk.offset + chunk.usable_size;
        for (auto it = std::next(head); it != pending_requests_.end() && current_offset < end_offset; ++it) {
            if (it->done) {
                continue;
            }

            const size_t aligned_offset = applyAllignment(current_offset);
            if (aligned_offset >= end_offset) {
                break;
            }

            const size_t remaining = end_offset - aligned_offset;
            if (it->requested_size > remaining) {
                continue;
            }

            it->chunk = DynamicChunk{chunk.allocation, chunk.total_size, aligned_offset, it->requested_size};
            it->done = true;
            current_offset = aligned_offset + it->requested_size;
        }

        dyn_cond_var_.notify_all();
    } else {
        // Head doesn't fit — free the chunk, then retry cudaMalloc for head
        // (freed GPU memory returns to CUDA pool, malloc may now succeed)
        { auto _ = std::move(chunk); }

        try {
            auto allocation = CUDA::DefaultStream::stream().malloc(head->requested_size);
            head->chunk = DynamicChunk{std::move(allocation), head->requested_size, 0, head->requested_size};
            head->done = true;
            dyn_cond_var_.notify_all();
        } catch (...) {
            // cudaMalloc still fails — head keeps waiting
        }
    }
}

}  // namespace nvidia_gpu
}  // namespace ov
