// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_thread_pool.hpp"

#include <fmt/format.h>

#include <details/ie_exception.hpp>

namespace CUDAPlugin {

static thread_local CUDA::ThreadContext* contextPtr = nullptr;

CudaThreadPool::CudaThreadPool(CUDA::Device d, unsigned _numThreads) {
    try {
        for (int i = 0; i < _numThreads; ++i) {
            threads_.emplace_back([this, d] {
                CUDA::ThreadContext context{d};
                contextPtr = &context;
                while (!is_stopped_.load(std::memory_order_acquire)) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> lock(mtx_);
                        queue_cond_var_.wait(lock,
                        [&] {
                            return !task_queue_.empty() ||
                                   is_stopped_.load(std::memory_order_acquire);
                        });
                        if (!task_queue_.empty()) {
                            task = std::move(task_queue_.front());
                            task_queue_.pop_front();
                        }
                    }
                    if (task) {
                        task();
                    }
                }
            });
        }
    } catch (...) {
        stopThreadPool();
        throw;
    }
}

CudaThreadPool::~CudaThreadPool() {
    stopThreadPool();
}

void CudaThreadPool::stopThreadPool() noexcept {
    is_stopped_.store(true, std::memory_order_release);
    queue_cond_var_.notify_all();
    threads_.clear();
}

const CUDA::ThreadContext& CudaThreadPool::GetThreadContext() {
    if (!contextPtr) {
        throwIEException(
            "Call GetThreadContext() not from ThreadPool owned thread is not "
            "allowed !!");
    }
    return *contextPtr;
}

void CudaThreadPool::run(Task task) {
    std::lock_guard<std::mutex> lock(mtx_);
    task_queue_.push_back(task);
    queue_cond_var_.notify_one();
}

}  // namespace CUDAPlugin
