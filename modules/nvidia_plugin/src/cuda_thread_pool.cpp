// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cuda_thread_pool.hpp"

#include <fmt/format.h>

#include "cuda_latch.hpp"

namespace ov {
namespace nvidia_gpu {

static thread_local ThreadContext* contextPtr = nullptr;

CudaThreadPool::CudaThreadPool(CUDA::Device d, unsigned _numThreads) {
    try {
        CudaLatch latch{_numThreads};
        for (int i = 0; i < _numThreads; ++i) {
            threads_.emplace_back([this, d, &latch] {
                ThreadContext context{d};
                contextPtr = &context;
                latch.count_down();
                while (true) {
                    Task task;
                    {
                        std::unique_lock<std::mutex> lock(mtx_);
                        queue_cond_var_.wait(lock, [&] { return !task_queue_.empty() || is_stopped_; });
                        if (is_stopped_) {
                            break;
                        }
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
        latch.wait();
    } catch (...) {
        stop_thread_pool();
        throw;
    }
}

CudaThreadPool::~CudaThreadPool() { stop_thread_pool(); }

void CudaThreadPool::stop_thread_pool() noexcept {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        is_stopped_ = true;
    }
    queue_cond_var_.notify_all();
    threads_.clear();
}

const ThreadContext& CudaThreadPool::get_thread_context() {
    if (!contextPtr) {
        throw_ov_exception(
            "Call get_thread_context() not from ThreadPool owned thread is not "
            "allowed !!");
    }
    return *contextPtr;
}

void CudaThreadPool::run(Task task) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        task_queue_.push_back(task);
    }
    queue_cond_var_.notify_one();
}

}  // namespace nvidia_gpu
}  // namespace ov
