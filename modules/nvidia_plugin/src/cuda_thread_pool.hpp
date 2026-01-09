// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <cuda_thread_context.hpp>
#include <deque>
#include <mutex>
#include <queue>
#include <thread>

#include "cuda_jthread.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace ov {
namespace nvidia_gpu {

class CudaThreadPool : public ov::threading::ITaskExecutor {
public:
    using Task = std::function<void()>;

    CudaThreadPool(CUDA::Device d, unsigned _numThreads);
    ~CudaThreadPool() override;
    const ThreadContext& get_thread_context();
    void run(Task task) override;

private:
    void stop_thread_pool() noexcept;

    std::mutex mtx_;
    bool is_stopped_ = false;
    std::condition_variable queue_cond_var_;
    std::deque<Task> task_queue_;
    std::vector<CudaJThread> threads_;
};

}  // namespace nvidia_gpu
}  // namespace ov
