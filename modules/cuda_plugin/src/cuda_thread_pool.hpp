// Copyright (C) 2018-2021 Intel Corporation
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
#include <threading/ie_itask_executor.hpp>

#include "cuda_jthread.hpp"

namespace CUDAPlugin {

class CudaThreadPool : public InferenceEngine::ITaskExecutor {
public:
    using Task = std::function<void()>;

    CudaThreadPool(CUDA::Device d, unsigned _numThreads);
    const ThreadContext& GetThreadContext();
    ~CudaThreadPool() override;
    void run(Task task) override;

private:
    void stopThreadPool() noexcept;

    std::mutex mtx_;
    bool is_stopped_ = false;
    std::condition_variable queue_cond_var_;
    std::deque<Task> task_queue_;
    std::vector<CudaJThread> threads_;
};

}  // namespace CUDAPlugin
