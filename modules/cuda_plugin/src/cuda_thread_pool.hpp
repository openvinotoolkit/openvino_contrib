// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <deque>

#include <threading/ie_itask_executor.hpp>
#include <atomic>
#include <cuda/stream.hpp>
#include "cuda_jthread.hpp"

namespace CUDAPlugin {

struct CudaThreadContext {
    CudaThreadContext() {
        cuda_stream_ = std::make_shared<CudaStream>();
    }

    std::shared_ptr<CudaStream> cuda_stream_;
};

class CudaThreadPool : public InferenceEngine::ITaskExecutor {
 public:
    using Task = std::function<void()>;

    explicit CudaThreadPool(unsigned _numThreads);
    CudaThreadContext& GetCudaThreadContext();
    ~CudaThreadPool() override;
    void run(Task task) override;

 private:
    void stopThreadPool() noexcept;

    std::mutex mtx_;
    std::atomic<bool> is_stopped_{false};
    std::condition_variable queue_cond_var_;
    std::deque<Task> task_queue_;
    std::vector<CudaJThread> threads_;
};

} // namespace CUDAPlugin
