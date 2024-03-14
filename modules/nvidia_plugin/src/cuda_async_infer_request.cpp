// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cuda_async_infer_request.hpp"
#include "cuda_itt.hpp"
#include "cuda_thread_pool.hpp"

namespace ov {
namespace nvidia_gpu {

CudaAsyncInferRequest::CudaAsyncInferRequest(const CudaInferRequest::Ptr& request,
                                             const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                             const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                                             const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor)
    : ov::IAsyncInferRequest(request, task_executor, callback_executor),
      request_(request) {
    // In current implementation we have CPU only tasks and no needs in 2 executors
    // So, by default single stage pipeline is created.
    // This stage executes InferRequest::Infer() using cpuTaskExecutor.
    // But if remote asynchronous device is used the pipeline can by splitted tasks that are executed by cpuTaskExecutor
    // and waiting tasks. Waiting tasks can lock execution thread so they use separate threads from other executor.
    constexpr const auto remoteDevice = true;

    auto cuda_thread_pool = std::dynamic_pointer_cast<CudaThreadPool>(wait_executor);
    if (remoteDevice) {
        m_pipeline = {{task_executor,
                      [this] {
                          OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "CudaAsyncInferRequest::infer_preprocess");
                          request_->infer_preprocess();
                      }},
                     {wait_executor,
                      [this, cuda_thread_pool] {
                          auto& threadContext = cuda_thread_pool->get_thread_context();
                          {
                              OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "CudaAsyncInferRequest::start_pipeline");
                              request_->start_pipeline(threadContext);
                          }
                          {
                              OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "CudaAsyncInferRequest::wait_pipeline");
                              request_->wait_pipeline(threadContext);
                          }
                      }},
                     {task_executor, [this] {
                          OV_ITT_SCOPED_TASK(itt::domains::nvidia_gpu, "CudaAsyncInferRequest::infer_postprocess");
                          request_->infer_postprocess();
                      }}};
    }
}

CudaAsyncInferRequest::~CudaAsyncInferRequest() {
    ov::IAsyncInferRequest::stop_and_wait();
}

void CudaAsyncInferRequest::cancel() {
    ov::IAsyncInferRequest::cancel();
    request_->cancel();
}

void CudaAsyncInferRequest::infer_thread_unsafe() {
    start_async_thread_unsafe();
}
}  // namespace nvidia_gpu
}  // namespace ov
