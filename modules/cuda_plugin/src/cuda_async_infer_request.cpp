// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <threading/ie_cpu_streams_executor.hpp>

#include "cuda_executable_network.hpp"
#include "cuda_async_infer_request.hpp"
#include "cuda_itt.hpp"
#include "cuda_thread_pool.hpp"

namespace CUDAPlugin {

CudaAsyncInferRequest::CudaAsyncInferRequest(
    const CudaInferRequest::Ptr&               inferRequest,
    const InferenceEngine::ITaskExecutor::Ptr& cpuTaskExecutor,
    const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
    const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor) :
    AsyncInferRequestThreadSafeDefault(inferRequest, cpuTaskExecutor, callbackExecutor),
    _inferRequest(inferRequest) {
    // In current implementation we have CPU only tasks and no needs in 2 executors
    // So, by default single stage pipeline is created.
    // This stage executes InferRequest::Infer() using cpuTaskExecutor.
    // But if remote asynchronous device is used the pipeline can by splitted tasks that are executed by cpuTaskExecutor
    // and waiting tasks. Waiting tasks can lock execution thread so they use separate threads from other executor.
    constexpr const auto remoteDevice = true;

    auto cudaThreadPool = std::dynamic_pointer_cast<CudaThreadPool>(waitExecutor);
    if (remoteDevice) {
        _pipeline = {
            {cpuTaskExecutor, [this] {
                OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin,
                                   "CudaAsyncInferRequest::Preprocessing");
                _inferRequest->inferPreprocess();
            }},
            {waitExecutor, [this, cudaThreadPool] {
                auto& threadContext = cudaThreadPool->GetThreadContext();
                {
                    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin,
                                       "CudaAsyncInferRequest::StartPipeline");
                    _inferRequest->startPipeline(threadContext);
                }
                {
                    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin,
                                       "CudaAsyncInferRequest::WaitPipeline");
                    _inferRequest->waitPipeline(threadContext);
                }
            }},
            {cpuTaskExecutor, [this] {
                OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin,
                                   "CudaAsyncInferRequest::Postprocessing");
                _inferRequest->inferPostprocess();
            }}
        };
    }
}

CudaAsyncInferRequest::~CudaAsyncInferRequest() {
    InferenceEngine::AsyncInferRequestThreadSafeDefault::StopAndWait();
}

void CudaAsyncInferRequest::Cancel() {
    InferenceEngine::AsyncInferRequestThreadSafeDefault::Cancel();
    _inferRequest->Cancel();
}

void CudaAsyncInferRequest::Infer_ThreadUnsafe() {
    StartAsync_ThreadUnsafe();
}

} // namespace CUDAPlugin
