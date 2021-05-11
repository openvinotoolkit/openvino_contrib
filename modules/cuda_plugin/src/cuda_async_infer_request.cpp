// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <threading/ie_cpu_streams_executor.hpp>

#include "cuda_executable_network.hpp"
#include "cuda_async_infer_request.hpp"
#include "cuda_itt.hpp"

using namespace CUDAPlugin;

CudaAsyncInferRequest::CudaAsyncInferRequest(
    const CudaInferRequest::Ptr&               inferRequest,
    const InferenceEngine::ITaskExecutor::Ptr& cpuTaskExecutor,
    const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
    const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor) :
    AsyncInferRequestThreadSafeDefault(inferRequest, cpuTaskExecutor, callbackExecutor),
    _inferRequest(inferRequest), _waitExecutor(waitExecutor) {
    // In current implementation we have CPU only tasks and no needs in 2 executors
    // So, by default single stage pipeline is created.
    // This stage executes InferRequest::Infer() using cpuTaskExecutor.
    // But if remote asynchronous device is used the pipeline can by splitted tasks that are executed by cpuTaskExecutor
    // and waiting tasks. Waiting tasks can lock execution thread so they use separate threads from other executor.
    constexpr const auto remoteDevice = false;

    if (remoteDevice) {
        _pipeline = {
            {cpuTaskExecutor, [this] {
                OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin,
                                   "CudaAsyncInferRequest::Preprocessing");
                _inferRequest->inferPreprocess();
            }},
            {_waitExecutor, [this] {
                auto cpuStreamExecutor = std::dynamic_pointer_cast<InferenceEngine::CPUStreamsExecutor>(_waitExecutor);
                auto streamId = cpuStreamExecutor->GetStreamId();
                auto execNetwork = _inferRequest->GetExecNetwork();
                auto cudaStream = execNetwork->GetCudaStream(streamId);
                _inferRequest->setCudaStream(cudaStream);
                {
                    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin,
                                       "CudaAsyncInferRequest::StartPipeline");
                    _inferRequest->startPipeline();
                }
                {
                    OV_ITT_SCOPED_TASK(itt::domains::CUDAPlugin,
                                       "CudaAsyncInferRequest::WaitPipeline");
                    _inferRequest->waitPipeline();
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
