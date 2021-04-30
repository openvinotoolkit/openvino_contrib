// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>

#include "cuda_infer_request.hpp"

namespace CUDAPlugin {

// ! [async_infer_request:header]
class CudaAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    CudaAsyncInferRequest(const CudaInferRequest::Ptr&           inferRequest,
                              const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                              const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                              const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);

    ~CudaAsyncInferRequest() override;

private:
    CudaInferRequest::Ptr           _inferRequest;
    InferenceEngine::ITaskExecutor::Ptr _waitExecutor;
};
// ! [async_infer_request:header]

}  // namespace CUDAPlugin
