// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>

#include "cuda_infer_request.hpp"

namespace CUDAPlugin {

class CudaAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    CudaAsyncInferRequest(const CudaInferRequest::Ptr&           inferRequest,
                          const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                          const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                          const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);

    /**
     * Cancel AsyncInferRequest
     */
    void Cancel() override;
    /**
     * Overrides default behaviour and run request asynchronous
     */
    void Infer_ThreadUnsafe() override;

private:
    CudaInferRequest::Ptr           _inferRequest;
};

}  // namespace CUDAPlugin
