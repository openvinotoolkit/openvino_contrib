// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>

#include "arm_infer_request.hpp"

namespace ArmPlugin {

class ArmAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    ArmAsyncInferRequest(const ArmInferRequest::Ptr&                   inferRequest,
                         const InferenceEngine::ITaskExecutor::Ptr&    taskExecutor,
                         const InferenceEngine::ITaskExecutor::Ptr&    callbackExecutor);

    ~ArmAsyncInferRequest() override;

private:
    ArmInferRequest::Ptr  _inferRequest;
};

}  // namespace ArmPlugin
