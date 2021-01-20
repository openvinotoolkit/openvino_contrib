// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include "arm_async_infer_request.hpp"
#include "arm_executable_network.hpp"

using namespace ArmPlugin;
using namespace InferenceEngine;

ArmAsyncInferRequest::ArmAsyncInferRequest(
    const ArmInferRequest::Ptr&   inferRequest,
    const ITaskExecutor::Ptr&       taskExecutor,
    const ITaskExecutor::Ptr&       callbackExecutor) :
    AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor),
    _inferRequest(inferRequest) {}

ArmAsyncInferRequest::~ArmAsyncInferRequest() {
    StopAndWait();
}