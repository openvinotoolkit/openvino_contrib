// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cuda_infer_request.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/iinfer_request.hpp"

namespace ov {
namespace nvidia_gpu {

class CudaAsyncInferRequest : public ov::IAsyncInferRequest {
public:
    CudaAsyncInferRequest(const CudaInferRequest::Ptr& request,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                          const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);

    ~CudaAsyncInferRequest();
    void cancel() override;
    void infer_thread_unsafe() override;

private:
    CudaInferRequest::Ptr request_;
};

}  // namespace nvidia_gpu
}  // namespace ov
