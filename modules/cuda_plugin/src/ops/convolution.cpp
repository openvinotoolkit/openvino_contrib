// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gsl/gsl_assert>

#include "convolution.hpp"

#include <sstream>
#include <fmt/format.h>

#include "cuda_operation_registry.hpp"

#include "convolution_cudnn.hpp"

#undef ENABLE_CUDNN_BACKEND_API_BASED_CONNVOLUTION
#ifdef ENABLE_CUDNN_BACKEND_API_BASED_CONNVOLUTION
#include "convolution_cudnn_be.hpp"
#endif // ENABLE_CUDNN_BACKEND_API_BASED_CONNVOLUTION

namespace CUDAPlugin {

ConvolutionOp::ConvolutionOp(const CUDA::CreationContext& context,
                             const NodeOp& node,
                             IndexCollection&& inputIds,
                             IndexCollection&& outputIds)
    : OperationCuDnn(context, node, std::move(inputIds), std::move(outputIds)) {
    CreateImpl(context, node);
}

void ConvolutionOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs, const Workbuffers& workbuffers) {
    impl_->Execute(context, inputs, outputs, workbuffers);
}

WorkbufferRequest ConvolutionOp::GetWorkBufferRequest() const {
    return impl_->GetWorkBufferRequest();
}

const WorkbufferIds& ConvolutionOp::GetWorkbufferIds() const {
    return impl_->GetWorkbufferIds();
}

IOperationExec::WorkbufferStatus ConvolutionOp::SetWorkbufferIds(
    WorkbufferIds&& workbufferIds) {
  return impl_->SetWorkbufferIds(std::move(workbufferIds));
}

void ConvolutionOp::CreateImpl(const CUDA::CreationContext& context, const NodeOp& node) {
    const Convolution::Details::ConvolutionParams params { node };

    std::stringstream exception_msg;

    try {
        impl_ = std::make_unique<ConvolutionCuDnn>(context, params);
        return;
    } catch(const std::exception& e) {
        exception_msg << "Failed to create ConvolutionCuDnn impl: " << e.what();
    }

#undef ENABLE_CUDNN_BACKEND_API_BASED_CONNVOLUTION
#ifdef ENABLE_CUDNN_BACKEND_API_BASED_CONNVOLUTION
    try {
        impl_ = std::make_unique<ConvolutionCuDnnBE>(params);
        return;
    } catch(const std::exception& e) {
        exception_msg << "\nFailed to create ConvolutionCuDnnBE impl: "
                      << e.what();
    }
#endif // ENABLE_CUDNN_BACKEND_API_BASED_CONNVOLUTION

    throwIEException(fmt::format("Convolution node is not supported:\n{}",
                                 exception_msg.str()));
}


OPERATION_REGISTER(ConvolutionOp, Convolution);
} // namespace CUDAPlugin
