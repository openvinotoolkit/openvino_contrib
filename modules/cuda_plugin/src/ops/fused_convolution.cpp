// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_convolution.hpp"

#include <fmt/format.h>

#include <gsl/gsl_assert>
#include <sstream>

#include "cuda_operation_registry.hpp"
#include "fused_convolution_cudnn.hpp"

namespace CUDAPlugin {

FusedConvolutionOp::FusedConvolutionOp(
    const CUDA::CreationContext& context,
    const NodeOp& node,
    IndexCollection&& inputIds,
    IndexCollection&& outputIds)
    : OperationCuDnn(context, node, std::move(inputIds), std::move(outputIds)) {
    const auto element_type = node.get_input_element_type(ArgIndices::input);
    Expects(element_type == node.get_input_element_type(ArgIndices::filter));
    Expects(element_type == node.get_input_element_type(ArgIndices::bias));
    Expects(element_type == node.get_output_element_type(ArgIndices::output));
    const bool includesOnlyBiasAdd = node.inputs().size() == 3;
    const bool includesSecondAddition = node.inputs().size() == 4;
    Expects(includesOnlyBiasAdd || includesSecondAddition); // Conv input, filters, Bias and optional Add

    CreateImpl(context, node);
}

void FusedConvolutionOp::Execute(
    const InferenceRequestContext& context, Inputs inputs, Outputs outputs,
    const Workbuffers& workbuffers) {
    impl_->Execute(context, inputs, outputs, workbuffers);
}

WorkbufferRequest FusedConvolutionOp::GetWorkBufferRequest() const {
    return impl_->GetWorkBufferRequest();
}

const WorkbufferIds& FusedConvolutionOp::GetWorkbufferIds() const {
    return impl_->GetWorkbufferIds();
}

IOperationExec::WorkbufferStatus FusedConvolutionOp::SetWorkbufferIds(
    WorkbufferIds&& workbufferIds) {
  return impl_->SetWorkbufferIds(std::move(workbufferIds));
}

void FusedConvolutionOp::CreateImpl(const CUDA::CreationContext& context, const NodeOp& node) {
    const Convolution::Details::FusedConvolutionParams params { node };

    std::stringstream exception_msg;

    try {
        impl_ = std::make_unique<FusedConvolutionCuDnn>(context, params);
        return;
    } catch(const std::exception& e) {
        exception_msg << "Failed to create FusedConvolutionCuDnn impl: " << e.what() << std::endl;
    }

    THROW_IE_EXCEPTION << fmt::format("unsupported `{}` node:\n {}",
                                      node.get_type_info().name, exception_msg.str());
}

OPERATION_REGISTER(FusedConvolutionOp, FusedConv2D);
} // namespace CUDAPlugin
