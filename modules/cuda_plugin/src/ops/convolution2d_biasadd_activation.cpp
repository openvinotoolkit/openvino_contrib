// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gsl/gsl_assert>

#include "convolution2d_biasadd_activation.hpp"

#include <sstream>
#include <fmt/format.h>

#include "cuda_operation_registry.hpp"
#include "convolution2d_biasadd_activation_cudnn.hpp"

namespace CUDAPlugin {

Convolution2DBiasAddActivationOp::Convolution2DBiasAddActivationOp(
                             const NodeOp& node,
                             IndexCollection&& inputIds,
                             IndexCollection&& outputIds)
    : OperationCuDnn(node, std::move(inputIds), std::move(outputIds)) {
    const auto element_type = node.get_input_element_type(ArgIndices::input);
    Expects(element_type == node.get_input_element_type(ArgIndices::filter));
    Expects(element_type == node.get_input_element_type(ArgIndices::bias));
    Expects(element_type == node.get_output_element_type(ArgIndices::output));
    Expects(node.inputs().size() == 3); // Conv input, filters, Bias

    CreateImpl(node);
}

void Convolution2DBiasAddActivationOp::Execute(
    const InferenceRequestContext& context, Inputs inputs, Outputs outputs,
    const Workbuffers& workbuffers) {
    impl_->Execute(context, inputs, outputs, workbuffers);
}

WorkbufferRequest Convolution2DBiasAddActivationOp::GetWorkBufferRequest() const {
    return impl_->GetWorkBufferRequest();
}

const WorkbufferIndices& Convolution2DBiasAddActivationOp::GetWorkbufferIds() const {
    return impl_->GetWorkbufferIds();
}

IOperationExec::WorkbufferStatus
Convolution2DBiasAddActivationOp::SetWorkbufferIds(WorkbufferIndices&& workbufferIds) {
  return impl_->SetWorkbufferIds(std::move(workbufferIds));
}

void Convolution2DBiasAddActivationOp::CreateImpl(const NodeOp& node) {
    const Convolution::Details::ConvolutionBiasAddActivationParams params { node };

    std::stringstream exception_msg;

    try {
        impl_ = std::make_unique<Convolution2DBiasAddActivationCuDnn>(params);
        return;
    } catch(const std::exception& e) {
        exception_msg << "Failed to create Convolution2DBiasAddActivationCuDnn impl: " << e.what() << std::endl;
    }

    THROW_IE_EXCEPTION << fmt::format("unsupported `{}` node:\n {}",
                                      node.get_type_info().name, exception_msg.str());
}

OPERATION_REGISTER(Convolution2DBiasAddActivationOp, Conv2DBiasAddActivation);
} // namespace CUDAPlugin
