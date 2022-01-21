// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_convolution.hpp"

#include <fmt/format.h>

#include <error.hpp>
#include <exception>
#include <gsl/gsl_assert>
#include <memory>

#include "cuda_operation_registry.hpp"
#include "fused_convolution_cudnn.hpp"
#include "fused_group_convolution.hpp"

namespace CUDAPlugin {

template <typename TOperation>
FusedConvolutionOp::FusedConvolutionOp(const CreationContext& context,
                                       const TOperation& op,
                                       IndexCollection&& inputIds,
                                       IndexCollection&& outputIds)
    : OperationCuDnn(context, op, std::move(inputIds), std::move(outputIds)) {
    const auto element_type = op.get_input_element_type(ArgIndices::input);
    Expects(element_type == op.get_input_element_type(ArgIndices::filter));
    Expects(element_type == op.get_input_element_type(ArgIndices::bias));
    Expects(element_type == op.get_output_element_type(ArgIndices::output));
    const bool includesOnlyBiasAdd = op.inputs().size() == 3;
    const bool includesSecondAddition = op.inputs().size() == 4;
    Expects(includesOnlyBiasAdd || includesSecondAddition);  // Conv input, filters, Bias and optional Add

    Convolution::Details::FusedConvolutionParams params{op};
    try {
        impl_ = std::make_unique<FusedConvolutionCuDnn>(context, params);
    } catch (const std::exception& e) {
        throwIEException(
            fmt::format("unsupported `{}` node: Failed to create "
                        "FusedConvolutionCuDnn impl: {}",
                        op.get_type_info().name,
                        e.what()));
    }
}

template FusedConvolutionOp::FusedConvolutionOp(const CreationContext& context,
                                                const nodes::FusedConvolution& op,
                                                IndexCollection&& inputIds,
                                                IndexCollection&& outputIds);

template FusedConvolutionOp::FusedConvolutionOp(const CreationContext& context,
                                                const nodes::FusedGroupConvolution& op,
                                                IndexCollection&& inputIds,
                                                IndexCollection&& outputIds);

void FusedConvolutionOp::Execute(const InferenceRequestContext& context,
                                 Inputs inputs,
                                 Outputs outputs,
                                 const Workbuffers& workbuffers) const {
    impl_->Execute(context, inputs, outputs, workbuffers);
}

WorkbufferRequest FusedConvolutionOp::GetWorkBufferRequest() const { return impl_->GetWorkBufferRequest(); }

const WorkbufferIds& FusedConvolutionOp::GetWorkbufferIds() const { return impl_->GetWorkbufferIds(); }

IOperationExec::WorkbufferStatus FusedConvolutionOp::SetWorkbufferIds(WorkbufferIds&& workbufferIds) {
    return impl_->SetWorkbufferIds(std::move(workbufferIds));
}

OPERATION_REGISTER(FusedConvolutionOp, FusedConvolution);

}  // namespace CUDAPlugin
