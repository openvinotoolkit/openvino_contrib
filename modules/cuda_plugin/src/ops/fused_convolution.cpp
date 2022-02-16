// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <error.hpp>
#include <exception>
#include <gsl/gsl_assert>
#include <memory>

#include "cuda_operation_registry.hpp"
#include "fused_convolution_cudnn.hpp"

namespace CUDAPlugin {

OperationBase::Ptr fusedConvolutionFactory(const CreationContext& context,
                                           const std::shared_ptr<ngraph::Node>& node,
                                           OperationBase::IndexCollection&& inputIds,
                                           OperationBase::IndexCollection&& outputIds) {
    using ArgIndices = Convolution::Details::FusedConvolutionIndices;
    const auto element_type = node->get_input_element_type(ArgIndices::input);
    Expects(element_type == node->get_input_element_type(ArgIndices::filter));
    Expects(element_type == node->get_input_element_type(ArgIndices::bias));
    Expects(element_type == node->get_output_element_type(ArgIndices::output));
    const bool includesOnlyBiasAdd = node->inputs().size() == 3;
    const bool includesSecondAddition = node->inputs().size() == 4;
    Expects(includesOnlyBiasAdd || includesSecondAddition);  // Conv input, filters, Bias and optional Add

    std::stringstream exception_msg;
    Convolution::Details::FusedConvolutionParams params{downcast<const nodes::FusedConvolution>(node)};
    try {
        return std::make_unique<FusedConvolutionCuDnn>(
            context, *node, std::move(inputIds), std::move(outputIds), params);
    } catch (const std::exception& e) {
        throwIEException(
            fmt::format("unsupported `{}` node: Failed to create "
                        "FusedConvolutionCuDnn impl: {}",
                        node->get_type_info().name,
                        e.what()));
    }
    throwIEException(fmt::format("Convolution node is not supported:\n{}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(fusedConvolutionFactory, FusedConvolution);

}  // namespace CUDAPlugin
