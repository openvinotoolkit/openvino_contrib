// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gsl/gsl_assert>

#include "convolution2d_biasadd_activation.hpp"

#include <sstream>
#include <fmt/format.h>

#include <ngraph/validation_util.hpp>

#include "cuda_operation_registry.hpp"
#include "convolution_cudnn.hpp"
#include "convolution_cudnn_be.hpp"

namespace CUDAPlugin {

constexpr int NON_SPATIAL_DIMS_NUMBER = 2;
constexpr int CONV_1D_DIMS_NUMBER = NON_SPATIAL_DIMS_NUMBER + 1;

Convolution2DBiasAddActivationOp::Convolution2DBiasAddActivationOp(const NodeOp& node,
                             IndexCollection&& inputIds,
                             IndexCollection&& outputIds)
    : OperationCuDnn(node, std::move(inputIds), std::move(outputIds)) {
    const auto element_type = node.get_input_element_type(ArgIndices::input);
    Expects(element_type == node.get_input_element_type(ArgIndices::filter));
    Expects(element_type == node.get_input_element_type(ArgIndices::bias));
    Expects(element_type == node.get_output_element_type(ArgIndices::output));
    Expects(node.inputs().size() == 3); // Conv input, filters, Bias
}

void Convolution2DBiasAddActivationOp::Execute(
    const InferenceRequestContext& context, Inputs inputs, Outputs outputs,
    const Workbuffers& workbuffers) {
  std::cout << "Error: Convolution2DBiasAddActivationOp not implemented!\n";
  if (impl_) impl_->Execute(context, inputs, outputs, workbuffers);
}

OPERATION_REGISTER(Convolution2DBiasAddActivationOp, ConvolutionBiasAddActivation);
} // namespace CUDAPlugin
