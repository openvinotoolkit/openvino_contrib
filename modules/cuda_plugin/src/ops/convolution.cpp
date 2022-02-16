// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <gsl/gsl_assert>
#include <sstream>

#include "convolution_cudnn.hpp"
#include "cuda_operation_registry.hpp"

#undef ENABLE_CUDNN_BACKEND_API_BASED_CONNVOLUTION
#ifdef ENABLE_CUDNN_BACKEND_API_BASED_CONNVOLUTION
#include "convolution_cudnn_be.hpp"
#endif  // ENABLE_CUDNN_BACKEND_API_BASED_CONNVOLUTION

namespace CUDAPlugin {

static OperationBase::Ptr convolutionFactory(const CreationContext& context,
                                             const std::shared_ptr<ngraph::Node>& node,
                                             OperationBase::IndexCollection&& inputIds,
                                             OperationBase::IndexCollection&& outputIds) {
    const Convolution::Details::ConvolutionParams params{downcast<const ngraph::op::v1::Convolution>(node)};
    std::stringstream exception_msg;
    try {
        return std::make_shared<ConvolutionCuDnn>(context, *node, move(inputIds), move(outputIds), params);
    } catch (const std::exception& e) {
        exception_msg << "Failed to create ConvolutionCuDnn impl: " << e.what();
    }
#ifdef ENABLE_CUDNN_BACKEND_API_BASED_CONNVOLUTION
    try {
        return std::make_shared<ConvolutionCuDnnBE>(context, *node, move(inputIds), move(outputIds), params);
    } catch (const std::exception& e) {
        exception_msg << "\nFailed to create ConvolutionCuDnnBE impl: " << e.what();
    }
#endif  // ENABLE_CUDNN_BACKEND_API_BASED_CONNVOLUTION
    throwIEException(fmt::format("Convolution node is not supported:\n{}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(convolutionFactory, Convolution);

}  // namespace CUDAPlugin
