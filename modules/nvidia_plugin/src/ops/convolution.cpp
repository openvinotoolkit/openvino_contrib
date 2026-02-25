// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fmt/format.h>

#include <openvino/core/except.hpp>
#include <sstream>

#include "convolution_cudnn.hpp"
#include "cuda_operation_registry.hpp"

#ifdef ENABLE_CUDNN_BACKEND_API
#include "convolution_cudnn_be.hpp"
#endif  // ENABLE_CUDNN_BACKEND_API

namespace ov {
namespace nvidia_gpu {

static OperationBase::Ptr convolutionFactory(const CreationContext& context,
                                             const std::shared_ptr<ov::Node>& node,
                                             OperationBase::IndexCollection&& inputIds,
                                             OperationBase::IndexCollection&& outputIds) {
    using IndexCollection = OperationBase::IndexCollection;
    const Convolution::Details::ConvolutionParams params{downcast<const ov::op::v1::Convolution>(node)};
    std::stringstream exception_msg;
#ifdef ENABLE_CUDNN_BACKEND_API
    try {
        return std::make_shared<ConvolutionCuDnnBE>(
            context, *node, IndexCollection{inputIds}, IndexCollection{outputIds}, params);
    } catch (const std::exception& e) {
        exception_msg << "\nFailed to create ConvolutionCuDnnBE impl: " << e.what();
    }
#endif  // ENABLE_CUDNN_BACKEND_API
    try {
        return std::make_shared<ConvolutionCuDnn>(
            context, *node, IndexCollection{inputIds}, IndexCollection{outputIds}, params);
    } catch (const std::exception& e) {
        exception_msg << "Failed to create ConvolutionCuDnn impl: " << e.what();
    }
    throw_ov_exception(fmt::format("Convolution node is not supported:\n{}", exception_msg.str()));
}

OPERATION_REGISTER_FACTORY(convolutionFactory, Convolution);

}  // namespace nvidia_gpu
}  // namespace ov
