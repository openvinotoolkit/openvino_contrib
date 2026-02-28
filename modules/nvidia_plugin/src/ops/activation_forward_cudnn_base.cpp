// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "activation_forward_cudnn_base.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cuda/constant_factory.hpp>
#include <cuda/descriptor_utils.hpp>

#include "converters.hpp"

namespace ov {
namespace nvidia_gpu {

ActivationForwardCuDnnOpBase::ActivationForwardCuDnnOpBase(std::unique_ptr<CUDA::DnnActivationDescriptor> opDesc,
                                                           const CreationContext& context,
                                                           const ov::Node& node,
                                                           IndexCollection&& inputIds,
                                                           IndexCollection&& outputIds)
    : OperationCuDnn{context, node, move(inputIds), move(outputIds)},
      op_desc_{move(opDesc)},
      x_desc_{CUDA::makeInputDnnTensorDescr(node, 0)},
      y_desc_{CUDA::makeOutputDnnTensorDescr(node, 0)},
      data_type_{convertDataType<cudnnDataType_t>(node.get_input_element_type(0))} {
    OPENVINO_ASSERT(node.get_input_size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(node.get_output_size() == 1, "Node name: ", GetName());

    if (std::find(supported_types.begin(), supported_types.end(), data_type_) == supported_types.end()) {
        throw_ov_exception(
            fmt::format("ov::nvidia_gpu::ActivationForwardCuDnnOpBase: unsupported data type: {}", toString(data_type_)));
    }

    const auto& shape = node.get_input_shape(0);
    OPENVINO_ASSERT(node.get_output_shape(0) == shape, "Node name: ", GetName());

    const auto in_shape_size = node.get_input_shape(0).size();
    if (in_shape_size > max_shape_size) {
        throw_ov_exception(
            fmt::format("ov::nvidia_gpu::ActivationForwardCuDnnOpBase: in_shape_size > max_shape_size: in_shape_size = {}, "
                        "max_shape_size = {}",
                        in_shape_size,
                        max_shape_size));
    }
}

void ActivationForwardCuDnnOpBase::Execute(const InferenceRequestContext& context,
                                           Inputs inputTensors,
                                           Outputs outputTensors,
                                           const Workbuffers&) const {
    context.getThreadContext().dnnHandle().activationForward(*op_desc_,
                                                             &CUDA::NumericConst<CUDA::constants::one>(data_type_),
                                                             x_desc_,
                                                             inputTensors[0].get(),
                                                             &CUDA::NumericConst<CUDA::constants::zero>(data_type_),
                                                             y_desc_,
                                                             outputTensors[0].get());
}

CudaGraphCompatibility ActivationForwardCuDnnOpBase::GetCudaGraphCompatibilityImpl() const {
    return CudaGraphCompatibility::FULL;
}

}  // namespace nvidia_gpu
}  // namespace ov
