// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "tanh.hpp"

#include <cuda/descriptor_utils.hpp>
#include <cuda_operation_base.hpp>
#include <cuda_operation_registry.hpp>

#include "constant_factory.hpp"
#include "converters.hpp"

namespace CUDAPlugin {

Tanh::Tanh(const CUDA::CreationContext& context,
           const std::shared_ptr<ngraph::Node>& node,
           IndexCollection&& inputIds,
           IndexCollection&& outputIds)
    : OperationCuDnn{context, node, move(inputIds), move(outputIds)},
      x_desc_{CUDA::makeInputDnnTensorDescr(*node, 0)},
      y_desc_{CUDA::makeOutputDnnTensorDescr(*node, 0)},
      data_type_{convertDataType<cudnnDataType_t>(node->get_input_element_type(0))} {}

void Tanh::Execute(const InferenceRequestContext& context,
                   Inputs inputTensors,
                   Outputs outputTensors,
                   const Workbuffers&) {
    context.getThreadContext().dnnHandle().activationForward(tanh_desc_,
                                                             &NumericConst<constants::one>(data_type_),
                                                             x_desc_,
                                                             inputTensors[0].get(),
                                                             &NumericConst<constants::zero>(data_type_),
                                                             y_desc_,
                                                             outputTensors[0].get());
}

OPERATION_REGISTER(Tanh, Tanh);
}  // namespace CUDAPlugin
