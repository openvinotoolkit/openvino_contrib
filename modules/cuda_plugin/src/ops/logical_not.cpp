// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "logical_not.hpp"

#include <cuda_operation_registry.hpp>
#include <ngraph/node.hpp>

namespace CUDAPlugin {

LogicalNotOp::LogicalNotOp(const CreationContext& context,
                           const std::shared_ptr<ov::Node>& node,
                           IndexCollection&& inputIds,
                           IndexCollection&& outputIds)
    : OperationBase{context, node, move(inputIds), move(outputIds)},
      kernel_{eltwise::KernelExecAttrs{
                  ov::Shape{ov::shape_size(node->get_output_shape(0))},
                  kernel::LogicalNot::kWarpsPerBlock * static_cast<unsigned>(context.device().props().warpSize),
                  kernel::LogicalNot::kElementsPerThread},
              1,  // since workload interpreted as 1D array in sake of performance
              ov::shape_size(node->get_output_shape(0))} {}

void LogicalNotOp::Execute(const InferenceRequestContext& context,
                           Inputs inputs,
                           Outputs outputs,
                           const Workbuffers& workbuffers) const {
    kernel_(
        context.getThreadContext().stream().get(), inputs[0].cast<const bool*>().get(), outputs[0].cast<bool*>().get());
    throwIfError(cudaPeekAtLastError());
}

OPERATION_REGISTER(LogicalNotOp, LogicalNot);

}  // namespace CUDAPlugin
