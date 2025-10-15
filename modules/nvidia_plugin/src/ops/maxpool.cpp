// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "maxpool.hpp"

#include <cuda_operation_registry.hpp>
#include <openvino/core/except.hpp>
#include <openvino/op/max_pool.hpp>

namespace ov {
namespace nvidia_gpu {

MaxPoolOp::MaxPoolOp(const CreationContext& context,
                     const std::shared_ptr<ov::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : OperationCuDnn(context, node, std::move(inputIds), std::move(outputIds)),
      impl_{dynamic_cast<const ov::op::v1::MaxPool&>(*node)} {}

void MaxPoolOp::Execute(const InferenceRequestContext& context,
                        Inputs inputs,
                        Outputs outputs,
                        const Workbuffers&) const {
    OPENVINO_ASSERT(inputs.size() == 1, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    impl_.Execute(context.getThreadContext().dnnHandle(),
                  inputs[PoolingImpl::input_index].get(),
                  outputs[PoolingImpl::output_index].get());
}

CudaGraphCompatibility MaxPoolOp::GetCudaGraphCompatibility() const { return CudaGraphCompatibility::FULL; }

OPERATION_REGISTER(MaxPoolOp, MaxPool);

}  // namespace nvidia_gpu
}  // namespace ov
