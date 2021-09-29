// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "maxpool.hpp"

#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <ngraph/op/max_pool.hpp>

namespace CUDAPlugin {

MaxPoolOp::MaxPoolOp(const CreationContext& context,
                     const std::shared_ptr<ngraph::Node>& node,
                     IndexCollection&& inputIds,
                     IndexCollection&& outputIds)
    : OperationCuDnn(context, node, std::move(inputIds), std::move(outputIds)),
      impl_{dynamic_cast<const ngraph::op::v1::MaxPool&>(*node)} {}

void MaxPoolOp::Execute(const InferenceRequestContext& context,
                        Inputs inputs,
                        Outputs outputs,
                        const Workbuffers&) const {
    Expects(inputs.size() == 1);
    Expects(outputs.size() == 1);

    impl_.Execute(context.getThreadContext().dnnHandle(),
                  inputs[PoolingImpl::input_index].get(),
                  outputs[PoolingImpl::output_index].get());
}

OPERATION_REGISTER(MaxPoolOp, MaxPool);

}  // namespace CUDAPlugin
