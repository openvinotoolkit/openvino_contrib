// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "avgpool.hpp"

#include <cuda_operation_registry.hpp>
#include <gsl/gsl_assert>
#include <ngraph/op/avg_pool.hpp>

namespace CUDAPlugin {

AvgPoolOp::AvgPoolOp(const std::shared_ptr<ngraph::Node>& node,
                     std::vector<unsigned>&& inputIds,
                     std::vector<unsigned>&& outputIds)
    : OperationCuDnn(node, std::move(inputIds), std::move(outputIds)),
      impl_{dynamic_cast<const ngraph::op::AvgPool&>(*node)} {}

void AvgPoolOp::Execute(const InferenceRequestContext& context, Inputs inputs,
                        Outputs outputs, const Workbuffers&) {
  Expects(inputs.size() == 1);
  Expects(outputs.size() == 1);

  impl_.Execute(context.getThreadContext().dnnHandle(),
                inputs[PoolingImpl::input_index].get(),
                outputs[PoolingImpl::output_index].get());
}

OPERATION_REGISTER(AvgPoolOp, AvgPool);

}  // namespace CUDAPlugin
