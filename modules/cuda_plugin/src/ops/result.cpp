// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gsl/gsl_assert>
#include "result.hpp"
#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

void ResultOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) {
  Expects(inputs.size() == 1);
  Expects(outputs.size() == 0);
  Expects(context.HasOutputBlob(GetName()));
  auto blob = context.GetOutputBlob(GetName());
  auto memory_ptr = blob->as<InferenceEngine::MemoryBlob>()->wmap();
  cudaMemcpyAsync(memory_ptr, inputs[0], blob->byteSize(),
                  cudaMemcpyDeviceToHost, context.GetCUDAStream());
}

OPERATION_REGISTER(ResultOp, "Result");
} // namespace CUDAPlugin
