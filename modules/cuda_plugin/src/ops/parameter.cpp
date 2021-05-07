// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cuda_runtime.h>
#include <gsl/gsl_assert>
#include "parameter.hpp"
#include <cuda_operation_registry.hpp>

namespace CUDAPlugin {

void ParameterOp::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs) {
  Expects(inputs.size() == 0);
  Expects(outputs.size() == 1);
  Expects(context.HasInputBlob(GetName()));
  auto blob = context.GetInputBlob(GetName());
  auto memory_ptr = blob->as<InferenceEngine::MemoryBlob>()->rmap();
  cudaMemcpyAsync(outputs[0], memory_ptr, blob->byteSize(),
                  cudaMemcpyHostToDevice, context.GetCUDAStream());
}

OPERATION_REGISTER(ParameterOp, "Parameter");
} // namespace CUDAPlugin
