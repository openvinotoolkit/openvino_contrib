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
  auto stream = context.GetCUDAStream();
  auto memory_ptr = blob->as<InferenceEngine::MemoryBlob>()->rmap();
  stream->memcpyAsync(outputs[0], static_cast<const void*>(memory_ptr), blob->byteSize());
}

OPERATION_REGISTER(ParameterOp, "Parameter");
} // namespace CUDAPlugin
