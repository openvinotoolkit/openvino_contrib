// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>

#include <cuda_operation_base.hpp>
#include <gpu/device_pointers.hpp>
#include <gpu/gpu_context_api_cuda.hpp>
#include <kernels/concat.hpp>
#include <ngraph/op/concat.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

namespace CUDAPlugin {

class ConcatOp : public OperationBase {
 public:
  using NodeOp = ngraph::op::Concat;
  ConcatOp(const CUDA::CreationContext& context,
            const NodeOp& node,
            IndexCollection&& inputIds,
            IndexCollection&& outputIds);
  void Execute(const InferenceRequestContext& context,
               Inputs inputTensors,
               Outputs outputTensors,
               const Workbuffers& workbuffers) const override;
  WorkbufferRequest GetWorkBufferRequest() const override;
  void InitSharedImmutableWorkbuffers(const Buffers&) override;

 private:
     size_t immutableWbSize() const { return concat_kernel_.value().immutableWbSize(); }
     size_t mutableWbSize() const { return concat_kernel_.value().mutableWbSize(); }

     std::size_t num_inputs_;
     std::optional<kernel::Concat> concat_kernel_;
};

} // namespace CUDAPlugin
