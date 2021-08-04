// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>
#include <ngraph/type/element_type.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/op/concat.hpp>
#include <cuda_operation_base.hpp>
#include <gpu/device_pointers.hpp>
#include <gpu/gpu_context_api_cuda.hpp>

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
                 const Workbuffers& workbuffers) override;
    WorkbufferRequest GetWorkBufferRequest() const override;
    void InitSharedImmutableWorkbuffers(const Buffers&) override;

    using VoidDevPtr = InferenceEngine::gpu::DevicePointer<void*>;
    using ConstVoidDevPtr = InferenceEngine::gpu::DevicePointer<const void*>;
    struct Chunk {
      size_t input;
      size_t offset;
    };

 private:
    size_t immutableWbSize() const { return sizeof(Chunk) * chunks_.size(); }
    size_t mutableWbSize() const { return sizeof(float *) * num_inputs_; }
    template <typename T>
    void Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs, const Workbuffers& buffers);

    ngraph::element::Type_t element_type_;
    size_t num_inputs_ {};
    std::vector<Chunk> chunks_;
    size_t chunk_size_;
    size_t all_chunk_size_ = 0;
    size_t num_blocks_;
    size_t threads_per_block_;
};

} // namespace CUDAPlugin
