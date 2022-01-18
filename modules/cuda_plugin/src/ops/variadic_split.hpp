// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_ops_infer.h>

#include <cuda/device_pointers.hpp>
#include <cuda_operation_base.hpp>
#include <gpu/gpu_context_api_cuda.hpp>
#include <kernels/split.hpp>
#include <ngraph/op/softmax.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

#include "kernels/variadic_split.hpp"

namespace CUDAPlugin {

class VariadicSplitOp : public OperationBase {
public:
    VariadicSplitOp(const CreationContext& context,
                    const ngraph::Node& node,
                    IndexCollection&& inputIds,
                    IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers& workbuffers) const override;

private:
    enum { kOutputPtrsMWBIdx = 0, kNumberOfMWBIdx };
    enum { kSplitIdxIWBIdx = 0, kAxisSizesIWBIdx, kAxisOffsetSizesIWBIdx, kNumberOfIWIdx };

    void buildAxisHelpers(const std::vector<int64_t>& split_lengths, size_t orig_axis_size);
    void buildSplitIndexHelper(const std::vector<int64_t>& split_lengths, size_t orig_axis_size);

    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers& buffers) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

    template <typename T>
    void Execute(const InferenceRequestContext& context,
                 Inputs inputs,
                 Outputs outputs,
                 const Workbuffers& buffers) const;

    std::vector<size_t> split_idx_;
    std::vector<size_t> axis_sizes_;
    std::vector<size_t> axis_offset_sizes_;
    std::optional<kernel::VariadicSplit> variadic_split_kernel_;
};

}  // namespace CUDAPlugin
