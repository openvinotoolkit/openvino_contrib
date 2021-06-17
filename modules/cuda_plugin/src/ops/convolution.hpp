// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/convolution.hpp>
#include <memory>
#include <cuda_operation_base.hpp>

namespace CUDAPlugin {

class ConvolutionOp : public OperationCuDnn {
public:
    using NodeOp = ngraph::op::v1::Convolution;
    ConvolutionOp(const NodeOp& node,
                  IndexCollection&& inputIds,
                  IndexCollection&& outputIds);
    void Execute(const InferenceRequestContext& context,
                 Inputs inputTensors,
                 Outputs outputTensors,
                 const Workbuffers&) override;
    WorkbufferRequest GetWorkBufferRequest() const override;

    void InitSharedImmutableWorkbuffers(const IOperationExec::Buffers&) override {}
    const WorkbufferIndices& GetWorkbufferIds() const override;
    WorkbufferStatus SetWorkbufferIds(WorkbufferIndices&& workbufferIds) override;

    struct ArgIndices {
        static constexpr size_t input = 0;
        static constexpr size_t filter = 1;
        static constexpr size_t output = 0;
    };

private:
    using PaddingBeforeAndAfter = std::pair<ngraph::CoordinateDiff,
                                            ngraph::CoordinateDiff>;
    static PaddingBeforeAndAfter InferPadding(const ngraph::op::v1::Convolution& op);

    void Create1DImpl(ngraph::element::Type_t element_type,
                      ngraph::Shape input_shape,
                      ngraph::Shape filter_shape,
                      ngraph::Shape output_shape,
                      ngraph::Strides strides,
                      ngraph::Strides dilations,
                      PaddingBeforeAndAfter padding);
    void Create2D3DImpl(ngraph::element::Type_t element_type,
                        const ngraph::Shape& input_shape,
                        const ngraph::Shape& filter_shape,
                        const ngraph::Shape& output_shape,
                        const ngraph::Strides& strides,
                        const ngraph::Strides& dilations,
                        const PaddingBeforeAndAfter& padding);

private:
    std::unique_ptr<IOperationExec> impl_;
};

} // namespace CUDAPlugin
