// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gsl/gsl_assert>
#include <details/ie_exception.hpp>

#include <algorithm>

#include "cuda/device.hpp"
#include "convolution_cudnn_be.hpp"
#include "convolution.hpp"
#include "converters.hpp"
#include "constant_factory.hpp"
#include <cudnn.h>

namespace CUDAPlugin {

constexpr int NON_SPATIAL_DIMS_NUMBER = 2;

struct DnnTensorID {
    static constexpr int64_t input = 'x';
    static constexpr int64_t filter = 'w';
    static constexpr int64_t output = 'y';
};

ConvolutionCuDnnBE::ConvolutionCuDnnBE(ngraph::element::Type_t element_type,
                                       const ngraph::Shape& input_shape,
                                       const ngraph::Shape& filter_shape,
                                       const ngraph::Shape& output_shape,
                                       const ngraph::Strides& strides,
                                       const ngraph::Strides& dilations,
                                       const ngraph::CoordinateDiff& padding_before,
                                       const ngraph::CoordinateDiff& padding_after)
    : exec_plan_index_hint_ {0} {
    const cudnnDataType_t tensor_element_type = convertDataType<cudnnDataType_t>(element_type);

    // Convolution dimension according to op spec (1D, 2D or 3D). 1D should already be
    // turned into 2D at this point.
    const int arrayLength = static_cast<int>(input_shape.size()) - NON_SPATIAL_DIMS_NUMBER;
    Expects((arrayLength == 2) || (arrayLength == 3));
    Expects(arrayLength == strides.size());
    Expects(arrayLength == dilations.size());
    Expects(arrayLength == padding_before.size());
    Expects(arrayLength == padding_after.size());

    CUDA::DnnBETensorDescriptor input_desc =
        MakeTensorDescriptor(DnnTensorID::input, tensor_element_type, input_shape);
    CUDA::DnnBETensorDescriptor filter_desc =
        MakeTensorDescriptor(DnnTensorID::filter, tensor_element_type, filter_shape);
    CUDA::DnnBETensorDescriptor output_desc =
        MakeTensorDescriptor(DnnTensorID::output, tensor_element_type, output_shape);

    CUDA::DnnBEConvolutionDescriptor conv_desc;
    conv_desc.setMode(CUDNN_CROSS_CORRELATION);
    conv_desc.setComputeType(tensor_element_type);
    conv_desc.setNumberOfSpatialDimensions(arrayLength);
    conv_desc.setPrePaddings(padding_before);
    conv_desc.setPostPaddings(padding_after);
    conv_desc.setDilations(dilations);
    conv_desc.setFilterStrides(strides);
    conv_desc.finalize();

    CUDA::DnnBEOperationConvolutionForwardDescriptor conv_op_desc;
    conv_op_desc.setX(input_desc);
    conv_op_desc.setW(filter_desc);
    conv_op_desc.setY(output_desc);
    conv_op_desc.setConv(conv_desc);
    if (tensor_element_type == CUDNN_DATA_DOUBLE) {
        conv_op_desc.setScalingParams<CUDNN_TYPE_DOUBLE>(1, 0);
    } else {
        conv_op_desc.setScalingParams<CUDNN_TYPE_FLOAT>(1, 0);
    }
    conv_op_desc.finalize();

    CUDA::DnnHandle dnnHandle {};

    CUDA::DnnBEOperationGraphDescriptor graph;
    graph.setDnnHandle(dnnHandle);
    std::array<cudnnBackendDescriptor_t, 1> ops { conv_op_desc.get() };
    graph.setOperations(ops);
    graph.finalize();

    CUDA::DnnBEEngineheurDescriptor heuristics;
    heuristics.setOpGraph(graph);
    heuristics.setMode(CUDNN_HEUR_MODE_INSTANT);
    heuristics.finalize();

    std::vector<CUDA::DnnBEEngineConfigDescriptor> configs = heuristics.getEngineConfigs();
    for (const auto& config : configs) {
        CUDA::DnnBEExecutionPlanDescriptor plan;
        plan.setDnnHandle(dnnHandle);
        plan.setEngineConfig(config);
        try {
            plan.finalize();
        } catch(const InferenceEngine::details::InferenceEngineException&) {
            continue;
        }
        exec_plans_.emplace_back(std::move(plan));
    }

    if (exec_plans_.empty())
        THROW_IE_EXCEPTION << "cuDNN BE API: Unsupported convolution";
}

WorkbufferRequest ConvolutionCuDnnBE::GetWorkBufferRequest() const {
    int64_t size {};
    // Finding the max memory size needed
    for(const auto& plan : exec_plans_) {
        size = std::max(size, plan.getWorkspaceSize());
    }
    if (size > 0)
        return {{}, {static_cast<size_t>(size)}};
    else
        return {};
}

void ConvolutionCuDnnBE::Execute(const InferenceRequestContext& context, Inputs inputs, Outputs outputs, const Workbuffers& workbuffers) {
    Expects(inputs.size() == 2);
    Expects(outputs.size() == 1);
    auto workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    for (size_t planIndex = exec_plan_index_hint_; planIndex < exec_plans_.size(); ++planIndex) {
        if (TryExecutePlan(context, inputs, outputs, workbuffer, exec_plans_[planIndex])) {
            exec_plan_index_hint_ = planIndex;
            return;
        }
    }

    THROW_IE_EXCEPTION << "cuDNN BE API: Unsupported convolution";
}

bool ConvolutionCuDnnBE::TryExecutePlan(const InferenceRequestContext& context,
                                        Inputs inputs, Outputs outputs,
                                        void* workbuffer,
                                        const CUDA::DnnBEExecutionPlanDescriptor& plan) {
    CUDA::DnnBEVariantPackDescriptor variantPack;
    std::array<int64_t, 3> uids = { DnnTensorID::input, DnnTensorID::filter, DnnTensorID::output };
    std::array<void*, 3> data_ptrs = {
        const_cast<void*>(inputs[ConvolutionOp::ArgIndices::input].get()),
        const_cast<void*>(inputs[ConvolutionOp::ArgIndices::filter].get()),
        outputs[ConvolutionOp::ArgIndices::output].get()
    };
    variantPack.setTensorPointers(uids, data_ptrs);

    variantPack.setWorkspase(workbuffer);
    variantPack.finalize();

    cudnnStatus_t status = ::cudnnBackendExecute(
                                context.getThreadContext().dnnHandle().get(),
                                plan.get(),
                                variantPack.get());
    if (status != CUDNN_STATUS_NOT_SUPPORTED) {
        CUDA::logIfError(status);
    }
    return (status == CUDNN_STATUS_SUCCESS);
}

CUDA::DnnBETensorDescriptor
ConvolutionCuDnnBE::MakeTensorDescriptor(int64_t id,
                                         cudnnDataType_t element_type,
                                         const ngraph::Shape& shape) {
    const int nbDims = shape.size();
    if (nbDims < 4 || nbDims > 5)
        THROW_IE_EXCEPTION << "Unexpected number of dimensions for Convolution input/output: "
                           << nbDims;

    CUDA::DnnBETensorDescriptor desc;
    desc.setUniqueId(id);
    desc.setDataType(element_type);
    desc.setShape(shape);
    desc.setStrides(ngraph::row_major_strides(shape));
    desc.setAlignment(CudaDevice::GetMemoryAllignment());
    desc.setIsVirtual(false);
    desc.finalize();
    return desc;
}

} // namespace CUDAPlugin
