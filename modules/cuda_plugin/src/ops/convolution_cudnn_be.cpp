// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_cudnn_be.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <details/ie_exception.hpp>
#include <gsl/gsl_assert>

#include "converters.hpp"
#include "cuda/constant_factory.hpp"

namespace CUDAPlugin {

constexpr int NON_SPATIAL_DIMS_NUMBER = 2;

struct DnnTensorID {
    static constexpr int64_t input = 'x';
    static constexpr int64_t filter = 'w';
    static constexpr int64_t output = 'y';
};

ConvolutionCuDnnBE::ConvolutionCuDnnBE(const CreationContext& context,
                                       const ngraph::Node& node,
                                       IndexCollection&& inputIds,
                                       IndexCollection&& outputIds,
                                       const Convolution::Details::ConvolutionParams& params)
    : OperationCuDnn{context, node, move(inputIds), move(outputIds)} {
    const cudnnDataType_t tensor_element_type = convertDataType<cudnnDataType_t>(params.element_type_);

    // Convolution dimension according to op spec (1D, 2D or 3D). 1D should
    // already be turned into 2D at this point.
    const int arrayLength = static_cast<int>(params.input_shape_.size()) - NON_SPATIAL_DIMS_NUMBER;
    Expects((arrayLength == 2) || (arrayLength == 3));
    Expects(arrayLength == params.strides_.size());
    Expects(arrayLength == params.dilations_.size());
    Expects(arrayLength == params.padding_before_.size());
    Expects(arrayLength == params.padding_after_.size());

    CUDA::DnnBETensorDescriptor input_desc =
        MakeTensorDescriptor(DnnTensorID::input, tensor_element_type, params.input_shape_);
    CUDA::DnnBETensorDescriptor filter_desc =
        MakeTensorDescriptor(DnnTensorID::filter, tensor_element_type, params.filter_shape_);
    CUDA::DnnBETensorDescriptor output_desc =
        MakeTensorDescriptor(DnnTensorID::output, tensor_element_type, params.output_shape_);

    CUDA::DnnBEConvolutionDescriptor conv_desc;
    conv_desc.setMode(CUDNN_CROSS_CORRELATION);
    conv_desc.setComputeType(tensor_element_type);
    conv_desc.setNumberOfSpatialDimensions(arrayLength);
    conv_desc.setPrePaddings(params.padding_before_);
    conv_desc.setPostPaddings(params.padding_after_);
    conv_desc.setDilations(params.dilations_);
    conv_desc.setFilterStrides(params.strides_);
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

    CUDA::DnnHandle dnnHandle{};

    CUDA::DnnBEOperationGraphDescriptor graph;
    graph.setDnnHandle(dnnHandle);
    std::array<cudnnBackendDescriptor_t, 1> ops{conv_op_desc.get()};
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
        } catch (const InferenceEngine::Exception&) {
            continue;
        }
        exec_plans_.emplace_back(std::move(plan));
    }

    if (exec_plans_.empty())
        throwIEException("cuDNN BE API: Unsupported convolution");
}

WorkbufferRequest ConvolutionCuDnnBE::GetWorkBufferRequest() const {
    int64_t size{};
    // Finding the max memory size needed
    for (const auto& plan : exec_plans_) {
        size = std::max(size, plan.getWorkspaceSize());
    }
    if (size > 0)
        return {{}, {static_cast<size_t>(size)}};
    else
        return {};
}

void ConvolutionCuDnnBE::Execute(const InferenceRequestContext& context,
                                 Inputs inputs,
                                 Outputs outputs,
                                 const Workbuffers& workbuffers) const {
    Expects(inputs.size() == 2);
    Expects(outputs.size() == 1);
    auto workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    for (size_t planIndex = exec_plan_index_hint_; planIndex < exec_plans_.size(); ++planIndex) {
        if (TryExecutePlan(context, inputs, outputs, workbuffer, exec_plans_[planIndex])) {
            exec_plan_index_hint_ = planIndex;
            return;
        }
    }

    throwIEException("cuDNN BE API: Unsupported convolution");
}

bool ConvolutionCuDnnBE::TryExecutePlan(const InferenceRequestContext& context,
                                        Inputs inputs,
                                        Outputs outputs,
                                        void* workbuffer,
                                        const CUDA::DnnBEExecutionPlanDescriptor& plan) const {
    CUDA::DnnBEVariantPackDescriptor variantPack;
    std::array<int64_t, 3> uids = {DnnTensorID::input, DnnTensorID::filter, DnnTensorID::output};
    std::array<void*, 3> data_ptrs = {const_cast<void*>(inputs[Convolution::Details::ConvArgIndices::input].get()),
                                      const_cast<void*>(inputs[Convolution::Details::ConvArgIndices::filter].get()),
                                      outputs[Convolution::Details::ConvArgIndices::output].get()};
    variantPack.setTensorPointers(uids, data_ptrs);

    variantPack.setWorkspase(workbuffer);
    variantPack.finalize();

    cudnnStatus_t status =
        ::cudnnBackendExecute(context.getThreadContext().dnnHandle().get(), plan.get(), variantPack.get());
    if (status != CUDNN_STATUS_NOT_SUPPORTED) {
        logIfError(status);
    }
    return (status == CUDNN_STATUS_SUCCESS);
}

CUDA::DnnBETensorDescriptor ConvolutionCuDnnBE::MakeTensorDescriptor(int64_t id, cudnnDataType_t element_type,
                                                                     const ngraph::Shape& shape) {
    const int nbDims = shape.size();
    if (nbDims < 4 || nbDims > 5)
        throwIEException(fmt::format("Unexpected number of dimensions for Convolution input/output: {}", nbDims));

    CUDA::DnnBETensorDescriptor desc;
    desc.setUniqueId(id);
    desc.setDataType(element_type);
    desc.setShape(shape);
    desc.setStrides(ngraph::row_major_strides(shape));
    desc.setAlignment(CUDA::memoryAlignment);
    desc.setIsVirtual(false);
    desc.finalize();
    return desc;
}

}  // namespace CUDAPlugin
