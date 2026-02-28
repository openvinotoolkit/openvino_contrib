// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_cudnn_be.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <openvino/core/except.hpp>
#include <ops/converters.hpp>

#include "cuda/constant_factory.hpp"
#include "cuda/dnn_be_algo.hpp"

namespace ov {
namespace nvidia_gpu {

constexpr int NON_SPATIAL_DIMS_NUMBER = 2;

struct DnnTensorID {
    static constexpr int64_t input = 'x';
    static constexpr int64_t filter = 'w';
    static constexpr int64_t output = 'y';
};

ConvolutionCuDnnBE::ConvolutionCuDnnBE(const CreationContext& context,
                                       const ov::Node& node,
                                       IndexCollection&& inputIds,
                                       IndexCollection&& outputIds,
                                       const Convolution::Details::ConvolutionParams& params)
    : OperationCuDnn{context, node, move(inputIds), move(outputIds)}, params_{params} {
    const cudnnDataType_t tensor_element_type = convertDataType<cudnnDataType_t>(params.element_type_);

    // Convolution dimension according to op spec (1D, 2D or 3D). 1D should
    // already be turned into 2D at this point.
    const int arrayLength = static_cast<int>(params.input_shape_.size()) - NON_SPATIAL_DIMS_NUMBER;
    OPENVINO_ASSERT((arrayLength == 2) || (arrayLength == 3), "Node name: ", GetName());
    OPENVINO_ASSERT(arrayLength == params.strides_.size(), "Node name: ", GetName());
    OPENVINO_ASSERT(arrayLength == params.dilations_.size(), "Node name: ", GetName());
    OPENVINO_ASSERT(arrayLength == params.padding_before_.size(), "Node name: ", GetName());
    OPENVINO_ASSERT(arrayLength == params.padding_after_.size(), "Node name: ", GetName());

    auto input_desc = MakeTensorDescriptor(DnnTensorID::input, tensor_element_type, params.input_shape_);
    auto filter_desc = MakeTensorDescriptor(DnnTensorID::filter, tensor_element_type, params.filter_shape_);
    auto output_desc = MakeTensorDescriptor(DnnTensorID::output, tensor_element_type, params.output_shape_);

    auto conv_desc = CUDA::DnnBEConvolutionDescriptorBuilder()
                         .setMode(CUDNN_CROSS_CORRELATION)
                         .setComputeType(tensor_element_type)
                         .setNumberOfSpatialDimensions(arrayLength)
                         .setPrePaddings(params.padding_before_)
                         .setPostPaddings(params.padding_after_)
                         .setDilations(params.dilations_)
                         .setFilterStrides(params.strides_)
                         .build();

    auto conv_op_desc_builder = CUDA::DnnBEOperationConvolutionForwardDescriptorBuilder()
                                    .setXDesc(input_desc)
                                    .setWDesc(filter_desc)
                                    .setYDesc(output_desc)
                                    .setConvDesc(conv_desc);
    if (tensor_element_type == CUDNN_DATA_DOUBLE) {
        conv_op_desc_builder.setScalingParams<CUDNN_TYPE_DOUBLE>(1, 0);
    } else {
        conv_op_desc_builder.setScalingParams<CUDNN_TYPE_FLOAT>(1, 0);
    }
    auto conv_op_desc = conv_op_desc_builder.build();

    auto dnnHandle = std::make_shared<CUDA::DnnHandle>();

    std::array<cudnnBackendDescriptor_t, 1> ops{conv_op_desc->get()};
    CUDA::DnnBEOperationGraphDescriptorBuilder graphBuilder;
    graphBuilder.setDnnHandle(dnnHandle);
    graphBuilder.setOperations(ops);
    auto graph = graphBuilder.build();

    auto plans = CUDA::getAllExecutionPlansFromHeuristics(graph, *dnnHandle);
    if (plans.empty()) {
        throw_ov_exception("cuDNN BE API: Unsupported convolution");
    }

    std::shared_ptr<CUDA::DnnBEExecutionPlan> plan;
    if (context.opBenchOption()) {
        plan = performBenchmarks(context.dnnHandle(), plans);
    } else {
        plan = std::move(plans[0]);
    }

    engine_config_ = plan->getConfigDesc();
    workspace_size_ = plan->getWorkspaceSize();
}

std::shared_ptr<CUDA::DnnBEExecutionPlan> ConvolutionCuDnnBE::performBenchmarks(
    const CUDA::DnnHandle& dnnHandle, std::vector<std::shared_ptr<CUDA::DnnBEExecutionPlan>>& plans) {
    auto input = CUDA::DefaultStream::stream().malloc(ov::element::Type{params_.element_type_}.size() *
                                                      ov::shape_size(params_.input_shape_));
    auto filter = CUDA::DefaultStream::stream().malloc(ov::element::Type{params_.element_type_}.size() *
                                                       ov::shape_size(params_.filter_shape_));
    auto output = CUDA::DefaultStream::stream().malloc(ov::element::Type{params_.element_type_}.size() *
                                                       ov::shape_size(params_.output_shape_));
    auto variantPackBuilder = CUDA::DnnBEVariantPackBuilder();
    std::array<int64_t, 3> uids = {DnnTensorID::input, DnnTensorID::filter, DnnTensorID::output};
    std::array<const void*, 3> data_ptrs = {input.get(), filter.get(), output.get()};
    variantPackBuilder.setTensorPointers(uids, data_ptrs);

    constexpr const size_t kNumBenchmarks = 100;
    return CUDA::performBenchmarks<kNumBenchmarks>(dnnHandle, plans, variantPackBuilder);
}

WorkbufferRequest ConvolutionCuDnnBE::GetWorkBufferRequest() const {
    OPENVINO_ASSERT(engine_config_, "Node name: ", GetName());
    if (workspace_size_ < 0) {
        ov::nvidia_gpu::throw_ov_exception(fmt::format("Workspace Size Invalid = {}", workspace_size_));
    }
    const size_t size = std::max(static_cast<int64_t>(0), workspace_size_);
    if (size > 0) {
        return {{}, {size}};
    } else {
        return {};
    }
}

void ConvolutionCuDnnBE::Execute(const InferenceRequestContext& context,
                                 Inputs inputs,
                                 Outputs outputs,
                                 const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(inputs.size() == 2, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    auto dnnHandle = context.getThreadContext().dnnHandle();
    auto workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    std::array<const void*, 3> dataPtrs = {inputs[Convolution::Details::FusedConvolutionIndices::input].get(),
                                           inputs[Convolution::Details::FusedConvolutionIndices::filter].get(),
                                           outputs[Convolution::Details::FusedConvolutionIndices::output].get()};
    const std::array<int64_t, 3> uids = {DnnTensorID::input, DnnTensorID::filter, DnnTensorID::output};
    auto variantPackBuilder = CUDA::DnnBEVariantPackBuilder();
    variantPackBuilder.setTensorPointers(uids, dataPtrs);
    variantPackBuilder.setWorkspase(workbuffer);
    const auto variantPack = variantPackBuilder.build();

    const auto plan = CUDA::DnnBEExecutionPlanBuilder()
                          .setDnnHandle(context.getThreadContext().dnnHandle())
                          .setEngineConfig(engine_config_)
                          .build();

    throwIfError(::cudnnBackendExecute(context.getThreadContext().dnnHandle().get(), plan->get(), variantPack->get()));
}

CudaGraphCompatibility ConvolutionCuDnnBE::GetCudaGraphCompatibilityImpl() const { return CudaGraphCompatibility::NONE; }

std::shared_ptr<CUDA::DnnBETensorDescriptor> ConvolutionCuDnnBE::MakeTensorDescriptor(int64_t id,
                                                                                      cudnnDataType_t element_type,
                                                                                      const ov::Shape& shape) {
    const int nbDims = shape.size();
    if (nbDims < 4 || nbDims > 5)
        throw_ov_exception(fmt::format("Unexpected number of dimensions for Convolution input/output: {}", nbDims));

    return CUDA::DnnBETensorDescriptorBuilder()
        .setUniqueId(id)
        .setDataType(element_type)
        .setShape(shape)
        .setStrides(ov::row_major_strides(shape))
        .setAlignment(CUDA::memoryAlignment)
        .setIsVirtual(false)
        .build();
}
}  // namespace nvidia_gpu
}  // namespace ov
