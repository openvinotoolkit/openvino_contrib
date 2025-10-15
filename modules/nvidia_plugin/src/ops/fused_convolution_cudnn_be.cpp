// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_convolution_cudnn_be.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cuda/constant_factory.hpp>
#include <openvino/core/except.hpp>
#include <ops/converters.hpp>

#include "cuda/dnn_be_algo.hpp"
#include "cuda/event.hpp"
#include "memory_manager/model/details/cuda_memory_utils.hpp"

namespace ov {
namespace nvidia_gpu {

constexpr int NON_SPATIAL_DIMS_NUMBER = 2;

struct DnnTensorID {
    static constexpr int64_t input = 'x';
    static constexpr int64_t filter = 'w';
    static constexpr int64_t output = 'y';
    static constexpr int64_t add = 'z';
    static constexpr int64_t bias = 'b';
    static constexpr int64_t conv_output = 'C';
    static constexpr int64_t add_output = 'A';
    static constexpr int64_t bias_output = 'B';
};

FusedConvolutionCuDnnBE::FusedConvolutionCuDnnBE(const CreationContext& context,
                                                 const ov::Node& node,
                                                 IndexCollection&& inputIds,
                                                 IndexCollection&& outputIds,
                                                 const Convolution::Details::FusedConvolutionParams& params)
    : OperationCuDnn{context, node, std::move(inputIds), std::move(outputIds)}, params_{params} {
    [[maybe_unused]] const cudnnDataType_t tensor_element_type =
        convertDataType<cudnnDataType_t>(params.conv_.element_type_);

    // Convolution dimension according to op spec (1D, 2D or 3D). 1D should
    // already be turned into 2D at this point.
    const int arrayLength = static_cast<int>(params.conv_.input_shape_.size()) - NON_SPATIAL_DIMS_NUMBER;
    OPENVINO_ASSERT((arrayLength == 2) || (arrayLength == 3), "Node name: ", GetName());
    OPENVINO_ASSERT(arrayLength == params.conv_.strides_.size(), "Node name: ", GetName());
    OPENVINO_ASSERT(arrayLength == params.conv_.dilations_.size(), "Node name: ", GetName());
    OPENVINO_ASSERT(arrayLength == params.conv_.padding_before_.size(), "Node name: ", GetName());
    OPENVINO_ASSERT(arrayLength == params.conv_.padding_after_.size(), "Node name: ", GetName());

    auto convertConvBackendDataType = [](int64_t index, cudnnDataType_t dataType) {
        if (dataType == CUDNN_DATA_INT8 || dataType == CUDNN_DATA_INT32) {
            switch (index) {
                case DnnTensorID::input:
                    return CUDNN_DATA_INT8;
                case DnnTensorID::filter:
                    return CUDNN_DATA_INT8;
                case DnnTensorID::output:
                    return CUDNN_DATA_INT8;
                case DnnTensorID::add:
                    return CUDNN_DATA_FLOAT;
                case DnnTensorID::bias:
                    return CUDNN_DATA_FLOAT;
                case DnnTensorID::conv_output:
                    return CUDNN_DATA_INT32;
                case DnnTensorID::add_output:
                    return CUDNN_DATA_FLOAT;
                case DnnTensorID::bias_output:
                    return CUDNN_DATA_FLOAT;
            }
        }
        return dataType;
    };

    std::vector<std::shared_ptr<CUDA::DnnBEExecutionPlan>> plans;
    auto dnnHandle = std::make_shared<CUDA::DnnHandle>();
    auto addPlans = [&](const cudnnTensorFormat_t& format, cudnnDataType_t dataType) {
        auto conv_tensor_desc = MakeTensorDescriptor(DnnTensorID::input,
                                                     convertConvBackendDataType(DnnTensorID::input, dataType),
                                                     params.conv_.input_shape_,
                                                     format);
        auto output_tensor_desc = MakeTensorDescriptor(DnnTensorID::output,
                                                       convertConvBackendDataType(DnnTensorID::output, dataType),
                                                       params.conv_.output_shape_,
                                                       format);
        auto filter_tensor_desc = MakeTensorDescriptor(DnnTensorID::filter,
                                                       convertConvBackendDataType(DnnTensorID::filter, dataType),
                                                       params.conv_.filter_shape_,
                                                       format);
        std::shared_ptr<CUDA::DnnBETensorDescriptor> add_tensor_desc;
        if (params.add_shape_) {
            add_tensor_desc = MakeTensorDescriptor(DnnTensorID::add,
                                                   convertConvBackendDataType(DnnTensorID::add, dataType),
                                                   params.add_shape_.value(),
                                                   format);
        } else {
            add_tensor_desc = MakeTensorDescriptor(DnnTensorID::add,
                                                   convertConvBackendDataType(DnnTensorID::add, dataType),
                                                   params.conv_.output_shape_,
                                                   format);
        }
        auto bias_tensor_desc = MakeTensorDescriptor(
            DnnTensorID::bias, convertConvBackendDataType(DnnTensorID::bias, dataType), params.bias_shape_, format);

        // Virtual tensors that will be used from workspace size
        auto output_add_tensor_desc =
            MakeTensorDescriptor(DnnTensorID::add_output,
                                 convertConvBackendDataType(DnnTensorID::add_output, dataType),
                                 params.conv_.output_shape_,
                                 format,
                                 true);
        auto output_bias_tensor_desc =
            MakeTensorDescriptor(DnnTensorID::bias_output,
                                 convertConvBackendDataType(DnnTensorID::bias_output, dataType),
                                 params.conv_.output_shape_,
                                 format,
                                 true);
        auto output_conv_tensor_desc =
            MakeTensorDescriptor(DnnTensorID::conv_output,
                                 convertConvBackendDataType(DnnTensorID::conv_output, dataType),
                                 params.conv_.output_shape_,
                                 format,
                                 true);

        auto conv_desc = CUDA::DnnBEConvolutionDescriptorBuilder()
                             .setMode(CUDNN_CROSS_CORRELATION)
                             .setComputeType(convertConvBackendDataType(DnnTensorID::conv_output, dataType))
                             .setNumberOfSpatialDimensions(arrayLength)
                             .setPrePaddings(params.conv_.padding_before_)
                             .setPostPaddings(params.conv_.padding_after_)
                             .setDilations(params.conv_.dilations_)
                             .setFilterStrides(params.conv_.strides_)
                             .build();

        auto bias_desc = CUDA::DnnBEPointwiseDescriptorBuilder()
                             .setMode(CUDNN_POINTWISE_ADD)
                             .setMathPrecision(convertConvBackendDataType(DnnTensorID::bias, dataType))
                             .build();

        auto add_desc = CUDA::DnnBEPointwiseDescriptorBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(convertConvBackendDataType(DnnTensorID::add, dataType))
                            .build();

        std::shared_ptr<CUDA::DnnBEPointwiseDescriptor> activation_desc;
        if (params.activation_ != ov::nvidia_gpu::nodes::ActivationMode::NO_ACTIVATION) {
            auto activation_desc_builder =
                CUDA::DnnBEPointwiseDescriptorBuilder()
                    .setMode(convertActivationModeToBE(params.activation_))
                    .setMathPrecision(convertConvBackendDataType(DnnTensorID::bias_output, dataType))
                    .setMode(CUDNN_PROPAGATE_NAN);
            if (convertConvBackendDataType(DnnTensorID::bias_output, dataType) == CUDNN_DATA_DOUBLE) {
                activation_desc_builder.setReluLowerClip(0.);
                activation_desc_builder.setReluUpperClip(std::numeric_limits<double>::max());
                activation_desc_builder.setReluLowerClipSlope(0.);
            } else {
                activation_desc_builder.setReluLowerClip(0.f);
                activation_desc_builder.setReluUpperClip(std::numeric_limits<float>::max());
                activation_desc_builder.setReluLowerClipSlope(0.f);
            }
            activation_desc = activation_desc_builder.build();
        }

        std::vector<std::shared_ptr<CUDA::DnnBackendDescriptor>> ops;

        CUDA::DnnBEOperationConvolutionForwardDescriptorBuilder conv_op_desc_builder;
        conv_op_desc_builder.setXDesc(conv_tensor_desc);
        conv_op_desc_builder.setWDesc(filter_tensor_desc);
        conv_op_desc_builder.setYDesc(output_conv_tensor_desc);
        if (convertConvBackendDataType(DnnTensorID::bias_output, dataType) == CUDNN_DATA_DOUBLE) {
            conv_op_desc_builder.setScalingParams<CUDNN_TYPE_DOUBLE>(1, 0);
        } else {
            conv_op_desc_builder.setScalingParams<CUDNN_TYPE_FLOAT>(1, 0);
        }
        conv_op_desc_builder.setConvDesc(conv_desc);
        auto conv_op_desc = conv_op_desc_builder.build();
        ops.push_back(conv_op_desc);

        auto add_op_desc_builder = CUDA::DnnBEOperationPointwiseDescriptorBuilder()
                                       .setXDesc(output_conv_tensor_desc)
                                       .setBDesc(add_tensor_desc);
        if (params.add_shape_) {
            add_op_desc_builder.setScalingParams<CUDNN_TYPE_FLOAT>(1., 1.);
        } else {
            add_op_desc_builder.setScalingParams<CUDNN_TYPE_FLOAT>(1., 0.);
        }
        add_op_desc_builder.setYDesc(output_add_tensor_desc);
        add_op_desc_builder.setPwDesc(add_desc);
        auto add_op_desc = add_op_desc_builder.build();
        ops.push_back(add_op_desc);

        auto bias_op_desc_builder = CUDA::DnnBEOperationPointwiseDescriptorBuilder()
                                        .setXDesc(output_add_tensor_desc)
                                        .setBDesc(bias_tensor_desc);
        if (activation_desc) {
            bias_op_desc_builder.setYDesc(output_bias_tensor_desc);
        } else {
            bias_op_desc_builder.setYDesc(output_tensor_desc);
        }
        auto bias_op_desc = bias_op_desc_builder.setPwDesc(bias_desc).build();
        ops.push_back(bias_op_desc);

        if (activation_desc) {
            auto activation_op_desc = CUDA::DnnBEOperationPointwiseDescriptorBuilder()
                                          .setXDesc(output_bias_tensor_desc)
                                          .setBDesc(output_bias_tensor_desc)
                                          .setYDesc(output_tensor_desc)
                                          .setPwDesc(activation_desc)
                                          .build();
            ops.push_back(activation_op_desc);
        }

        CUDA::DnnBEOperationGraphDescriptorBuilder graphBuilder;
        graphBuilder.setDnnHandle(dnnHandle);
        graphBuilder.setOperations(ops);
        auto graph = graphBuilder.build();

        auto new_plans = CUDA::getAllExecutionPlansFromHeuristics(graph, *dnnHandle);
        plans.insert(plans.end(), new_plans.begin(), new_plans.end());
    };

    addPlans(cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, tensor_element_type);
    // TODO: Add other modes for testing purposes
    // addPlans(cudnnTensorFormat_t::CUDNN_TENSOR_NHWC, tensor_element_type);

    if (plans.empty()) {
        throw_ov_exception("No available plans for backend version of fused convolution !!");
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

std::shared_ptr<CUDA::DnnBEExecutionPlan> FusedConvolutionCuDnnBE::performBenchmarks(
    const CUDA::DnnHandle& dnnHandle, std::vector<std::shared_ptr<CUDA::DnnBEExecutionPlan>>& plans) {
    auto input = CUDA::DefaultStream::stream().malloc(ov::element::Type{params_.conv_.element_type_}.size() *
                                                      ov::shape_size(params_.conv_.input_shape_));
    auto filter = CUDA::DefaultStream::stream().malloc(ov::element::Type{params_.conv_.element_type_}.size() *
                                                       ov::shape_size(params_.conv_.filter_shape_));
    auto output = CUDA::DefaultStream::stream().malloc(ov::element::Type{params_.conv_.element_type_}.size() *
                                                       ov::shape_size(params_.conv_.output_shape_));
    auto bias = CUDA::DefaultStream::stream().malloc(ov::element::Type{params_.conv_.element_type_}.size() *
                                                     ov::shape_size(params_.bias_shape_));
    std::optional<CUDA::DefaultAllocation> add;
    if (params_.add_shape_) {
        add = CUDA::DefaultStream::stream().malloc(ov::element::Type{params_.conv_.element_type_}.size() *
                                                   ov::shape_size(params_.add_shape_.value()));
    }
    auto variantPackBuilder = CUDA::DnnBEVariantPackBuilder();
    if (params_.add_shape_) {
        std::array<int64_t, 5> uids = {
            DnnTensorID::input, DnnTensorID::filter, DnnTensorID::output, DnnTensorID::bias, DnnTensorID::add};
        std::array<const void*, 5> data_ptrs = {input.get(), filter.get(), output.get(), bias.get(), add.value().get()};
        variantPackBuilder.setTensorPointers(uids, data_ptrs);
    } else {
        std::array<int64_t, 4> uids = {DnnTensorID::input, DnnTensorID::filter, DnnTensorID::output, DnnTensorID::bias};
        std::array<const void*, 4> data_ptrs = {input.get(), filter.get(), output.get(), bias.get()};
        variantPackBuilder.setTensorPointers(uids, data_ptrs);
    }

    constexpr const size_t kNumBenchmarks = 100;
    return CUDA::performBenchmarks<kNumBenchmarks>(dnnHandle, plans, variantPackBuilder);
}

WorkbufferRequest FusedConvolutionCuDnnBE::GetWorkBufferRequest() const {
    OPENVINO_ASSERT(engine_config_, "Node name: ", GetName());
    if (workspace_size_ < 0) {
        ov::nvidia_gpu::throw_ov_exception(fmt::format("Workspace Size Invalid = {}", workspace_size_));
    }
    const size_t size = std::max(static_cast<int64_t>(0), workspace_size_);
    if (size > 0) {
        return {{}, {applyAllignment(size)}};
    } else {
        return {};
    }
}

void FusedConvolutionCuDnnBE::Execute(const InferenceRequestContext& context,
                                      Inputs inputs,
                                      Outputs outputs,
                                      const Workbuffers& workbuffers) const {
    OPENVINO_ASSERT(inputs.size() == 3 || inputs.size() == 4, "Node name: ", GetName());
    OPENVINO_ASSERT(outputs.size() == 1, "Node name: ", GetName());

    auto dnnHandle = context.getThreadContext().dnnHandle();
    auto workbuffer = workbuffers.mutable_buffers.empty() ? nullptr : workbuffers.mutable_buffers[0].get();
    std::array<const void*, 5> dataPtrs;
    if (params_.add_shape_) {
        dataPtrs = {inputs[Convolution::Details::FusedConvolutionIndices::input].get(),
                    inputs[Convolution::Details::FusedConvolutionIndices::filter].get(),
                    outputs[Convolution::Details::FusedConvolutionIndices::output].get(),
                    inputs[Convolution::Details::FusedConvolutionIndices::bias].get(),
                    inputs[Convolution::Details::FusedConvolutionIndices::add].get()};
    } else {
        dataPtrs = {inputs[Convolution::Details::FusedConvolutionIndices::input].get(),
                    inputs[Convolution::Details::FusedConvolutionIndices::filter].get(),
                    outputs[Convolution::Details::FusedConvolutionIndices::output].get(),
                    inputs[Convolution::Details::FusedConvolutionIndices::bias].get(),
                    nullptr};
    }

    const std::array<int64_t, 5> uids = {
        DnnTensorID::input, DnnTensorID::filter, DnnTensorID::output, DnnTensorID::bias, DnnTensorID::add};
    auto variantPackBuilder = CUDA::DnnBEVariantPackBuilder();
    if (params_.add_shape_) {
        variantPackBuilder.setTensorPointers(uids, dataPtrs);
    } else {
        variantPackBuilder.setTensorPointers(uids, dataPtrs);
    }
    variantPackBuilder.setWorkspase(workbuffer);
    const auto variantPack = variantPackBuilder.build();

    const auto plan = CUDA::DnnBEExecutionPlanBuilder()
                          .setDnnHandle(context.getThreadContext().dnnHandle())
                          .setEngineConfig(engine_config_)
                          .build();

    throwIfError(::cudnnBackendExecute(context.getThreadContext().dnnHandle().get(), plan->get(), variantPack->get()));
}

CudaGraphCompatibility FusedConvolutionCuDnnBE::GetCudaGraphCompatibility() const {
    return CudaGraphCompatibility::NONE;
}

std::shared_ptr<CUDA::DnnBETensorDescriptor> FusedConvolutionCuDnnBE::MakeTensorDescriptor(
    int64_t id,
    cudnnDataType_t element_type,
    const ov::Shape& shape,
    const cudnnTensorFormat_t format,
    bool isVirtual) {
    const int nbDims = shape.size();
    if (nbDims < 4 || nbDims > 5) {
        ov::nvidia_gpu::throw_ov_exception(
            fmt::format("Unexpected number of dimensions for Convolution input/output: {}", nbDims));
    }

    auto desc_builder = CUDA::DnnBETensorDescriptorBuilder().setDataType(element_type).setShape(shape);
    if (format == cudnnTensorFormat_t::CUDNN_TENSOR_NCHW) {
        desc_builder.setStrides(CUDA::generateStrides(shape, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW));
    } else {
        desc_builder.setStrides(CUDA::generateStrides(shape, cudnnTensorFormat_t::CUDNN_TENSOR_NHWC));
    }
    return desc_builder.setIsVirtual(isVirtual).setUniqueId(id).setAlignment(16).build();
}

}  // namespace nvidia_gpu
}  // namespace ov
