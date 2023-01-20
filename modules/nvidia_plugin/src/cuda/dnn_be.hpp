// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dnn.hpp"
#include "dnn_be_desc.hpp"

namespace CUDA {

/**
 * @brief Describes generic tensor.
 *
 * A tensor is identified by a unique identifier and described by its data type,
 * its data byte-alignment requirements, and the extents and strides of its
 * dimensions.
 */
class DnnBETensorDescriptor : public DnnBackendDescriptor {
public:
    friend class DnnBETensorDescriptorBuilder;

    DnnBETensorDescriptor() : DnnBackendDescriptor{CUDNN_BACKEND_TENSOR_DESCRIPTOR} {}
};

/**
 * @brief Builder for generic tensor.
 */
class DnnBETensorDescriptorBuilder : public DnnBackendDescriptorBuilder<DnnBETensorDescriptor> {
public:
    DnnBETensorDescriptorBuilder() {}

    auto& setUniqueId(int64_t id) {
        setAttributeValue<CUDNN_ATTR_TENSOR_UNIQUE_ID>(id);
        return *this;
    }
    auto& setDataType(cudnnDataType_t dataType) {
        setAttributeValue<CUDNN_ATTR_TENSOR_DATA_TYPE>(dataType);
        return *this;
    }
    template <typename T>
    auto& setShape(const std::vector<T>& shape) {
        Expects(shape.size() <= CUDNN_DIM_MAX);
        std::array<int64_t, CUDNN_DIM_MAX> dimensions;
        std::copy(shape.begin(), shape.end(), dimensions.begin());
        setAttributeValues<CUDNN_ATTR_TENSOR_DIMENSIONS>(gsl::span<int64_t>(dimensions.data(), shape.size()));
        return *this;
    }
    template <typename T>
    auto& setStrides(const std::vector<T>& strides) {
        Expects(strides.size() <= CUDNN_DIM_MAX);
        std::array<int64_t, CUDNN_DIM_MAX> values;
        std::copy(strides.begin(), strides.end(), values.begin());
        setAttributeValues<CUDNN_ATTR_TENSOR_STRIDES>(gsl::span<int64_t>(values.data(), strides.size()));
        return *this;
    }
    auto& setAlignment(int64_t byteAlignment) {
        setAttributeValue<CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT>(byteAlignment);
        return *this;
    }
    auto& setIsVirtual(bool value) {
        setAttributeValue<CUDNN_ATTR_TENSOR_IS_VIRTUAL>(value);
        return *this;
    }
};

/**
 * @brief Describes the parameters for a convolution operator for both
 * forward and backward propagation.
 */
class DnnBEConvolutionDescriptor : public DnnBackendDescriptor {
public:
    friend class DnnBEConvolutionDescriptorBuilder;

    DnnBEConvolutionDescriptor() : DnnBackendDescriptor{CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR} {}
};

/**
 * @brief Builder for a convolution operator descriptor for both
 * forward and backward propagation.
 */
class DnnBEConvolutionDescriptorBuilder : public DnnBackendDescriptorBuilder<DnnBEConvolutionDescriptor> {
public:
    DnnBEConvolutionDescriptorBuilder() {}

    auto& setMode(cudnnConvolutionMode_t mode) {
        setAttributeValue<CUDNN_ATTR_CONVOLUTION_CONV_MODE>(mode);
        return *this;
    }
    auto& setComputeType(cudnnDataType_t computeType) {
        setAttributeValue<CUDNN_ATTR_CONVOLUTION_COMP_TYPE>(computeType);
        return *this;
    }
    auto& setNumberOfSpatialDimensions(int64_t nDims) {
        setAttributeValue<CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS>(nDims);
        return *this;
    }
    template <typename T>
    auto& setPrePaddings(const std::vector<T>& prePaddings) {
        Expects(prePaddings.size() <= CUDNN_DIM_MAX);
        std::array<int64_t, CUDNN_DIM_MAX> values;
        std::copy(prePaddings.begin(), prePaddings.end(), values.begin());
        setAttributeValues<CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS>(gsl::span<int64_t>(values.data(), prePaddings.size()));
        return *this;
    }
    template <typename T>
    auto& setPostPaddings(const std::vector<T>& postPaddings) {
        Expects(postPaddings.size() <= CUDNN_DIM_MAX);
        std::array<int64_t, CUDNN_DIM_MAX> values;
        std::copy(postPaddings.begin(), postPaddings.end(), values.begin());
        setAttributeValues<CUDNN_ATTR_CONVOLUTION_POST_PADDINGS>(
            gsl::span<int64_t>(values.data(), postPaddings.size()));
        return *this;
    }
    template <typename T>
    auto& setDilations(const std::vector<T>& dilations) {
        Expects(dilations.size() <= CUDNN_DIM_MAX);
        std::array<int64_t, CUDNN_DIM_MAX> values;
        std::copy(dilations.begin(), dilations.end(), values.begin());
        setAttributeValues<CUDNN_ATTR_CONVOLUTION_DILATIONS>(gsl::span<int64_t>(values.data(), dilations.size()));
        return *this;
    }
    template <typename T>
    auto& setFilterStrides(const std::vector<T>& strides) {
        Expects(strides.size() <= CUDNN_DIM_MAX);
        std::array<int64_t, CUDNN_DIM_MAX> values;
        std::copy(strides.begin(), strides.end(), values.begin());
        setAttributeValues<CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES>(gsl::span<int64_t>(values.data(), strides.size()));
        return *this;
    }
};

/**
 * @brief Pointwise operator descriptor for following operations: add, mul, min and etc.
 */
class DnnBEPointwiseDescriptor : public DnnBackendDescriptor {
public:
    friend class DnnBEPointwiseDescriptorBuilder;

    DnnBEPointwiseDescriptor() : DnnBackendDescriptor{CUDNN_BACKEND_POINTWISE_DESCRIPTOR} {}
};

/**
 * @brief Builder for a pointwise operator descriptor.
 */
class DnnBEPointwiseDescriptorBuilder : public DnnBackendDescriptorBuilder<DnnBEPointwiseDescriptor> {
public:
    DnnBEPointwiseDescriptorBuilder() {}

    auto& setMathPrecision(cudnnDataType_t data_type) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_MATH_PREC>(data_type);
        return *this;
    }

    auto& setClipping(double l, double u) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP>(l);
        setAttributeValue<CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP>(u);
        return *this;
    }

    auto& setMode(cudnnPointwiseMode_t mode) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_MODE>(mode);
        return *this;
    }

    auto& setMode(cudnnNanPropagation_t nan_mode) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_NAN_PROPAGATION>(nan_mode);
        return *this;
    }

    auto& setReluLowerClip(float lower_clip) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP>(lower_clip);
        return *this;
    }

    auto& setReluLowerClip(double lower_clip) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP>(lower_clip);
        return *this;
    }

    auto& setReluUpperClip(float upper_clip) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP>(upper_clip);
        return *this;
    }

    auto& setReluUpperClip(double upper_clip) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP>(upper_clip);
        return *this;
    }

    auto& setReluLowerClipSlope(float lower_clip_slope) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE>(lower_clip_slope);
        return *this;
    }

    auto& setReluLowerClipSlope(double lower_clip_slope) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE>(lower_clip_slope);
        return *this;
    }

    auto& setEluAlpha(double elu_alpha) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_ELU_ALPHA>(elu_alpha);
        return *this;
    }

    auto& setSoftplusBeta(double softplus_beta) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA>(softplus_beta);
        return *this;
    }

    auto& setSwishBeta(double swish_beta) {
        setAttributeValue<CUDNN_ATTR_POINTWISE_SWISH_BETA>(swish_beta);
        return *this;
    }
};

/**
 * @brief Describes an operation graph descriptor, a small network of one or more operations
 * connected by virtual tensors.
 */
class DnnBEOperationGraphDescriptor : public DnnBackendDescriptor {
public:
    friend class DnnBEOperationGraphDescriptorBuilder;

    DnnBEOperationGraphDescriptor() : DnnBackendDescriptor{CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR} {}

    int64_t getEngineCount(void) const { return getAttributeValue<CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT>(); }

private:
    std::shared_ptr<Handle<cudnnHandle_t>> dnn_handle_;
    std::vector<std::shared_ptr<CUDA::DnnBackendDescriptor>> ops_;
    std::vector<cudnnBackendDescriptor_t> ops_decs_;
};

/**
 * @brief Builder for an operation graph descriptor.
 */
class DnnBEOperationGraphDescriptorBuilder : public DnnBackendDescriptorBuilder<DnnBEOperationGraphDescriptor> {
public:
    DnnBEOperationGraphDescriptorBuilder() {}

    auto& setDnnHandle(const std::shared_ptr<CUDA::DnnHandle>& dnnHandle) {
        desc_->dnn_handle_ = dnnHandle;
        setAttributeValue<CUDNN_ATTR_OPERATIONGRAPH_HANDLE>(dnnHandle->get());
        return *this;
    }

    auto& setOperations(gsl::span<cudnnBackendDescriptor_t> ops) {
        setAttributeValues<CUDNN_ATTR_OPERATIONGRAPH_OPS>(ops);
        return *this;
    }

    auto& setOperations(gsl::span<std::shared_ptr<CUDA::DnnBackendDescriptor>> ops) {
        desc_->ops_ = {ops.begin(), ops.end()};
        std::transform(ops.begin(),
                       ops.end(),
                       std::back_inserter(desc_->ops_decs_),
                       [](const auto& op) -> cudnnBackendDescriptor_t { return (*op).get(); });
        setAttributeValues<CUDNN_ATTR_OPERATIONGRAPH_OPS>(desc_->ops_decs_);
        return *this;
    }
};

/**
 * @brief Specifies an operation node to represent forward convolution in operation graph.
 */
class DnnBEOperationConvolutionForwardDescriptor : public DnnBackendDescriptor {
public:
    friend class DnnBEOperationConvolutionForwardDescriptorBuilder;

    DnnBEOperationConvolutionForwardDescriptor()
        : DnnBackendDescriptor{CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR} {}

private:
    std::shared_ptr<const DnnBackendDescriptor> x_desc_;
    std::shared_ptr<const DnnBackendDescriptor> w_desc_;
    std::shared_ptr<const DnnBackendDescriptor> y_desc_;
    std::shared_ptr<const DnnBackendDescriptor> conv_desc_;
};

/**
 * @brief Builder for a forward convolution descriptor.
 */
class DnnBEOperationConvolutionForwardDescriptorBuilder
    : public DnnBackendDescriptorBuilder<DnnBEOperationConvolutionForwardDescriptor> {
public:
    DnnBEOperationConvolutionForwardDescriptorBuilder() {}

    auto& setXDesc(const std::shared_ptr<DnnBETensorDescriptor>& tensor_desc) {
        desc_->x_desc_ = tensor_desc;
        setAttributeValue<CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X>(tensor_desc->get());
        return *this;
    }
    auto& setWDesc(const std::shared_ptr<DnnBETensorDescriptor>& tensor_desc) {
        desc_->w_desc_ = tensor_desc;
        setAttributeValue<CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W>(tensor_desc->get());
        return *this;
    }
    auto& setYDesc(const std::shared_ptr<DnnBETensorDescriptor>& tensor_desc) {
        desc_->y_desc_ = tensor_desc;
        setAttributeValue<CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y>(tensor_desc->get());
        return *this;
    }
    auto& setConvDesc(const std::shared_ptr<DnnBEConvolutionDescriptor>& conv_desc) {
        desc_->conv_desc_ = conv_desc;
        setAttributeValue<CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC>(conv_desc->get());
        return *this;
    }
    template <cudnnBackendAttributeType_t TypeID>
    auto& setScalingParams(typename DnnBEAttrType<TypeID>::ValueType alpha,
                           typename DnnBEAttrType<TypeID>::ValueType beta) {
        setAttributeValue<TypeID>(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, alpha);
        setAttributeValue<TypeID>(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, beta);
        return *this;
    }
};

/**
 * @brief Specifies an operation node to represent forward convolution in operation graph.
 */
class DnnBEOperationPointwiseDescriptor : public DnnBackendDescriptor {
public:
    friend class DnnBEOperationPointwiseDescriptorBuilder;

    DnnBEOperationPointwiseDescriptor() : DnnBackendDescriptor{CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR} {}

private:
    std::shared_ptr<const DnnBackendDescriptor> x_desc_;
    std::shared_ptr<const DnnBackendDescriptor> b_desc_;
    std::shared_ptr<const DnnBackendDescriptor> y_desc_;
    std::shared_ptr<const DnnBackendDescriptor> dx_desc_;
    std::shared_ptr<const DnnBackendDescriptor> dy_desc_;
    std::shared_ptr<const DnnBackendDescriptor> pw_desc_;
};

/**
 * @brief Builder for a forward convolution in operation graph.
 */
class DnnBEOperationPointwiseDescriptorBuilder : public DnnBackendDescriptorBuilder<DnnBEOperationPointwiseDescriptor> {
public:
    DnnBEOperationPointwiseDescriptorBuilder() {}

    auto& setXDesc(const std::shared_ptr<DnnBETensorDescriptor>& tensor_desc) {
        desc_->x_desc_ = tensor_desc;
        setAttributeValue<CUDNN_ATTR_OPERATION_POINTWISE_XDESC>(tensor_desc->get());
        return *this;
    }
    auto& setBDesc(const std::shared_ptr<DnnBETensorDescriptor>& tensor_desc) {
        desc_->b_desc_ = tensor_desc;
        setAttributeValue<CUDNN_ATTR_OPERATION_POINTWISE_BDESC>(tensor_desc->get());
        return *this;
    }
    auto& setYDesc(const std::shared_ptr<DnnBETensorDescriptor>& tensor_desc) {
        desc_->y_desc_ = tensor_desc;
        setAttributeValue<CUDNN_ATTR_OPERATION_POINTWISE_YDESC>(tensor_desc->get());
        return *this;
    }
    auto& setDxDesc(const std::shared_ptr<DnnBEConvolutionDescriptor>& conv_desc) {
        desc_->dx_desc_ = conv_desc;
        setAttributeValue<CUDNN_ATTR_OPERATION_POINTWISE_DXDESC>(conv_desc->get());
        return *this;
    }
    auto& setDyDesc(const std::shared_ptr<DnnBEConvolutionDescriptor>& conv_desc) {
        desc_->dy_desc_ = conv_desc;
        setAttributeValue<CUDNN_ATTR_OPERATION_POINTWISE_DYDESC>(conv_desc->get());
        return *this;
    }
    auto& setPwDesc(const std::shared_ptr<DnnBEPointwiseDescriptor>& tensor_desc) {
        desc_->pw_desc_ = tensor_desc;
        setAttributeValue<CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR>(tensor_desc->get());
        return *this;
    }
    template <cudnnBackendAttributeType_t TypeID>
    auto& setScalingParams(typename DnnBEAttrType<TypeID>::ValueType alpha1,
                           typename DnnBEAttrType<TypeID>::ValueType alpha2) {
        setAttributeValue<TypeID>(CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, alpha1);
        setAttributeValue<TypeID>(CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2, alpha2);
        return *this;
    }
};

/**
 * @brief Describes an engine to compute an operation graph. An engine is
 * a grouping of kernels with similar compute and numerical attributes.
 */
class DnnBEEngine : public DnnBackendDescriptor {
public:
    friend class DnnBEEngineBuilder;
    friend class DnnBEEngineConfigDescriptorBuilder;

    DnnBEEngine() : DnnBackendDescriptor{CUDNN_BACKEND_ENGINE_DESCRIPTOR} {}

    DnnBEEngine(DnnBEEngine&& engine)
        : DnnBackendDescriptor{engine},
          idx_{engine.idx_},
          num_knobs_{engine.num_knobs_},
          knobs_{engine.knobs_},
          bknobs_{engine.bknobs_},
          graph_{engine.graph_} {
        for (uint64_t i = 0; i < bknobs_.size(); i++) {
            bknobs_[i] = std::make_shared<DnnBackendDescriptor>(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR);
        }

        std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> bKnobs_ =
            {};  //!< Opaque pointer to the backend knobs_
        for (std::uint32_t i = 0; i < bknobs_.size(); i++) {
            bKnobs_[i] = bknobs_[i]->get();
        }
        throwIfError(cudnnBackendGetAttribute(get(),
                                              CUDNN_ATTR_ENGINE_KNOB_INFO,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              CUDNN_KNOB_TYPE_COUNTS,
                                              &num_knobs_,
                                              bKnobs_.data()));
        buildKnobs();
    }

    int64_t getGlobalIndex() const { return getAttributeValue<CUDNN_ATTR_ENGINE_GLOBAL_INDEX>(); }

    bool isKnobsSet() const {
        bool is_knob_set = false;
        for (auto i = 0; i < num_knobs_; i++) {
            if (knobs_[i].getChoice() != -1) {
                is_knob_set = true;
                break;
            }
        }
        return is_knob_set;
    }

private:
    class Knob {
    public:
        Knob(cudnnBackendKnobType_t type_, int64_t max, int64_t min, int64_t stride_)
            : knobType(type_), maxValue(max), minValue(min), stride(stride_) {}

        void setChoice(uint64_t val) { choice = val; }

        int64_t getChoice() const { return choice; }

        cudnnBackendKnobType_t getKnobType() const { return knobType; }

        int64_t getMinValue() const { return minValue; }

        int64_t getMaxValue() const { return minValue; }

        int64_t getStride() const { return stride; }

    private:
        cudnnBackendKnobType_t knobType = CUDNN_KNOB_TYPE_COUNTS;
        int64_t maxValue = 0, minValue = 0, stride = 0;  //!< min, max and stride of the knob value
        int64_t choice = -1;                             //!< Choice set by the user
    };

    void buildKnobs() {
        for (auto i = 0; i < num_knobs_; i++) {
            auto bKnob = bknobs_[i]->get();
            cudnnBackendKnobType_t type;
            int64_t maxValue, minValue, stride, elemCount;
            throwIfError(
                cudnnBackendGetAttribute(bKnob, CUDNN_ATTR_KNOB_INFO_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &elemCount, &type));
            throwIfError(cudnnBackendGetAttribute(
                bKnob, CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, CUDNN_TYPE_INT64, 1, &elemCount, &maxValue));
            throwIfError(cudnnBackendGetAttribute(
                bKnob, CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, CUDNN_TYPE_INT64, 1, &elemCount, &minValue));
            throwIfError(
                cudnnBackendGetAttribute(bKnob, CUDNN_ATTR_KNOB_INFO_STRIDE, CUDNN_TYPE_INT64, 1, &elemCount, &stride));
            knobs_.emplace_back(Knob(type, maxValue, minValue, stride));
        }
    }

    int64_t idx_ = -1;
    int64_t num_knobs_ = 0;
    std::vector<Knob> knobs_;
    std::array<std::shared_ptr<DnnBackendDescriptor>, CUDNN_KNOB_TYPE_COUNTS> bknobs_ = {};
    std::shared_ptr<const DnnBackendDescriptor> graph_;
};

/**
 * @brief Builder for an engine to compute an operation graph.
 */
class DnnBEEngineBuilder : public DnnBackendDescriptorBuilder<DnnBEEngine> {
public:
    DnnBEEngineBuilder() {}

    auto& setOpEngineGraph(const std::shared_ptr<DnnBEOperationGraphDescriptor>& graph) {
        desc_->graph_ = graph;
        setAttributeValue<CUDNN_ATTR_ENGINE_OPERATION_GRAPH>(graph->get());
        return *this;
    }
    auto& setGlobalIndex(int64_t index) {
        setAttributeValue<CUDNN_ATTR_ENGINE_GLOBAL_INDEX>(index);
        return *this;
    }

    std::shared_ptr<DnnBEEngine> build() override {
        auto engine = DnnBackendDescriptorBuilder<DnnBEEngine>::build();
        return std::make_shared<DnnBEEngine>(std::move(*engine));
    }
};

/**
 * @brief Describes backend engine configuration.
 *
 * This descriptor is initialized by DnnBEEngineHeuristicsDescriptor in
 * `getEngineConfigs()` method.
 */
class DnnBEEngineConfigDescriptor : public DnnBackendDescriptor {
public:
    friend class DnnBEEngineConfigDescriptorBuilder;

    DnnBEEngineConfigDescriptor() : DnnBackendDescriptor{CUDNN_BACKEND_ENGINECFG_DESCRIPTOR} {
        for (uint64_t i = 0; i < bchoices_.size(); i++) {
            bchoices_[i] = std::make_shared<DnnBackendDescriptor>(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
        }
    }

    DnnBEEngine getEngine() const {
        auto engines = getBEDescAttributeValues<CUDNN_ATTR_ENGINECFG_ENGINE, DnnBEEngine>();
        if (engines.size() != 1) ov::nvidia_gpu::throwIEException("Unexpected number of cuDNN Backend engines");
        return std::move(*engines[0]);
    }

private:
    int64_t num_knobs_ = 0;
    bool set_knobs_attr_ = false;
    std::array<std::shared_ptr<const DnnBackendDescriptor>, CUDNN_KNOB_TYPE_COUNTS> bchoices_{};
    std::shared_ptr<const DnnBackendDescriptor> engine_;
    std::shared_ptr<const DnnBackendDescriptor> graph_;
};

/**
 * @brief Builder for a backend engine configuration descriptor.
 */
class DnnBEEngineConfigDescriptorBuilder : public DnnBackendDescriptorBuilder<DnnBEEngineConfigDescriptor> {
public:
    DnnBEEngineConfigDescriptorBuilder() {}

    auto& setEngine(const std::shared_ptr<DnnBEEngine>& engine) {
        desc_->engine_ = engine;
        auto& knobs = engine->knobs_;
        desc_->num_knobs_ = knobs.size();
        desc_->set_knobs_attr_ = engine->isKnobsSet();

        for (std::uint32_t i = 0; i < knobs.size(); i++) {
            cudnnBackendKnobType_t type = knobs[i].getKnobType();
            int64_t value = knobs[i].getChoice();
            throwIfError(cudnnBackendSetAttribute(
                desc_->bchoices_[i]->get(), CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &type));
            throwIfError(cudnnBackendSetAttribute(
                desc_->bchoices_[i]->get(), CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, &value));
            throwIfError(cudnnBackendFinalize(desc_->bchoices_[i]->get()));
        }

        auto handle = engine->get();
        throwIfError(cudnnBackendSetAttribute(
            desc_->get(), CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &handle));

        if (desc_->set_knobs_attr_ && desc_->num_knobs_ > 0) {
            std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> bchoices;
            for (auto i = 0; i < desc_->num_knobs_; i++) {
                bchoices[i] = desc_->bchoices_[i]->get();
            }
            throwIfError(cudnnBackendSetAttribute(desc_->get(),
                                                  CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                                  CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  desc_->num_knobs_,
                                                  bchoices.data()));
        }

        return *this;
    }
};

/**
 * @brief Provides engine configs ranked by performance according to
 * cuDNN’s heuristics.
 */
class DnnBEEngineHeuristicsDescriptor : public DnnBackendDescriptor {
public:
    friend class DnnBEEngineHeuristicsDescriptorBuilder;

    DnnBEEngineHeuristicsDescriptor() : DnnBackendDescriptor{CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR} {}

    std::vector<std::shared_ptr<DnnBEEngineConfigDescriptor>> getEngineConfigs() {
        return getBEDescAttributeValues<CUDNN_ATTR_ENGINEHEUR_RESULTS, DnnBEEngineConfigDescriptor>();
    }

private:
    std::shared_ptr<const DnnBackendDescriptor> graph_;
};

/**
 * @brief Builder for an engine heuristic descriptor.
 */
class DnnBEEngineHeuristicsDescriptorBuilder : public DnnBackendDescriptorBuilder<DnnBEEngineHeuristicsDescriptor> {
public:
    DnnBEEngineHeuristicsDescriptorBuilder() {}

    auto& setOpGraph(const std::shared_ptr<DnnBEOperationGraphDescriptor>& graph) {
        graph_ = graph;
        setAttributeValue<CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH>(graph->get());
        return *this;
    }
    auto& setMode(cudnnBackendHeurMode_t mode) {
        setAttributeValue<CUDNN_ATTR_ENGINEHEUR_MODE>(mode);
        return *this;
    }

private:
    std::shared_ptr<const DnnBackendDescriptor> graph_;
};

/**
 * @brief Describes execution plan.
 */
class DnnBEExecutionPlan : public DnnBackendDescriptor {
public:
    friend class DnnBEExecutionPlanBuilder;

    DnnBEExecutionPlan() : DnnBackendDescriptor{CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR} {}

    int64_t getWorkspaceSize() const { return getAttributeValue<CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE>(); }
    std::shared_ptr<DnnBEEngineConfigDescriptor> getConfigDesc() const { return config_desc_; }

private:
    std::shared_ptr<const CUDA::DnnHandle> dnn_handle_;
    std::shared_ptr<DnnBEEngineConfigDescriptor> config_desc_;
};

/**
 * @brief Builder for an execution plan.
 */
class DnnBEExecutionPlanBuilder : public DnnBackendDescriptorBuilder<DnnBEExecutionPlan> {
public:
    DnnBEExecutionPlanBuilder() {}

    auto& setEngineConfig(const std::shared_ptr<DnnBEEngineConfigDescriptor>& config) {
        desc_->config_desc_ = std::static_pointer_cast<DnnBEEngineConfigDescriptor>(config);
        setAttributeValue<CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG>(config->get());
        return *this;
    }
    auto& setDnnHandle(const CUDA::DnnHandle& dnnHandle) {
        setAttributeValue<CUDNN_ATTR_EXECUTION_PLAN_HANDLE>(dnnHandle.get());
        return *this;
    }
    auto& setDnnHandle(const std::shared_ptr<CUDA::DnnHandle>& dnnHandle) {
        desc_->dnn_handle_ = dnnHandle;
        setAttributeValue<CUDNN_ATTR_EXECUTION_PLAN_HANDLE>(dnnHandle->get());
        return *this;
    }
};

/**
 * @brief Describes device pointers for tensors and workspace.
 * Passed to `cudnnBackendExecute()`.
 */
class DnnBEVariantPack : public DnnBackendDescriptor {
public:
    friend class DnnBEVariantPackBuilder;

    DnnBEVariantPack() : DnnBackendDescriptor{CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR} {}
};

/**
 * @brief Builder for descriptor that specifies device pointers for tensors and workspace.
 */
class DnnBEVariantPackBuilder : public DnnBackendDescriptorBuilder<DnnBEVariantPack> {
public:
    DnnBEVariantPackBuilder() {}
    auto& setTensorPointers(gsl::span<const int64_t> uids, gsl::span<const void*> ptrs) {
        setAttributeValues<CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS>(uids);
        setAttributeValues<CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS>(ptrs);
        return *this;
    }
    auto& setWorkspase(const void* workspace) {
        setAttributeValue<CUDNN_ATTR_VARIANT_PACK_WORKSPACE>(workspace);
        return *this;
    }
};

}  // namespace CUDA
