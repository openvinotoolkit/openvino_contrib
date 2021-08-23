// Copyright (C) 2021 Intel Corporation
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
    DnnBETensorDescriptor()
        : DnnBackendDescriptor { CUDNN_BACKEND_TENSOR_DESCRIPTOR } {
    }

    void setUniqueId(int64_t id) {
        setAttributeValue<CUDNN_ATTR_TENSOR_UNIQUE_ID>(id);
    }
    void setDataType(cudnnDataType_t dataType) {
        setAttributeValue<CUDNN_ATTR_TENSOR_DATA_TYPE>(dataType);
    }
    template <typename T>
    void setShape(const std::vector<T>& shape) {
        Expects(shape.size() <= CUDNN_DIM_MAX);
        std::array<int64_t, CUDNN_DIM_MAX> dimensions;
        std::copy(shape.begin(), shape.end(), dimensions.begin());
        setAttributeValues<CUDNN_ATTR_TENSOR_DIMENSIONS>(
            gsl::span<int64_t>(dimensions.data(), shape.size()));
    }
    template <typename T>
    void setStrides(const std::vector<T>& strides) {
        Expects(strides.size() <= CUDNN_DIM_MAX);
        std::array<int64_t, CUDNN_DIM_MAX> values;
        std::copy(strides.begin(), strides.end(), values.begin());
        setAttributeValues<CUDNN_ATTR_TENSOR_STRIDES>(
            gsl::span<int64_t>(values.data(), strides.size()));
    }
    void setAlignment(int64_t byteAlignment) {
        setAttributeValue<CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT>(byteAlignment);
    }
    void setIsVirtual(bool value) {
        setAttributeValue<CUDNN_ATTR_TENSOR_IS_VIRTUAL>(value);
    }
};


/**
 * @brief Describes the parameters for a convolution operator for both
 * forward and backward propagation.
 */
class DnnBEConvolutionDescriptor : public DnnBackendDescriptor {
public:
    DnnBEConvolutionDescriptor()
        : DnnBackendDescriptor { CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR } {
    }

    void setMode(cudnnConvolutionMode_t mode) {
        setAttributeValue<CUDNN_ATTR_CONVOLUTION_CONV_MODE>(mode);
    }
    void setComputeType(cudnnDataType_t computeType) {
        setAttributeValue<CUDNN_ATTR_CONVOLUTION_COMP_TYPE>(computeType);
    }
    void setNumberOfSpatialDimensions(int64_t nDims) {
        setAttributeValue<CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS>(nDims);
    }
    template<typename T>
    void setPrePaddings(const std::vector<T>& prePaddings) {
        Expects(prePaddings.size() <= CUDNN_DIM_MAX);
        std::array<int64_t, CUDNN_DIM_MAX> values;
        std::copy(prePaddings.begin(), prePaddings.end(), values.begin());
        setAttributeValues<CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS>(
            gsl::span<int64_t>(values.data(), prePaddings.size()));
    }
    template<typename T>
    void setPostPaddings(const std::vector<T>& postPaddings) {
        Expects(postPaddings.size() <= CUDNN_DIM_MAX);
        std::array<int64_t, CUDNN_DIM_MAX> values;
        std::copy(postPaddings.begin(), postPaddings.end(), values.begin());
        setAttributeValues<CUDNN_ATTR_CONVOLUTION_POST_PADDINGS>(
            gsl::span<int64_t>(values.data(), postPaddings.size()));
    }
    template<typename T>
    void setDilations(const std::vector<T>& dilations) {
        Expects(dilations.size() <= CUDNN_DIM_MAX);
        std::array<int64_t, CUDNN_DIM_MAX> values;
        std::copy(dilations.begin(), dilations.end(), values.begin());
        setAttributeValues<CUDNN_ATTR_CONVOLUTION_DILATIONS>(
            gsl::span<int64_t>(values.data(), dilations.size()));
    }
    template<typename T>
    void setFilterStrides(const std::vector<T>& strides) {
        Expects(strides.size() <= CUDNN_DIM_MAX);
        std::array<int64_t, CUDNN_DIM_MAX> values;
        std::copy(strides.begin(), strides.end(), values.begin());
        setAttributeValues<CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES>(
            gsl::span<int64_t>(values.data(), strides.size()));
    }
};


/**
 * @brief Describes an operation graph, a small network of one or more operations
 * connected by virtual tensors.
 */
class DnnBEOperationGraphDescriptor : public DnnBackendDescriptor {
public:
    DnnBEOperationGraphDescriptor()
        : DnnBackendDescriptor { CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR } {
    }

    void setDnnHandle(const CUDA::DnnHandle& dnnHandle) {
        setAttributeValue<CUDNN_ATTR_OPERATIONGRAPH_HANDLE>(dnnHandle.get());
    }

    void setOperations(gsl::span<cudnnBackendDescriptor_t> ops) {
        setAttributeValues<CUDNN_ATTR_OPERATIONGRAPH_OPS>(ops);
    }
};


/**
 * @brief Specifies an operation node to represent forward convolution in operation graph.
 */
class DnnBEOperationConvolutionForwardDescriptor : public DnnBackendDescriptor {
public:
    DnnBEOperationConvolutionForwardDescriptor()
        : DnnBackendDescriptor { CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR } {
    }

    void setX(const DnnBETensorDescriptor& tensor_desc) {
        setAttributeValue<CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X>(tensor_desc.get());
    }
    void setW(const DnnBETensorDescriptor& tensor_desc) {
        setAttributeValue<CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W>(tensor_desc.get());
    }
    void setY(const DnnBETensorDescriptor& tensor_desc) {
        setAttributeValue<CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y>(tensor_desc.get());
    }
    void setConv(const DnnBEConvolutionDescriptor& conv_desc) {
        setAttributeValue<CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC>(conv_desc.get());
    }
    template <cudnnBackendAttributeType_t TypeID>
    void setScalingParams(typename DnnBEAttrTypeID<TypeID>::ValueType alpha,
                          typename DnnBEAttrTypeID<TypeID>::ValueType beta) {
        setAttributeValue<TypeID>(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, alpha);
        setAttributeValue<TypeID>(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, beta);
    }
};


/**
 * @brief Describes an engine to compute an operation graph. An engine is
 * a grouping of kernels with similar compute and numerical attributes.
 */
class DnnBEEngineDescriptor : public DnnBackendDescriptor {
public:
    DnnBEEngineDescriptor()
        : DnnBackendDescriptor { CUDNN_BACKEND_ENGINE_DESCRIPTOR } {
    }

    int64_t getGlobalIndex() const {
        return getAttributeValue<CUDNN_ATTR_ENGINE_GLOBAL_INDEX>();
    }
};


/**
 * @brief Describes backend engine configuration.
 *
 * This descriptor is initialized by DnnBEEngineheurDescriptor in
 * `getEngineConfigs()` method.
 */
class DnnBEEngineConfigDescriptor : public DnnBackendDescriptor {
public:
    DnnBEEngineConfigDescriptor()
        : DnnBackendDescriptor { CUDNN_BACKEND_ENGINECFG_DESCRIPTOR } {
    }

    DnnBEEngineDescriptor getEngine() const {
        auto engines = getBEDescAttributeValues<CUDNN_ATTR_ENGINECFG_ENGINE, DnnBEEngineDescriptor>();
        if (engines.size() != 1)
            CUDAPlugin::throwIEException(
                "Unexpected number of cuDNN Backend engines");
        return std::move(engines[0]);
    }
};


/**
 * @brief Provides engine configs ranked by performance according to
 * cuDNN’s heuristics.
 */
class DnnBEEngineheurDescriptor : public DnnBackendDescriptor {
public:
    DnnBEEngineheurDescriptor()
        : DnnBackendDescriptor { CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR } {
    }

    void setOpGraph(const DnnBEOperationGraphDescriptor& graph) {
        setAttributeValue<CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH>(graph.get());
    }
    void setMode(cudnnBackendHeurMode_t mode) {
        setAttributeValue<CUDNN_ATTR_ENGINEHEUR_MODE>(mode);
    }

    std::vector<DnnBEEngineConfigDescriptor> getEngineConfigs() {
        return getBEDescAttributeValues<CUDNN_ATTR_ENGINEHEUR_RESULTS, DnnBEEngineConfigDescriptor>();
    }
};


/**
 * @brief Describes execution plan.
 *
 * An exception from `finalize()` method indicates that execution plan
 * is not appropriate for computation.
 */
class DnnBEExecutionPlanDescriptor : public DnnBackendDescriptor {
public:
    DnnBEExecutionPlanDescriptor()
        : DnnBackendDescriptor { CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR } {
    }

    void setEngineConfig(const DnnBEEngineConfigDescriptor& config) {
        setAttributeValue<CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG>(config.get());
    }
    void setDnnHandle(const CUDA::DnnHandle& dnnHandle) {
        setAttributeValue<CUDNN_ATTR_EXECUTION_PLAN_HANDLE>(dnnHandle.get());
    }
    int64_t getWorkspaceSize() const {
        return getAttributeValue<CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE>();
    }
};


/**
 * @brief Describes device pointers for tensors and workspace.
 * Passed to `cudnnBackendExecute()`.
 */
class DnnBEVariantPackDescriptor : public DnnBackendDescriptor {
public:
    DnnBEVariantPackDescriptor()
        : DnnBackendDescriptor { CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR } {
    }
    void setTensorPointers(gsl::span<int64_t> uids, gsl::span<void*> ptrs) {
        setAttributeValues<CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS>(uids);
        setAttributeValues<CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS>(ptrs);
    }
    void setWorkspase(void* workspace) {
        setAttributeValue<CUDNN_ATTR_VARIANT_PACK_WORKSPACE>(workspace);
    }
};


} // namespace CUDA
