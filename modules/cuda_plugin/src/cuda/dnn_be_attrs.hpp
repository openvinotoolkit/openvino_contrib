// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cudnn_backend.h>

namespace CUDA {

/**
 * @brief Attribute type-id traits of cuDNN backend descriptor.
 *
 * Binds together attribute type identifier and attribute value type.
 */
template<cudnnBackendAttributeType_t> struct DnnBEAttrTypeID;

template<> struct DnnBEAttrTypeID<CUDNN_TYPE_HANDLE> {
    using ValueType = cudnnHandle_t;
};
template<> struct DnnBEAttrTypeID<CUDNN_TYPE_DATA_TYPE> {
    using ValueType = cudnnDataType_t;
};
template<> struct DnnBEAttrTypeID<CUDNN_TYPE_CONVOLUTION_MODE> {
    using ValueType = cudnnConvolutionMode_t;
};
template<> struct DnnBEAttrTypeID<CUDNN_TYPE_BACKEND_DESCRIPTOR> {
    using ValueType = cudnnBackendDescriptor_t;
};
template<> struct DnnBEAttrTypeID<CUDNN_TYPE_HEUR_MODE> {
    using ValueType = cudnnBackendHeurMode_t;
};
template<> struct DnnBEAttrTypeID<CUDNN_TYPE_VOID_PTR> {
    using ValueType = void*;
};
template<> struct DnnBEAttrTypeID<CUDNN_TYPE_DOUBLE> {
    using ValueType = double;
};
template<> struct DnnBEAttrTypeID<CUDNN_TYPE_FLOAT> {
    using ValueType = float;
};
template<> struct DnnBEAttrTypeID<CUDNN_TYPE_INT64> {
    using ValueType = int64_t;
};
template<> struct DnnBEAttrTypeID<CUDNN_TYPE_BOOLEAN> {
    using ValueType = bool;
};


/**
 * @brief Traits of cuDNN backend descriptor attributes.
 *
 * Binds together attribute name and default type-id of it's value.
 * Most of attributes have a single value type. Some attributes can
 * accept several numeric types.
 */
template<cudnnBackendAttributeName_t> struct DnnBEAttrName;

template<> struct DnnBEAttrName<CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_EXECUTION_PLAN_HANDLE> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_HANDLE;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_INT64;
};

template<> struct DnnBEAttrName<CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_INT64;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_VARIANT_PACK_WORKSPACE> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_VOID_PTR;
};

template<> struct DnnBEAttrName<CUDNN_ATTR_TENSOR_DATA_TYPE> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_TENSOR_DIMENSIONS> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_INT64;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_TENSOR_STRIDES> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_INT64;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_TENSOR_UNIQUE_ID> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_INT64;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_INT64;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_TENSOR_IS_VIRTUAL> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_BOOLEAN;
};

template<> struct DnnBEAttrName<CUDNN_ATTR_CONVOLUTION_COMP_TYPE> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_DATA_TYPE;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_CONVOLUTION_CONV_MODE> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_CONVOLUTION_MODE;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_INT64;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_INT64;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_CONVOLUTION_POST_PADDINGS> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_INT64;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_CONVOLUTION_DILATIONS> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_INT64;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_INT64;
};

template<> struct DnnBEAttrName<CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR;
};

template<> struct DnnBEAttrName<CUDNN_ATTR_OPERATIONGRAPH_OPS> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_OPERATIONGRAPH_HANDLE> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_HANDLE;
};

template<> struct DnnBEAttrName<CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_ENGINEHEUR_MODE> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_HEUR_MODE;
};
template<> struct DnnBEAttrName<CUDNN_ATTR_ENGINEHEUR_RESULTS> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR;
};

template<> struct DnnBEAttrName<CUDNN_ATTR_ENGINE_GLOBAL_INDEX> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_INT64;
};

template<> struct DnnBEAttrName<CUDNN_ATTR_ENGINECFG_ENGINE> {
    constexpr static auto TypeID = cudnnBackendAttributeType_t::CUDNN_TYPE_BACKEND_DESCRIPTOR;
};

} // namespace CUDA
