// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn_backend.h>

#include <cstdint>

namespace CUDA {

/**
 * @brief Attribute type-id traits of cuDNN backend descriptor.
 *
 * Binds together attribute type identifier and attribute value type.
 */
template <cudnnBackendAttributeType_t TypeId>
struct DnnBEAttrType;

template <>
struct DnnBEAttrType<CUDNN_TYPE_HANDLE> {
    using ValueType = cudnnHandle_t;
};
template <>
struct DnnBEAttrType<CUDNN_TYPE_NAN_PROPOGATION> {
    using ValueType = cudnnNanPropagation_t;
};
template <>
struct DnnBEAttrType<CUDNN_TYPE_DATA_TYPE> {
    using ValueType = cudnnDataType_t;
};
template <>
struct DnnBEAttrType<CUDNN_TYPE_CONVOLUTION_MODE> {
    using ValueType = cudnnConvolutionMode_t;
};
template <>
struct DnnBEAttrType<CUDNN_TYPE_BACKEND_DESCRIPTOR> {
    using ValueType = cudnnBackendDescriptor_t;
};
template <>
struct DnnBEAttrType<CUDNN_TYPE_POINTWISE_MODE> {
    using ValueType = cudnnPointwiseMode_t;
};
template <>
struct DnnBEAttrType<CUDNN_TYPE_HEUR_MODE> {
    using ValueType = cudnnBackendHeurMode_t;
};
template <>
struct DnnBEAttrType<CUDNN_TYPE_VOID_PTR> {
    using ValueType = const void*;
};
template <>
struct DnnBEAttrType<CUDNN_TYPE_DOUBLE> {
    using ValueType = double;
};
template <>
struct DnnBEAttrType<CUDNN_TYPE_FLOAT> {
    using ValueType = float;
};
template <>
struct DnnBEAttrType<CUDNN_TYPE_INT64> {
    using ValueType = int64_t;
};
template <>
struct DnnBEAttrType<CUDNN_TYPE_BOOLEAN> {
    using ValueType = bool;
};

/**
 * @brief Traits of cuDNN backend descriptor attributes.
 *
 * Binds together attribute name and default type-id of it's value.
 * Most of attributes have a single value type. Some attributes can
 * accept several numeric types.
 */
template <cudnnBackendAttributeName_t Name>
constexpr cudnnBackendAttributeType_t GetDnnBEAttrTypeId() {
    switch (Name) {
        case CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG:
        case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X:
        case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W:
        case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y:
        case CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC:
        case CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR:
        case CUDNN_ATTR_OPERATION_POINTWISE_XDESC:
        case CUDNN_ATTR_OPERATION_POINTWISE_BDESC:
        case CUDNN_ATTR_OPERATION_POINTWISE_YDESC:
        case CUDNN_ATTR_OPERATION_POINTWISE_DXDESC:
        case CUDNN_ATTR_OPERATION_POINTWISE_DYDESC:
        case CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1:
        case CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2:
        case CUDNN_ATTR_OPERATIONGRAPH_OPS:
        case CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH:
        case CUDNN_ATTR_ENGINE_OPERATION_GRAPH:
        case CUDNN_ATTR_ENGINEHEUR_RESULTS:
        case CUDNN_ATTR_ENGINECFG_ENGINE: {
            return CUDNN_TYPE_BACKEND_DESCRIPTOR;
        }
        case CUDNN_ATTR_EXECUTION_PLAN_HANDLE:
        case CUDNN_ATTR_OPERATIONGRAPH_HANDLE: {
            return CUDNN_TYPE_HANDLE;
        }
        case CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP:
        case CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP:
        case CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE:
        case CUDNN_ATTR_POINTWISE_ELU_ALPHA:
        case CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA:
        case CUDNN_ATTR_POINTWISE_SWISH_BETA: {
            return CUDNN_TYPE_DOUBLE;
        }
        case CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT:
        case CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE:
        case CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS:
        case CUDNN_ATTR_TENSOR_DIMENSIONS:
        case CUDNN_ATTR_TENSOR_STRIDES:
        case CUDNN_ATTR_TENSOR_UNIQUE_ID:
        case CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT:
        case CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS:
        case CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS:
        case CUDNN_ATTR_CONVOLUTION_POST_PADDINGS:
        case CUDNN_ATTR_CONVOLUTION_DILATIONS:
        case CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES:
        case CUDNN_ATTR_ENGINE_GLOBAL_INDEX: {
            return CUDNN_TYPE_INT64;
        }
        case CUDNN_ATTR_POINTWISE_MATH_PREC:
        case CUDNN_ATTR_TENSOR_DATA_TYPE:
        case CUDNN_ATTR_CONVOLUTION_COMP_TYPE: {
            return CUDNN_TYPE_DATA_TYPE;
        }
        case CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS:
        case CUDNN_ATTR_VARIANT_PACK_WORKSPACE: {
            return CUDNN_TYPE_VOID_PTR;
        }
        case CUDNN_ATTR_TENSOR_IS_VIRTUAL: {
            return CUDNN_TYPE_BOOLEAN;
        }
        case CUDNN_ATTR_CONVOLUTION_CONV_MODE: {
            return CUDNN_TYPE_CONVOLUTION_MODE;
        }
        case CUDNN_ATTR_POINTWISE_MODE: {
            return CUDNN_TYPE_POINTWISE_MODE;
        }
        case CUDNN_ATTR_POINTWISE_NAN_PROPAGATION: {
            return CUDNN_TYPE_NAN_PROPOGATION;
        }
        case CUDNN_ATTR_ENGINEHEUR_MODE: {
            return CUDNN_TYPE_HEUR_MODE;
        }
    }
}

}  // namespace CUDA
