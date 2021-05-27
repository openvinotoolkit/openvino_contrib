// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runtime.hpp"
#include <ngraph/type/element_type.hpp>

namespace CUDA {

class DnnOpTensorDescriptor : public UniqueBase<cudnnCreateOpTensorDescriptor,
                                                cudnnDestroyOpTensorDescriptor,
                                                cudnnOpTensorDescriptor_t> {
 public:
  DnnOpTensorDescriptor(cudnnOpTensorOp_t opTensorOp,
                        cudnnDataType_t opTensorCompType,
                        cudnnNanPropagation_t opTensorNanOpt) {
    set(opTensorOp, opTensorCompType, opTensorNanOpt);
  }
  void set(cudnnOpTensorOp_t opTensorOp, cudnnDataType_t opTensorCompType,
           cudnnNanPropagation_t opTensorNanOpt) {
    throwIfError(cudnnSetOpTensorDescriptor(get(), opTensorOp, opTensorCompType,
                                            opTensorNanOpt));
  }
};

class DnnTensorDescriptor
    : public UniqueBase<cudnnCreateTensorDescriptor,
                        cudnnDestroyTensorDescriptor, cudnnTensorDescriptor_t> {
 public:
  DnnTensorDescriptor() {}
  DnnTensorDescriptor(cudnnDataType_t dataType, int nbDims, const int dimA[],
                      const int strideA[]) {
    set(dataType, nbDims, dimA, strideA);
  }
  /**
   * @brief creates a 4D tensor description with given format (NCHW or NHWC)
   */
  DnnTensorDescriptor(cudnnDataType_t dataType, cudnnTensorFormat_t format,
                      const int dimA[4]) {
    set(dataType, format, dimA);
  }
  void set(cudnnDataType_t dataType, int nbDims, const int dimA[],
           const int strideA[]) {
    throwIfError(
        cudnnSetTensorNdDescriptor(  // there are two other signatures available
            get(), dataType, nbDims, dimA, strideA));
  }
  /**
   * @brief setup a 4D tensor with given format (NCHW or NHWC)
   */
  void set(cudnnDataType_t dataType, cudnnTensorFormat_t format, const int dimA[4]) {
    throwIfError(cudnnSetTensorNdDescriptorEx(get(), format, dataType, 4, dimA));
  }
};

class DnnActivationDescriptor
    : public UniqueBase<cudnnCreateActivationDescriptor,
                        cudnnDestroyActivationDescriptor,
                        cudnnActivationDescriptor_t> {
 public:
  DnnActivationDescriptor(cudnnActivationMode_t mode,
                          cudnnNanPropagation_t reluNanOpt, double coef) {
    set(mode, reluNanOpt, coef);
  }
  void set(cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt,
           double coef) {
    throwIfError(cudnnSetActivationDescriptor(get(), mode, reluNanOpt, coef));
  }
};

class DnnPoolingDescriptor
    : public UniqueBase<cudnnCreatePoolingDescriptor,
                        cudnnDestroyPoolingDescriptor, cudnnPoolingDescriptor_t> {
 public:
  DnnPoolingDescriptor(const cudnnPoolingMode_t mode,
                       const cudnnNanPropagation_t nanPropagation, int nbDims,
                       const int windowDimA[], const int paddingA[], const int strideA[]) {
    set(mode, nanPropagation, nbDims, windowDimA, paddingA, strideA);
  }
  void set(const cudnnPoolingMode_t mode,
           const cudnnNanPropagation_t nanPropagation, int nbDims,
           const int windowDimA[], const int paddingA[], const int strideA[]) {
    throwIfError(cudnnSetPoolingNdDescriptor(
        get(), mode, nanPropagation, nbDims, windowDimA, paddingA, strideA));
  }
};

class ReluDescriptor : public DnnActivationDescriptor {
  using DnnActivationDescriptor::set;

 public:
  ReluDescriptor()
      : DnnActivationDescriptor{CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0} {
  }
};

class SigmoidDescriptor : public DnnActivationDescriptor {
  using DnnActivationDescriptor::set;

 public:
  SigmoidDescriptor()
      : DnnActivationDescriptor{CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN,
                                0} {}
};

class DnnHandle : public UniqueBase<cudnnCreate, cudnnDestroy, cudnnHandle_t> {
 public:
  explicit DnnHandle(const Stream& stream) {
    throwIfError(cudnnSetStream(get(), stream.get()));
  }
  void opTensor(const DnnOpTensorDescriptor& opTensorDesc, const void* alpha1,
                const DnnTensorDescriptor& aDesc, const void* A,
                const void* alpha2, const DnnTensorDescriptor& bDesc,
                const void* B, const void* beta,
                const DnnTensorDescriptor& cDesc, void* C) const {
    throwIfError(cudnnOpTensor(get(), opTensorDesc.get(), alpha1, aDesc.get(),
                               A, alpha2, bDesc.get(), B, beta, cDesc.get(),
                               C));
  }
  // TODO: accept device pointers for x and y
  void activationForward(const DnnActivationDescriptor& activationDesc,
                         const void* alpha, const DnnTensorDescriptor& xDesc,
                         const void* x, const void* beta,
                         const DnnTensorDescriptor& yDesc, void* y) const {
    throwIfError(cudnnActivationForward(get(), activationDesc.get(), alpha,
                                        xDesc.get(), x, beta, yDesc.get(), y));
  }
};

/**
 * @brief Holds scaling parameters, required for some CUDNN functions
 * @ref https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#scaling-parameters
 */
struct ScalingParameters {
  float alpha;
  float beta;
};

/* Note: the doc says:
 * The storage data types for alpha and beta are:
 *   float for HALF and FLOAT tensors, and
 *   double for DOUBLE tensors.
 * Since CUDAPlugin supports only HALF and FLOAT tensors,
 * the storage data is set to float
 */

/**
 * @brief Defines default scaling parameters { 1.0, 0.0 } that make no scaling
 */
inline constexpr ScalingParameters NoScaling { 1.0, 0.0 };


/**
 * @brief Converts an OpenVino data type to a corresponding cuDNN data type, throws
 *        if incompatible.
 */
constexpr cudnnDataType_t toDataType(const ngraph::element::Type& type) {
  using ngraph::element::Type_t;
  switch (static_cast<Type_t>(type)) {
    case Type_t::bf16:
      return CUDNN_DATA_BFLOAT16;
    case Type_t::f16:
      return CUDNN_DATA_HALF;
    case Type_t::f32:
      return CUDNN_DATA_FLOAT;
    case Type_t::f64:
      return CUDNN_DATA_DOUBLE;
    case Type_t::i8:
      return CUDNN_DATA_INT8;
    case Type_t::i32:
      return CUDNN_DATA_INT32;
    case Type_t::i64:
      return CUDNN_DATA_INT64;
    default:
      CUDA::throwIEException("unsupported ngraph element type " +
                             type.c_type_string());
  }
}


}  // namespace CUDA
