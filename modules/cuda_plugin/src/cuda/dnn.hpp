// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "runtime.hpp"

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

}  // namespace CUDA
