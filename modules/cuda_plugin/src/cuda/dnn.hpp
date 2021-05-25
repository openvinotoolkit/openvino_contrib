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

    DnnTensorDescriptor(cudnnDataType_t dataType, int nbDims, const int dimA[], const int strideA[]) {
        set(dataType, nbDims, dimA, strideA);
    }

    DnnTensorDescriptor(cudnnTensorFormat_t format, cudnnDataType_t dataType, int nbDims, const int dimA[]) {
        set(format, dataType, nbDims, dimA);
    }

    DnnTensorDescriptor(cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w) {
        set(format, dataType, n, c, h, w);
    }

public:
    void set(cudnnDataType_t dataType, int nbDims, const int dimA[], const int strideA[]) {
        throwIfError(cudnnSetTensorNdDescriptor(get(), dataType, nbDims, dimA, strideA));
    }

    void set(cudnnTensorFormat_t format, cudnnDataType_t dataType, int nbDims, const int dimA[]) {
        throwIfError(cudnnSetTensorNdDescriptorEx(get(), format, dataType, nbDims, dimA));
    }

    void set(cudnnTensorFormat_t format, cudnnDataType_t dataType,
             int n, int c, int h, int w) {
        throwIfError(cudnnSetTensor4dDescriptor(get(), format, dataType, n, c, h, w));
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

class DnnFilterDescriptor
    : public UniqueBase<cudnnCreateFilterDescriptor,
                        cudnnDestroyFilterDescriptor,
                        cudnnFilterDescriptor_t> {
 public:
    DnnFilterDescriptor() {}
    DnnFilterDescriptor(cudnnDataType_t dataType, cudnnTensorFormat_t format,
                        int nbDims, const int filterDimA[]) {
        set(dataType, format, nbDims, filterDimA);
    }
    void set(cudnnDataType_t dataType, cudnnTensorFormat_t format,
             int nbDims, const int filterDimA[]) {
        throwIfError(cudnnSetFilterNdDescriptor(get(), dataType, format, nbDims, filterDimA));
    }
};

class DnnConvolutionDescriptor
    : public UniqueBase<cudnnCreateConvolutionDescriptor,
                        cudnnDestroyConvolutionDescriptor,
                        cudnnConvolutionDescriptor_t> {
 public:
    DnnConvolutionDescriptor() {}
    DnnConvolutionDescriptor(int arrayLength, const int padA[], const int filterStrideA[],
             const int dilationA[], cudnnConvolutionMode_t mode, cudnnDataType_t dataType) {
        set(arrayLength, padA, filterStrideA, dilationA, mode, dataType);
    }
 public:
    void set(int arrayLength, const int padA[], const int filterStrideA[],
             const int dilationA[], cudnnConvolutionMode_t mode, cudnnDataType_t dataType) {
        throwIfError(cudnnSetConvolutionNdDescriptor(get(), arrayLength, padA, filterStrideA,
                                                     dilationA, mode, dataType));
    }
};

class DnnHandle : public UniqueBase<cudnnCreate, cudnnDestroy, cudnnHandle_t> {
 public:
  DnnHandle() {}
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
class ScalingParameters {
public:
    union Param { double fp64; float fp32; };

    void set(cudnnDataType_t type, double alpha = 1.0, double beta = 0.0) {
        if (type == CUDNN_DATA_DOUBLE) {
            alpha_.fp64 = alpha;
            beta_.fp64 = beta;
        } else {
            alpha_.fp32 = static_cast<float>(alpha);
            beta_.fp32 = static_cast<float>(beta);
        }
    }

    Param alpha() const { return alpha_; }
    Param beta() const { return beta_; }

private:
    Param alpha_ = {}, beta_ = {};
};

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
