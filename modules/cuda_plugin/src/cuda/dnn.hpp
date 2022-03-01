// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cudnn.h>

#include <functional>
#include <ngraph/type/element_type.hpp>
#include <optional>

#include "constant_factory.hpp"
#include "runtime.hpp"

inline std::string cudnnGetErrorString(cudnnConvolutionFwdAlgo_t algo) {
    switch (algo) {
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
            return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
            return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
            return "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";
        case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
            return "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
            return "CUDNN_CONVOLUTION_FWD_ALGO_FFT";
        case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
            return "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
            return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
        case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
            return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
        default:
            return "UNKNOWN CUDNN_CONVOLUTION_ALGO";
    }
}

inline void throwIfError(
    cudnnStatus_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != CUDNN_STATUS_SUCCESS) CUDAPlugin::throwIEException(cudnnGetErrorString(err), location);
}

inline void logIfError(
    cudnnStatus_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != CUDNN_STATUS_SUCCESS) CUDAPlugin::logError(cudnnGetErrorString(err), location);
}

namespace CUDA {

class DnnOpTensorDescriptor : public Handle<cudnnOpTensorDescriptor_t> {
public:
    DnnOpTensorDescriptor() : Handle(cudnnCreateOpTensorDescriptor, cudnnDestroyOpTensorDescriptor) {}
    auto&& set(cudnnOpTensorOp_t opTensorOp, cudnnDataType_t opTensorCompType, cudnnNanPropagation_t opTensorNanOpt) {
        throwIfError(cudnnSetOpTensorDescriptor(get(), opTensorOp, opTensorCompType, opTensorNanOpt));
        return std::move(*this);
    }
};

class DnnTensorDescriptor
    : public Handle<cudnnTensorDescriptor_t> {
public:
    using CRef = std::reference_wrapper<const DnnTensorDescriptor>;

    DnnTensorDescriptor() : Handle(cudnnCreateTensorDescriptor, cudnnDestroyTensorDescriptor) {}

    auto&& set(cudnnDataType_t dataType, int nbDims, const int dimA[], const int strideA[]) {
        throwIfError(cudnnSetTensorNdDescriptor(get(), dataType, nbDims, dimA, strideA));
        return std::move(*this);
    }

    auto&& set(cudnnTensorFormat_t format, cudnnDataType_t dataType, int nbDims, const int dimA[]) {
        throwIfError(cudnnSetTensorNdDescriptorEx(get(), format, dataType, nbDims, dimA));
        return std::move(*this);
    }

    auto&& set(cudnnTensorFormat_t format, cudnnDataType_t dataType, int n, int c, int h, int w) {
        throwIfError(cudnnSetTensor4dDescriptor(get(), format, dataType, n, c, h, w));
        return std::move(*this);
    }

    void getTensorNdDescriptor(int nbDimsRequested, cudnnDataType_t& dataType, int& nbDims, int dimA[], int strideA[]) {
        throwIfError(cudnnGetTensorNdDescriptor(get(), nbDimsRequested, &dataType, &nbDims, dimA, strideA));
    }

    size_t getTensorSizeInBytes() {
        size_t size = 0;
        throwIfError(cudnnGetTensorSizeInBytes(get(), &size));
        return size;
    }
};

class DnnActivationDescriptor : public Handle<cudnnActivationDescriptor_t> {
public:
    DnnActivationDescriptor() : Handle(cudnnCreateActivationDescriptor, cudnnDestroyActivationDescriptor) {}
    auto&& set(cudnnActivationMode_t mode, cudnnNanPropagation_t reluNanOpt, double coef) {
        throwIfError(cudnnSetActivationDescriptor(get(), mode, reluNanOpt, coef));
        return std::move(*this);
    }
};

class SigmoidDescriptor : public DnnActivationDescriptor {
public:
    SigmoidDescriptor() { set(CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0); }
};

class ReluDescriptor : public DnnActivationDescriptor {
public:
    ReluDescriptor() : DnnActivationDescriptor{} { set(CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0); }
};

class TanhDescriptor : public DnnActivationDescriptor {
public:
    TanhDescriptor() { set(CUDNN_ACTIVATION_TANH, CUDNN_PROPAGATE_NAN, 0); }
};

class ClippedReluDescriptor : public DnnActivationDescriptor {
public:
    explicit ClippedReluDescriptor(double threshold) {
        set(CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, threshold);
    }
};

class DnnPoolingDescriptor : public Handle<cudnnPoolingDescriptor_t> {
public:
    DnnPoolingDescriptor() : Handle(cudnnCreatePoolingDescriptor, cudnnDestroyPoolingDescriptor) {}
    auto&& set(const cudnnPoolingMode_t mode,
               const cudnnNanPropagation_t nanPropagation,
               int nbDims,
               const int windowDimA[],
               const int paddingA[],
               const int strideA[]) {
        throwIfError(cudnnSetPoolingNdDescriptor(get(), mode, nanPropagation, nbDims, windowDimA, paddingA, strideA));
        return std::move(*this);
    }
};

class DnnFilterDescriptor : public Handle<cudnnFilterDescriptor_t> {
public:
    DnnFilterDescriptor() : Handle(cudnnCreateFilterDescriptor, cudnnDestroyFilterDescriptor) {}
    auto&& set(cudnnDataType_t dataType, cudnnTensorFormat_t format, int nbDims, const int filterDimA[]) {
        throwIfError(cudnnSetFilterNdDescriptor(get(), dataType, format, nbDims, filterDimA));
        return std::move(*this);
    }
};

class DnnRnnDataDescriptor
    : public Handle<cudnnRNNDataDescriptor_t> {
public:
    DnnRnnDataDescriptor() : Handle(cudnnCreateRNNDataDescriptor, cudnnDestroyRNNDataDescriptor) {}
    auto&& set(cudnnDataType_t dataType,
               cudnnRNNDataLayout_t layout,
               int maxSeqLength,
               int batchSize,
               int vectorSize,
               const int seqLengthArray[],
               void* paddingFill) {
        throwIfError(cudnnSetRNNDataDescriptor(
            get(), dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill));
        return std::move(*this);
    }
};

class DnnConvolutionDescriptor : public Handle<cudnnConvolutionDescriptor_t> {
public:
    DnnConvolutionDescriptor() : Handle(cudnnCreateConvolutionDescriptor, cudnnDestroyConvolutionDescriptor) {}
    auto&& set(int arrayLength,
               const int padA[],
               const int filterStrideA[],
               const int dilationA[],
               cudnnConvolutionMode_t mode,
               cudnnDataType_t dataType) {
        throwIfError(
            cudnnSetConvolutionNdDescriptor(get(), arrayLength, padA, filterStrideA, dilationA, mode, dataType));
        return std::move(*this);
    }
};

class DnnRnnDescriptor : public Handle<cudnnRNNDescriptor_t> {
public:
    DnnRnnDescriptor() : Handle(cudnnCreateRNNDescriptor, cudnnDestroyRNNDescriptor) {}
    auto&& set(cudnnRNNAlgo_t algo,
               cudnnRNNMode_t cellMode,
               cudnnRNNBiasMode_t biasMode,
               cudnnDirectionMode_t dirMode,
               cudnnRNNInputMode_t inputMode,
               cudnnDataType_t dataType,
               cudnnDataType_t mathPrec,
               cudnnMathType_t mathType,
               int32_t inputSize,
               int32_t hiddenSize,
               int32_t projSize,
               int32_t numLayers,
               cudnnDropoutDescriptor_t dropoutDesc,
               uint32_t auxFlags) {
        throwIfError(cudnnSetRNNDescriptor_v8(get(),
                                              algo,
                                              cellMode,
                                              biasMode,
                                              dirMode,
                                              inputMode,
                                              dataType,
                                              mathPrec,
                                              mathType,
                                              inputSize,
                                              hiddenSize,
                                              projSize,
                                              numLayers,
                                              dropoutDesc,
                                              auxFlags));
        return std::move(*this);
    }
    auto&& setClip(cudnnRNNClipMode_t clipMode, cudnnNanPropagation_t clipNanOpt, double lclip, double rclip) {
        throwIfError(cudnnRNNSetClip_v8(get(), clipMode, clipNanOpt, lclip, rclip));
        return std::move(*this);
    }
};

class DnnReduceTensorDescriptor : public Handle<cudnnReduceTensorDescriptor_t> {
public:
    DnnReduceTensorDescriptor()
        : Handle<cudnnReduceTensorDescriptor_t>(cudnnCreateReduceTensorDescriptor, cudnnDestroyReduceTensorDescriptor) {
    }
    auto&& set(cudnnReduceTensorOp_t op,
               cudnnDataType_t compType,
               cudnnNanPropagation_t nanOpt,
               cudnnReduceTensorIndices_t indices,
               cudnnIndicesType_t indicesType) {
        throwIfError(cudnnSetReduceTensorDescriptor(get(), op, compType, nanOpt, indices, indicesType));
        return std::move(*this);
    }
};

class DnnReduceAddDescriptor : public DnnReduceTensorDescriptor {
public:
    explicit DnnReduceAddDescriptor(cudnnDataType_t compType) {
        set(CUDNN_REDUCE_TENSOR_ADD,
            compType,
            CUDNN_PROPAGATE_NAN,
            CUDNN_REDUCE_TENSOR_NO_INDICES,
            CUDNN_32BIT_INDICES);
    }
};

class DnnReduceAvgDescriptor : public DnnReduceTensorDescriptor {
public:
    explicit DnnReduceAvgDescriptor(cudnnDataType_t compType) {
        set(CUDNN_REDUCE_TENSOR_AVG,
            compType,
            CUDNN_PROPAGATE_NAN,
            CUDNN_REDUCE_TENSOR_NO_INDICES,
            CUDNN_32BIT_INDICES);
    }
};

class DnnScaleFactor {
public:
    constexpr const void* get() const noexcept { return scaling_factor_; }

protected:
    explicit constexpr DnnScaleFactor(const void* scalingFactor) noexcept : scaling_factor_{scalingFactor} {}

private:
    const void* scaling_factor_;
};

class DnnScaleFactorZero : public DnnScaleFactor {
public:
    explicit constexpr DnnScaleFactorZero(cudnnDataType_t compType) noexcept
        : DnnScaleFactor{&CUDA::NumericConst<CUDA::constants::zero>(compType)} {}
};

class DnnScaleFactorOne : public DnnScaleFactor {
public:
    explicit constexpr DnnScaleFactorOne(cudnnDataType_t compType) noexcept
        : DnnScaleFactor{&CUDA::NumericConst<CUDA::constants::one>(compType)} {}
};

class DnnHandle : public Handle<cudnnHandle_t> {
public:
    DnnHandle() : Handle(cudnnCreate, cudnnDestroy) {}
    void setStream(const Stream& stream) { throwIfError(cudnnSetStream(get(), stream.get())); }
    void opTensor(const DnnOpTensorDescriptor& opTensorDesc,
                  const void* alpha1,
                  const DnnTensorDescriptor& aDesc,
                  const void* A,
                  const void* alpha2,
                  const DnnTensorDescriptor& bDesc,
                  const void* B,
                  const void* beta,
                  const DnnTensorDescriptor& cDesc,
                  void* C) const {
        throwIfError(cudnnOpTensor(
            get(), opTensorDesc.get(), alpha1, aDesc.get(), A, alpha2, bDesc.get(), B, beta, cDesc.get(), C));
    }
    // TODO: accept device pointers for x and y
    void activationForward(const DnnActivationDescriptor& activationDesc,
                           const void* alpha,
                           const DnnTensorDescriptor& xDesc,
                           const void* x,
                           const void* beta,
                           const DnnTensorDescriptor& yDesc,
                           void* y) const {
        throwIfError(cudnnActivationForward(get(), activationDesc.get(), alpha, xDesc.get(), x, beta, yDesc.get(), y));
    }
    void rnnForward(const DnnRnnDescriptor& rnnDesc,
                    cudnnForwardMode_t fwdMode,
                    const int32_t devSeqLengths[],
                    const DnnRnnDataDescriptor& xDesc,
                    const void* x,
                    const DnnRnnDataDescriptor& yDesc,
                    void* y,
                    const DnnTensorDescriptor& hDesc,
                    const void* hx,
                    void* hy,
                    std::optional<DnnTensorDescriptor::CRef> cDesc,
                    const void* cx,
                    void* cy,
                    size_t weightSpaceSize,
                    const void* weightSpace,
                    size_t workSpaceSize,
                    void* workSpace,
                    size_t reserveSpaceSize,
                    void* reserveSpace) const {
        throwIfError(cudnnRNNForward(get(),
                                     rnnDesc.get(),
                                     fwdMode,
                                     devSeqLengths,
                                     xDesc.get(),
                                     x,
                                     yDesc.get(),
                                     y,
                                     hDesc.get(),
                                     hx,
                                     hy,
                                     cDesc ? cDesc->get().get() : nullptr,
                                     cx,
                                     cy,
                                     weightSpaceSize,
                                     weightSpace,
                                     workSpaceSize,
                                     workSpace,
                                     reserveSpaceSize,
                                     reserveSpace));
    }
    size_t getReductionWorkspaceSize(const DnnReduceTensorDescriptor& reduceDesc,
                                     const DnnTensorDescriptor& aDesc,
                                     const DnnTensorDescriptor& cDesc) const {
        return createLastArg(cudnnGetReductionWorkspaceSize, get(), reduceDesc, aDesc, cDesc);
    }
    void reduceTensor(const DnnReduceTensorDescriptor& reduceTensorDesc,
                      CUDA::DeviceBuffer<std::uint8_t> workspace,
                      const DnnScaleFactor& alpha,
                      const DnnTensorDescriptor& aDesc,
                      CUDA::DevicePointer<const void*> a,
                      const DnnScaleFactor& beta,
                      const DnnTensorDescriptor& cDesc,
                      CUDA::DevicePointer<void*> c) const {
        throwIfError(cudnnReduceTensor(get(),
                                       reduceTensorDesc.get(),
                                       nullptr,
                                       0,
                                       workspace.data(),
                                       workspace.size_bytes(),
                                       alpha.get(),
                                       aDesc.get(),
                                       a.get(),
                                       beta.get(),
                                       cDesc.get(),
                                       c.get()));
    }
};

}  // namespace CUDA
