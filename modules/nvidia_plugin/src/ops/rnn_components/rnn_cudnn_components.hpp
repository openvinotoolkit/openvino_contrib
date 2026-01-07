// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/dnn.hpp>
#include <cuda_creation_context.hpp>

#include "rnn_components.hpp"

namespace ov::nvidia_gpu::RNN::Details {

/**
 * @brief Presents LSTM Cell parameters in a form suitable for cuDNN API.
 */
class LSTMCellParamsCuDnn {
public:
    LSTMCellParamsCuDnn(const CreationContext& context, const LSTMCellParams& params);

    int32_t numLayers() const;
    int32_t projSize() const;
    int maxSeqLength() const;
    cudnnRNNDataLayout_t layout() const;
    void* paddingFill() const;
    static constexpr int nbDims() { return 3; }

    cudnnForwardMode_t dnnForwardMode() const;

    CUDA::DnnRnnDescriptor makeRNNDescriptor() const;
    CUDA::DnnRnnDataDescriptor makeXDescriptor() const;
    CUDA::DnnRnnDataDescriptor makeYDescriptor() const;
    CUDA::DnnTensorDescriptor makeHDescriptor() const;
    CUDA::DnnTensorDescriptor makeCDescriptor() const;

    cudnnDataType_t dataType() const { return data_type_; }

    std::size_t elementSize() const { return element_size_; }

    int32_t inputSize() const { return static_cast<int32_t>(lstm_cell_params_.input_size_); }

    int32_t hiddenSize() const { return static_cast<int32_t>(lstm_cell_params_.hidden_size_); }

    int batchSize() const { return static_cast<int>(lstm_cell_params_.batch_size_); }

    double clip() const { return static_cast<double>(lstm_cell_params_.clip_); }

    int32_t linLayerCount() const { return static_cast<int32_t>(LSTMCellParams::lin_layer_count); }

    const int* seqLengthArray() { return seq_length_array_.data(); }

    std::size_t seqLengthArraySizeBytes() const { return seq_length_array_.size() * sizeof(int); }

    gsl::span<const uint8_t> wHostBuffers() const { return lstm_cell_params_.w_host_buffers_; }

    gsl::span<const uint8_t> rHostBuffers() const { return lstm_cell_params_.r_host_buffers_; }

    gsl::span<const uint8_t> bHostBuffers() const { return lstm_cell_params_.b_host_buffers_; }

    std::size_t ySizeBytes() const { return batchSize() * projSize() * elementSize(); }

    bool isHalfSupported() const { return is_half_supported_; }

private:
    const LSTMCellParams& lstm_cell_params_;
    const cudnnDataType_t data_type_;
    const std::size_t element_size_;
    const bool is_half_supported_;
    std::vector<int> seq_length_array_;
};

/**
 * @brief Prepares all data required for cuDNN LSTM Cell API invocation.
 */
class LSTMCellDescriptorsCuDnn {
public:
    using DevPtr = CUDA::DevicePointer<void*>;

    LSTMCellDescriptorsCuDnn(const LSTMCellParamsCuDnn& params);

    const CUDA::DnnRnnDescriptor& rnnDesc() const { return rnn_desc_; }
    const CUDA::DnnRnnDataDescriptor& xDesc() const { return x_desc_; }
    const CUDA::DnnRnnDataDescriptor& yDesc() const { return y_desc_; }
    const CUDA::DnnTensorDescriptor& hDesc() const { return h_desc_; }
    const CUDA::DnnTensorDescriptor& cDesc() const { return c_desc_; }

    size_t seqLengthArraySizeBytes() const { return params_.seqLengthArraySizeBytes(); }
    size_t weightSpaceSize() const { return weight_space_size_; }
    size_t ySizeBytes() const { return params_.ySizeBytes(); }
    size_t workSpaceSize() const { return work_space_size_; }
    cudnnForwardMode_t dnnForwardMode() const { return params_.dnnForwardMode(); }

    void initDevSeqLengthArray(DevPtr buffer);
    void initWeightSpace(DevPtr buffer);

private:
    using DevBuffers = std::vector<CUDA::DeviceBuffer<uint8_t>>;

    bool weightBuffersFit(DevPtr buffer) const;
    void calculateWeightBuffers(DevPtr buffer);

    LSTMCellParamsCuDnn params_;

    CUDA::DnnRnnDescriptor rnn_desc_;
    CUDA::DnnRnnDataDescriptor x_desc_;
    CUDA::DnnRnnDataDescriptor y_desc_;
    CUDA::DnnTensorDescriptor h_desc_;
    CUDA::DnnTensorDescriptor c_desc_;

    size_t weight_space_size_;
    size_t work_space_size_;

    DevBuffers w_dev_buffers_;
    DevBuffers r_dev_buffers_;
    DevBuffers b1_dev_buffers_;
    DevBuffers b2_dev_buffers_;
};

/**
 * @brief Presents LSTM Cell parameters in a form suitable for cuDNN API.
 */
class GRUCellParamsCuDnn {
public:
    GRUCellParamsCuDnn(const CreationContext& context, const GRUCellParams& params);

    int32_t numLayers() const;
    int32_t projSize() const;
    int maxSeqLength() const;
    cudnnRNNDataLayout_t layout() const;
    void* paddingFill() const;
    static constexpr int nbDims() { return 3; }

    cudnnForwardMode_t dnnForwardMode() const;

    CUDA::DnnRnnDescriptor makeRNNDescriptor() const;
    CUDA::DnnRnnDataDescriptor makeXDescriptor() const;
    CUDA::DnnRnnDataDescriptor makeYDescriptor() const;
    CUDA::DnnTensorDescriptor makeHDescriptor() const;

    cudnnDataType_t dataType() const { return data_type_; }

    std::size_t elementSize() const { return element_size_; }

    int32_t inputSize() const { return static_cast<int32_t>(gru_cell_params_.input_size_); }

    int32_t hiddenSize() const { return static_cast<int32_t>(gru_cell_params_.hidden_size_); }

    int batchSize() const { return static_cast<int>(gru_cell_params_.batch_size_); }

    double clip() const { return static_cast<double>(gru_cell_params_.clip_); }

    int32_t linLayerCount() const { return static_cast<int32_t>(GRUCellParams::lin_layer_count); }

    const int* seqLengthArray() { return seq_length_array_.data(); }

    std::size_t seqLengthArraySizeBytes() const { return seq_length_array_.size() * sizeof(int); }

    gsl::span<const uint8_t> wHostBuffers() const { return gru_cell_params_.w_host_buffers_; }

    gsl::span<const uint8_t> rHostBuffers() const { return gru_cell_params_.r_host_buffers_; }

    gsl::span<const uint8_t> bHostBuffers() const { return gru_cell_params_.b_host_buffers_; }

    std::size_t ySizeBytes() const { return batchSize() * projSize() * elementSize(); }

    bool isHalfSupported() const { return is_half_supported_; }

    bool linearBeforeReset() const { return gru_cell_params_.linear_before_reset_; }

private:
    const GRUCellParams& gru_cell_params_;
    const cudnnDataType_t data_type_;
    const std::size_t element_size_;
    const bool is_half_supported_;
    std::vector<int> seq_length_array_;
};

/**
 * @brief Prepares all data required for cuDNN LSTM Cell API invocation.
 */
class GRUCellDescriptorsCuDnn {
public:
    using DevPtr = CUDA::DevicePointer<void*>;

    GRUCellDescriptorsCuDnn(const GRUCellParamsCuDnn& params);

    const CUDA::DnnRnnDescriptor& rnnDesc() const { return rnn_desc_; }
    const CUDA::DnnRnnDataDescriptor& xDesc() const { return x_desc_; }
    const CUDA::DnnRnnDataDescriptor& yDesc() const { return y_desc_; }
    const CUDA::DnnTensorDescriptor& hDesc() const { return h_desc_; }

    size_t seqLengthArraySizeBytes() const { return params_.seqLengthArraySizeBytes(); }
    size_t weightSpaceSize() const { return weight_space_size_; }
    size_t ySizeBytes() const { return params_.ySizeBytes(); }
    size_t workSpaceSize() const { return work_space_size_; }
    cudnnForwardMode_t dnnForwardMode() const { return params_.dnnForwardMode(); }

    void initDevSeqLengthArray(DevPtr buffer);
    void initWeightSpace(DevPtr buffer);

private:
    using DevBuffers = std::vector<CUDA::DeviceBuffer<uint8_t>>;

    bool weightBuffersFit(DevPtr buffer) const;
    void calculateWeightBuffers(DevPtr buffer);

    GRUCellParamsCuDnn params_;

    CUDA::DnnRnnDescriptor rnn_desc_;
    CUDA::DnnRnnDataDescriptor x_desc_;
    CUDA::DnnRnnDataDescriptor y_desc_;
    CUDA::DnnTensorDescriptor h_desc_;

    size_t weight_space_size_;
    size_t work_space_size_;

    DevBuffers w_dev_buffers_;
    DevBuffers r_dev_buffers_;
    DevBuffers b1_dev_buffers_;
    DevBuffers b2_dev_buffers_;
};

}  // namespace ov::nvidia_gpu::RNN::Details
