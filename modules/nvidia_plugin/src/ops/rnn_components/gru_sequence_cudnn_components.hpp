// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda/dnn.hpp>
#include <cuda_creation_context.hpp>

#include "gru_sequence_components.hpp"

namespace ov::nvidia_gpu::RNN::Details {

/**
 * @brief Presents GRU Sequence parameters in a form suitable for cuDNN API.
 *        Represents parameters from ngraph.
 */
class GRUSequenceParamsCuDnn {
public:
    GRUSequenceParamsCuDnn(const GRUSequenceParams& params);

    cudnnDataType_t element_type_;
    cudaDataType_t element_type_cuda_;
    size_t element_size_;
    cudnnDirectionMode_t direction_;
    bool linear_before_reset_;

    int32_t batch_size_;
    int32_t max_seq_length_;
    int32_t input_size_;
    int32_t hidden_size_;

    gsl::span<const uint8_t> w_host_buffers_;
    gsl::span<const uint8_t> r_host_buffers_;
    gsl::span<const uint8_t> b_host_buffers_;

    std::vector<int32_t> seq_length_array_;

    int32_t numDirections() const { return direction_ == CUDNN_BIDIRECTIONAL ? 2 : 1; }
    int32_t projSize() const { return hidden_size_; }
};

/**
 * @brief Prepares all data required for cuDNN LSTM Sequence API invocation.
 */
class GRUSequenceDescriptorsCuDnn {
public:
    using DevPtr = CUDA::DevicePointer<void*>;

    /// @brief Implementation specific configuration.
    struct Config {
        cudnnRNNDataLayout_t rnn_data_layout{CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED};
    };

    GRUSequenceDescriptorsCuDnn(const CreationContext& context,
                                const GRUSequenceParamsCuDnn& params,
                                const Config& config);

    cudnnForwardMode_t dnnForwardMode() const { return CUDNN_FWD_MODE_INFERENCE; }

    const CUDA::DnnRnnDescriptor& rnnDesc() const { return rnn_desc_; }
    const CUDA::DnnRnnDataDescriptor& xDesc() const { return x_desc_; }
    const CUDA::DnnRnnDataDescriptor& yDesc() const { return y_desc_; }
    const CUDA::DnnTensorDescriptor& hDesc() const { return h_desc_; }

    size_t seqLengthArraySizeBytes() const { return params_.seq_length_array_.size() * sizeof(int32_t); }
    void initDevSeqLengthArray(DevPtr buffer);

    size_t weightSpaceSize() const { return weight_space_size_; }
    void initWeightSpace(DevPtr buffer);

    size_t workSpaceSize() const { return work_space_size_; }

private:
    void createRNNDescriptor(const CreationContext& context);
    void createXDescriptor();
    void createYDescriptor();
    void createHDescriptor();

    bool weightBuffersFit(DevPtr buffer) const;
    void calculateWeightBuffers(DevPtr buffer);

private:
    const GRUSequenceParamsCuDnn& params_;
    const Config config_;

    CUDA::DnnRnnDescriptor rnn_desc_;
    CUDA::DnnRnnDataDescriptor x_desc_;
    CUDA::DnnRnnDataDescriptor y_desc_;
    CUDA::DnnTensorDescriptor h_desc_;

    size_t weight_space_size_;
    size_t work_space_size_;

    using DevBuffers = std::vector<CUDA::DeviceBuffer<uint8_t>>;
    DevBuffers w_dev_buffers_;
    DevBuffers r_dev_buffers_;
    DevBuffers b1_dev_buffers_;
    DevBuffers b2_dev_buffers_;
};

}  // namespace ov::nvidia_gpu::RNN::Details
