// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_operation_base.hpp>
#include <gsl/span>
#include <ngraph/op/lstm_sequence.hpp>
#include <transformer/nodes/lstm_sequence_optimized.hpp>

namespace CUDAPlugin::RNN::Details {

/**
 * @brief Defines tensor indices for `ngraph::op::v5::LSTMSequence` node.
 */
struct LSTMSequenceArgIndices {
    static constexpr size_t x = 0;
    static constexpr size_t hidden_input = 1;
    static constexpr size_t cell_input = 2;
    static constexpr size_t sequence_lengths = 3;
    static constexpr size_t weights = 4;
    static constexpr size_t recurrence_weights = 5;
    static constexpr size_t biases = 6;
    static constexpr size_t y = 0;
    static constexpr size_t hidden_output = 1;
    static constexpr size_t cell_output = 2;
};

/**
 * @brief Extracted and validated parameters from ngraph operation.
 */
struct LSTMSequenceParams {
    LSTMSequenceParams(const ngraph::op::v5::LSTMSequence& node);
    LSTMSequenceParams(const CUDAPlugin::nodes::LSTMSequenceOptimized& node);

    static constexpr int lin_layer_count = 4;

    ngraph::element::Type element_type_;
    using direction = ngraph::op::v5::LSTMSequence::direction;
    direction direction_;
    std::vector<std::string> activations_;
    std::vector<float> activations_alpha_;
    std::vector<float> activations_beta_;
    float clip_;

    gsl::span<const uint8_t> w_host_buffers_;
    gsl::span<const uint8_t> r_host_buffers_;
    gsl::span<const uint8_t> b_host_buffers_;

    size_t batch_size_;
    size_t max_seq_length_;
    size_t input_size_;
    size_t hidden_size_;

private:
    void validate(const ngraph::op::v5::LSTMSequence& node);
};

/**
 * Helps operation to handle multiple workbuffers. Allows to easily skip
 * some allocations without the need to track workbuffer indices.
 */
class WorkbufferDesc {
public:
    WorkbufferDesc() : bsize_{0}, index_{-1} {}
    void addRequest(std::vector<WorkbufferRequest::size_in_bytes_t>& workbuffers_sizes, size_t requested_bsize) {
        if (requested_bsize > 0) {
            bsize_ = requested_bsize;
            index_ = workbuffers_sizes.size();
            workbuffers_sizes.emplace_back(bsize_);
        }
    }
    size_t size() const { return bsize_; }
    template <typename T>
    T* optionalPtr(const std::vector<CUDA::DevicePointer<T*>>& buffers) const {
        if (index_ < 0) {
            return nullptr;
        }
        return requiredPtr(buffers);
    }
    template <typename T>
    T* requiredPtr(const std::vector<CUDA::DevicePointer<T*>>& buffers) const {
        Expects((index_ >= 0) && (index_ < buffers.size()));
        return buffers[index_].get();
    }
    operator bool() const { return index_ >= 0; }

private:
    size_t bsize_;
    int index_;
};

/**
 * Base class for `TransposeInputTensorAdapter` and `TransposeOutputTensorAdapter`
 *
 * TODO: Consider to refactor using `TransposeOp` using aggregation or by creating
 *       reusable class that will be used in both places.
 */
class TransposeTensorAdapterBase {
public:
    TransposeTensorAdapterBase(cudaDataType_t element_tpe,
                               size_t element_size,
                               const std::vector<int64_t>& src_shape,
                               const std::vector<int64_t>& dst_shape,
                               const std::vector<int>& mode);

    void requestWorkbuffer(std::vector<size_t>& workbuffers_sizes);

    void* dnnApiPtr(const std::vector<Workbuffers::mutable_buffer>& mutable_buffers) const;

protected:
    void execute(const InferenceRequestContext& context, const void* src, void* dst) const;

    WorkbufferDesc workbuffer_;

private:
    void initCuTensorDescriptor(const CUDA::CuTensorHandle& handle,
                                const std::vector<int64_t>& shape,
                                cutensorTensorDescriptor_t& desc) const;

    cudaDataType_t element_type_;
    size_t element_size_;
    std::vector<int64_t> src_shape_;
    std::vector<int64_t> dst_shape_;
    std::vector<int> src_mode_;
    std::vector<int> dst_mode_;
};

class TransposeInputTensorAdapter : public TransposeTensorAdapterBase {
public:
    using TransposeTensorAdapterBase::TransposeTensorAdapterBase;
    void execute(const InferenceRequestContext& context,
                 CUDA::DevicePointer<const void*> input,
                 const std::vector<Workbuffers::mutable_buffer>& dst) const;
};

class TransposeOutputTensorAdapter : public TransposeTensorAdapterBase {
public:
    using TransposeTensorAdapterBase::TransposeTensorAdapterBase;
    void execute(const InferenceRequestContext& context,
                 const std::vector<Workbuffers::mutable_buffer>& src,
                 CUDA::DevicePointer<void*> output) const;
};

}  // namespace CUDAPlugin::RNN::Details
