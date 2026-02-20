// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_cudnn_components.hpp"

#include <error.hpp>
#include <openvino/core/except.hpp>
#include <ops/converters.hpp>

#define CUDNN_VERSION_MIN(major, minor, patch) (CUDNN_VERSION >= ((major)*1000 + (minor)*100 + (patch)))

namespace ov::nvidia_gpu::RNN::Details {

LSTMCellParamsCuDnn::LSTMCellParamsCuDnn(const CreationContext& context, const LSTMCellParams& params)
    : lstm_cell_params_(params),
      data_type_{convertDataType<cudnnDataType_t>(lstm_cell_params_.element_type_)},
      element_size_{ov::nvidia_gpu::elementSize(data_type_)},
      is_half_supported_(CUDA::isHalfSupported(context.device())) {
    if (inputSize() == 1 && hiddenSize() == 1) {
        throw_ov_exception(
            "Currently LSTMCell cuDNN implementation doesn't support combination of "
            "input_size == 1 and hidden_size == 1 simultaneously");
    }

    const auto supported_activations = std::vector<std::string>{"sigmoid", "tanh", "tanh"};
    if (lstm_cell_params_.activations_ != supported_activations) {
        throw_ov_exception(
            "Currently LSTMCell cuDNN implementation supports only default LSTM activations of "
            "\"sigmoid\", \"tanh\", \"tanh\"");
    }

    const auto supported_alphas = std::vector<float>{1.0f, 1.0f, 1.0f};
    const auto supported_betas = std::vector<float>{0.0f, 0.0f, 0.0f};

    const bool are_supported_alphas =
        lstm_cell_params_.activations_alpha_.size() == 0 || lstm_cell_params_.activations_alpha_ == supported_alphas;

    const bool are_supported_betas =
        lstm_cell_params_.activations_beta_.size() == 0 || lstm_cell_params_.activations_beta_ == supported_betas;

    if (!are_supported_alphas || !are_supported_betas) {
        throw_ov_exception(
            "Currently LSTMCell cuDNN implementation supports only default activation "
            "alphas = {1.0f, 1.0f, 1.0f} and betas = {0.0f, 0.0f, 0.0f}");
    }

    OPENVINO_ASSERT(lstm_cell_params_.element_type_.size() == elementSize());
    OPENVINO_ASSERT(sizeof(int32_t) == sizeof(int));
    seq_length_array_.resize(batchSize(), maxSeqLength());
}

int32_t LSTMCellParamsCuDnn::numLayers() const { return 1; }

int32_t LSTMCellParamsCuDnn::projSize() const { return hiddenSize(); }

int LSTMCellParamsCuDnn::maxSeqLength() const { return 1; }

cudnnRNNDataLayout_t LSTMCellParamsCuDnn::layout() const { return CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED; }

// Paddings not used for single forward cell
void* LSTMCellParamsCuDnn::paddingFill() const { return nullptr; }

cudnnForwardMode_t LSTMCellParamsCuDnn::dnnForwardMode() const { return CUDNN_FWD_MODE_INFERENCE; }

CUDA::DnnRnnDescriptor LSTMCellParamsCuDnn::makeRNNDescriptor() const {
    const auto rnn_algo = CUDNN_RNN_ALGO_STANDARD;
    const auto rnn_mode = CUDNN_LSTM;
    const auto bias_mode = CUDNN_RNN_SINGLE_INP_BIAS;
    const auto dir_mode = CUDNN_UNIDIRECTIONAL;
    const auto input_mode = CUDNN_LINEAR_INPUT;
    const bool can_use_half = dataType() == CUDNN_DATA_HALF && isHalfSupported();
    const auto math_prec = (dataType() == CUDNN_DATA_DOUBLE || can_use_half) ? dataType() : CUDNN_DATA_FLOAT;

    // Possible optimization: down type conversion can be forced with CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION option
    // to utilize Tensor Cores on the supported devices at the price of precision
    const auto math_type = dataType() == CUDNN_DATA_DOUBLE ? CUDNN_DEFAULT_MATH : CUDNN_TENSOR_OP_MATH;

    // A single layer network will have no dropout applied. Dropout is used in the training mode only.
    const cudnnDropoutDescriptor_t drop_out_desc = nullptr;
    const uint32_t aux_flags = CUDNN_RNN_PADDED_IO_DISABLED;
    CUDA::DnnRnnDescriptor rnn_desc;
    rnn_desc.set(rnn_algo,
                 rnn_mode,
                 bias_mode,
                 dir_mode,
                 input_mode,
                 dataType(),
                 math_prec,
                 math_type,
                 inputSize(),
                 hiddenSize(),
                 projSize(),
                 numLayers(),
                 drop_out_desc,
                 aux_flags);

    const bool is_clipped = clip() != 0.0f && !std::isinf(clip());
    if (is_clipped) {
        throw_ov_exception("Currently LSTMCell cuDNN implementation doesn't support clipping");
    }
    // TODO: If cuDNN starts supporting similar clipping as OpenVino, remove the 'throw' above and uncomment:
    // const auto clip_mode = is_clipped ? CUDNN_RNN_CLIP_MINMAX : CUDNN_RNN_CLIP_NONE;
    // const auto clip_nan_opt = CUDNN_PROPAGATE_NAN;
    // const auto lclip = -clip();
    // const auto rclip = clip();
    // rnn_desc.setClip(clip_mode, clip_nan_opt, lclip, rclip);
    return rnn_desc;
}

CUDA::DnnRnnDataDescriptor LSTMCellParamsCuDnn::makeXDescriptor() const {
    const auto x_vector_size = inputSize();
    return CUDA::DnnRnnDataDescriptor{}.set(
        dataType(), layout(), maxSeqLength(), batchSize(), x_vector_size, seq_length_array_.data(), paddingFill());
}

CUDA::DnnRnnDataDescriptor LSTMCellParamsCuDnn::makeYDescriptor() const {
    const auto y_vector_size = projSize();
    return CUDA::DnnRnnDataDescriptor{}.set(
        dataType(), layout(), maxSeqLength(), batchSize(), y_vector_size, seq_length_array_.data(), paddingFill());
}

CUDA::DnnTensorDescriptor LSTMCellParamsCuDnn::makeHDescriptor() const {
    const int h_dim_a[nbDims()] = {numLayers(), batchSize(), projSize()};
    const int h_stride_a[nbDims()] = {batchSize() * projSize(), projSize(), 1};
    return CUDA::DnnTensorDescriptor{}.set(dataType(), nbDims(), h_dim_a, h_stride_a);
}

CUDA::DnnTensorDescriptor LSTMCellParamsCuDnn::makeCDescriptor() const {
    const int c_dim_a[nbDims()] = {numLayers(), batchSize(), hiddenSize()};
    const int c_stride_a[nbDims()] = {batchSize() * hiddenSize(), hiddenSize(), 1};
    return CUDA::DnnTensorDescriptor{}.set(dataType(), nbDims(), c_dim_a, c_stride_a);
}

LSTMCellDescriptorsCuDnn::LSTMCellDescriptorsCuDnn(const LSTMCellParamsCuDnn& params)
    : params_{params},
      rnn_desc_{params_.makeRNNDescriptor()},
      x_desc_{params_.makeXDescriptor()},
      y_desc_{params_.makeYDescriptor()},
      h_desc_{params_.makeHDescriptor()},
      c_desc_{params_.makeCDescriptor()} {
    CUDA::DnnHandle dnn_handle{};
    weight_space_size_ = 0;
    throwIfError(cudnnGetRNNWeightSpaceSize(dnn_handle.get(), rnn_desc_.get(), &weight_space_size_));
    OPENVINO_ASSERT(weight_space_size_ >= params_.wHostBuffers().size_bytes() + params_.rHostBuffers().size_bytes() +
                                      params_.bHostBuffers().size_bytes());

    work_space_size_ = 0;
    size_t reserve_space_size = 0;
    throwIfError(cudnnGetRNNTempSpaceSizes(
        dnn_handle.get(), rnn_desc_.get(), dnnForwardMode(), x_desc_.get(), &work_space_size_, &reserve_space_size));
    // the returned size of the reserve space buffer will be zero when the fMode argument is CUDNN_FWD_MODE_INFERENCE
    OPENVINO_ASSERT(reserve_space_size == 0);
}

void LSTMCellDescriptorsCuDnn::initDevSeqLengthArray(DevPtr buffer) {
    // The devSeqLengths array must be stored in GPU memory as it is accessed asynchronously
    // by GPU kernels, possibly after the cudnnRNNForward() function exists
    CUDA::DefaultStream::stream().upload(buffer, params_.seqLengthArray(), params_.seqLengthArraySizeBytes());
}

void LSTMCellDescriptorsCuDnn::initWeightSpace(DevPtr buffer) {
    calculateWeightBuffers(buffer);

    const auto lin_layer_count = params_.linLayerCount();

    const auto w_host_layer_size = params_.wHostBuffers().size_bytes() / lin_layer_count;
    const auto r_host_layer_size = params_.rHostBuffers().size_bytes() / lin_layer_count;
    const auto b1_host_layer_size = params_.bHostBuffers().size_bytes() / lin_layer_count;

    const uint8_t* w_host_addr = params_.wHostBuffers().data();
    const uint8_t* r_host_addr = params_.rHostBuffers().data();
    const uint8_t* b1_host_addr = params_.bHostBuffers().data();

    const auto& stream = CUDA::DefaultStream::stream();

    for (int i = 0; i < lin_layer_count; ++i) {
        // OpenVINO: linear layer indices are FICO (forget, input, candidate, output)
        //      https://docs.openvino.ai/2022.1/openvino_docs_ops_sequence_LSTMCell_1.html
        // In cuDNN they are IFCO
        //      https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNWeightParams
        //
        // So we swap the first 2 buffers:
        const int j = (i == 0) ? 1 : ((i == 1) ? 0 : i);

        stream.upload(DevPtr{w_dev_buffers_[j].data()}, w_host_addr, w_host_layer_size);
        w_host_addr += w_host_layer_size;

        stream.upload(DevPtr{r_dev_buffers_[j].data()}, r_host_addr, r_host_layer_size);
        r_host_addr += r_host_layer_size;

        stream.upload(DevPtr{b1_dev_buffers_[j].data()}, b1_host_addr, b1_host_layer_size);
        b1_host_addr += b1_host_layer_size;

        // The 2nd bias data isn't used in OpenVino
        if (j < b2_dev_buffers_.size()) {
            stream.memset(DevPtr{b2_dev_buffers_[j].data()}, 0, b2_dev_buffers_[j].size_bytes());
        }
    }
}

bool LSTMCellDescriptorsCuDnn::weightBuffersFit(DevPtr buffer) const {
    auto weight_space = buffer.get();
    OPENVINO_ASSERT(weight_space);

    DevBuffers all_weights;
    all_weights.reserve(w_dev_buffers_.size() + r_dev_buffers_.size() + b1_dev_buffers_.size() +
                        b2_dev_buffers_.size());
    for (const auto v : w_dev_buffers_) {
        all_weights.push_back(v);
    }
    for (const auto v : r_dev_buffers_) {
        all_weights.push_back(v);
    }
    for (const auto v : b1_dev_buffers_) {
        all_weights.push_back(v);
    }
    for (const auto v : b2_dev_buffers_) {
        all_weights.push_back(v);
    }

    std::sort(all_weights.begin(), all_weights.end(), [](DevBuffers::value_type a, DevBuffers::value_type b) {
        return a.data() < b.data();
    });

    const auto size = all_weights.size();
    uint8_t* addr_a = static_cast<uint8_t*>(all_weights[0].data());
    uint8_t* addr_b = static_cast<uint8_t*>(buffer.get());

    if (addr_a < addr_b) {
        return false;
    }

    addr_a = static_cast<uint8_t*>(all_weights[size - 1].data());
    size_t size_a = all_weights[size - 1].size_bytes();
    addr_b = static_cast<uint8_t*>(buffer.get());

    if (addr_a + size_a > addr_b + weightSpaceSize()) {
        return false;
    }

    for (int i = 1; i < size; ++i) {
        addr_a = static_cast<uint8_t*>(all_weights[i - 1].data());
        size_a = all_weights[i - 1].size_bytes();
        addr_b = static_cast<uint8_t*>(all_weights[i].data());
        if (addr_a + size_a > addr_b) {
            return false;
        }
    }

    return true;
}

void LSTMCellDescriptorsCuDnn::calculateWeightBuffers(DevPtr buffer) {
    constexpr int32_t pseudo_layer = 0;

    auto weight_space = buffer.get();
    OPENVINO_ASSERT(weight_space);

    const auto data_type = params_.dataType();
    const auto input_size = params_.inputSize();
    const auto hidden_size = params_.hiddenSize();
    const auto element_size = params_.elementSize();

    w_dev_buffers_.clear();
    r_dev_buffers_.clear();
    b1_dev_buffers_.clear();
    b2_dev_buffers_.clear();
    size_t w_total_bytes = 0;
    size_t r_total_bytes = 0;
    size_t b1_total_bytes = 0;
    size_t b2_total_bytes = 0;

    CUDA::DnnHandle dnn_handle{};
    int32_t lin_layer_id = 0;
    const int wb_nb_dims_requested = 3;
    cudnnDataType_t wb_data_type;
    int wb_nb_dims = 0;
    int wb_dim_a[wb_nb_dims_requested] = {};
    int wb_stride_a[wb_nb_dims_requested] = {};
    size_t tensor_size_bytes = 0;

    const auto lin_layer_count = params_.linLayerCount();
    for (int i = 0; i < lin_layer_count; ++i) {
        lin_layer_id = i;
        CUDA::DnnTensorDescriptor w_desc{};
        CUDA::DnnTensorDescriptor b1_desc{};
        void* w_addr = nullptr;
        void* b1_addr = nullptr;
        throwIfError(cudnnGetRNNWeightParams(dnn_handle.get(),
                                             rnn_desc_.get(),
                                             pseudo_layer,
                                             weight_space_size_,
                                             weight_space,
                                             lin_layer_id,
                                             w_desc.get(),
                                             &w_addr,
                                             b1_desc.get(),
                                             &b1_addr));
        OPENVINO_ASSERT(w_addr);
        w_desc.getTensorNdDescriptor(wb_nb_dims_requested, wb_data_type, wb_nb_dims, wb_dim_a, wb_stride_a);
        OPENVINO_ASSERT(wb_nb_dims == wb_nb_dims_requested);
        OPENVINO_ASSERT(wb_data_type == data_type);
        OPENVINO_ASSERT(wb_dim_a[0] == 1 && wb_dim_a[1] == hidden_size && wb_dim_a[2] == input_size);
        OPENVINO_ASSERT(wb_stride_a[0] == hidden_size * input_size && wb_stride_a[1] == input_size &&
                        wb_stride_a[2] == 1);
        tensor_size_bytes = w_desc.getTensorSizeInBytes();
        OPENVINO_ASSERT(tensor_size_bytes >= hidden_size * input_size * element_size);
        w_dev_buffers_.emplace_back(static_cast<uint8_t*>(w_addr), tensor_size_bytes);
        w_total_bytes += tensor_size_bytes;

        OPENVINO_ASSERT(b1_addr);
        b1_desc.getTensorNdDescriptor(wb_nb_dims_requested, wb_data_type, wb_nb_dims, wb_dim_a, wb_stride_a);
        OPENVINO_ASSERT(wb_nb_dims == wb_nb_dims_requested);
        OPENVINO_ASSERT(wb_data_type == data_type);
        OPENVINO_ASSERT(wb_dim_a[0] == 1 && wb_dim_a[1] == hidden_size && wb_dim_a[2] == 1);
        OPENVINO_ASSERT(wb_stride_a[0] == hidden_size && wb_stride_a[1] == 1 && wb_stride_a[2] == 1);
        tensor_size_bytes = b1_desc.getTensorSizeInBytes();
        OPENVINO_ASSERT(tensor_size_bytes >= hidden_size * element_size);
        b1_dev_buffers_.emplace_back(static_cast<uint8_t*>(b1_addr), tensor_size_bytes);
        b1_total_bytes += tensor_size_bytes;

        lin_layer_id = i + lin_layer_count;
        CUDA::DnnTensorDescriptor r_desc{};
        CUDA::DnnTensorDescriptor b2_desc{};
        void* r_addr = nullptr;
        void* b2_addr = nullptr;
        throwIfError(cudnnGetRNNWeightParams(dnn_handle.get(),
                                             rnn_desc_.get(),
                                             pseudo_layer,
                                             weight_space_size_,
                                             weight_space,
                                             lin_layer_id,
                                             r_desc.get(),
                                             &r_addr,
                                             b2_desc.get(),
                                             &b2_addr));
        OPENVINO_ASSERT(r_addr);
        r_desc.getTensorNdDescriptor(wb_nb_dims_requested, wb_data_type, wb_nb_dims, wb_dim_a, wb_stride_a);
        OPENVINO_ASSERT(wb_nb_dims == wb_nb_dims_requested);
        OPENVINO_ASSERT(wb_data_type == data_type);
        OPENVINO_ASSERT(wb_dim_a[0] == 1 && wb_dim_a[1] == hidden_size && wb_dim_a[2] == hidden_size);
        OPENVINO_ASSERT(wb_stride_a[0] == hidden_size * hidden_size && wb_stride_a[1] == hidden_size &&
                        wb_stride_a[2] == 1);
        tensor_size_bytes = r_desc.getTensorSizeInBytes();
        OPENVINO_ASSERT(tensor_size_bytes >= hidden_size * hidden_size * element_size);
        r_dev_buffers_.emplace_back(static_cast<uint8_t*>(r_addr), tensor_size_bytes);
        r_total_bytes += tensor_size_bytes;

        if (b2_addr) {
            b2_desc.getTensorNdDescriptor(wb_nb_dims_requested, wb_data_type, wb_nb_dims, wb_dim_a, wb_stride_a);
            tensor_size_bytes = b2_desc.getTensorSizeInBytes();
            b2_dev_buffers_.emplace_back(static_cast<uint8_t*>(b2_addr), tensor_size_bytes);
            b2_total_bytes += tensor_size_bytes;
        }
    }

    OPENVINO_ASSERT(weightBuffersFit(buffer));
    OPENVINO_ASSERT(w_total_bytes >= params_.wHostBuffers().size_bytes() &&
                    r_total_bytes >= params_.rHostBuffers().size_bytes() &&
                    b1_total_bytes >= params_.bHostBuffers().size_bytes());
    OPENVINO_ASSERT(weight_space_size_ >= w_total_bytes + r_total_bytes + b1_total_bytes + b2_total_bytes);
}

GRUCellParamsCuDnn::GRUCellParamsCuDnn(const CreationContext& context, const GRUCellParams& params)
    : gru_cell_params_(params),
      data_type_{convertDataType<cudnnDataType_t>(gru_cell_params_.element_type_)},
      element_size_{ov::nvidia_gpu::elementSize(data_type_)},
      is_half_supported_(CUDA::isHalfSupported(context.device())) {
    const auto supported_activations = std::vector<std::string>{"sigmoid", "tanh"};
    if (gru_cell_params_.activations_ != supported_activations) {
        throw_ov_exception(
            "Currently GRUCell cuDNN implementation supports only default GRU activations of \"sigmoid\", \"tanh\"");
    }

    const auto supported_alphas = std::vector<float>{1.0f, 1.0f};
    const auto supported_betas = std::vector<float>{0.0f, 0.0f};

    const bool are_supported_alphas =
        gru_cell_params_.activations_alpha_.size() == 0 || gru_cell_params_.activations_alpha_ == supported_alphas;

    const bool are_supported_betas =
        gru_cell_params_.activations_beta_.size() == 0 || gru_cell_params_.activations_beta_ == supported_betas;

    if (!are_supported_alphas || !are_supported_betas) {
        throw_ov_exception(
            "Currently GRUCell cuDNN implementation supports only default activation "
            "alphas = {1.0f, 1.0f} and betas = {0.0f, 0.0f}");
    }

    OPENVINO_ASSERT(gru_cell_params_.element_type_.size() == elementSize());
    OPENVINO_ASSERT(sizeof(int32_t) == sizeof(int));
    seq_length_array_.resize(batchSize(), maxSeqLength());
}

int32_t GRUCellParamsCuDnn::numLayers() const { return 1; }

int32_t GRUCellParamsCuDnn::projSize() const { return hiddenSize(); }

int GRUCellParamsCuDnn::maxSeqLength() const { return 1; }

cudnnRNNDataLayout_t GRUCellParamsCuDnn::layout() const { return CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED; }

// Paddings not used for single forward cell
void* GRUCellParamsCuDnn::paddingFill() const { return nullptr; }

cudnnForwardMode_t GRUCellParamsCuDnn::dnnForwardMode() const { return CUDNN_FWD_MODE_INFERENCE; }

CUDA::DnnRnnDescriptor GRUCellParamsCuDnn::makeRNNDescriptor() const {
// now standard algo cause issue under cudnn 8.1.X version
// see https://forums.developer.nvidia.com/t/cudnn-crash-in-v8-1-x/194346
#if CUDNN_VERSION_MIN(8, 2, 0)
    const auto rnn_algo = CUDNN_RNN_ALGO_STANDARD;
#else
    const auto rnn_algo = CUDNN_RNN_ALGO_PERSIST_STATIC;
#endif
    const auto rnn_mode = CUDNN_GRU;
    const auto bias_mode = gru_cell_params_.linear_before_reset_ ? CUDNN_RNN_DOUBLE_BIAS : CUDNN_RNN_SINGLE_INP_BIAS;
    const auto dir_mode = CUDNN_UNIDIRECTIONAL;
    const auto input_mode = CUDNN_LINEAR_INPUT;
    const bool can_use_half = dataType() == CUDNN_DATA_HALF && isHalfSupported();
    const auto math_prec = (dataType() == CUDNN_DATA_DOUBLE || can_use_half) ? dataType() : CUDNN_DATA_FLOAT;

    // Possible optimization: down type conversion can be forced with CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION option
    // to utilize Tensor Cores on the supported devices at the price of precision
    const auto math_type =
        dataType() == CUDNN_DATA_DOUBLE || dataType() == CUDNN_DATA_FLOAT ? CUDNN_DEFAULT_MATH : CUDNN_TENSOR_OP_MATH;

    // A single layer network will have no dropout applied. Dropout is used in the training mode only.
    const cudnnDropoutDescriptor_t drop_out_desc = nullptr;
    const uint32_t aux_flags = CUDNN_RNN_PADDED_IO_DISABLED;
    CUDA::DnnRnnDescriptor rnn_desc;
    rnn_desc.set(rnn_algo,
                 rnn_mode,
                 bias_mode,
                 dir_mode,
                 input_mode,
                 dataType(),
                 math_prec,
                 math_type,
                 inputSize(),
                 hiddenSize(),
                 projSize(),
                 numLayers(),
                 drop_out_desc,
                 aux_flags);

    const bool is_clipped = clip() != 0.0f && !std::isinf(clip());
    if (is_clipped) {
        throw_ov_exception("Currently LSTMCell cuDNN implementation doesn't support clipping");
    }
    // TODO: If cuDNN starts supporting similar clipping as OpenVino, remove the 'throw' above and uncomment:
    // const auto clip_mode = is_clipped ? CUDNN_RNN_CLIP_MINMAX : CUDNN_RNN_CLIP_NONE;
    // const auto clip_nan_opt = CUDNN_PROPAGATE_NAN;
    // const auto lclip = -clip();
    // const auto rclip = clip();
    // rnn_desc.setClip(clip_mode, clip_nan_opt, lclip, rclip);
    return rnn_desc;
}

CUDA::DnnRnnDataDescriptor GRUCellParamsCuDnn::makeXDescriptor() const {
    const auto x_vector_size = inputSize();
    return CUDA::DnnRnnDataDescriptor{}.set(
        dataType(), layout(), maxSeqLength(), batchSize(), x_vector_size, seq_length_array_.data(), paddingFill());
}

CUDA::DnnRnnDataDescriptor GRUCellParamsCuDnn::makeYDescriptor() const {
    const auto y_vector_size = projSize();
    return CUDA::DnnRnnDataDescriptor{}.set(
        dataType(), layout(), maxSeqLength(), batchSize(), y_vector_size, seq_length_array_.data(), paddingFill());
}

CUDA::DnnTensorDescriptor GRUCellParamsCuDnn::makeHDescriptor() const {
    const int h_dim_a[nbDims()] = {numLayers(), batchSize(), projSize()};
    const int h_stride_a[nbDims()] = {batchSize() * projSize(), projSize(), 1};
    return CUDA::DnnTensorDescriptor{}.set(dataType(), nbDims(), h_dim_a, h_stride_a);
}

GRUCellDescriptorsCuDnn::GRUCellDescriptorsCuDnn(const GRUCellParamsCuDnn& params)
    : params_{params},
      rnn_desc_{params_.makeRNNDescriptor()},
      x_desc_{params_.makeXDescriptor()},
      y_desc_{params_.makeYDescriptor()},
      h_desc_{params_.makeHDescriptor()} {
    CUDA::DnnHandle dnn_handle{};
    weight_space_size_ = 0;
    throwIfError(cudnnGetRNNWeightSpaceSize(dnn_handle.get(), rnn_desc_.get(), &weight_space_size_));
    OPENVINO_ASSERT(weight_space_size_ >= params_.wHostBuffers().size_bytes() + params_.rHostBuffers().size_bytes() +
                                      params_.bHostBuffers().size_bytes());

    work_space_size_ = 0;
    size_t reserve_space_size = 0;
    throwIfError(cudnnGetRNNTempSpaceSizes(
        dnn_handle.get(), rnn_desc_.get(), dnnForwardMode(), x_desc_.get(), &work_space_size_, &reserve_space_size));
    // the returned size of the reserve space buffer will be zero when the fMode argument is CUDNN_FWD_MODE_INFERENCE
    OPENVINO_ASSERT(reserve_space_size == 0);
}

void GRUCellDescriptorsCuDnn::initDevSeqLengthArray(DevPtr buffer) {
    // The devSeqLengths array must be stored in GPU memory as it is accessed asynchronously
    // by GPU kernels, possibly after the cudnnRNNForward() function exists
    CUDA::DefaultStream::stream().upload(buffer, params_.seqLengthArray(), params_.seqLengthArraySizeBytes());
}

void GRUCellDescriptorsCuDnn::initWeightSpace(DevPtr buffer) {
    calculateWeightBuffers(buffer);

    const size_t lin_layer_count = params_.linLayerCount();

    const auto w_host_layer_size = params_.wHostBuffers().size_bytes() / lin_layer_count;
    const auto r_host_layer_size = params_.rHostBuffers().size_bytes() / lin_layer_count;
    // TODO: make it more generic
    const size_t b1_host_layer_size = params_.hiddenSize() * params_.elementSize();

    const uint8_t* w_host_addr = params_.wHostBuffers().data();
    const uint8_t* r_host_addr = params_.rHostBuffers().data();
    const uint8_t* b1_host_addr = params_.bHostBuffers().data();

    const auto& stream = CUDA::DefaultStream::stream();

    for (size_t i = 0; i < lin_layer_count; ++i) {
        // openvino and CUDA use different order of bias weights
        // swap the first and second row
        const int j = (i == 0) ? 1 : ((i == 1) ? 0 : i);

        stream.upload(DevPtr{w_dev_buffers_[j].data()}, w_host_addr, w_host_layer_size);
        w_host_addr += w_host_layer_size;

        stream.upload(DevPtr{r_dev_buffers_[j].data()}, r_host_addr, r_host_layer_size);
        r_host_addr += r_host_layer_size;

        stream.upload(DevPtr{b1_dev_buffers_[j].data()}, b1_host_addr, b1_host_layer_size);
        b1_host_addr += b1_host_layer_size;

        if (i < b2_dev_buffers_.size()) {
            if (params_.linearBeforeReset() && i == 2) {
                // the tensor shape is [4 * hidden_size], get the fourth element from an array
                const size_t idx = b1_host_layer_size * (i + 1);
                OPENVINO_ASSERT(idx + b1_host_layer_size <= params_.bHostBuffers().size_bytes());
                const uint8_t* b2_host_addr = params_.bHostBuffers().data() + idx;
                stream.upload(DevPtr{b2_dev_buffers_[j].data()}, b2_host_addr, b1_host_layer_size);
            } else {
                stream.memset(DevPtr{b2_dev_buffers_[j].data()}, 0, b2_dev_buffers_[j].size_bytes());
            }
        }
    }
}

bool GRUCellDescriptorsCuDnn::weightBuffersFit(DevPtr buffer) const {
    auto weight_space = buffer.get();
    OPENVINO_ASSERT(weight_space);

    DevBuffers all_weights;
    all_weights.reserve(w_dev_buffers_.size() + r_dev_buffers_.size() + b1_dev_buffers_.size() +
                        b2_dev_buffers_.size());
    for (const auto v : w_dev_buffers_) {
        all_weights.push_back(v);
    }
    for (const auto v : r_dev_buffers_) {
        all_weights.push_back(v);
    }
    for (const auto v : b1_dev_buffers_) {
        all_weights.push_back(v);
    }
    for (const auto v : b2_dev_buffers_) {
        all_weights.push_back(v);
    }

    std::sort(all_weights.begin(), all_weights.end(), [](DevBuffers::value_type a, DevBuffers::value_type b) {
        return a.data() < b.data();
    });

    const auto size = all_weights.size();
    uint8_t* addr_a = static_cast<uint8_t*>(all_weights[0].data());
    uint8_t* addr_b = static_cast<uint8_t*>(buffer.get());

    if (addr_a < addr_b) {
        return false;
    }

    addr_a = static_cast<uint8_t*>(all_weights[size - 1].data());
    size_t size_a = all_weights[size - 1].size_bytes();
    addr_b = static_cast<uint8_t*>(buffer.get());

    if (addr_a + size_a > addr_b + weightSpaceSize()) {
        return false;
    }

    for (int i = 1; i < size; ++i) {
        addr_a = static_cast<uint8_t*>(all_weights[i - 1].data());
        size_a = all_weights[i - 1].size_bytes();
        addr_b = static_cast<uint8_t*>(all_weights[i].data());
        if (addr_a + size_a > addr_b) {
            return false;
        }
    }

    return true;
}

void GRUCellDescriptorsCuDnn::calculateWeightBuffers(DevPtr buffer) {
    constexpr int32_t pseudo_layer = 0;

    auto weight_space = buffer.get();
    OPENVINO_ASSERT(weight_space);

    const auto data_type = params_.dataType();
    const auto input_size = params_.inputSize();
    const auto hidden_size = params_.hiddenSize();
    const auto element_size = params_.elementSize();

    w_dev_buffers_.clear();
    r_dev_buffers_.clear();
    b1_dev_buffers_.clear();
    b2_dev_buffers_.clear();
    size_t w_total_bytes = 0;
    size_t r_total_bytes = 0;
    size_t b1_total_bytes = 0;
    size_t b2_total_bytes = 0;

    CUDA::DnnHandle dnn_handle{};
    int32_t lin_layer_id = 0;
    const int wb_nb_dims_requested = 3;
    cudnnDataType_t wb_data_type;
    int wb_nb_dims = 0;
    int wb_dim_a[wb_nb_dims_requested + 1] = {};
    int wb_stride_a[wb_nb_dims_requested + 1] = {};
    size_t tensor_size_bytes = 0;

    const auto lin_layer_count = params_.linLayerCount();
    for (int i = 0; i < lin_layer_count; ++i) {
        lin_layer_id = i;
        CUDA::DnnTensorDescriptor w_desc{};
        CUDA::DnnTensorDescriptor b1_desc{};
        void* w_addr = nullptr;
        void* b1_addr = nullptr;
        throwIfError(cudnnGetRNNWeightParams(dnn_handle.get(),
                                             rnn_desc_.get(),
                                             pseudo_layer,
                                             weight_space_size_,
                                             weight_space,
                                             lin_layer_id,
                                             w_desc.get(),
                                             &w_addr,
                                             b1_desc.get(),
                                             &b1_addr));
        OPENVINO_ASSERT(w_addr);
        w_desc.getTensorNdDescriptor(wb_nb_dims_requested, wb_data_type, wb_nb_dims, wb_dim_a, wb_stride_a);
        OPENVINO_ASSERT(wb_nb_dims == wb_nb_dims_requested);
        OPENVINO_ASSERT(wb_data_type == data_type);
        OPENVINO_ASSERT(wb_dim_a[0] == 1 && wb_dim_a[1] == hidden_size && wb_dim_a[2] == input_size);
        OPENVINO_ASSERT(wb_stride_a[0] == hidden_size * input_size && wb_stride_a[1] == input_size &&
                        wb_stride_a[2] == 1);
        tensor_size_bytes = w_desc.getTensorSizeInBytes();
        OPENVINO_ASSERT(tensor_size_bytes >= hidden_size * input_size * element_size);
        w_dev_buffers_.emplace_back(static_cast<uint8_t*>(w_addr), tensor_size_bytes);
        w_total_bytes += tensor_size_bytes;

        OPENVINO_ASSERT(b1_addr);
        b1_desc.getTensorNdDescriptor(wb_nb_dims_requested, wb_data_type, wb_nb_dims, wb_dim_a, wb_stride_a);
        OPENVINO_ASSERT(wb_nb_dims == wb_nb_dims_requested);
        OPENVINO_ASSERT(wb_data_type == data_type);
        OPENVINO_ASSERT(wb_dim_a[0] == 1 && wb_dim_a[1] == hidden_size && wb_dim_a[2] == 1);
        OPENVINO_ASSERT(wb_stride_a[0] == hidden_size && wb_stride_a[1] == 1 && wb_stride_a[2] == 1);
        tensor_size_bytes = b1_desc.getTensorSizeInBytes();
        OPENVINO_ASSERT(tensor_size_bytes >= hidden_size * element_size);
        b1_dev_buffers_.emplace_back(static_cast<uint8_t*>(b1_addr), tensor_size_bytes);
        b1_total_bytes += tensor_size_bytes;

        lin_layer_id = i + lin_layer_count;
        CUDA::DnnTensorDescriptor r_desc{};
        CUDA::DnnTensorDescriptor b2_desc{};
        void* r_addr = nullptr;
        void* b2_addr = nullptr;
        throwIfError(cudnnGetRNNWeightParams(dnn_handle.get(),
                                             rnn_desc_.get(),
                                             pseudo_layer,
                                             weight_space_size_,
                                             weight_space,
                                             lin_layer_id,
                                             r_desc.get(),
                                             &r_addr,
                                             b2_desc.get(),
                                             &b2_addr));
        OPENVINO_ASSERT(r_addr);
        r_desc.getTensorNdDescriptor(wb_nb_dims_requested, wb_data_type, wb_nb_dims, wb_dim_a, wb_stride_a);
        OPENVINO_ASSERT(wb_nb_dims == wb_nb_dims_requested);
        OPENVINO_ASSERT(wb_data_type == data_type);
        OPENVINO_ASSERT(wb_dim_a[0] == 1 && wb_dim_a[1] == hidden_size && wb_dim_a[2] == hidden_size);
        OPENVINO_ASSERT(wb_stride_a[0] == hidden_size * hidden_size && wb_stride_a[1] == hidden_size &&
                        wb_stride_a[2] == 1);
        tensor_size_bytes = r_desc.getTensorSizeInBytes();
        OPENVINO_ASSERT(tensor_size_bytes >= hidden_size * hidden_size * element_size);
        r_dev_buffers_.emplace_back(static_cast<uint8_t*>(r_addr), tensor_size_bytes);
        r_total_bytes += tensor_size_bytes;

        if (b2_addr) {
            b2_desc.getTensorNdDescriptor(wb_nb_dims_requested, wb_data_type, wb_nb_dims, wb_dim_a, wb_stride_a);
            tensor_size_bytes = b2_desc.getTensorSizeInBytes();
            b2_dev_buffers_.emplace_back(static_cast<uint8_t*>(b2_addr), tensor_size_bytes);
            b2_total_bytes += tensor_size_bytes;
        }
    }

    OPENVINO_ASSERT(weightBuffersFit(buffer));
    OPENVINO_ASSERT(w_total_bytes >= params_.wHostBuffers().size_bytes() &&
                    r_total_bytes >= params_.rHostBuffers().size_bytes());
    OPENVINO_ASSERT(weight_space_size_ >= w_total_bytes + r_total_bytes + b1_total_bytes + b2_total_bytes);
}

}  // namespace ov::nvidia_gpu::RNN::Details
