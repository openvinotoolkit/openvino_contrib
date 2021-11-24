// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>
#include <arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {

enum ConvInput {Features, Weights, Bias};
template<typename Conv>
static auto ConvParameters(const Conv& node) {
    unsigned int pad_l    = node.get_pads_begin().at(D2::W);
    unsigned int pad_r    = node.get_pads_end().at(D2::W);
    unsigned int pad_t    = node.get_pads_begin().at(D2::H);
    unsigned int pad_b    = node.get_pads_end().at(D2::H);
    unsigned int stride_x = node.get_strides().at(D2::W);
    unsigned int stride_y = node.get_strides().at(D2::H);

    return std::make_pair(
        arm_compute::PadStrideInfo {stride_x, stride_y, pad_l, pad_r, pad_t, pad_b, arm_compute::DimensionRoundingType::FLOOR},
        arm_compute::Size2D {node.get_dilations().at(D2::W), node.get_dilations().at(D2::H)});
}

static arm_compute::ActivationLayerInfo GetActivationInfo(const ngraph::Node& node) {
    auto itInfo = node.get_rt_info().find("ActivationLayerInfo");
    if (itInfo != node.get_rt_info().end()) {
        return itInfo->second.as<arm_compute::ActivationLayerInfo>();
    } else {
        return {};
    }
}

struct NEConvolutionLayerQI final: public arm_compute::IFunction {
public:
    NEConvolutionLayerQI(std::shared_ptr<arm_compute::IMemoryManager> memory_manager = nullptr):
        _memory_manager(memory_manager), _conv(), _weights(nullptr), _wp(nullptr), _weightsqi(), _output(nullptr), _qi(nullptr), _outputqi() {}
    NEConvolutionLayerQI(const NEConvolutionLayerQI &) = delete;
    NEConvolutionLayerQI &operator=(const NEConvolutionLayerQI &) = delete;
    NEConvolutionLayerQI(NEConvolutionLayerQI &&) = delete;
    NEConvolutionLayerQI &operator=(NEConvolutionLayerQI &&) = delete;
    ~NEConvolutionLayerQI() = default;
    void configure(arm_compute::ITensor *input, const arm_compute::ITensor *weights, const arm_compute::ITensor *biases, arm_compute::ITensor *output,
                   const arm_compute::PadStrideInfo &conv_info, const arm_compute::WeightsInfo &weights_info, const arm_compute::Size2D &dilation,
                   const arm_compute::ActivationLayerInfo &act_info, const arm_compute::QuantizationInfo *wp, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
        ARM_COMPUTE_UNUSED(num_groups);
        ARM_COMPUTE_ERROR_THROW_ON(NEConvolutionLayerQI::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr),
                                                                  output->info(), conv_info, weights_info, dilation, act_info, wp, qi));

        _weights = weights;
        _wp = wp;
        if (_wp) {
            bool setPerChannel = _weights->info()->data_type() == QASYMM8_SIGNED ||
                                 _weights->info()->data_type() == QASYMM8 ||
                                 _weights->info()->data_type() == QSYMM8;
            _weightsqi.allocator()->init(setPerChannel ? _weights->info()->set_data_type(QSYMM8_PER_CHANNEL) : *(_weights->info()));
            _weightsqi.info()->set_quantization_info(*wp);
        }

        _output = output;
        _qi = qi;
        if (_qi) {
            _outputqi.allocator()->init(*(_output->info()));
            _outputqi.info()->set_quantization_info(*qi);
        }

        _conv = std::make_unique<arm_compute::NEConvolutionLayer>(_memory_manager);
        _conv->configure(input, _wp ? &_weightsqi : _weights, biases, _qi ? &_outputqi : _output, conv_info, weights_info, dilation, act_info);
    }
    static Status validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *weights, const arm_compute::ITensorInfo *biases,
                           const arm_compute::ITensorInfo *output, const arm_compute::PadStrideInfo &conv_info, const arm_compute::WeightsInfo &weights_info,
                           const arm_compute::Size2D &dilation, const arm_compute::ActivationLayerInfo &act_info,
                           const arm_compute::QuantizationInfo *wp, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(weights, output);
        //At the moment quantization info isn't checked actually, but just in case
        return arm_compute::NEConvolutionLayer::validate(input, &arm_compute::TensorInfo(*weights).set_quantization_info(wp), biases,
                                                         &arm_compute::TensorInfo(*output).set_quantization_info(qi),
                                                         conv_info, weights_info, dilation, act_info);
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_conv.get(), "Kernel didn't configured");
        if (_wp) {
            if (_weightsqi.info()->padding() != _weights->info()->padding()) _weightsqi.info()->extend_padding(_weights->info()->padding());
            _weightsqi.allocator()->import_memory(_weights->buffer());
        }
        if (_qi) {
            if (_outputqi.info()->padding() != _output->info()->padding()) _outputqi.info()->extend_padding(_output->info()->padding());
            _outputqi.allocator()->import_memory(_output->buffer());
        }
        _conv->run();
        if (_wp) _weightsqi.allocator()->free();
        if (_qi) _outputqi.allocator()->free();
    }

protected:
    std::shared_ptr<arm_compute::IMemoryManager> _memory_manager;
    const arm_compute::ITensor *_weights;
    const arm_compute::QuantizationInfo *_wp;
    arm_compute::Tensor _weightsqi;
    const arm_compute::QuantizationInfo *_qi;
    const arm_compute::ITensor *_output;
    arm_compute::Tensor _outputqi;
    std::unique_ptr<arm_compute::NEConvolutionLayer> _conv;
};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmConvolution& node) {
    arm_compute::PadStrideInfo conv_info;
    arm_compute::Size2D dilation;
    std::tie(conv_info, dilation) = ConvParameters(node);

    auto wInfoIt = node.get_rt_info().find("WeightsPrescaleInfo");
    arm_compute::QuantizationInfo* wInfo = wInfoIt == node.get_rt_info().end() ? nullptr
                                           &(safe_cast<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(wInfoIt->second)->get());
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    arm_compute::QuantizationInfo* qInfo = qInfoIt == node.get_rt_info().end() ? nullptr
                                           &(safe_cast<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(qInfoIt->second)->get());

    if (node.get_input_size() == 3) {
        return MakeConversion<NEConvolutionLayerQI>(
            node.input(Features), node.input(Weights), node.input(Bias), node.output(0),
            conv_info, arm_compute::WeightsInfo{}, dilation, GetActivationInfo(node), wInfo, qInfo);
    } else {
        return MakeConversion<NEConvolutionLayerQI>(
            node.input(Features), node.input(Weights), nullptr, node.output(0),
            conv_info, arm_compute::WeightsInfo{}, dilation, GetActivationInfo(node), wInfo, qInfo);
    }
}

struct NEDepthwiseConvolutionLayerQI final: public arm_compute::IFunction {
public:
    NEDepthwiseConvolutionLayerQI(std::shared_ptr<arm_compute::IMemoryManager> memory_manager = nullptr):
        _memory_manager(memory_manager), _conv(), _weights(nullptr), _wp(nullptr), _weightsqi(), _output(nullptr), _qi(nullptr), _outputqi() {}
    NEDepthwiseConvolutionLayerQI(const NEDepthwiseConvolutionLayerQI &) = delete;
    NEDepthwiseConvolutionLayerQI &operator=(const NEDepthwiseConvolutionLayerQI &) = delete;
    NEDepthwiseConvolutionLayerQI(NEDepthwiseConvolutionLayerQI &&) = delete;
    NEDepthwiseConvolutionLayerQI &operator=(NEDepthwiseConvolutionLayerQI &&) = delete;
    ~NEDepthwiseConvolutionLayerQI() = default;
    void configure(arm_compute::ITensor *input, const arm_compute::ITensor *weights, const arm_compute::ITensor *biases, arm_compute::ITensor *output,
                   const arm_compute::PadStrideInfo &conv_info, unsigned int depth_multiplier, const arm_compute::ActivationLayerInfo &act_info,
                   const arm_compute::Size2D &dilation, const arm_compute::QuantizationInfo *wp, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
        ARM_COMPUTE_UNUSED(num_groups);
        ARM_COMPUTE_ERROR_THROW_ON(NEDepthwiseConvolutionLayerQI::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr),
                                                                           output->info(), conv_info, depth_multiplier, act_info, dilation, wp, qi));

        _weights = weights;
        _wp = wp;
        if (_wp) {
            bool setPerChannel = _weights->info()->data_type() == QASYMM8_SIGNED ||
                                 _weights->info()->data_type() == QASYMM8 ||
                                 _weights->info()->data_type() == QSYMM8;
            _weightsqi.allocator()->init(setPerChannel ? _weights->info()->set_data_type(QSYMM8_PER_CHANNEL) : *(_weights->info()));
            _weightsqi.info()->set_quantization_info(*wp);
        }

        _output = output;
        _qi = qi;
        if (_qi) {
            _outputqi.allocator()->init(*(_output->info()));
            _outputqi.info()->set_quantization_info(*qi);
        }

        _conv = std::make_unique<arm_compute::NEDepthwiseConvolutionLayer>(_memory_manager);
        _conv->configure(input, _wp ? &_weightsqi : _weights, biases, _qi ? &_outputqi : _output, conv_info, depth_multiplier, act_info, dilation);
    }
    static Status validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *weights, const arm_compute::ITensorInfo *biases,
                           const arm_compute::ITensorInfo *output, const arm_compute::PadStrideInfo &conv_info, unsigned int depth_multiplier,
                           const arm_compute::ActivationLayerInfo &act_info, const arm_compute::Size2D &dilation,
                           const arm_compute::QuantizationInfo *wp, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(weights, output);
        //At the moment quantization info isn't checked actually, but just in case
        return arm_compute::NEDepthwiseConvolutionLayer::validate(input, &arm_compute::TensorInfo(*weights).set_quantization_info(wp), biases,
                                                                  &arm_compute::TensorInfo(*output).set_quantization_info(qi),
                                                                  conv_info, depth_multiplier, act_info, dilation);
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_conv.get(), "Kernel didn't configured");
        if (_wp) {
            if (_weightsqi.info()->padding() != _weights->info()->padding()) _weightsqi.info()->extend_padding(_weights->info()->padding());
            _weightsqi.allocator()->import_memory(_weights->buffer());
        }
        if (_qi) {
            if (_outputqi.info()->padding() != _output->info()->padding()) _outputqi.info()->extend_padding(_output->info()->padding());
            _outputqi.allocator()->import_memory(_output->buffer());
        }
        _conv->run();
        if (_wp) _weightsqi.allocator()->free();
        if (_qi) _outputqi.allocator()->free();
    }

protected:
    std::shared_ptr<arm_compute::IMemoryManager> _memory_manager;
    const arm_compute::ITensor *_weights;
    const arm_compute::QuantizationInfo *_wp;
    arm_compute::Tensor _weightsqi;
    const arm_compute::QuantizationInfo *_qi;
    const arm_compute::ITensor *_output;
    arm_compute::Tensor _outputqi;
    std::unique_ptr<arm_compute::NEDepthwiseConvolutionLayer> _conv;
};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmGroupConvolution& node) {
    arm_compute::PadStrideInfo conv_info;
    arm_compute::Size2D dilation;
    std::tie(conv_info, dilation) = ConvParameters(node);
    auto ngraphWeightsShape = node.input(Weights).get_shape();
    _layers.at(node.get_instance_id())._inputs.at(node.input(Weights))->_tensor->info()->set_tensor_shape(ShapeCast({
        ngraphWeightsShape[1],
        ngraphWeightsShape[0]*ngraphWeightsShape[2],
        ngraphWeightsShape[3],
        ngraphWeightsShape[4]
    }));

    auto wInfoIt = node.get_rt_info().find("WeightsPrescaleInfo");
    arm_compute::QuantizationInfo* wInfo = wInfoIt == node.get_rt_info().end() ? nullptr
                                           &(safe_cast<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(wInfoIt->second)->get());
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    arm_compute::QuantizationInfo* qInfo = qInfoIt == node.get_rt_info().end() ? nullptr
                                           &(safe_cast<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(qInfoIt->second)->get());

    if (node.get_input_size() == 3) {
        return MakeConversion<NEDepthwiseConvolutionLayerQI>(
            node.input(Features), node.input(Weights), node.input(Bias), node.output(0),
            conv_info, 1u, GetActivationInfo(node), dilation, wInfo, qInfo);
    } else {
        return MakeConversion<NEDepthwiseConvolutionLayerQI>(
            node.input(Features), node.input(Weights), nullptr, node.output(0),
            conv_info, 1u, GetActivationInfo(node), dilation, wInfo, qInfo);
    }
}
}  //  namespace ArmPlugin
