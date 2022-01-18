// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <src/cpu/kernels/CpuConvertQuantizedSignednessKernel.h>
#include <arm_compute/runtime/NEON/NEScheduler.h>
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
        _memory_manager(memory_manager), _memory_group{std::make_unique<arm_compute::MemoryGroup>(memory_manager)},
        _i_sgn(nullptr), _w_sgn(nullptr), _conv(nullptr),
        _input(nullptr), _ip(nullptr), _inputqi(),
        _weights(nullptr), _wp(nullptr), _weightsqi(),
        _output(nullptr), _qi(nullptr), _outputqi() {}
    NEConvolutionLayerQI(const NEConvolutionLayerQI &) = delete;
    NEConvolutionLayerQI &operator=(const NEConvolutionLayerQI &) = delete;
    NEConvolutionLayerQI(NEConvolutionLayerQI &&) = delete;
    NEConvolutionLayerQI &operator=(NEConvolutionLayerQI &&) = delete;
    ~NEConvolutionLayerQI() = default;
    void configure(arm_compute::ITensor *input, const arm_compute::ITensor *weights, const arm_compute::ITensor *biases, arm_compute::ITensor *output,
                   const arm_compute::PadStrideInfo &conv_info, const arm_compute::WeightsInfo &weights_info, const arm_compute::Size2D &dilation,
                   const arm_compute::ActivationLayerInfo &act_info,
                   const arm_compute::QuantizationInfo *ip, const arm_compute::QuantizationInfo *wp, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
        ARM_COMPUTE_ERROR_THROW_ON(NEConvolutionLayerQI::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr),
                                                                  output->info(), conv_info, weights_info, dilation, act_info, ip, wp, qi));

        _input = input;
        arm_compute::ITensor *conv_input = input;
        _ip = ip;
        if (output->info()->data_type() == arm_compute::DataType::QASYMM8_SIGNED && _input->info()->data_type() == arm_compute::DataType::QASYMM8 ||
            output->info()->data_type() == arm_compute::DataType::QASYMM8 && _input->info()->data_type() == arm_compute::DataType::QASYMM8_SIGNED) {
            _i_sgn = std::make_unique<arm_compute::cpu::kernels::CpuConvertQuantizedSignednessKernel>();
            _memory_group->manage(&_inputqi);
            _inputqi.allocator()->init(*(_input->info()));
            float scale = 1.f;
            std::int32_t offset = output->info()->data_type() == arm_compute::DataType::QASYMM8 ? 128 : -128;
            if (ip) {
                scale = ip->scale()[0];
                offset += ip->offset()[0];
            }
            _inputqi.info()->set_data_type(output->info()->data_type()).set_quantization_info(arm_compute::QuantizationInfo(scale, offset));
            _i_sgn->configure(_input->info(), _inputqi.info());
            conv_input = &_inputqi;
        } else if (_ip) {
            _inputqi.allocator()->init(*(_input->info()));
            _inputqi.info()->set_quantization_info(*ip);
            conv_input = &_inputqi;
        }

        _weights = weights;
        const arm_compute::ITensor *conv_weights = weights;
        _wp = wp;
        if (_wp) {
            _weightsqi.allocator()->init(*(_weights->info()));
            _weightsqi.info()->set_data_type(arm_compute::DataType::QSYMM8_PER_CHANNEL).set_quantization_info(*wp);
            conv_weights = &_weightsqi;
        } else if (output->info()->data_type() == arm_compute::DataType::QASYMM8_SIGNED && _weights->info()->data_type() == arm_compute::DataType::QASYMM8 ||
                   output->info()->data_type() == arm_compute::DataType::QASYMM8 && _weights->info()->data_type() == arm_compute::DataType::QASYMM8_SIGNED) {
            _w_sgn = std::make_unique<arm_compute::cpu::kernels::CpuConvertQuantizedSignednessKernel>();
            _memory_group->manage(&_weightsqi);
            _weightsqi.allocator()->init(*(_weights->info()));
            _weightsqi.info()->set_data_type(output->info()->data_type()).set_quantization_info(arm_compute::QuantizationInfo(1.f,
                                                                output->info()->data_type() == arm_compute::DataType::QASYMM8 ? 128 : -128));
            _w_sgn->configure(_weights->info(), _weightsqi.info());
            conv_weights = &_weightsqi;
        }

        _output = output;
        _qi = qi;
        if (_qi) {
            _outputqi.allocator()->init(*(_output->info()));
            _outputqi.info()->set_quantization_info(*qi);
        }

        _conv = std::make_unique<arm_compute::NEConvolutionLayer>(_memory_manager);
        _conv->configure(conv_input, conv_weights, biases, _qi ? &_outputqi : _output, conv_info, weights_info, dilation, act_info);

        if (_i_sgn) {
            _inputqi.allocator()->allocate();
        } else if (_ip && _inputqi.info()->padding() != _input->info()->padding()) {
            //Backpropagate possible input padding change
            _input->info()->extend_padding(_inputqi.info()->padding());
        }
        if (_w_sgn) {
            _weightsqi.allocator()->allocate();
        } else if (_wp && _weightsqi.info()->padding() != _weights->info()->padding()) {
            //Backpropagate possible weights padding change
            _weights->info()->extend_padding(_weightsqi.info()->padding());
        }
    }
    static arm_compute::Status validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *weights,
                                        const arm_compute::ITensorInfo *biases, const arm_compute::ITensorInfo *output,
                                        const arm_compute::PadStrideInfo &conv_info, const arm_compute::WeightsInfo &weights_info,
                                        const arm_compute::Size2D &dilation, const arm_compute::ActivationLayerInfo &act_info,
                                        const arm_compute::QuantizationInfo *ip, const arm_compute::QuantizationInfo *wp,
                                        const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
        arm_compute::TensorInfo vld_input(*input);
        if (output->data_type() == arm_compute::DataType::QASYMM8_SIGNED && input->data_type() == arm_compute::DataType::QASYMM8 ||
            output->data_type() == arm_compute::DataType::QASYMM8 && input->data_type() == arm_compute::DataType::QASYMM8_SIGNED) {
            vld_input.set_data_type(output->data_type());
            float scale = 1.f;
            std::int32_t offset = output->data_type() == arm_compute::DataType::QASYMM8 ? 128 : -128;
            if (ip) {
                scale = ip->scale()[0];
                offset += ip->offset()[0];
            }
            vld_input.set_quantization_info(arm_compute::QuantizationInfo(scale, offset));
            ARM_COMPUTE_RETURN_ON_ERROR(arm_compute::cpu::kernels::CpuConvertQuantizedSignednessKernel::validate(input, &vld_input));
        } else if (ip) {
            vld_input.set_quantization_info(*ip);
        }
        arm_compute::TensorInfo vld_weights(*weights);
        if (wp) {
            vld_weights.set_data_type(arm_compute::DataType::QSYMM8_PER_CHANNEL).set_quantization_info(*wp);
        } else if (output->data_type() == arm_compute::DataType::QASYMM8_SIGNED && weights->data_type() == arm_compute::DataType::QASYMM8 ||
                   output->data_type() == arm_compute::DataType::QASYMM8 && weights->data_type() == arm_compute::DataType::QASYMM8_SIGNED) {
            vld_weights.set_data_type(output->data_type());
            vld_weights.set_quantization_info(arm_compute::QuantizationInfo(1.f, output->data_type() == arm_compute::DataType::QASYMM8 ? 128 : -128));
            ARM_COMPUTE_RETURN_ON_ERROR(arm_compute::cpu::kernels::CpuConvertQuantizedSignednessKernel::validate(weights, &vld_weights));
        }
        arm_compute::TensorInfo vld_output(*output);
        if (qi) vld_output.set_quantization_info(*qi);

        return arm_compute::NEConvolutionLayer::validate(&vld_input, &vld_weights, biases, &vld_output, conv_info, weights_info, dilation, act_info);
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_conv.get(), "Kernel didn't configured");
        std::unique_ptr<arm_compute::MemoryGroupResourceScope> _sgn_scope = _i_sgn || _w_sgn ?
                                                                std::make_unique<arm_compute::MemoryGroupResourceScope>(*_memory_group) : nullptr;
        if (_i_sgn) {
            arm_compute::ITensorPack pack = {
                { arm_compute::TensorType::ACL_SRC, _input },
                { arm_compute::TensorType::ACL_DST, &_inputqi }
            };
            arm_compute::NEScheduler::get().schedule_op(_i_sgn.get(), arm_compute::Window::DimY, _i_sgn->window(), pack);
        } else if (_ip) {
            if (_inputqi.info()->padding() != _input->info()->padding()) _inputqi.info()->extend_padding(_input->info()->padding());
            _inputqi.allocator()->import_memory(_input->buffer());
        }
        if (_w_sgn) {
            arm_compute::ITensorPack pack = {
                { arm_compute::TensorType::ACL_SRC, _weights },
                { arm_compute::TensorType::ACL_DST, &_weightsqi }
            };
            arm_compute::NEScheduler::get().schedule_op(_w_sgn.get(), arm_compute::Window::DimY, _w_sgn->window(), pack);
        } else if (_wp) {
            if (_weightsqi.info()->padding() != _weights->info()->padding()) _weightsqi.info()->extend_padding(_weights->info()->padding());
            _weightsqi.allocator()->import_memory(_weights->buffer());
        }
        if (_qi) {
            if (_outputqi.info()->padding() != _output->info()->padding()) _outputqi.info()->extend_padding(_output->info()->padding());
            _outputqi.allocator()->import_memory(_output->buffer());
        }
        _conv->run();
        if (!_i_sgn && _ip) _inputqi.allocator()->free();
        if (_wp) _weightsqi.allocator()->free();
        if (_qi) _outputqi.allocator()->free();
    }

protected:
    std::shared_ptr<arm_compute::IMemoryManager> _memory_manager;
    std::unique_ptr<arm_compute::MemoryGroup> _memory_group;
    const arm_compute::QuantizationInfo *_ip;
    arm_compute::ITensor *_input;
    arm_compute::Tensor _inputqi;
    const arm_compute::QuantizationInfo *_wp;
    const arm_compute::ITensor *_weights;
    arm_compute::Tensor _weightsqi;
    const arm_compute::QuantizationInfo *_qi;
    arm_compute::ITensor *_output;
    arm_compute::Tensor _outputqi;
    std::unique_ptr<arm_compute::cpu::kernels::CpuConvertQuantizedSignednessKernel> _i_sgn, _w_sgn;
    std::unique_ptr<arm_compute::NEConvolutionLayer> _conv;
};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmConvolution& node) {
    arm_compute::PadStrideInfo conv_info;
    arm_compute::Size2D dilation;
    std::tie(conv_info, dilation) = ConvParameters(node);

    auto iInfoIt = node.get_rt_info().find("InputPrescaleInfo");
    const arm_compute::QuantizationInfo* iInfo = iInfoIt == node.get_rt_info().end() ? nullptr :
                                               &(iInfoIt->second.as<arm_compute::QuantizationInfo>());
    auto wInfoIt = node.get_rt_info().find("WeightsPrescaleInfo");
    const arm_compute::QuantizationInfo* wInfo = wInfoIt == node.get_rt_info().end() ? nullptr :
                                               &(wInfoIt->second.as<arm_compute::QuantizationInfo>());
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    const arm_compute::QuantizationInfo* qInfo = qInfoIt == node.get_rt_info().end() ? nullptr :
                                               &(qInfoIt->second.as<arm_compute::QuantizationInfo>());

    if (node.get_input_size() == 3) {
        return MakeConversion<NEConvolutionLayerQI>(
            node.input(Features), node.input(Weights), node.input(Bias), node.output(0),
            conv_info, arm_compute::WeightsInfo{}, dilation, GetActivationInfo(node), iInfo, wInfo, qInfo);
    } else {
        return MakeConversion<NEConvolutionLayerQI>(
            node.input(Features), node.input(Weights), nullptr, node.output(0),
            conv_info, arm_compute::WeightsInfo{}, dilation, GetActivationInfo(node), iInfo, wInfo, qInfo);
    }
}

struct NEDepthwiseConvolutionLayerQI final: public arm_compute::IFunction {
public:
    NEDepthwiseConvolutionLayerQI(std::shared_ptr<arm_compute::IMemoryManager> memory_manager = nullptr):
        _memory_manager(memory_manager), _memory_group{std::make_unique<arm_compute::MemoryGroup>(memory_manager)},
        _i_sgn(nullptr), _w_sgn(nullptr), _conv(nullptr),
        _input(nullptr), _ip(nullptr), _inputqi(),
        _weights(nullptr), _wp(nullptr), _weightsqi(),
        _output(nullptr), _qi(nullptr), _outputqi() {}
    NEDepthwiseConvolutionLayerQI(const NEDepthwiseConvolutionLayerQI &) = delete;
    NEDepthwiseConvolutionLayerQI &operator=(const NEDepthwiseConvolutionLayerQI &) = delete;
    NEDepthwiseConvolutionLayerQI(NEDepthwiseConvolutionLayerQI &&) = delete;
    NEDepthwiseConvolutionLayerQI &operator=(NEDepthwiseConvolutionLayerQI &&) = delete;
    ~NEDepthwiseConvolutionLayerQI() = default;
    void configure(arm_compute::ITensor *input, const arm_compute::ITensor *weights, const arm_compute::ITensor *biases, arm_compute::ITensor *output,
                   const arm_compute::PadStrideInfo &conv_info, unsigned int depth_multiplier,
                   const arm_compute::ActivationLayerInfo &act_info, const arm_compute::Size2D &dilation,
                   const arm_compute::QuantizationInfo *ip, const arm_compute::QuantizationInfo *wp, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
        ARM_COMPUTE_ERROR_THROW_ON(NEDepthwiseConvolutionLayerQI::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr),
                                                                           output->info(), conv_info, depth_multiplier, act_info, dilation, ip, wp, qi));

        _input = input;
        arm_compute::ITensor *conv_input = input;
        _ip = ip;
        if (output->info()->data_type() == arm_compute::DataType::QASYMM8_SIGNED && _input->info()->data_type() == arm_compute::DataType::QASYMM8 ||
            output->info()->data_type() == arm_compute::DataType::QASYMM8 && _input->info()->data_type() == arm_compute::DataType::QASYMM8_SIGNED) {
            _i_sgn = std::make_unique<arm_compute::cpu::kernels::CpuConvertQuantizedSignednessKernel>();
            _memory_group->manage(&_inputqi);
            _inputqi.allocator()->init(*(_input->info()));
            float scale = 1.f;
            std::int32_t offset = output->info()->data_type() == arm_compute::DataType::QASYMM8 ? 128 : -128;
            if (ip) {
                scale = ip->scale()[0];
                offset += ip->offset()[0];
            }
            _inputqi.info()->set_data_type(output->info()->data_type()).set_quantization_info(arm_compute::QuantizationInfo(scale, offset));
            _i_sgn->configure(_input->info(), _inputqi.info());
            conv_input = &_inputqi;
        } else if (_ip) {
            _inputqi.allocator()->init(*(_input->info()));
            _inputqi.info()->set_quantization_info(*ip);
            conv_input = &_inputqi;
        }

        _weights = weights;
        const arm_compute::ITensor *conv_weights = weights;
        _wp = wp;
        if (_wp) {
            _weightsqi.allocator()->init(*(_weights->info()));
            _weightsqi.info()->set_data_type(arm_compute::DataType::QSYMM8_PER_CHANNEL).set_quantization_info(*wp);
            conv_weights = &_weightsqi;
        } else if (output->info()->data_type() == arm_compute::DataType::QASYMM8_SIGNED && _weights->info()->data_type() == arm_compute::DataType::QASYMM8 ||
                   output->info()->data_type() == arm_compute::DataType::QASYMM8 && _weights->info()->data_type() == arm_compute::DataType::QASYMM8_SIGNED) {
            _w_sgn = std::make_unique<arm_compute::cpu::kernels::CpuConvertQuantizedSignednessKernel>();
            _memory_group->manage(&_weightsqi);
            _weightsqi.allocator()->init(*(_weights->info()));
            _weightsqi.info()->set_data_type(output->info()->data_type()).set_quantization_info(arm_compute::QuantizationInfo(1.f,
                                                                output->info()->data_type() == arm_compute::DataType::QASYMM8 ? 128 : -128));
            _w_sgn->configure(_weights->info(), _weightsqi.info());
            conv_weights = &_weightsqi;
        }

        _output = output;
        _qi = qi;
        if (_qi) {
            _outputqi.allocator()->init(*(_output->info()));
            _outputqi.info()->set_quantization_info(*qi);
        }

        _conv = std::make_unique<arm_compute::NEDepthwiseConvolutionLayer>(_memory_manager);
        _conv->configure(conv_input, conv_weights, biases, _qi ? &_outputqi : _output, conv_info, depth_multiplier, act_info, dilation);

        if (_i_sgn) {
            _inputqi.allocator()->allocate();
        } else if (_ip && _inputqi.info()->padding() != _input->info()->padding()) {
            //Backpropagate possible input padding change
            _input->info()->extend_padding(_inputqi.info()->padding());
        }
        if (_w_sgn) {
            _weightsqi.allocator()->allocate();
        } else if (_wp && _weightsqi.info()->padding() != _weights->info()->padding()) {
            //Backpropagate possible weights padding change
            _weights->info()->extend_padding(_weightsqi.info()->padding());
        }
    }
    static arm_compute::Status validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *weights,
                                        const arm_compute::ITensorInfo *biases, const arm_compute::ITensorInfo *output,
                                        const arm_compute::PadStrideInfo &conv_info, unsigned int depth_multiplier,
                                        const arm_compute::ActivationLayerInfo &act_info, const arm_compute::Size2D &dilation,
                                        const arm_compute::QuantizationInfo *ip, const arm_compute::QuantizationInfo *wp,
                                        const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
        arm_compute::TensorInfo vld_input(*input);
        if (output->data_type() == arm_compute::DataType::QASYMM8_SIGNED && input->data_type() == arm_compute::DataType::QASYMM8 ||
            output->data_type() == arm_compute::DataType::QASYMM8 && input->data_type() == arm_compute::DataType::QASYMM8_SIGNED) {
            vld_input.set_data_type(output->data_type());
            float scale = 1.f;
            std::int32_t offset = output->data_type() == arm_compute::DataType::QASYMM8 ? 128 : -128;
            if (ip) {
                scale = ip->scale()[0];
                offset += ip->offset()[0];
            }
            vld_input.set_quantization_info(arm_compute::QuantizationInfo(scale, offset));
            ARM_COMPUTE_RETURN_ON_ERROR(arm_compute::cpu::kernels::CpuConvertQuantizedSignednessKernel::validate(input, &vld_input));
        } else if (ip) {
            vld_input.set_quantization_info(*ip);
        }
        arm_compute::TensorInfo vld_weights(*weights);
        if (wp) {
            vld_weights.set_data_type(arm_compute::DataType::QSYMM8_PER_CHANNEL).set_quantization_info(*wp);
        } else if (output->data_type() == arm_compute::DataType::QASYMM8_SIGNED && weights->data_type() == arm_compute::DataType::QASYMM8 ||
                   output->data_type() == arm_compute::DataType::QASYMM8 && weights->data_type() == arm_compute::DataType::QASYMM8_SIGNED) {
            vld_weights.set_data_type(output->data_type());
            vld_weights.set_quantization_info(arm_compute::QuantizationInfo(1.f, output->data_type() == arm_compute::DataType::QASYMM8 ? 128 : -128));
            ARM_COMPUTE_RETURN_ON_ERROR(arm_compute::cpu::kernels::CpuConvertQuantizedSignednessKernel::validate(weights, &vld_weights));
        }
        arm_compute::TensorInfo vld_output(*output);
        if (qi) vld_output.set_quantization_info(*qi);

        return arm_compute::NEDepthwiseConvolutionLayer::validate(&vld_input, &vld_weights, biases, &vld_output,
                                                                  conv_info, depth_multiplier, act_info, dilation);
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_conv.get(), "Kernel didn't configured");
        std::unique_ptr<arm_compute::MemoryGroupResourceScope> _sgn_scope = _i_sgn || _w_sgn ?
                                                                std::make_unique<arm_compute::MemoryGroupResourceScope>(*_memory_group) : nullptr;
        if (_i_sgn) {
            arm_compute::ITensorPack pack = {
                { arm_compute::TensorType::ACL_SRC, _input },
                { arm_compute::TensorType::ACL_DST, &_inputqi }
            };
            arm_compute::NEScheduler::get().schedule_op(_i_sgn.get(), arm_compute::Window::DimY, _i_sgn->window(), pack);
        } else if (_ip) {
            if (_inputqi.info()->padding() != _input->info()->padding()) _inputqi.info()->extend_padding(_input->info()->padding());
            _inputqi.allocator()->import_memory(_input->buffer());
        }
        if (_w_sgn) {
            arm_compute::ITensorPack pack = {
                { arm_compute::TensorType::ACL_SRC, _weights },
                { arm_compute::TensorType::ACL_DST, &_weightsqi }
            };
            arm_compute::NEScheduler::get().schedule_op(_w_sgn.get(), arm_compute::Window::DimY, _w_sgn->window(), pack);
        } else if (_wp) {
            if (_weightsqi.info()->padding() != _weights->info()->padding()) _weightsqi.info()->extend_padding(_weights->info()->padding());
            _weightsqi.allocator()->import_memory(_weights->buffer());
        }
        if (_qi) {
            if (_outputqi.info()->padding() != _output->info()->padding()) _outputqi.info()->extend_padding(_output->info()->padding());
            _outputqi.allocator()->import_memory(_output->buffer());
        }
        _conv->run();
        if (!_i_sgn && _ip) _inputqi.allocator()->free();
        if (_wp) _weightsqi.allocator()->free();
        if (_qi) _outputqi.allocator()->free();
    }

protected:
    std::shared_ptr<arm_compute::IMemoryManager> _memory_manager;
    std::unique_ptr<arm_compute::MemoryGroup> _memory_group;
    const arm_compute::QuantizationInfo *_ip;
    arm_compute::ITensor *_input;
    arm_compute::Tensor _inputqi;
    const arm_compute::QuantizationInfo *_wp;
    const arm_compute::ITensor *_weights;
    arm_compute::Tensor _weightsqi;
    const arm_compute::QuantizationInfo *_qi;
    arm_compute::ITensor *_output;
    arm_compute::Tensor _outputqi;
    std::unique_ptr<arm_compute::cpu::kernels::CpuConvertQuantizedSignednessKernel> _i_sgn, _w_sgn;
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

    auto iInfoIt = node.get_rt_info().find("InputPrescaleInfo");
    const arm_compute::QuantizationInfo* iInfo = iInfoIt == node.get_rt_info().end() ? nullptr :
                                               &(iInfoIt->second.as<arm_compute::QuantizationInfo>());
    auto wInfoIt = node.get_rt_info().find("WeightsPrescaleInfo");
    const arm_compute::QuantizationInfo* wInfo = wInfoIt == node.get_rt_info().end() ? nullptr :
                                               &(wInfoIt->second.as<arm_compute::QuantizationInfo>());
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    const arm_compute::QuantizationInfo* qInfo = qInfoIt == node.get_rt_info().end() ? nullptr :
                                               &(qInfoIt->second.as<arm_compute::QuantizationInfo>());

    if (node.get_input_size() == 3) {
        return MakeConversion<NEDepthwiseConvolutionLayerQI>(
            node.input(Features), node.input(Weights), node.input(Bias), node.output(0),
            conv_info, 1u, GetActivationInfo(node), dilation, iInfo, wInfo, qInfo);
    } else {
        return MakeConversion<NEDepthwiseConvolutionLayerQI>(
            node.input(Features), node.input(Weights), nullptr, node.output(0),
            conv_info, 1u, GetActivationInfo(node), dilation, iInfo, wInfo, qInfo);
    }
}
}  //  namespace ArmPlugin
