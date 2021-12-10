// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <src/core/NEON/kernels/NEConvertQuantizedSignednessKernel.h>
#include <src/runtime/Utils.h>
#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
enum InputArg {Features, Weights, Bias};
struct NEFullyConnectedLayerQI final: public arm_compute::IFunction {
public:
    NEFullyConnectedLayerQI(std::shared_ptr<arm_compute::IMemoryManager> memory_manager = nullptr):
        _memory_manager(memory_manager), _memory_group{std::make_unique<arm_compute::MemoryGroup>(memory_manager)},
        _i_sgn(nullptr), _w_sgn(nullptr), _fconn(nullptr),
        _input(nullptr), _ip(nullptr), _inputqi(),
        _weights(nullptr), _wp(nullptr), _weightsqi(),
        _output(nullptr), _qi(nullptr), _outputqi() {}
    NEFullyConnectedLayerQI(const NEFullyConnectedLayerQI &) = delete;
    NEFullyConnectedLayerQI &operator=(const NEFullyConnectedLayerQI &) = delete;
    NEFullyConnectedLayerQI(NEFullyConnectedLayerQI &&) = delete;
    NEFullyConnectedLayerQI &operator=(NEFullyConnectedLayerQI &&) = delete;
    ~NEFullyConnectedLayerQI() = default;
    void configure(arm_compute::ITensor *input, const arm_compute::ITensor *weights, const arm_compute::ITensor *biases, arm_compute::ITensor *output,
                   const arm_compute::QuantizationInfo *ip, const arm_compute::QuantizationInfo *wp, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
        ARM_COMPUTE_ERROR_THROW_ON(NEFullyConnectedLayerQI::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr),
                                                                     output->info(), ip, wp, qi));

        _input = input;
        arm_compute::ITensor *conv_input = input;
        _ip = ip;
        if (output->info()->data_type() == arm_compute::DataType::QASYMM8_SIGNED && input->info()->data_type() == arm_compute::DataType::QASYMM8 ||
            output->info()->data_type() == arm_compute::DataType::QASYMM8 && input->info()->data_type() == arm_compute::DataType::QASYMM8_SIGNED) {
            _i_sgn = std::make_unique<arm_compute::NEConvertQuantizedSignednessKernel>();
            _memory_group->manage(&_inputqi);
            _inputqi.allocator()->init(*(input->info()));
            float scale = 1.f;
            std::int32_t offset = output->info()->data_type() == arm_compute::DataType::QASYMM8 ? 128 : -128;
            if (ip) {
                scale = ip->scale()[0];
                offset += ip->offset()[0];
            }
            _inputqi.info()->set_data_type(output->info()->data_type()).set_quantization_info(arm_compute::QuantizationInfo(scale, offset));
            _i_sgn->configure(input, &_inputqi);
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
        } else if (output->info()->data_type() == arm_compute::DataType::QASYMM8_SIGNED && weights->info()->data_type() == arm_compute::DataType::QASYMM8 ||
                   output->info()->data_type() == arm_compute::DataType::QASYMM8 && weights->info()->data_type() == arm_compute::DataType::QASYMM8_SIGNED) {
            _w_sgn = std::make_unique<arm_compute::NEConvertQuantizedSignednessKernel>();
            _memory_group->manage(&_weightsqi);
            _weightsqi.info()->set_data_type(output->info()->data_type()).set_quantization_info(arm_compute::QuantizationInfo(1.f,
                                                                output->info()->data_type() == arm_compute::DataType::QASYMM8 ? 128 : -128));
            _w_sgn->configure(weights, &_weightsqi);
            conv_weights = &_weightsqi;
        }

        _output = output;
        _qi = qi;
        if (_qi) {
            _outputqi.allocator()->init(*(_output->info()));
            _outputqi.info()->set_quantization_info(*qi);
        }

        _fconn = std::make_unique<arm_compute::NEFullyConnectedLayer>(_memory_manager);
        _fconn->configure(conv_input, conv_weights, biases, _qi ? &_outputqi : _output);

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
        }
        arm_compute::TensorInfo vld_output(*output);
        if (qi) vld_output.set_quantization_info(*qi);

        return arm_compute::NEFullyConnectedLayer::validate(&vld_input, &vld_weights, biases, &vld_output);
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_fconn.get(), "Kernel didn't configured");
        std::unique_ptr<arm_compute::MemoryGroupResourceScope> _sgn_scope = _i_sgn || _w_sgn ?
                                                                std::make_unique<arm_compute::MemoryGroupResourceScope>(*_memory_group) : nullptr;
        if (_i_sgn) {
            arm_compute::utils::schedule_kernel_on_ctx(nullptr, _i_sgn.get(), arm_compute::Window::DimY);
        } else if (_ip) {
            if (_inputqi.info()->padding() != _input->info()->padding()) _inputqi.info()->extend_padding(_input->info()->padding());
            _inputqi.allocator()->import_memory(_input->buffer());
        }
        if (_w_sgn) {
            arm_compute::utils::schedule_kernel_on_ctx(nullptr, _w_sgn.get(), arm_compute::Window::DimY);
        } else if (_wp) {
            if (_weightsqi.info()->padding() != _weights->info()->padding()) _weightsqi.info()->extend_padding(_weights->info()->padding());
            _weightsqi.allocator()->import_memory(_weights->buffer());
        }
        if (_qi) {
            if (_outputqi.info()->padding() != _output->info()->padding()) _outputqi.info()->extend_padding(_output->info()->padding());
            _outputqi.allocator()->import_memory(_output->buffer());
        }
        _fconn->run();
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
    std::unique_ptr<arm_compute::NEConvertQuantizedSignednessKernel> _i_sgn, _w_sgn;
    std::unique_ptr<arm_compute::NEFullyConnectedLayer> _fconn;
};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::MatMul& node) {
    if (node.get_transpose_a()) {
        IE_THROW() << "Can not create MatMul layer with transpose first input";
    }
    auto iInfoIt = node.get_rt_info().find("InputPrescaleInfo");
    arm_compute::QuantizationInfo* iInfo = iInfoIt == node.get_rt_info().end() ? nullptr :
                                           &(iInfoIt->second.as<arm_compute::QuantizationInfo>());
    auto wInfoIt = node.get_rt_info().find("WeightsPrescaleInfo");
    arm_compute::QuantizationInfo* wInfo = wInfoIt == node.get_rt_info().end() ? nullptr :
                                           &(wInfoIt->second.as<arm_compute::QuantizationInfo>());
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    arm_compute::QuantizationInfo* qInfo = qInfoIt == node.get_rt_info().end() ? nullptr :
                                           &(qInfoIt->second.as<arm_compute::QuantizationInfo>());
    return MakeConversion<NEFullyConnectedLayerQI>(node.input(Features), node.input(Weights), nullptr, node.output(0), iInfo, wInfo, qInfo);
}
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmMatMulBias& node) {
    if (node.get_transpose_a()) {
        IE_THROW() << "Can not create MatMul layer with transpose first input";
    }
    auto iInfoIt = node.get_rt_info().find("InputPrescaleInfo");
    arm_compute::QuantizationInfo* iInfo = iInfoIt == node.get_rt_info().end() ? nullptr :
                                           &(iInfoIt->second.as<arm_compute::QuantizationInfo>());
    auto wInfoIt = node.get_rt_info().find("WeightsPrescaleInfo");
    arm_compute::QuantizationInfo* wInfo = wInfoIt == node.get_rt_info().end() ? nullptr :
                                           &(wInfoIt->second.as<arm_compute::QuantizationInfo>());
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    arm_compute::QuantizationInfo* qInfo = qInfoIt == node.get_rt_info().end() ? nullptr :
                                           &(qInfoIt->second.as<arm_compute::QuantizationInfo>());
    return MakeConversion<NEFullyConnectedLayerQI>(node.input(Features), node.input(Weights), node.input(Bias), node.output(0), iInfo, wInfo, qInfo);
}
}  //  namespace ArmPlugin
