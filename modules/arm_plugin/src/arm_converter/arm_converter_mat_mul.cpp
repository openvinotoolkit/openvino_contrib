// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
enum InputArg {Features, Weights, Bias};
struct NEFullyConnectedLayerQI final: public arm_compute::IFunction {
public:
    NEFullyConnectedLayerQI(std::shared_ptr<arm_compute::IMemoryManager> memory_manager = nullptr):
        _memory_manager(memory_manager), _fconn(), _weights(nullptr), _wp(nullptr), _weightsqi(), _output(nullptr), _qi(nullptr), _outputqi() {}
    NEFullyConnectedLayerQI(const NEFullyConnectedLayerQI &) = delete;
    NEFullyConnectedLayerQI &operator=(const NEFullyConnectedLayerQI &) = delete;
    NEFullyConnectedLayerQI(NEFullyConnectedLayerQI &&) = delete;
    NEFullyConnectedLayerQI &operator=(NEFullyConnectedLayerQI &&) = delete;
    ~NEFullyConnectedLayerQI() = default;
    void configure(arm_compute::ITensor *input, const arm_compute::ITensor *weights, const arm_compute::ITensor *biases, arm_compute::ITensor *output,
                   const arm_compute::QuantizationInfo *wp, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
        ARM_COMPUTE_ERROR_THROW_ON(NEFullyConnectedLayerQI::validate(input->info(), weights->info(), ((biases != nullptr) ? biases->info() : nullptr),
                                                                     output->info(), wp, qi));

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

        _fconn = std::make_unique<arm_compute::NEFullyConnectedLayer>(_memory_manager);
        _fconn->configure(input, _wp ? &_weightsqi : _weights, biases, _qi ? &_outputqi : _output);
    }
    static Status validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *weights, const arm_compute::ITensorInfo *biases,
                           const arm_compute::ITensorInfo *output, const arm_compute::QuantizationInfo *wp, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(weights, output);
        //At the moment quantization info isn't checked actually, but just in case
        return arm_compute::NEFullyConnectedLayer::validate(input, &arm_compute::TensorInfo(*weights).set_quantization_info(wp), biases,
                                                            &arm_compute::TensorInfo(*output).set_quantization_info(qi));
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_fconn.get(), "Kernel didn't configured");
        if (_wp) {
            if (_weightsqi.info()->padding() != _weights->info()->padding()) _weightsqi.info()->extend_padding(_weights->info()->padding());
            _weightsqi.allocator()->import_memory(_weights->buffer());
        }
        if (_qi) {
            if (_outputqi.info()->padding() != _output->info()->padding()) _outputqi.info()->extend_padding(_output->info()->padding());
            _outputqi.allocator()->import_memory(_output->buffer());
        }
        _fconn->run();
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
    std::unique_ptr<arm_compute::NEFullyConnectedLayer> _fconn;
};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::MatMul& node) {
    if (node.get_transpose_a()) {
        IE_THROW() << "Can not create MatMul layer with transpose first input";
    }
    auto wInfoIt = node.get_rt_info().find("WeightsPrescaleInfo");
    arm_compute::QuantizationInfo* wInfo = wInfoIt == node.get_rt_info().end() ? nullptr
                                           &(safe_cast<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(wInfoIt->second)->get());
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    arm_compute::QuantizationInfo* qInfo = qInfoIt == node.get_rt_info().end() ? nullptr
                                           &(safe_cast<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(qInfoIt->second)->get());
    return MakeConversion<NEFullyConnectedLayerQI>(node.input(Features), node.input(Weights), nullptr, node.output(0), wInfo, qInfo);
}
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmMatMulBias& node) {
    if (node.get_transpose_a()) {
        IE_THROW() << "Can not create MatMul layer with transpose first input";
    }
    auto wInfoIt = node.get_rt_info().find("WeightsPrescaleInfo");
    arm_compute::QuantizationInfo* wInfo = wInfoIt == node.get_rt_info().end() ? nullptr
                                           &(safe_cast<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(wInfoIt->second)->get());
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    arm_compute::QuantizationInfo* qInfo = qInfoIt == node.get_rt_info().end() ? nullptr
                                           &(safe_cast<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(qInfoIt->second)->get());
    return MakeConversion<NEFullyConnectedLayerQI>(node.input(Features), node.input(Weights), node.input(Bias), node.output(0), wInfo, qInfo);
}
}  //  namespace ArmPlugin
