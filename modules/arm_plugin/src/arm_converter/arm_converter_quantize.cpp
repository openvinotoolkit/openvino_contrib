// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "opset/quantize.hpp"
#include <arm_compute/runtime/NEON/functions/NEQuantizationLayer.h>
#include <arm_compute/runtime/NEON/functions/NEDequantizationLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
struct NEQuantizationLayerQI final: public arm_compute::IFunction {
public:
    NEQuantizationLayerQI():
        _quant(), _output(nullptr), _outputqi() {}
    NEQuantizationLayerQI(const NEQuantizationLayerQI &) = delete;
    NEQuantizationLayerQI &operator=(const NEQuantizationLayerQI &) = delete;
    NEQuantizationLayerQI(NEQuantizationLayerQI &&) = delete;
    NEQuantizationLayerQI &operator=(NEQuantizationLayerQI &&) = delete;
    ~NEQuantizationLayerQI() = default;
    void configure(const arm_compute::ITensor *input, arm_compute::ITensor *output, const arm_compute::QuantizationInfo &qi) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
        ARM_COMPUTE_ERROR_THROW_ON(NEQuantizationLayerQI::validate(input->info(), output->info(), qi));
        _output = output;
        _outputqi.allocator()->init(*(_output->info()));
        _outputqi.info()->set_quantization_info(qi);
        _quant = std::make_unique<arm_compute::NEQuantizationLayer>();
        _quant->configure(input, &_outputqi);
    }
    static arm_compute::Status validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *output,
                                        const arm_compute::QuantizationInfo &qi) {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
        //At the moment quantization info isn't checked actually, but just in case
        return arm_compute::NEQuantizationLayer::validate(input, &arm_compute::TensorInfo(*output).set_quantization_info(qi));
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_quant.get(), "Kernel didn't configured");
        if (_outputqi.info()->padding() != _output->info()->padding()) _outputqi.info()->extend_padding(_output->info()->padding());
        _outputqi.allocator()->import_memory(_output->buffer());
        _quant->run();
        _outputqi.allocator()->free();
    }

protected:
    const arm_compute::ITensor *_output;
    arm_compute::Tensor _outputqi;
    std::unique_ptr<arm_compute::NEQuantizationLayer> _quant;
};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmQuantize& node) {
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    if (qInfoIt == node.get_rt_info().end()) {
        IE_THROW() << "No quantization info available for ArmQuantize";
    }
    auto qInfo = qInfoIt->second.as<arm_compute::QuantizationInfo>();
    return MakeConversion<NEQuantizationLayerQI>(node.input(0), node.output(0), qInfo);
}

struct NEDequantizationLayerQI final: public arm_compute::IFunction {
public:
    NEDequantizationLayerQI():
        _dequant(), _input(nullptr), _inputqi() {}
    NEDequantizationLayerQI(const NEDequantizationLayerQI &) = delete;
    NEDequantizationLayerQI &operator=(const NEDequantizationLayerQI &) = delete;
    NEDequantizationLayerQI(NEDequantizationLayerQI &&) = delete;
    NEDequantizationLayerQI &operator=(NEDequantizationLayerQI &&) = delete;
    ~NEDequantizationLayerQI() = default;
    void configure(const arm_compute::ITensor *input, arm_compute::ITensor *output, const arm_compute::QuantizationInfo &qi) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
        ARM_COMPUTE_ERROR_THROW_ON(NEDequantizationLayerQI::validate(input->info(), output->info(), qi));
        _input = input;
        _inputqi.allocator()->init(*(_input->info()));
        _inputqi.info()->set_quantization_info(qi);
        _dequant = std::make_unique<arm_compute::NEDequantizationLayer>();
        _dequant->configure(&_inputqi, output);
    }
    static arm_compute::Status validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *output,
                                        const arm_compute::QuantizationInfo &qi) {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
        //At the moment quantization info isn't checked actually, but just in case
        return arm_compute::NEDequantizationLayer::validate(&arm_compute::TensorInfo(*input).set_quantization_info(qi), output);
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_dequant.get(), "Kernel didn't configured");
        if (_inputqi.info()->padding() != _input->info()->padding()) _inputqi.info()->extend_padding(_input->info()->padding());
        _inputqi.allocator()->import_memory(_input->buffer());
        _dequant->run();
        _inputqi.allocator()->free();
    }

protected:
    const arm_compute::ITensor *_input;
    arm_compute::Tensor _inputqi;
    std::unique_ptr<arm_compute::NEDequantizationLayer> _dequant;
};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmDequantize& node) {
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    if (qInfoIt == node.get_rt_info().end()) {
        IE_THROW() << "No quantization info available for ArmDequantize";
    }
    auto qInfo = qInfoIt->second.as<arm_compute::QuantizationInfo>();
    return MakeConversion<NEDequantizationLayerQI>(node.input(0), node.output(0), qInfo);
}
} // namespace ArmPlugin