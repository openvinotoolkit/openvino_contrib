// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEPoolingLayer.h>
#include "arm_converter/arm_converter.hpp"


namespace ArmPlugin {

template<typename Pool>
static void FillLayerInfo(const Pool& node, arm_compute::PoolingLayerInfo& pool_info) {
    unsigned int pad_left   = node.get_pads_begin().at(D2::W);
    unsigned int pad_right  = node.get_pads_end().at(D2::W);
    unsigned int pad_top    = node.get_pads_begin().at(D2::H);
    unsigned int pad_bottom = node.get_pads_end().at(D2::H);
    unsigned int kernel_w   = node.get_kernel().at(D2::W);
    unsigned int kernel_h   = node.get_kernel().at(D2::H);
    unsigned int stride_x   = node.get_strides().at(D2::W);
    unsigned int stride_y   = node.get_strides().at(D2::H);

    arm_compute::DimensionRoundingType round = (node.get_rounding_type() == ngraph::op::RoundingType::FLOOR)
                                             ? arm_compute::DimensionRoundingType::FLOOR
                                             : arm_compute::DimensionRoundingType::CEIL;

    pool_info.data_layout       = arm_compute::DataLayout::NCHW;
    pool_info.pool_size         = arm_compute::Size2D(kernel_w, kernel_h);
    pool_info.pad_stride_info   = arm_compute::PadStrideInfo(stride_x, stride_y, pad_left, pad_right, pad_top, pad_bottom, round);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::MaxPool& node) {
    arm_compute::PoolingLayerInfo pool_info;
    FillLayerInfo(node, pool_info);
    pool_info.pool_type = arm_compute::PoolingType::MAX;
    return MakeConversion<arm_compute::NEPoolingLayer>(node.input(0), node.output(0), pool_info);
}

struct NEPoolingLayerQI final: public arm_compute::IFunction {
public:
    NEPoolingLayerQI(std::shared_ptr<arm_compute::IMemoryManager> memory_manager = nullptr):
        _memory_manager(memory_manager), _pool(), _output(nullptr), _qi(nullptr), _outputqi() {}
    NEPoolingLayerQI(const NEPoolingLayerQI &) = delete;
    NEPoolingLayerQI &operator=(const NEPoolingLayerQI &) = delete;
    NEPoolingLayerQI(NEPoolingLayerQI &&) = delete;
    NEPoolingLayerQI &operator=(NEPoolingLayerQI &&) = delete;
    ~NEPoolingLayerQI() = default;
    void configure(arm_compute::ITensor *input, arm_compute::ITensor *output, const arm_compute::PoolingLayerInfo &pool_info,
                   const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
        ARM_COMPUTE_ERROR_THROW_ON(NEPoolingLayerQI::validate(input->info(), output->info(), pool_info, qi));
        _output = output;
        _qi = qi;
        if (_qi) {
            _outputqi.allocator()->init(*(_output->info()));
            _outputqi.info()->set_quantization_info(*qi);
        }
        _pool = std::make_unique<arm_compute::NEPoolingLayer>(_memory_manager);
        _pool->configure(input, _qi ? &_outputqi : _output, pool_info);
    }
    static Status validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *output, const arm_compute::PoolingLayerInfo &pool_info,
                           const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
        //At the moment quantization info isn't checked actually, but just in case
        return arm_compute::NEPoolingLayer::validate(input, qi ? &arm_compute::TensorInfo(*output).set_quantization_info(qi) : output, pool_info);
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_pool.get(), "Kernel didn't configured");
        if (_qi) {
            if (_outputqi.info()->padding() != _output->info()->padding()) _outputqi.info()->extend_padding(_output->info()->padding());
            _outputqi.allocator()->import_memory(_output->buffer());
        }
        _pool->run();
        if (_qi) _outputqi.allocator()->free();
    }

protected:
    std::shared_ptr<arm_compute::IMemoryManager> _memory_manager;
    const arm_compute::QuantizationInfo *_qi;
    const arm_compute::ITensor *_output;
    arm_compute::Tensor _outputqi;
    std::unique_ptr<arm_compute::NEPoolingLayer> _pool;
};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::AvgPool& node) {
    arm_compute::PoolingLayerInfo pool_info;
    FillLayerInfo(node, pool_info);
    pool_info.pool_type       = arm_compute::PoolingType::AVG;
    pool_info.exclude_padding = node.get_exclude_pad();
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    arm_compute::QuantizationInfo* qInfo = qInfoIt == node.get_rt_info().end() ? nullptr
                                           &(safe_cast<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(qInfoIt->second)->get());
    return MakeConversion<NEPoolingLayerQI>(node.input(0), node.output(0), pool_info, qInfo);
}
}  // namespace ArmPlugin
