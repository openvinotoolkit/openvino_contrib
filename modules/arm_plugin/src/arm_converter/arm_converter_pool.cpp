// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <src/cpu/kernels/CpuConvertQuantizedSignednessKernel.h>
#include <arm_compute/runtime/NEON/NEScheduler.h>
#include <arm_compute/runtime/NEON/functions/NEPoolingLayer.h>
#include <ngraph/runtime/reference/max_pool.hpp>
#include <ngraph/runtime/reference/avg_pool.hpp>
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
    if (node.get_auto_pad() != ngraph::op::PadType::EXPLICIT) {
        pool_info.exclude_padding = true;
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::MaxPool& node) {
    if (node.get_input_shape(0).size() == 4) {
        arm_compute::PoolingLayerInfo pool_info;
        FillLayerInfo(node, pool_info);
        pool_info.pool_type = arm_compute::PoolingType::MAX;
        return MakeConversion<arm_compute::NEPoolingLayer>(node.input(0), node.output(0), pool_info);
    } else {
        auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_output_shape(0),
                                    node.get_kernel(),
                                    node.get_strides(),
                                    node.get_pads_begin(),
                                    node.get_pads_end());
        };
        return CallSwitch(
            AP_WRAP(make, ngraph::runtime::reference::max_pool),
            node.input(0), allTypes);
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v8::MaxPool& node) {
    if ((node.get_input_shape(0).size() == 4) &&
       (node.get_output_element_type(1) == ngraph::element::u32) &&
       (node.get_kernel() == ngraph::Shape{2, 2}) &&
       (node.get_dilations() == ngraph::Strides{1, 1}) &&
       (node.get_axis() == 0)) {
        arm_compute::PoolingLayerInfo pool_info;
        FillLayerInfo(node, pool_info);
        pool_info.pool_type = arm_compute::PoolingType::MAX;
        return MakeConversion<arm_compute::NEPoolingLayer>(node.input(0), node.output(0), pool_info, node.output(1));
    } else {
        auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.output(0),
                                    node.output(1),
                                    node.get_input_shape(0),
                                    node.get_output_shape(0),
                                    node.get_kernel(),
                                    node.get_strides(),
                                    node.get_dilations(),
                                    node.get_pads_begin(),
                                    node.get_pads_end(),
                                    node.get_axis());
        };
        return CallSwitch(
            AP_WRAP(make, ngraph::runtime::reference::max_pool),
            node.input(0), allTypes,
            node.output(1), indexTypes);
    }
}

struct NEPoolingLayerQI final: public arm_compute::IFunction {
public:
    NEPoolingLayerQI(std::shared_ptr<arm_compute::IMemoryManager> memory_manager = nullptr):
        _memory_manager(memory_manager), _memory_group{std::make_unique<arm_compute::MemoryGroup>(memory_manager)},
        _i_sgn(nullptr), _pool(nullptr), _input(nullptr), _ip(nullptr), _inputqi(), _output(nullptr), _qi(nullptr), _outputqi() {}
    NEPoolingLayerQI(const NEPoolingLayerQI &) = delete;
    NEPoolingLayerQI &operator=(const NEPoolingLayerQI &) = delete;
    NEPoolingLayerQI(NEPoolingLayerQI &&) = delete;
    NEPoolingLayerQI &operator=(NEPoolingLayerQI &&) = delete;
    ~NEPoolingLayerQI() = default;
    void configure(arm_compute::ITensor *input, arm_compute::ITensor *output, const arm_compute::PoolingLayerInfo &pool_info,
                   const arm_compute::QuantizationInfo *ip, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
        ARM_COMPUTE_ERROR_THROW_ON(NEPoolingLayerQI::validate(input->info(), output->info(), pool_info, ip, qi));

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

        _output = output;
        _qi = qi;
        if (_qi) {
            _outputqi.allocator()->init(*(_output->info()));
            _outputqi.info()->set_quantization_info(*qi);
        }
        _pool = std::make_unique<arm_compute::NEPoolingLayer>(_memory_manager);
        _pool->configure(conv_input, _qi ? &_outputqi : _output, pool_info);

        if (_i_sgn) {
            _inputqi.allocator()->allocate();
        } else if (_ip && _inputqi.info()->padding() != _input->info()->padding()) {
            //Backpropagate possible input padding change
            _input->info()->extend_padding(_inputqi.info()->padding());
        }
    }
    static arm_compute::Status validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *output,
                                        const arm_compute::PoolingLayerInfo &pool_info,
                                        const arm_compute::QuantizationInfo *ip, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
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
        arm_compute::TensorInfo vld_output(*output);
        if (qi) vld_output.set_quantization_info(*qi);

        return arm_compute::NEPoolingLayer::validate(&vld_input, &vld_output, pool_info);
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_pool.get(), "Kernel didn't configured");
        std::unique_ptr<arm_compute::MemoryGroupResourceScope> _sgn_scope;
        if (_i_sgn) {
            _sgn_scope = std::make_unique<arm_compute::MemoryGroupResourceScope>(*_memory_group);
            arm_compute::ITensorPack pack = {
                { arm_compute::TensorType::ACL_SRC, _input },
                { arm_compute::TensorType::ACL_DST, &_inputqi }
            };
            arm_compute::NEScheduler::get().schedule_op(_i_sgn.get(), arm_compute::Window::DimY, _i_sgn->window(), pack);
        } else if (_ip) {
            if (_inputqi.info()->padding() != _input->info()->padding()) _inputqi.info()->extend_padding(_input->info()->padding());
            _inputqi.allocator()->import_memory(_input->buffer());
        }
        if (_qi) {
            if (_outputqi.info()->padding() != _output->info()->padding()) _outputqi.info()->extend_padding(_output->info()->padding());
            _outputqi.allocator()->import_memory(_output->buffer());
        }
        _pool->run();
        if (!_i_sgn && _ip) _inputqi.allocator()->free();
        if (_qi) _outputqi.allocator()->free();
    }

protected:
    std::shared_ptr<arm_compute::IMemoryManager> _memory_manager;
    std::unique_ptr<arm_compute::MemoryGroup> _memory_group;
    const arm_compute::QuantizationInfo *_ip;
    arm_compute::ITensor *_input;
    arm_compute::Tensor _inputqi;
    const arm_compute::QuantizationInfo *_qi;
    arm_compute::ITensor *_output;
    arm_compute::Tensor _outputqi;
    std::unique_ptr<arm_compute::cpu::kernels::CpuConvertQuantizedSignednessKernel> _i_sgn;
    std::unique_ptr<arm_compute::NEPoolingLayer> _pool;
};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::AvgPool& node) {
    auto iInfoIt = node.get_rt_info().find("InputPrescaleInfo");
    const arm_compute::QuantizationInfo* iInfo = iInfoIt == node.get_rt_info().end() ? nullptr :
                                               &(iInfoIt->second.as<arm_compute::QuantizationInfo>());
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    const arm_compute::QuantizationInfo* qInfo = qInfoIt == node.get_rt_info().end() ? nullptr :
                                               &(qInfoIt->second.as<arm_compute::QuantizationInfo>());
    if (node.get_input_shape(0).size() == 4) {
        arm_compute::PoolingLayerInfo pool_info;
        FillLayerInfo(node, pool_info);
        pool_info.pool_type       = arm_compute::PoolingType::AVG;
        pool_info.exclude_padding = node.get_exclude_pad();
        return MakeConversion<NEPoolingLayerQI>(node.input(0), node.output(0), pool_info, iInfo, qInfo);
    } else if (!iInfo && !qInfo) {
        auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_output_shape(0),
                                    node.get_kernel(),
                                    node.get_strides(),
                                    node.get_pads_begin(),
                                    node.get_pads_end(),
                                    !node.get_exclude_pad());
        };
        return CallSwitch(
            AP_WRAP(make, ngraph::runtime::reference::avg_pool),
            node.input(0), allTypes);
    } else {
        IE_THROW() << "AvgPool node doesn't support quantization for " << node.get_input_shape(0) << " input shape.";
        return {};
    }
}
}  // namespace ArmPlugin
