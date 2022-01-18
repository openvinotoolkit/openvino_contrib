// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NEReductionOperation.h>
#include <src/cpu/kernels/CpuConvertQuantizedSignednessKernel.h>
#include <arm_compute/runtime/NEON/NEScheduler.h>
#include <arm_compute/runtime/NEON/functions/NEReduceMean.h>
#include <ngraph/runtime/reference/logical_reduction.hpp>
#include "arm_converter/arm_converter.hpp"
#include "opset/utils.hpp"

namespace ArmPlugin {
template<typename Reduce>
static auto ConvertReduce(const Reduce& node, const arm_compute::ReductionOperation& op, Converter* converter) {
    auto axes = safe_cast<opset::Constant>(node.input_value(1).get_node())->template cast_vector<std::int64_t>();
    if (axes.size() != 1) {
        IE_THROW() << "Arm Plugin: Multiple reduction axes aren't supported";
    }
    unsigned int axis = AxisCast(axes[0], node.get_input_shape(0).size());
    return converter->MakeConversion<arm_compute::NEReductionOperation>(node.input(0), node.output(0), axis, op, node.get_keep_dims());
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceProd& node) {
    return ConvertReduce(node, arm_compute::ReductionOperation::PROD, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceMin& node) {
    return ConvertReduce(node, arm_compute::ReductionOperation::MIN, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceMax& node) {
    return ConvertReduce(node, arm_compute::ReductionOperation::MAX, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceSum& node) {
    return ConvertReduce(node, arm_compute::ReductionOperation::SUM, this);
}

struct NEReduceMeanQI final: public arm_compute::IFunction {
public:
    NEReduceMeanQI(std::shared_ptr<arm_compute::IMemoryManager> memory_manager = nullptr):
        _memory_manager(memory_manager), _memory_group{std::make_unique<arm_compute::MemoryGroup>(memory_manager)},
        _i_sgn(nullptr), _rmean(nullptr), _input(nullptr), _ip(nullptr), _inputqi(), _output(nullptr), _qi(nullptr), _outputqi() {}
    NEReduceMeanQI(const NEReduceMeanQI &) = delete;
    NEReduceMeanQI &operator=(const NEReduceMeanQI &) = delete;
    NEReduceMeanQI(NEReduceMeanQI &&) = delete;
    NEReduceMeanQI &operator=(NEReduceMeanQI &&) = delete;
    ~NEReduceMeanQI() = default;
    void configure(arm_compute::ITensor *input, const arm_compute::Coordinates &reduction_axis, bool keep_dims, arm_compute::ITensor *output,
                   const arm_compute::QuantizationInfo *ip, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
        ARM_COMPUTE_ERROR_THROW_ON(NEReduceMeanQI::validate(input->info(), reduction_axis, keep_dims, output->info(), ip, qi));

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
        _rmean = std::make_unique<arm_compute::NEReduceMean>(_memory_manager);
        _rmean->configure(conv_input, reduction_axis, keep_dims, _qi ? &_outputqi : _output);

        if (_i_sgn) {
            _inputqi.allocator()->allocate();
        } else if (_ip && _inputqi.info()->padding() != _input->info()->padding()) {
            //Backpropagate possible input padding change
            _input->info()->extend_padding(_inputqi.info()->padding());
        }
    }
    static arm_compute::Status validate(const arm_compute::ITensorInfo *input, const arm_compute::Coordinates &reduction_axis, bool keep_dims,
                                        const arm_compute::ITensorInfo *output,
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

        return arm_compute::NEReduceMean::validate(&vld_input, reduction_axis, keep_dims, &vld_output);
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_rmean.get(), "Kernel didn't configured");
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
        _rmean->run();
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
    std::unique_ptr<arm_compute::NEReduceMean> _rmean;
};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceMean& node) {
    arm_compute::Coordinates axes;
    auto reduction_axes = safe_cast<opset::Constant>(node.input_value(1).get_node())->cast_vector<std::int64_t>();
    for (size_t i = 0; i < reduction_axes.size(); ++i) {
        auto pos = AxisCast(i, reduction_axes.size());
        axes.set(pos, reduction_axes[i]);
    }
    auto iInfoIt = node.get_rt_info().find("InputPrescaleInfo");
    const arm_compute::QuantizationInfo* iInfo = iInfoIt == node.get_rt_info().end() ? nullptr :
                                               &(iInfoIt->second.as<arm_compute::QuantizationInfo>());
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    const arm_compute::QuantizationInfo* qInfo = qInfoIt == node.get_rt_info().end() ? nullptr :
                                               &(qInfoIt->second.as<arm_compute::QuantizationInfo>());
    return MakeConversion<NEReduceMeanQI>(node.input(0), axes, node.get_keep_dims(), node.output(0), iInfo, qInfo);
}

static void wrap_reduce_logical_and(const std::uint8_t* arg,
                                    std::uint8_t* out,
                                    const ngraph::Shape& input_shape,
                                    const ngraph::AxisSet& reduction_axes) {
    ngraph::runtime::reference::reduce_logical_and(reinterpret_cast<const char*>(arg),
                                                   reinterpret_cast<char*>(out),
                                                   input_shape,
                                                   reduction_axes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceLogicalAnd& node) {
    if (node.get_input_element_type(0) != ngraph::element::u8) {
        IE_THROW() << "Arm Plugin: Unsupported Type: " << node.get_input_element_type(0);
    }

    return MakeConversion(wrap_reduce_logical_and,
                          node.input(0),
                          node.output(0),
                          node.get_input_shape(0),
                          node.get_reduction_axes());
}

static void wrap_reduce_logical_or(const std::uint8_t* arg,
                                   std::uint8_t* out,
                                   const ngraph::Shape& input_shape,
                                   const ngraph::AxisSet& reduction_axes) {
    ngraph::runtime::reference::reduce_logical_or(reinterpret_cast<const char*>(arg),
                                                  reinterpret_cast<char*>(out),
                                                  input_shape,
                                                  reduction_axes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceLogicalOr& node) {
    if (node.get_input_element_type(0) != ngraph::element::u8) {
        IE_THROW() << "Arm Plugin: Unsupported Type: " << node.get_input_element_type(0);
    }

    return MakeConversion(wrap_reduce_logical_or,
                          node.input(0),
                          node.output(0),
                          node.get_input_shape(0),
                          node.get_reduction_axes());
}

} // namespace ArmPlugin
