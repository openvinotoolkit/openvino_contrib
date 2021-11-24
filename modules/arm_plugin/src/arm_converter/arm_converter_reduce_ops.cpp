// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NEReductionOperation.h>
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
        _memory_manager(memory_manager), _rmean(), _output(nullptr), _qi(nullptr), _outputqi() {}
    NEReduceMeanQI(const NEReduceMeanQI &) = delete;
    NEReduceMeanQI &operator=(const NEReduceMeanQI &) = delete;
    NEReduceMeanQI(NEReduceMeanQI &&) = delete;
    NEReduceMeanQI &operator=(NEReduceMeanQI &&) = delete;
    ~NEReduceMeanQI() = default;
    void configure(arm_compute::ITensor *input, const arm_compute::Coordinates &reduction_axis, bool keep_dims, arm_compute::ITensor *output,
                   const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
        ARM_COMPUTE_ERROR_THROW_ON(NEReduceMeanQI::validate(input->info(), reduction_axis, keep_dims, output->info(), qi));
        _output = output;
        _qi = qi;
        if (_qi) {
            _outputqi.allocator()->init(*(_output->info()));
            _outputqi.info()->set_quantization_info(*qi);
        }
        _rmean = std::make_unique<arm_compute::NEReduceMean>(_memory_manager);
        _rmean->configure(input, reduction_axis, keep_dims, _qi ? &_outputqi : _output);
    }
    static Status validate(const arm_compute::ITensorInfo *input, const arm_compute::Coordinates &reduction_axis, bool keep_dims,
                           const arm_compute::ITensorInfo *output, const arm_compute::QuantizationInfo *qi) {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
        //At the moment quantization info isn't checked actually, but just in case
        return arm_compute::NEReduceMean::validate(input, reduction_axis, keep_dims,
                                                   qi ? &arm_compute::TensorInfo(*output).set_quantization_info(qi) : output);
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_rmean.get(), "Kernel didn't configured");
        if (_qi) {
            if (_outputqi.info()->padding() != _output->info()->padding()) _outputqi.info()->extend_padding(_output->info()->padding());
            _outputqi.allocator()->import_memory(_output->buffer());
        }
        _rmean->run();
        if (_qi) _outputqi.allocator()->free();
    }

protected:
    std::shared_ptr<arm_compute::IMemoryManager> _memory_manager;
    const arm_compute::QuantizationInfo *_qi;
    const arm_compute::ITensor *_output;
    arm_compute::Tensor _outputqi;
    std::unique_ptr<arm_compute::NEReduceMean> _rmean;
};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ReduceMean& node) {
    arm_compute::Coordinates axes;
    auto reduction_axes = safe_cast<opset::Constant>(node.input_value(1).get_node())->cast_vector<std::int64_t>();
    for (size_t i = 0; i < reduction_axes.size(); ++i) {
        auto pos = AxisCast(i, reduction_axes.size());
        axes.set(pos, reduction_axes[i]);
    }
    auto qInfoIt = node.get_rt_info().find("QuantizationInfo");
    arm_compute::QuantizationInfo* qInfo = qInfoIt == node.get_rt_info().end() ? nullptr
                                           &(safe_cast<ngraph::VariantWrapper<arm_compute::QuantizationInfo>>(qInfoIt->second)->get());
    return MakeConversion<NEReduceMeanQI>(node.input(0), axes, node.get_keep_dims(), node.output(0), qInfo);
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
