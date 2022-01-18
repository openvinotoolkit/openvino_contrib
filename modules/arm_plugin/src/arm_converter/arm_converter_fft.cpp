// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEFFT1D.h>
#include <ngraph/runtime/reference/fft.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {

template<typename D>
void wrap_fft(const D* data,
              const ngraph::Shape& data_shape,
              const std::vector<std::int64_t>& axes,
              const ngraph::Shape& axes_shape,
              D* out,
              const ngraph::Shape& output_shape,
              ngraph::runtime::reference::FFTKind kind) {
    std::vector<float> fft_data(data_shape.size());
    for (size_t i = 0; i < fft_data.size(); ++i) {
        fft_data[i] = static_cast<float>(data[i]);
    }
    std::vector<float> fft_result(ngraph::shape_size(output_shape), 0.0f);
    wrap_fft<float>(fft_data.data(), data_shape, axes, axes_shape, fft_result.data(), output_shape, kind);
    for (size_t i = 0; i < fft_result.size(); ++i) {
        out[i] = D(fft_result[i]);
    }
}
template<>
void wrap_fft<float>(const float* data,
                     const ngraph::Shape& data_shape,
                     const std::vector<std::int64_t>& axes,
                     const ngraph::Shape& axes_shape,
                     float* out,
                     const ngraph::Shape& output_shape,
                     ngraph::runtime::reference::FFTKind kind) {
    ngraph::runtime::reference::fft(data,
                                    data_shape,
                                    axes.data(),
                                    axes_shape,
                                    out,
                                    output_shape,
                                    kind);
}

static void verify_fft_args(const ngraph::op::util::FFTBase& node, std::vector<int64_t>& axes_vals, ngraph::Shape& output_shape) {
    output_shape = node.get_input_shape(0);
    auto axes = safe_cast<opset::Constant>(node.input_value(1).get_node_shared_ptr());
    axes_vals = axes->cast_vector<int64_t>();

    if (node.get_input_size() == 3) {
        if (node.get_input_shape(2).size() != node.get_input_shape(1).size()) {
            IE_THROW() << "Signal size input length should be equal to axis input length";
        }
        auto signal_sizes = safe_cast<opset::Constant>(node.input_value(2).get_node_shared_ptr());
        std::vector<int64_t> signal_size_vals = signal_sizes->cast_vector<int64_t>();

        std::int64_t input_rank = static_cast<std::int64_t>(output_shape.size());
        for (size_t i = 0; i < axes_vals.size(); ++i) {
            if (signal_size_vals[i] != -1) {
                int64_t current_axis = axes_vals[i];
                if (current_axis < 0) {
                    current_axis += input_rank - 1;
                }
                output_shape[current_axis] = signal_size_vals[i];
            }
        }
    }
    //node.get_output_shape(0) = output_shape;
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::DFT& node) {
    auto make = [&] (auto refFunction) {
        std::vector<std::int64_t> axes_vals;
        ngraph::Shape output_shape;
        verify_fft_args(node, axes_vals, output_shape);
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.get_input_shape(0),
                                    axes_vals,
                                    node.get_input_shape(1),
                                    node.output(0),
                                    output_shape,
                                    ngraph::runtime::reference::FFTKind::Forward);
    };
    return CallSwitch(
        AP_WRAP(make, wrap_fft),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::IDFT& node) {
    auto make = [&] (auto refFunction) {
        std::vector<std::int64_t> axes_vals;
        ngraph::Shape output_shape;
        verify_fft_args(node, axes_vals, output_shape);
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.get_input_shape(0),
                                    axes_vals,
                                    node.get_input_shape(1),
                                    node.output(0),
                                    output_shape,
                                    ngraph::runtime::reference::FFTKind::Inverse);
    };
    return CallSwitch(
        AP_WRAP(make, wrap_fft),
        node.input(0), floatTypes);
}

struct NEFFT1Ds2 final: public arm_compute::IFunction {
public:
    NEFFT1Ds2(const std::shared_ptr<arm_compute::IMemoryManager>& memory_manager):
        _memory_manager(memory_manager), _fft(), _input(nullptr), _output(nullptr), _input2c(), _output2c() {}
    NEFFT1Ds2(const NEFFT1Ds2 &) = delete;
    NEFFT1Ds2 &operator=(const NEFFT1Ds2 &) = delete;
    NEFFT1Ds2(NEFFT1Ds2 &&) = delete;
    NEFFT1Ds2 &operator=(NEFFT1Ds2 &&) = delete;
    ~NEFFT1Ds2() = default;
    void configure(const arm_compute::ITensor *input, arm_compute::ITensor *output, const arm_compute::FFT1DInfo &config) {
        ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
        ARM_COMPUTE_ERROR_THROW_ON(NEFFT1Ds2::validate(input->info(), output->info(), config));
        _input = input;
        _output = output;
        _input2c.allocator()->init(shape2chan(_input->info()));
        _output2c.allocator()->init(shape2chan(_output->info()));

        _fft = std::make_unique<arm_compute::NEFFT1D>(_memory_manager);
        _fft->configure(&_input2c, &_output2c, config);
    }
    static arm_compute::Status validate(const arm_compute::ITensorInfo *input, const arm_compute::ITensorInfo *output, const arm_compute::FFT1DInfo &config) {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
        arm_compute::TensorInfo ishape = shape2chan(input);
        arm_compute::TensorInfo oshape = shape2chan(output);
        return arm_compute::NEFFT1D::validate(&ishape, &oshape, config);
    }
    void run() override {
        ARM_COMPUTE_ERROR_ON_MSG(!_fft.get(), "Kernel didn't configured");
        _input2c.allocator()->import_memory(_input->buffer());
        _output2c.allocator()->import_memory(_output->buffer());
        _fft->run();
    }

protected:
    static arm_compute::TensorInfo shape2chan(const arm_compute::ITensorInfo *t_info) {
        arm_compute::TensorShape shape = t_info->tensor_shape();
        int num_channels = shape[0];
        shape.remove_dimension(0);
        return arm_compute::TensorInfo(shape, num_channels, t_info->data_type());
    }
    std::shared_ptr<arm_compute::IMemoryManager> _memory_manager;
    const arm_compute::ITensor *_input;
    arm_compute::ITensor *_output;
    arm_compute::Tensor                          _input2c;
    arm_compute::Tensor                          _output2c;
    std::unique_ptr<arm_compute::NEFFT1D>        _fft;
};

template <> Converter::Conversion::Ptr Converter::Convert(const opset::ArmFFT& node) {
    arm_compute::FFT1DInfo fft_cfg{node.get_arm_axis(),
                                   node.is_inverse() ? arm_compute::FFTDirection::Inverse : arm_compute::FFTDirection::Forward};
    return MakeConversion<NEFFT1Ds2>(node.input(0), node.output(0), fft_cfg);
}
} // namespace ArmPlugin
