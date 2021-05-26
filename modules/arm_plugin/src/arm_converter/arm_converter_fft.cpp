// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


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
    wrap_fft(fft_data.data(), data_shape, axes, axes_shape, fft_result.data(), output_shape, kind);
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
    auto axes = std::dynamic_pointer_cast<opset::Constant>(node.input_value(1).get_node_shared_ptr());
    if (!axes) {
        IE_THROW() << "Supported FFT op with constant axes only";
    }
    axes_vals = axes->cast_vector<int64_t>();

    if (node.get_input_size() == 3) {
        if (node.get_input_shape(2).size() != node.get_input_shape(1).size()) {
            IE_THROW() << "Signal size input length should be equal to axis input length";
        }
        auto signal_sizes = std::dynamic_pointer_cast<opset::Constant>(node.input_value(2).get_node_shared_ptr());
        if (!signal_sizes) {
            IE_THROW() << "Supported FFT op with constant signal sizes only";
        }
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
} // namespace ArmPlugin
