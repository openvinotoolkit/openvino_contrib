// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEActivationLayer.h>
#include <arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h>
#include <arm_compute/runtime/NEON/functions/NEFloor.h>
#include <arm_compute/runtime/NEON/functions/NEPReluLayer.h>
#include <ngraph/runtime/reference/hsigmoid.hpp>
#include <ngraph/runtime/reference/hard_sigmoid.hpp>
#include <ngraph/runtime/reference/selu.hpp>
#include <ngraph/runtime/reference/gelu.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<typename Activation>
static auto ConvertActivation(const Activation& node, const arm_compute::ActivationLayerInfo& info, Converter* converter) {
    return converter->MakeConversion<arm_compute::NEActivationLayer>(node.input(0), node.output(0), info);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Sigmoid& node) {
    arm_compute::ActivationLayerInfo info(arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC);
    return ConvertActivation(node, info, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Tanh& node) {
    // TanH(x, a, b) = a * std::tanh(b * x);
    arm_compute::ActivationLayerInfo info(arm_compute::ActivationLayerInfo::ActivationFunction::TANH, 1.0f, 1.0f);
    return ConvertActivation(node, info, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Relu& node) {
    arm_compute::ActivationLayerInfo info(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
    return ConvertActivation(node, info, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::PRelu& node) {
    return MakeConversion<arm_compute::NEPReluLayer>(node.input(0), node.input(1), node.output(0));
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Abs& node) {
    arm_compute::ActivationLayerInfo info(arm_compute::ActivationLayerInfo::ActivationFunction::ABS);
    return ConvertActivation(node, info, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Clamp& node) {
    arm_compute::ActivationLayerInfo info(arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, node.get_max(), node.get_min());
    return ConvertActivation(node, info, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Sqrt& node) {
    arm_compute::ActivationLayerInfo info(arm_compute::ActivationLayerInfo::ActivationFunction::SQRT);
    return ConvertActivation(node, info, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Elu& node) {
    arm_compute::ActivationLayerInfo info(arm_compute::ActivationLayerInfo::ActivationFunction::ELU, node.get_alpha());
    return ConvertActivation(node, info, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Negative& node) {
    return MakeConversion<arm_compute::NENegLayer>(node.input(0), node.output(0));
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Exp& node) {
    return MakeConversion<arm_compute::NEExpLayer>(node.input(0), node.output(0));
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Floor& node) {
    return MakeConversion<arm_compute::NEFloor>(node.input(0), node.output(0));
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::HSwish& node) {
    arm_compute::ActivationLayerInfo info(arm_compute::ActivationLayerInfo::ActivationFunction::HARD_SWISH);
    return ConvertActivation(node, info, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::SoftPlus& node) {
    arm_compute::ActivationLayerInfo info(arm_compute::ActivationLayerInfo::ActivationFunction::SOFT_RELU);
    return ConvertActivation(node, info, this);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Log& node) {
    return MakeConversion<arm_compute::NELogLayer>(node.input(0), node.output(0));
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::HSigmoid& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::hsigmoid),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Gelu& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), node.get_approximation_mode(), ngraph::shape_size(node.get_output_shape(0)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::gelu),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::HardSigmoid& node) {
    auto make = [&] (auto refFunction, auto alpha, auto beta) {
        return this->MakeConversion(refFunction, node.input(0), alpha, beta, node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };

    auto alpha_node = safe_cast<opset::Constant>(node.input_value(1).get_node_shared_ptr());
    auto beta_node  = safe_cast<opset::Constant>(node.input_value(2).get_node_shared_ptr());

    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::f16 : {
            auto alpha = alpha_node->cast_vector<ngraph::float16>()[0];
            auto beta  = beta_node->cast_vector<ngraph::float16>()[0];
            return make(ngraph::runtime::reference::hard_sigmoid<ngraph::float16>, alpha, beta);
        }
        case ngraph::element::Type_t::f32 : {
            auto alpha = alpha_node->cast_vector<float>()[0];
            auto beta  = beta_node->cast_vector<float>()[0];
            return make(ngraph::runtime::reference::hard_sigmoid<float>, alpha, beta);
        }
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Selu& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(2),
                                    node.output(0),
                                    ngraph::shape_size(node.get_input_shape(0)),
                                    ngraph::shape_size(node.get_input_shape(1)),
                                    ngraph::shape_size(node.get_input_shape(2)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::selu),
        node.input(0), floatTypes);
}
} // namespace ArmPlugin
