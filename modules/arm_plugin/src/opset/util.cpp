// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_arm.hpp"
#include "arm_compute/core/Rounding.h"

namespace ArmPlugin {
namespace opset {

float round(const float v) {
#ifdef __aarch64__
            return arm_compute::round(v, arm_compute::RoundingPolicy::TO_NEAREST_EVEN);
#else  // __aarch64__
            return arm_compute::round(v, arm_compute::RoundingPolicy::TO_ZERO);
#endif // __aarch64__z
}

arm_compute::QuantizationInfo makeQuantizationInfo(
                const ngraph::Output<ngraph::Node>& input_low_output,
                const ngraph::Output<ngraph::Node>& input_high_output,
                const ngraph::Output<ngraph::Node>& output_low_output,
                const ngraph::Output<ngraph::Node>& output_high_output) {
    auto data_type = input_low_output.get_element_type();
    std::vector<float> scale_vector;
    std::vector<std::int32_t> zero_point_vector;
    auto add_chanel = [&](float min, float max, float qMin, float qMax) {
        auto scale = (max - min) / (qMax - qMin);
        auto zeroPointReal = qMin - min / scale;
        std::int32_t zeroPointNudged = 0;
        if (zeroPointReal < qMin) {
            zeroPointNudged = qMin;
        } else if (zeroPointReal > qMax) {
            zeroPointNudged = qMax;
        } else {
            zeroPointNudged = static_cast<std::int32_t>(std::round(zeroPointReal));
        }
        scale_vector.emplace_back(scale);
        zero_point_vector.emplace_back(zeroPointNudged);
    };
    auto init = [&] (auto get_vector) {
        auto input_low = get_vector(input_low_output);
        auto input_high = get_vector(input_high_output);
        auto output_low = get_vector(output_low_output);
        auto output_high = get_vector(output_high_output);
        IE_ASSERT(input_low.size() == input_high.size());
        for (std::size_t i = 0; i < input_low.size(); ++i) {
            add_chanel(input_low[i], input_high[i], output_low[0], output_high[0]);
        }
    };
    if (data_type == ngraph::element::Type_t::f16) {
        init([&](const ngraph::Output<ngraph::Node>& input) {
            return ngraph::as_type<opset::Constant>(input.get_node())->cast_vector<ngraph::float16>();
        });
    } else if (data_type == ngraph::element::Type_t::f32) {
        init([&](const ngraph::Output<ngraph::Node>& input) {
            return ngraph::as_type<opset::Constant>(input.get_node())->cast_vector<float>();
        });
    } else {
        IE_THROW() << "Arm Plugin: Unsupported Data type: " << data_type;
    }
    return {scale_vector, zero_point_vector};
}

arm_compute::ActivationLayerInfo makeActivationLayerInfo(ngraph::Node* node) {
    if (ngraph::is_type<opset::Sigmoid>(node)) {
        return {arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC};
    } else if (ngraph::is_type<opset::Tanh>(node)) {
        return {arm_compute::ActivationLayerInfo::ActivationFunction::TANH};
    } else if (ngraph::is_type<opset::Relu>(node)) {
        return {arm_compute::ActivationLayerInfo::ActivationFunction::RELU};
    } else if (ngraph::is_type<opset::Abs>(node)) {
        return {arm_compute::ActivationLayerInfo::ActivationFunction::ABS};
    } else if (ngraph::is_type<opset::Elu>(node)) {
        return {arm_compute::ActivationLayerInfo::ActivationFunction::ELU,
                static_cast<float>(ngraph::as_type<opset::Elu>(node)->get_alpha())};
    } else if (ngraph::is_type<opset::Sqrt>(node)) {
        return {arm_compute::ActivationLayerInfo::ActivationFunction::SQRT};
    } else if (ngraph::is_type<opset::SoftPlus>(node)) {
        return {arm_compute::ActivationLayerInfo::ActivationFunction::SOFT_RELU};
    } else if (ngraph::is_type<opset::HSwish>(node)) {
        return {arm_compute::ActivationLayerInfo::ActivationFunction::HARD_SWISH};
    } else if (ngraph::is_type<opset::PRelu>(node)) {
        auto prelu = ngraph::as_type<opset::PRelu>(node);
        auto a = ngraph::as_type<opset::Constant>(prelu->input_value(1).get_node())->get_data_ptr<ngraph::element::f32>()[0];
        return {arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, a};
    } else if (ngraph::is_type<opset::Clamp>(node)) {
        auto clamp = ngraph::as_type<opset::Clamp>(node);
        return {arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                static_cast<float>(clamp->get_max()),
                static_cast<float>(clamp->get_min())};
    } else {
        return {};
    }
}
}  // namespace opset
}  // namespace ArmPlugin

namespace ngraph {
NGRAPH_RTTI_DEFINITION(VariantWrapper<arm_compute::QuantizationInfo>, "Variant::arm_compute::QuantizationInfo", 0);
VariantWrapper<arm_compute::QuantizationInfo>::~VariantWrapper() {}
NGRAPH_RTTI_DEFINITION(VariantWrapper<arm_compute::ActivationLayerInfo>, "Variant::arm_compute::ActivationLayerInfo", 0);
VariantWrapper<arm_compute::ActivationLayerInfo>::~VariantWrapper() {}
}  // namespace ngraph