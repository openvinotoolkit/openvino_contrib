// Copyright (C) 2020-2022 Intel Corporation
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
                static_cast<float>(safe_cast<opset::Elu>(node)->get_alpha())};
    } else if (ngraph::is_type<opset::Sqrt>(node)) {
        return {arm_compute::ActivationLayerInfo::ActivationFunction::SQRT};
    } else if (ngraph::is_type<opset::SoftPlus>(node)) {
        return {arm_compute::ActivationLayerInfo::ActivationFunction::SOFT_RELU};
    } else if (ngraph::is_type<opset::HSwish>(node)) {
        return {arm_compute::ActivationLayerInfo::ActivationFunction::HARD_SWISH};
    } else if (ngraph::is_type<opset::PRelu>(node)) {
        auto prelu = safe_cast<opset::PRelu>(node);
        auto a = safe_cast<opset::Constant>(prelu->input_value(1).get_node())->get_data_ptr<ngraph::element::f32>()[0];
        return {arm_compute::ActivationLayerInfo::ActivationFunction::LEAKY_RELU, a};
    } else if (ngraph::is_type<opset::Clamp>(node)) {
        auto clamp = safe_cast<opset::Clamp>(node);
        return {arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
                static_cast<float>(clamp->get_max()),
                static_cast<float>(clamp->get_min())};
    } else {
        return {};
    }
}
}  // namespace opset
}  // namespace ArmPlugin
