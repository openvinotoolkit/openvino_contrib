// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "arm_converter/arm_converter.hpp"
#include <arm_compute/runtime/NEON/functions/NERNNLayer.h>

typedef arm_compute::ActivationLayerInfo::ActivationFunction ArmActivationFunction;

static arm_compute::ActivationLayerInfo GetActivationInfo(const std::vector<std::string>& activation,
                                                          const std::vector<float>& a,
                                                          const std::vector<float>& b,
                                                          float clip) {
    if (activation[0] == "tanh") {
        std::cout << "tanh" << std::endl;
            return {ArmActivationFunction::TANH, a.empty() ? 1.f : a[0], b.empty() ? 1.f : b[0]};
    } else if (activation[0] == "relu") {
        std::cout << "relu" << std::endl;
            std::cout << "relu - 0" << std::endl;
            return ArmActivationFunction::RELU;
    } else if (activation[0] == "sigmoid") {
        std::cout << "sigmoid" << std::endl;
        return { ArmActivationFunction::LOGISTIC };
    }
    return arm_compute::ActivationLayerInfo{};
}

namespace ArmPlugin {
enum RNNInput {InputData, HiddenState, Weights, RecurrenceWeights, Bias};

template<> Converter::Conversion::Ptr Converter::Convert(const opset::RNNCell& node) {
    arm_compute::ActivationLayerInfo activationLayerInfo = GetActivationInfo(node.get_activations(),
                                                                             node.get_activations_alpha(),
                                                                             node.get_activations_beta(),
                                                                             node.get_clip());
    return MakeConversion<arm_compute::NERNNLayer>(node.input(RNNInput::InputData),
                                                   node.input(RNNInput::Weights),
                                                   node.input(RNNInput::RecurrenceWeights),
                                                   node.input(RNNInput::Bias),
                                                   node.input(RNNInput::HiddenState),
                                                   node.output(0),
                                                   activationLayerInfo);
}
} // namespace ArmPlugin