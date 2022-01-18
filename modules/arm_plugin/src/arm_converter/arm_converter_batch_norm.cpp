// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NEBatchNormalizationLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template <> Converter::Conversion::Ptr Converter::Convert(const opset::BatchNormInference& node) {
    enum Input {Features, Gamma, Beta, Mean, Variance};
    float eps = static_cast<float>(node.get_eps_value());
    return MakeConversion<arm_compute::NEBatchNormalizationLayer>(node.input(Input::Features), node.output(0),
                                                                  node.input(Input::Mean),     node.input(Input::Variance),
                                                                  node.input(Input::Beta),     node.input(Input::Gamma), eps);
}
}  //  namespace ArmPlugin