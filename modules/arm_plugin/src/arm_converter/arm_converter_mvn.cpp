// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <details/ie_exception.hpp>

#include <arm_compute/runtime/NEON/functions/NEMeanStdDevNormalizationLayer.h>
#include <ngraph/runtime/reference/mvn.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template <> Converter::Conversion::Ptr Converter::Convert(const opset::MVN& node) {
    if (node.get_input_shape(0).size() != 2 || !node.get_normalize_variance() || !node.get_across_channels()) {
        auto make = [&] (auto refFunction) {
            return MakeConversion(refFunction,
                                  node.input(0),
                                  node.output(0),
                                  node.get_input_shape(0),
                                  node.get_normalize_variance(),
                                  node.get_reduction_axes(),
                                  node.get_eps());
        };
        switch (node.get_input_element_type(0)) {
            // case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::mvn<half_float::half>);
            case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::mvn<float>);
            default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
        }
    }

    float eps = node.get_eps();
    return MakeConversion<arm_compute::NEMeanStdDevNormalizationLayer>(node.input(0), node.output(0), eps);
}
}  //  namespace ArmPlugin