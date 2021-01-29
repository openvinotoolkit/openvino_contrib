// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/detection_output.hpp>

namespace ArmPlugin {
template <typename T>
void detection_output(const T* _location,
                      const T* _confidence,
                      const T* _priors,
                      const T* _armConfidence,
                      const T* _armLocation,
                      T* result,
                      const ngraph::op::DetectionOutputAttrs& attrs,
                      const ngraph::Shape& locShape,
                      const ngraph::Shape& priorsShape,
                      const ngraph::Shape& outShape) {
    ngraph::runtime::reference::referenceDetectionOutput<float> refDetOut(attrs, locShape, priorsShape, outShape);
    refDetOut.run(_location, _confidence, _priors, _armConfidence, _armLocation, result);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::DetectionOutput& node) {
    auto make = [&] (auto refFunction) {
        if (node.get_input_size() == 3) {
            return MakeConversion(refFunction,
                                  node.input(0),
                                  node.input(1),
                                  node.input(2),
                                  nullptr,
                                  nullptr,
                                  node.output(0),
                                  node.get_attrs(),
                                  node.get_input_shape(0),
                                  node.get_input_shape(2),
                                  node.get_output_shape(0));
        }
        return MakeConversion(refFunction,
                              node.input(0),
                              node.input(1),
                              node.input(2),
                              node.input(3),
                              node.input(4),
                              node.output(0),
                              node.get_attrs(),
                              node.get_input_shape(0),
                              node.get_input_shape(2),
                              node.get_output_shape(0));
    };

    switch (node.input(0).get_element_type()) {
        // case ngraph::element::Type_t::f16 :
        //     return make(detection_output<half_float::half>);
        case ngraph::element::Type_t::f32 : {
            return make(detection_output<float>);
        }
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}
}  //  namespace ArmPlugin
