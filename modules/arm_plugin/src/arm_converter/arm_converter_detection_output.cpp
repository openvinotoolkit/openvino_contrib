// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


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
    IE_THROW(NotImplemented) << "Not implemented";
}

template<> void detection_output<ngraph::float16>(const ngraph::float16* _location,
                                                  const ngraph::float16* _confidence,
                                                  const ngraph::float16* _priors,
                                                  const ngraph::float16* _armConfidence,
                                                  const ngraph::float16* _armLocation,
                                                  ngraph::float16* result,
                                                  const ngraph::op::DetectionOutputAttrs& attrs,
                                                  const ngraph::Shape& locShape,
                                                  const ngraph::Shape& priorsShape,
                                                  const ngraph::Shape& outShape) {
    ngraph::runtime::reference::referenceDetectionOutput<ngraph::float16> refDet(attrs, locShape, priorsShape, outShape);
    refDet.run(_location, _confidence, _priors, _armConfidence, _armLocation, result);
}

template<> void detection_output<float>(const float* _location,
                                        const float* _confidence,
                                        const float* _priors,
                                        const float* _armConfidence,
                                        const float* _armLocation,
                                        float* result,
                                        const ngraph::op::DetectionOutputAttrs& attrs,
                                        const ngraph::Shape& locShape,
                                        const ngraph::Shape& priorsShape,
                                        const ngraph::Shape& outShape) {
    ngraph::runtime::reference::referenceDetectionOutput<float> refDet(attrs, locShape, priorsShape, outShape);
    refDet.run(_location, _confidence, _priors, _armConfidence, _armLocation, result);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::DetectionOutput& node) {
    auto make = [&] (auto refFunction) {
        if (node.get_input_size() == 3) {
            return this->MakeConversion(refFunction,
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
        return this->MakeConversion(refFunction,
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
    return CallSwitch(
        AP_WRAP(make, detection_output),
        node.input(0), floatTypes);
}

template <typename T>
void detection_output_v8(const T* _location,
                      const T* _confidence,
                      const T* _priors,
                      const T* _armConfidence,
                      const T* _armLocation,
                      T* result,
                      const ngraph::op::util::DetectionOutputBase::AttributesBase& attrs,
                      const ngraph::Shape& locShape,
                      const ngraph::Shape& classPredShape,
                      const ngraph::Shape& priorsShape,
                      const ngraph::Shape& outShape) {
    ngraph::runtime::reference::referenceDetectionOutput<T> refDet(attrs, locShape, classPredShape, priorsShape, outShape);
    refDet.run(_location, _confidence, _priors, _armConfidence, _armLocation, result);
}

template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v8::DetectionOutput& node) {
    auto make = [&] (auto refFunction) {
        if (node.get_input_size() == 3) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        node.input(1),
                                        node.input(2),
                                        nullptr,
                                        nullptr,
                                        node.output(0),
                                        node.get_attrs(),
                                        node.get_input_shape(0),
                                        node.get_input_shape(1),
                                        node.get_input_shape(2),
                                        node.get_output_shape(0));
        }
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(2),
                                    node.input(3),
                                    node.input(4),
                                    node.output(0),
                                    node.get_attrs(),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_input_shape(2),
                                    node.get_output_shape(0));
    };
    return CallSwitch(
        AP_WRAP(make, detection_output_v8),
        node.input(0), floatTypes);
}
}  //  namespace ArmPlugin
