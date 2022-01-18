// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <cfenv>
#include <ngraph/op/fake_quantize.hpp>
#include <ngraph/runtime/reference/fake_quantize.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::FakeQuantize& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0), node.input(1), node.input(2), node.input(3), node.input(4),
                                    node.output(0),
                                    node.input(0).get_shape(),
                                    node.input(1).get_shape(), node.input(2).get_shape(), node.input(3).get_shape(), node.input(4).get_shape(),
                                    node.get_levels(),
                                    node.get_auto_broadcast());
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::fake_quantize<ngraph::float16>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::fake_quantize<float>);
        default: IE_THROW() << "Arm Plugin: Unsupported Type: " << node.get_element_type();
    }
}
}  // namespace ArmPlugin
