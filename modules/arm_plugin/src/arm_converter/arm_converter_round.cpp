// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/round.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Round& node) {
    auto make = [&] (auto refFunction) {
        return MakeConversion(refFunction,
                              node.input(0),
                              node.output(0),
                              ngraph::shape_size(node.get_input_shape(0)),
                              node.get_mode());
    };

    if (node.get_mode() != ngraph::op::v5::Round::RoundMode::HALF_TO_EVEN) {
        THROW_IE_EXCEPTION << "Use ConvertRound transformation";
    }
    return make(ngraph::runtime::reference::round<float>);
}

}  //  namespace ArmPlugin
