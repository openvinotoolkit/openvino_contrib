// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <arm_compute/runtime/NEON/NEScheduler.h>
#include <arm_compute/runtime/NEON/functions/NEGEMM.h>
#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/matmul.hpp>

namespace ArmPlugin {
enum InputArg {Features, Weights, Bias};
template<> Converter::Conversion::Ptr Converter::Convert(const opset::MatMul& node) {
    if ((node.input(Features).get_shape().at(0) > 1 && node.input(Features).get_shape().size() == 4) ||
        (node.input(Weights).get_shape().at(0)  > 1 && node.input(Weights).get_shape().size() == 4)) {
        auto make = [&] (auto refFunction) {
            return this->MakeConversion(refFunction,
                                        node.input(Features),
                                        node.input(Weights),
                                        node.output(0),
                                        node.get_input_shape(Features),
                                        node.get_input_shape(Weights),
                                        node.get_output_shape(0),
                                        node.get_transpose_a(),
                                        node.get_transpose_b());
        };

        return CallSwitch(
                AP_WRAP(make, ngraph::runtime::reference::matmul),
                node.input(0), allTypes);
    } else {
        arm_compute::GEMMInfo gemmInfo;
        gemmInfo.set_pretranspose_A(node.get_transpose_a());
        gemmInfo.set_pretranspose_B(node.get_transpose_b());
        return MakeConversion<arm_compute::NEGEMM>(node.input(Features), node.input(Weights), nullptr, node.output(0),
                                                   1.f, 1.f, gemmInfo);
    }
}
template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmMatMulBias& node) {
    if (node.get_transpose_a()) {
        IE_THROW() << "Can not create MatMul layer with transpose first input";
    }
    if ((node.input(0).get_shape().at(0) > 1 && node.input(0).get_shape().size() == 4) ||
        (node.input(1).get_shape().at(0) > 1 && node.input(1).get_shape().size() == 4)) {
        auto make = [&] (auto refFunction) {
            return this->MakeConversion(refFunction,
                                        node.input(Features),
                                        node.input(Weights),
                                        node.output(0),
                                        node.get_input_shape(Features),
                                        node.get_input_shape(Weights),
                                        node.get_output_shape(0),
                                        node.get_transpose_a(),
                                        node.get_transpose_b());
        };

        return CallSwitch(
                AP_WRAP(make, ngraph::runtime::reference::matmul),
                node.input(0), allTypes);
    } else {
        arm_compute::GEMMInfo gemmInfo;
        gemmInfo.set_pretranspose_A(node.get_transpose_a());
        gemmInfo.set_pretranspose_B(node.get_transpose_b());
        return MakeConversion<arm_compute::NEGEMM>(node.input(Features), node.input(Weights), node.input(Bias), node.output(0),
                                                   1.f, 1.f, gemmInfo);
    }
}
}  //  namespace ArmPlugin
