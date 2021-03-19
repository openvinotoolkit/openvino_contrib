// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/ctc_loss.hpp>

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::CTCLoss& node) {
    if (node.get_input_size() < 4) {
        THROW_IE_EXCEPTION << "Unsupported CTCLoss op with num inputs = " << node.get_input_size();
    }

    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.get_input_shape(0),
                                    node.input(1),
                                    node.input(2),
                                    node.input(3),
                                    node.input(4),
                                    node.get_preprocess_collapse_repeated(),
                                    node.get_ctc_merge_repeated(),
                                    node.get_unique(),
                                    node.output(0));
    };
    switch (node.get_input_element_type(0)) {
        // case ngraph::element::Type_t::f16 :
        //     if (node.get_input_element_type(1) == ngraph::element::i32) {
        //         return make(ngraph::runtime::reference::CTCLoss<half_float::half, std::int32_t>);
        //     }
        //     return make(ngraph::runtime::reference::CTCLoss<half_float::half, std::int64_t>);
        case ngraph::element::Type_t::f32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(ngraph::runtime::reference::CTCLoss<float, std::int32_t>);
            }
            return make(ngraph::runtime::reference::CTCLoss<float, std::int64_t>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

}  //  namespace ArmPlugin
