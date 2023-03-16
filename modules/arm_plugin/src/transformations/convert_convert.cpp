// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_convert.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>


using type = ngraph::element::Type_t;

OPENVINO_OP(ArmPlugin::pass::ConvertArmConvertBase, "ConvertArmConvertBase");
template <class T>
ngraph::matcher_pass_callback ArmPlugin::pass::ConvertArmConvertBase::convert_to_arm_convert() {
    return [&](ngraph::pattern::Matcher& m) {
        auto convert = m.get_match_root();
        if (!std::dynamic_pointer_cast<T>(convert)) {
            return false;
        }

        auto src_type = convert->get_input_element_type(0);
        auto dst_type = convert->get_output_element_type(0);

        if ((src_type == type::u8  && (dst_type == type::u16 || dst_type == type::i16  || dst_type == type::i32 || dst_type == type::f16 || dst_type == type::f32)) ||
            (src_type == type::u16 && (dst_type == type::u8  || dst_type == type::u32)) || (src_type == dst_type) ||
            (src_type == type::i16 && (dst_type == type::u8  || dst_type == type::i32)) ||
            (src_type == type::i32 && (dst_type == type::f16 || dst_type == type::f32 || dst_type == type::u8)) ||
            (src_type == type::f16 && (dst_type == type::f32 || dst_type == type::i32 || dst_type == type::u8)) ||
            (src_type == type::f32 && (dst_type == type::f16 || dst_type == type::i32 || dst_type == type::u8))) {
            auto arm_convert = std::make_shared<opset::ArmConvert>(convert->input_value(0), dst_type);
            arm_convert->set_friendly_name(convert->get_friendly_name());
            ngraph::copy_runtime_info(convert, arm_convert);
            ngraph::replace_node(convert, arm_convert);
            return true;
        }
        return false;
    };
}

OPENVINO_OP(ArmPlugin::pass::ConvertArmConvert, "ConvertArmConvert");
ArmPlugin::pass::ConvertArmConvert::ConvertArmConvert() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Convert>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape())}),
                                                        "ConvertConvert");
    register_matcher(m, convert_to_arm_convert<opset::Convert>());
}

OPENVINO_OP(ArmPlugin::pass::ConvertArmConvertLike, "ConvertArmConvertLike");
ArmPlugin::pass::ConvertArmConvertLike::ConvertArmConvertLike() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::ConvertLike>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                            ngraph::pattern::any_input(ngraph::pattern::has_static_shape())}),
                                                            "ConvertConvertLike");
    register_matcher(m, convert_to_arm_convert<opset::Convert>());
}