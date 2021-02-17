// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/one_hot.hpp>

namespace ArmPlugin {
template <typename Indices, typename OutputType>
void wrap_one_hot(const Indices* arg,
                  OutputType* out,
                  const ngraph::Shape& in_shape,
                  const ngraph::Shape& out_shape,
                  size_t one_hot_axis,
                  const OutputType* on_values,
                  const OutputType* off_values) {
    ngraph::runtime::reference::one_hot(arg,
                                        out,
                                        in_shape,
                                        out_shape,
                                        one_hot_axis,
                                        on_values[0],
                                        off_values[0]);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::OneHot& node) {
    auto make = [&] (auto refFunction) {
        return MakeConversion(refFunction,
                              node.input(0),
                              node.output(0),
                              node.get_input_shape(0),
                              node.get_output_shape(0),
                              static_cast<size_t>(node.get_axis()),
                              node.input(2),
                              node.input(3));
    };

    ngraph::element::Type_t outType = node.get_output_element_type(0);
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            switch (outType) {
                case ngraph::element::Type_t::u8  : return make(wrap_one_hot<std::uint8_t, std::uint8_t>);
                case ngraph::element::Type_t::i16 : return make(wrap_one_hot<std::uint8_t, std::int16_t>);
                case ngraph::element::Type_t::u16 : return make(wrap_one_hot<std::uint8_t, std::uint16_t>);
                case ngraph::element::Type_t::i32 : return make(wrap_one_hot<std::uint8_t, std::int32_t>);
                case ngraph::element::Type_t::f32 : return make(wrap_one_hot<std::uint8_t, float>);
                default: THROW_IE_EXCEPTION << "Unsupported Type: " << outType; return {};
            }
        case ngraph::element::Type_t::i16 :
            switch (outType) {
                case ngraph::element::Type_t::u8  : return make(wrap_one_hot<std::int16_t, std::uint8_t>);
                case ngraph::element::Type_t::i16 : return make(wrap_one_hot<std::int16_t, std::int16_t>);
                case ngraph::element::Type_t::u16 : return make(wrap_one_hot<std::int16_t, std::uint16_t>);
                case ngraph::element::Type_t::i32 : return make(wrap_one_hot<std::int16_t, std::int32_t>);
                case ngraph::element::Type_t::f32 : return make(wrap_one_hot<std::int16_t, float>);
                default: THROW_IE_EXCEPTION << "Unsupported Type: " << outType; return {};
            }
        case ngraph::element::Type_t::u16 :
            switch (outType) {
                case ngraph::element::Type_t::u8  : return make(wrap_one_hot<std::uint16_t, std::uint8_t>);
                case ngraph::element::Type_t::i16 : return make(wrap_one_hot<std::uint16_t, std::int16_t>);
                case ngraph::element::Type_t::u16 : return make(wrap_one_hot<std::uint16_t, std::uint16_t>);
                case ngraph::element::Type_t::i32 : return make(wrap_one_hot<std::uint16_t, std::int32_t>);
                case ngraph::element::Type_t::f32 : return make(wrap_one_hot<std::uint16_t, float>);
                default: THROW_IE_EXCEPTION << "Unsupported Type: " << outType; return {};
            }
        case ngraph::element::Type_t::u32 :
            switch (outType) {
                case ngraph::element::Type_t::u8  : return make(wrap_one_hot<std::uint32_t, std::uint8_t>);
                case ngraph::element::Type_t::i16 : return make(wrap_one_hot<std::uint32_t, std::int16_t>);
                case ngraph::element::Type_t::u16 : return make(wrap_one_hot<std::uint32_t, std::uint16_t>);
                case ngraph::element::Type_t::i32 : return make(wrap_one_hot<std::uint32_t, std::int32_t>);
                case ngraph::element::Type_t::f32 : return make(wrap_one_hot<std::uint32_t, float>);
                default: THROW_IE_EXCEPTION << "Unsupported Type: " << outType; return {};
            }
        case ngraph::element::Type_t::i32 :
            switch (outType) {
                case ngraph::element::Type_t::u8  : return make(wrap_one_hot<std::int32_t, std::uint8_t>);
                case ngraph::element::Type_t::i16 : return make(wrap_one_hot<std::int32_t, std::int16_t>);
                case ngraph::element::Type_t::u16 : return make(wrap_one_hot<std::int32_t, std::uint16_t>);
                case ngraph::element::Type_t::i32 : return make(wrap_one_hot<std::int32_t, std::int32_t>);
                case ngraph::element::Type_t::f32 : return make(wrap_one_hot<std::int32_t, float>);
                default: THROW_IE_EXCEPTION << "Unsupported Type: " << outType; return {};
            }
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}
}  //  namespace ArmPlugin
