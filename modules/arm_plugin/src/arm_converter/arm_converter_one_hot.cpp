// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/one_hot.hpp>

namespace ArmPlugin {
template <typename INPUT_TYPE>
void wrap_one_hot(const INPUT_TYPE* indices,
             const ngraph::Shape& indices_shape,
             char* out,
             const size_t out_elem_size,
             const size_t depth,
             const int64_t one_hot_axis,
             const char* on_value,
             const char* off_value) {
    ngraph::runtime::reference::one_hot<INPUT_TYPE>(indices,
                                            indices_shape,
                                            out,
                                            out_elem_size,
                                            depth,
                                            one_hot_axis,
                                            on_value,
                                            off_value);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::OneHot& node) {
    if (node.get_input_size() != 4 || node.get_output_size() != 1)
        THROW_IE_EXCEPTION << "Invalid number of inputs or outputs. Expected 4 and 1, got " <<
                            node.get_input_size() << " and " << node.get_output_size();

    const auto& out_shape = node.get_output_shape(0);
    const int64_t axis = node.get_axis();
    if (axis < 0 || axis >= out_shape.size())
        THROW_IE_EXCEPTION << "Invalid axis value. Expected in [0, " << out_shape.size() <<
                            "), got " << axis;
    const auto& depth_const = std::dynamic_pointer_cast<opset::Constant>(node.input_value(1).get_node_shared_ptr());
    const auto depth = depth_const -> cast_vector<int64_t>()[0];

    const auto& ind_shape = node.get_input_shape(0);
    if (shape_size(ind_shape) * depth != shape_size(out_shape))
        THROW_IE_EXCEPTION << "Incompatible I/O shapes or wrong depth value.";
    if (depth != out_shape[axis])
        THROW_IE_EXCEPTION << "Incompatible depth and axis values.";

    auto make = [&] (auto refFunction) {
        return MakeConversion(refFunction,
                              node.input(0),
                              ind_shape,
                              node.output(0),
                              node.get_output_element_type(0).size(),
                              static_cast<size_t>(depth),
                              axis,
                              node.input(2),
                              node.input(3));
    };
    ngraph::element::Type_t inputType = node.get_input_element_type(0);
    switch (inputType) {
        case ngraph::element::Type_t::i32: return make(wrap_one_hot<std::int32_t>);
        case ngraph::element::Type_t::i64: return make(wrap_one_hot<std::int64_t>);
        default: THROW_IE_EXCEPTION << "Unsupported input type: " << inputType; return {};
    }
}

/*
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
*/

/*
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
 */
}  //  namespace ArmPlugin
