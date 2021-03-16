// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>
#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/one_hot.hpp>

template <typename InputType, typename DepthType, typename OutputType>
void wrap_one_hot(const InputType* indices,
                  const ngraph::Shape& indices_shape,
                  OutputType* out,
                  const ngraph::Shape& out_shape,
                  const size_t out_elem_size,
                  const DepthType* depth,
                  const std::int64_t one_hot_axis,
                  const OutputType* on_value,
                  const OutputType* off_value) {
    const size_t depth_val = static_cast<size_t>(depth[0]);
    if (ngraph::shape_size(indices_shape) * depth_val != ngraph::shape_size(out_shape))
        THROW_IE_EXCEPTION << "Incompatible I/O shapes or wrong depth value.";
    if (depth_val != out_shape[one_hot_axis])
        THROW_IE_EXCEPTION << "Incompatible depth and axis values.";
    ngraph::runtime::reference::one_hot<InputType>(indices,
                                                   indices_shape,
                                                   reinterpret_cast<char*>(out),
                                                   out_elem_size,
                                                   depth_val,
                                                   one_hot_axis,
                                                   reinterpret_cast<const char*>(on_value),
                                                   reinterpret_cast<const char*>(off_value));
}

namespace ArmPlugin {
    template<> Converter::Conversion::Ptr Converter::Convert(const opset::OneHot& node) {
        const auto& ind_shape = node.get_input_shape(0);
        const auto& out_shape = node.get_output_shape(0);
        std::int64_t axis = node.get_axis();
        const std::int64_t out_rank = out_shape.size();
        if (axis < -out_rank || axis >= out_rank)
            THROW_IE_EXCEPTION << "Invalid axis value. Expected in [" << -out_rank
                               << ", " << out_rank-1 << "]. Got " << axis;
        if (axis < 0)
            axis += out_rank;
        auto make = [&] (auto refFunction) {
            return this->MakeConversion(refFunction,
                                        node.input(0),
                                        ind_shape,
                                        node.output(0),
                                        out_shape,
                                        node.get_output_element_type(0).size(),
                                        node.input(1),
                                        axis,
                                        node.input(2),
                                        node.input(3));
        };
        ngraph::element::Type_t inputType = node.get_input_element_type(0);
        ngraph::element::Type_t depthType = node.get_input_element_type(1);
        ngraph::element::Type_t outType = node.get_output_element_type(0);
        #define ONEHOT_TYPE_OUT_CASE(input_type, depth_type)\
            switch (outType) {\
                case ngraph::element::Type_t::u8  : return make(wrap_one_hot<input_type, depth_type, std::uint8_t>);\
                case ngraph::element::Type_t::i16 : return make(wrap_one_hot<input_type, depth_type, std::int16_t>);\
                case ngraph::element::Type_t::u16 : return make(wrap_one_hot<input_type, depth_type, std::uint16_t>);\
                case ngraph::element::Type_t::i32 : return make(wrap_one_hot<input_type, depth_type, std::int32_t>);\
                case ngraph::element::Type_t::f32 : return make(wrap_one_hot<input_type, depth_type, float>);\
                default: THROW_IE_EXCEPTION << "Unsupported Output Type: " << outType; return {};\
            }\
            break
        switch (inputType) {
            case ngraph::element::Type_t::i32 : {
                switch (depthType) {
                    case ngraph::element::Type_t::u8 : ONEHOT_TYPE_OUT_CASE(std::int32_t, std::uint8_t);
                    case ngraph::element::Type_t::u16 : ONEHOT_TYPE_OUT_CASE(std::int32_t, std::uint16_t);
                    case ngraph::element::Type_t::i16 : ONEHOT_TYPE_OUT_CASE(std::int32_t, std::int16_t);
                    case ngraph::element::Type_t::i32 : ONEHOT_TYPE_OUT_CASE(std::int32_t, std::int32_t);
                    case ngraph::element::Type_t::i64 : ONEHOT_TYPE_OUT_CASE(std::int32_t, std::int64_t);
                    default: THROW_IE_EXCEPTION << "Unsupported Depth Type: " << depthType; return {};
                }
                break;
            }
            case ngraph::element::Type_t::i64 : {
                switch (depthType) {
                    case ngraph::element::Type_t::u8 : ONEHOT_TYPE_OUT_CASE(std::int64_t, std::uint8_t);
                    case ngraph::element::Type_t::u16 : ONEHOT_TYPE_OUT_CASE(std::int64_t, std::uint16_t);
                    case ngraph::element::Type_t::i16 : ONEHOT_TYPE_OUT_CASE(std::int64_t, std::int16_t);
                    case ngraph::element::Type_t::i32 : ONEHOT_TYPE_OUT_CASE(std::int64_t, std::int32_t);
                    case ngraph::element::Type_t::i64 : ONEHOT_TYPE_OUT_CASE(std::int64_t, std::int64_t);
                    default: THROW_IE_EXCEPTION << "Unsupported Depth Type: " << depthType; return {};
                }
                break;
            }
            default: THROW_IE_EXCEPTION << "Unsupported Input Type. Expected int32 or int64, got " << inputType; return {};
        }
        #undef ONEHOT_TYPE_OUT_CASE
    }
}  //  namespace ArmPlugin
