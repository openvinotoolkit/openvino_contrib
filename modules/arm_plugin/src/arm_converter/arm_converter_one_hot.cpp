// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>
#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/one_hot.hpp>

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

namespace ArmPlugin {
    template<> Converter::Conversion::Ptr Converter::Convert(const opset::OneHot& node) {
        const auto& ind_shape = node.get_input_shape(0);
        const auto& out_shape = node.get_output_shape(0);
        std::int64_t axis = node.get_axis();
        const auto out_rank = out_shape.size();
        if (axis < -out_rank || axis >= out_rank)
            THROW_IE_EXCEPTION << "Invalid axis value. Expected in [" << -out_rank
                               << ", " << out_rank-1 << "]. Got " << axis;
        if (axis < 0)
            axis += out_rank;

        const auto& depth_const = std::dynamic_pointer_cast<opset::Constant>(node.input_value(1).get_node_shared_ptr());
        if (!depth_const)
            THROW_IE_EXCEPTION << "Depth value must be constant.";

        const auto depth = depth_const -> cast_vector<int64_t>()[0];

        if (ngraph::shape_size(ind_shape) * depth != ngraph::shape_size(out_shape))
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
            case ngraph::element::Type_t::u8 : return make(wrap_one_hot<std::uint8_t>);
            case ngraph::element::Type_t::i16 : return make(wrap_one_hot<std::int16_t>);
            case ngraph::element::Type_t::u16 : return make(wrap_one_hot<std::uint16_t>);
            case ngraph::element::Type_t::i32 : return make(wrap_one_hot<std::int32_t>);
            case ngraph::element::Type_t::u32 : return make(wrap_one_hot<std::uint32_t>);
            default : THROW_IE_EXCEPTION << "Unsupported input type: " << inputType; return {};
        }
    }
}  //  namespace ArmPlugin
