// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/scatter_update.hpp>

namespace ArmPlugin {
template <typename T, typename U>
void wrap_scatter_update(const T* input_data,
                         const U* indices,
                         const T* updates,
                         const U* axis,
                         T* out,
                         const size_t elem_size,
                         const ngraph::Shape& data_shape,
                         const ngraph::Shape& indices_shape,
                         const ngraph::Shape& updates_shape) {
    auto indices_size = ngraph::shape_size(indices_shape);
    std::vector<std::int64_t> converted_indices(indices_size);
    for (size_t i = 0; i < indices_size; i++) {
        converted_indices[i] = static_cast<std::int64_t>(indices[i]);
    }
    std::int64_t axis_val = static_cast<std::int64_t>(axis[0]);
    if (axis_val < 0) {
        axis_val = static_cast<std::int64_t>(data_shape.size()) + axis_val;
    }
    ngraph::runtime::reference::scatter_update(reinterpret_cast<const char*>(input_data),
                                               converted_indices.data(),
                                               reinterpret_cast<const char*>(updates),
                                               axis_val,
                                               reinterpret_cast<char*>(out),
                                               elem_size,
                                               data_shape,
                                               indices_shape,
                                               updates_shape);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ScatterUpdate& node) {
    auto make = [&] (auto refFunction) {
        return MakeConversion(refFunction,
                              node.input(0),
                              node.input(1),
                              node.input(2),
                              node.input(3),
                              node.output(0),
                              node.input(0).get_element_type().size(),
                              node.get_input_shape(0),
                              node.get_input_shape(1),
                              node.get_input_shape(2));
    };

    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_update<std::uint8_t, std::int32_t>);
            }
            return make(wrap_scatter_update<std::uint8_t, std::int64_t>);
        case ngraph::element::Type_t::i16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_update<std::int16_t, std::int32_t>);
            }
            return make(wrap_scatter_update<std::int16_t, std::int64_t>);
        case ngraph::element::Type_t::u16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_update<std::uint16_t, std::int32_t>);
            }
            return make(wrap_scatter_update<std::uint16_t, std::int64_t>);
        case ngraph::element::Type_t::u32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_update<std::uint32_t, std::int32_t>);
            }
            return make(wrap_scatter_update<std::uint32_t, std::int64_t>);
        case ngraph::element::Type_t::i32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_update<std::int32_t, std::int32_t>);
            }
            return make(wrap_scatter_update<std::int32_t, std::int64_t>);
        case ngraph::element::Type_t::i64 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_update<std::int64_t, std::int32_t>);
            }
            return make(wrap_scatter_update<std::int64_t, std::int64_t>);
        case ngraph::element::Type_t::f16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_update<half_float::half, std::int32_t>);
            }
            return make(wrap_scatter_update<half_float::half, std::int64_t>);
        case ngraph::element::Type_t::f32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_update<float, std::int32_t>);
            }
            return make(wrap_scatter_update<float, std::int64_t>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

}  //  namespace ArmPlugin
