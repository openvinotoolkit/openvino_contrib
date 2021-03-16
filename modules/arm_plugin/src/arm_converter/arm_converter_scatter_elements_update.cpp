// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/scatter_elements_update.hpp>

namespace ArmPlugin {
template <typename DataType, typename IndicesType>
void wrap_scatter_elem_update(const DataType* input_data,
                              const IndicesType* indices,
                              const DataType* updates,
                              const IndicesType* axes,
                              DataType* out_buf,
                              const ngraph::Shape& data_shape,
                              const ngraph::Shape& indices_shape) {
    std::int64_t axis_val = static_cast<std::int64_t>(axes[0]);
    if (axis_val < 0) {
        axis_val = static_cast<std::int64_t>(data_shape.size()) + axis_val;
    }
    ngraph::runtime::reference::scatter_elem_update<DataType, IndicesType>(input_data,
                                                                           indices,
                                                                           updates,
                                                                           axis_val,
                                                                           out_buf,
                                                                           data_shape,
                                                                           indices_shape);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ScatterElementsUpdate& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(2),
                                    node.input(3),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1));
    };
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::u8 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_elem_update<std::uint8_t, std::int32_t>);
            }
            return make(wrap_scatter_elem_update<std::uint8_t, std::int64_t>);
        case ngraph::element::Type_t::i16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_elem_update<std::int16_t, std::int32_t>);
            }
            return make(wrap_scatter_elem_update<std::int16_t, std::int64_t>);
        case ngraph::element::Type_t::u16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_elem_update<std::uint16_t, std::int32_t>);
            }
            return make(wrap_scatter_elem_update<std::uint16_t, std::int64_t>);
        case ngraph::element::Type_t::u32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_elem_update<std::uint32_t, std::int32_t>);
            }
            return make(wrap_scatter_elem_update<std::uint32_t, std::int64_t>);
        case ngraph::element::Type_t::i32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_elem_update<std::int32_t, std::int32_t>);
            }
            return make(wrap_scatter_elem_update<std::int32_t, std::int64_t>);
        case ngraph::element::Type_t::i64 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_elem_update<std::int64_t, std::int32_t>);
            }
            return make(wrap_scatter_elem_update<std::int64_t, std::int64_t>);
        case ngraph::element::Type_t::f16 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_elem_update<half_float::half, std::int32_t>);
            }
            return make(wrap_scatter_elem_update<half_float::half, std::int64_t>);
        case ngraph::element::Type_t::f32 :
            if (node.get_input_element_type(1) == ngraph::element::i32) {
                return make(wrap_scatter_elem_update<float, std::int32_t>);
            }
            return make(wrap_scatter_elem_update<float, std::int64_t>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

}  //  namespace ArmPlugin
