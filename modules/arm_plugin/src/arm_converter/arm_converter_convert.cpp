// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <arm_compute/runtime/NEON/functions/NEDepthConvertLayer.h>
#include <arm_compute/runtime/NEON/functions/NECopy.h>
#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/convert.hpp>

using type = ngraph::element::Type_t;

namespace ArmPlugin {
template <> Converter::Conversion::Ptr Converter::Convert(const opset::Convert& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.output(0),
                                    ngraph::shape_size(node.get_input_shape(0)));
    };

    auto src = node.get_input_element_type(0);
    auto dst = node.get_convert_element_type();

    switch (src) {
        case ngraph::element::Type_t::i8 :
            switch (dst) {
                case ngraph::element::Type_t::i32 :
                    return make(ngraph::runtime::reference::convert<std::int8_t, std::int32_t>);
                case ngraph::element::Type_t::u32 :
                    return make(ngraph::runtime::reference::convert<std::int8_t, std::uint32_t>);
                case ngraph::element::Type_t::f16 :
                    return make(ngraph::runtime::reference::convert<std::int8_t, ngraph::float16>);
                case ngraph::element::Type_t::f32 :
                    return make(ngraph::runtime::reference::convert<std::int8_t, float>);
            default:
                IE_THROW() << "Unsupported convertion from " << src << " to " << dst; return {};
            }
        case ngraph::element::Type_t::u8 :
            switch (dst) {
                case ngraph::element::Type_t::i32 :
                    return make(ngraph::runtime::reference::convert<std::uint8_t, std::int32_t>);
                case ngraph::element::Type_t::u32 :
                    return make(ngraph::runtime::reference::convert<std::uint8_t, std::uint32_t>);
                case ngraph::element::Type_t::f16 :
                    return make(ngraph::runtime::reference::convert<std::uint8_t, ngraph::float16>);
                case ngraph::element::Type_t::f32 :
                    return make(ngraph::runtime::reference::convert<std::uint8_t, float>);
            default:
                IE_THROW() << "Unsupported convertion from " << src << " to " << dst; return {};
            }
        case ngraph::element::Type_t::i16 :
            switch (dst) {
                case ngraph::element::Type_t::u16 :
                    return make(ngraph::runtime::reference::convert<std::int16_t, std::uint16_t>);
                case ngraph::element::Type_t::i32 :
                    return make(ngraph::runtime::reference::convert<std::int16_t, std::int32_t>);
                case ngraph::element::Type_t::u32 :
                    return make(ngraph::runtime::reference::convert<std::int16_t, std::uint32_t>);
                case ngraph::element::Type_t::f16 :
                    return make(ngraph::runtime::reference::convert<std::int16_t, ngraph::float16>);
                case ngraph::element::Type_t::f32 :
                    return make(ngraph::runtime::reference::convert<std::int16_t, float>);
            default:
                IE_THROW() << "Unsupported convertion from " << src << " to " << dst; return {};
            }
        case ngraph::element::Type_t::u16 :
            switch (dst) {
                case ngraph::element::Type_t::i32 :
                    return make(ngraph::runtime::reference::convert<std::uint16_t, std::int32_t>);
                case ngraph::element::Type_t::f16 :
                    return make(ngraph::runtime::reference::convert<std::uint16_t, ngraph::float16>);
                case ngraph::element::Type_t::f32 :
                    return make(ngraph::runtime::reference::convert<std::uint16_t, float>);
            default:
                IE_THROW() << "Unsupported convertion from " << src << " to " << dst; return {};
            }
        case ngraph::element::Type_t::i32 :
            switch (dst) {
                case ngraph::element::Type_t::u8 :
                    return make(ngraph::runtime::reference::convert<std::int32_t, std::uint8_t>);
                case ngraph::element::Type_t::i16 :
                    return make(ngraph::runtime::reference::convert<std::int32_t, std::int16_t>);
                case ngraph::element::Type_t::u32 :
                    return make(ngraph::runtime::reference::convert<std::int32_t, std::uint32_t>);
                case ngraph::element::Type_t::f16 :
                    return make(ngraph::runtime::reference::convert<std::int32_t, ngraph::float16>);
                case ngraph::element::Type_t::f32 :
                    return make(ngraph::runtime::reference::convert<std::int32_t, float>);
            default:
                IE_THROW() << "Unsupported convertion from " << src << " to " << dst; return {};
            }
        case ngraph::element::Type_t::u32 :
            switch (dst) {
                case ngraph::element::Type_t::u8 :
                    return make(ngraph::runtime::reference::convert<std::uint32_t, std::uint8_t>);
                case ngraph::element::Type_t::i32 :
                    return make(ngraph::runtime::reference::convert<std::uint32_t, std::int32_t>);
                case ngraph::element::Type_t::f16 :
                    return make(ngraph::runtime::reference::convert<std::uint32_t, ngraph::float16>);
                case ngraph::element::Type_t::f32 :
                    return make(ngraph::runtime::reference::convert<std::uint32_t, float>);
            default:
                IE_THROW() << "Unsupported convertion from " << src << " to " << dst; return {};
            }
        case ngraph::element::Type_t::f16 :
            switch (dst) {
                case ngraph::element::Type_t::u8 :
                    return make(ngraph::runtime::reference::convert<ngraph::float16, std::uint8_t>);
                case ngraph::element::Type_t::i16 :
                    return make(ngraph::runtime::reference::convert<ngraph::float16, std::int16_t>);
                case ngraph::element::Type_t::i32 :
                    return make(ngraph::runtime::reference::convert<ngraph::float16, std::int32_t>);
            default:
                IE_THROW() << "Unsupported convertion from " << src << " to " << dst; return {};
            }
        case ngraph::element::Type_t::f32 :
            switch (dst) {
                case ngraph::element::Type_t::i8 :
                    return make(ngraph::runtime::reference::convert<float, std::int8_t>);
                case ngraph::element::Type_t::u8 :
                    return make(ngraph::runtime::reference::convert<float, std::uint8_t>);
                case ngraph::element::Type_t::i16 :
                    return make(ngraph::runtime::reference::convert<float, std::int16_t>);
                case ngraph::element::Type_t::i32 :
                    return make(ngraph::runtime::reference::convert<float, std::int32_t>);
            default:
                IE_THROW() << "Unsupported convertion from " << src << " to " << dst; return {};
            }
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

template <> Converter::Conversion::Ptr Converter::Convert(const opset::ArmConvert& node) {
    if (node.get_input_element_type(0) == node.get_convert_element_type()) {
        return MakeConversion<arm_compute::NECopy>(node.input(0), node.output(0));
    }
    return MakeConversion<arm_compute::NEDepthConvertLayer>(node.input(0),
                                                                node.output(0),
                                                                arm_compute::ConvertPolicy::SATURATE);
}
}  //  namespace ArmPlugin
