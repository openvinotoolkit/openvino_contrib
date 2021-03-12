// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include <arm_compute/runtime/NEON/functions/NEElementwiseUnaryLayer.h>
#include <ngraph/runtime/reference/acos.hpp>
#include <ngraph/runtime/reference/acosh.hpp>
#include <ngraph/runtime/reference/asin.hpp>
#include <ngraph/runtime/reference/asinh.hpp>
#include <ngraph/runtime/reference/atan.hpp>
#include <ngraph/runtime/reference/atanh.hpp>
#include <ngraph/runtime/reference/cos.hpp>
#include <ngraph/runtime/reference/cosh.hpp>
#include <ngraph/runtime/reference/erf.hpp>
#include <ngraph/runtime/reference/sinh.hpp>
#include <ngraph/runtime/reference/tan.hpp>
#include <ngraph/runtime/reference/tanh.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template<> Converter::Conversion::Ptr Converter::Convert(const opset::Acos& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::acos<half_float::half>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::acos<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Acosh& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::acosh<half_float::half>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::acosh<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Asin& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::asin<half_float::half>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::asin<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Asinh& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::asinh<half_float::half>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::asinh<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Atan& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::atan<half_float::half>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::atan<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Atanh& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::atanh<half_float::half>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::atanh<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Cos& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::cos<half_float::half>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::cos<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Cosh& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::cosh<half_float::half>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::cosh<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Sin& node) {
    return MakeConversion<arm_compute::NESinLayer>(node.input(0), node.output(0));
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Sinh& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::sinh<half_float::half>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::sinh<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Tan& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::tan<half_float::half>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::tan<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Erf& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    switch (node.input(0).get_element_type()) {
        case ngraph::element::Type_t::f16 : return make(ngraph::runtime::reference::erf<half_float::half>);
        case ngraph::element::Type_t::f32 : return make(ngraph::runtime::reference::erf<float>);
        default: THROW_IE_EXCEPTION << "Unsupported Type: " << node.get_element_type(); return {};
    }
}
}  //  namespace ArmPlugin
