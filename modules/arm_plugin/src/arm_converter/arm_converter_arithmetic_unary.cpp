// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


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
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::acos),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Acosh& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::acosh),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Asin& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::asin),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Asinh& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::asinh),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Atan& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::atan),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Atanh& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::atanh),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Cos& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::cos),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Cosh& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::cosh),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Sin& node) {
    return MakeConversion<arm_compute::NESinLayer>(node.input(0), node.output(0));
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Sinh& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::sinh),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Tan& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::tan),
        node.input(0), floatTypes);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Erf& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction, node.input(0), node.output(0), ngraph::shape_size(node.get_output_shape(0)));
    };
    return CallSwitch(
        AP_WRAP(make, ngraph::runtime::reference::erf),
        node.input(0), floatTypes);
}
}  //  namespace ArmPlugin
