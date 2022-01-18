// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0



#include <arm_compute/runtime/NEON/functions/NEMeanStdDevNormalizationLayer.h>
#include <ngraph/runtime/reference/mvn.hpp>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {
template <typename U>
ngraph::AxisSet mvn_6_reduction_axes(const U* axes, const ngraph::Shape& axes_shape) {
    auto rank = axes_shape.size();
    auto v = std::vector<U>(axes, axes + axes_shape[0]);
    std::vector<size_t> reduction_axes(v.size(), 0);
    for (int i = 0; i < v.size(); i++) {
        if (v[i] < 0) {
            if (rank + v[i] < 0) {
                IE_THROW() << "Unexpected axis";
            }
            reduction_axes[i] = rank + static_cast<size_t>(v[i]);
        } else {
            reduction_axes[i] = static_cast<size_t>(v[i]);
        }
    }
    return ngraph::AxisSet(reduction_axes);
}

template <typename T, typename U>
void wrap_mvn_6(const T* arg,
                const U* axes,
                T* out,
                const ngraph::Shape& in_shape,
                const ngraph::Shape& axes_shape,
                bool normalize_variance,
                double eps,
                ngraph::op::MVNEpsMode eps_mode) {
    ngraph::AxisSet reduction_axes = mvn_6_reduction_axes(axes, axes_shape);
    ngraph::runtime::reference::mvn_6(arg,
                                      out,
                                      in_shape,
                                      reduction_axes,
                                      normalize_variance,
                                      eps,
                                      eps_mode);
}

template <> Converter::Conversion::Ptr Converter::Convert(const opset::MVN& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_normalize_variance(),
                                    static_cast<double>(node.get_eps()),
                                    node.get_eps_mode());
    };

    return CallSwitch(
        AP_WRAP(make, wrap_mvn_6),
        node.input(0), floatTypes,
        node.input(1), indexTypes);
}

template <> Converter::Conversion::Ptr Converter::Convert(const opset::ArmMVN& node) {
    return MakeConversion<arm_compute::NEMeanStdDevNormalizationLayer>(node.input(0), node.output(0), node.get_eps());
}
}  //  namespace ArmPlugin