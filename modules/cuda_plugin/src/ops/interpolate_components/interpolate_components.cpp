// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "interpolate_components.hpp"

#include <fmt/format.h>

#include <gsl/gsl_assert>

#include "error.hpp"
#include "ngraph/validation_util.hpp"

namespace CUDAPlugin::Interpolate::Details {

void getAxesAndScales(const ngraph::op::v4::Interpolate& node, std::vector<size_t>& axes, std::vector<float>& scales) {
    axes = ngraph::get_constant_from_source(node.input_value(3))->cast_vector<size_t>();
    switch (node.get_attrs().shape_calculation_mode) {
        case ngraph::op::v4::Interpolate::ShapeCalcMode::sizes: {
            const auto& input_shape = node.get_input_shape(0);
            const auto& output_shape = node.get_output_shape(0);
            scales.resize(axes.size());
            for (size_t i = 0; i < axes.size(); ++i) {
                const auto axe = axes[i];
                scales[i] = static_cast<float>(output_shape[axe]) / static_cast<float>(input_shape[axe]);
            }
        } break;
        case ngraph::op::v4::Interpolate::ShapeCalcMode::scales:
            scales = ngraph::get_constant_from_source(node.input_value(2))->cast_vector<float>();
            Expects(axes.size() == scales.size());
            break;
        default:
            throwIEException(fmt::format("Interpolate operation: unsupported shape calculation mode {}.",
                                         node.get_attrs().shape_calculation_mode));
    }
}

}  // namespace CUDAPlugin::Interpolate::Details
