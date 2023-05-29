// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <fmt/format.h>

#include "interpolate_components.hpp"

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"

#include "error.hpp"

namespace ov::nvidia_gpu::Interpolate::Details {

void getAxesAndScales(const ov::op::v4::Interpolate& node, std::vector<size_t>& axes, std::vector<float>& scales) {
    axes = ov::as_type_ptr<op::v0::Constant>(node.input_value(3).get_node_shared_ptr())->cast_vector<size_t>();
    switch (node.get_attrs().shape_calculation_mode) {
        case ov::op::v4::Interpolate::ShapeCalcMode::SIZES: {
            const auto& input_shape = node.get_input_shape(0);
            const auto& output_shape = node.get_output_shape(0);
            scales.resize(axes.size());
            for (size_t i = 0; i < axes.size(); ++i) {
                const auto axe = axes[i];
                scales[i] = static_cast<float>(output_shape[axe]) / static_cast<float>(input_shape[axe]);
            }
        } break;
        case ov::op::v4::Interpolate::ShapeCalcMode::SCALES:
            scales = ov::as_type_ptr<op::v0::Constant>(node.input_value(2).get_node_shared_ptr())->cast_vector<float>();
            OPENVINO_ASSERT(axes.size() == scales.size());
            break;
        default:
            throw_ov_exception(fmt::format("Interpolate operation: unsupported shape calculation mode {}.",
                                         node.get_attrs().shape_calculation_mode));
    }
}

}  // namespace ov::nvidia_gpu::Interpolate::Details
