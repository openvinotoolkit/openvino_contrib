// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_interpolate_arm.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>


using Nearest_mode    = ngraph::op::v4::Interpolate::NearestMode;
using Transform_mode  = ngraph::op::v4::Interpolate::CoordinateTransformMode;
using InterpolateMode = ngraph::op::v4::Interpolate::InterpolateMode;

bool isSupportedConfiguration(const ngraph::op::v4::Interpolate& node) {
    auto& inp_shape = node.get_input_shape(0);
    auto& out_shape = node.get_output_shape(0);

    float scale_h = out_shape[2] / inp_shape[2];
    float scale_w = out_shape[3] / inp_shape[3];
    bool is_upsample = scale_h > 1 && scale_w > 1;

    auto& attrs = node.get_attrs();
    auto& coord_mode = attrs.coordinate_transformation_mode;
    auto& nearest_mode = attrs.nearest_mode;

    if (coord_mode == Transform_mode::asymmetric && nearest_mode == Nearest_mode::floor) {
        return is_upsample;
    }

    if (coord_mode == Transform_mode::align_corners && nearest_mode == Nearest_mode::round_prefer_ceil) {
        return true;
    }

    if (coord_mode == Transform_mode::half_pixel &&
        (nearest_mode == Nearest_mode::simple || nearest_mode == Nearest_mode::round_prefer_ceil)) {
        return false;
    }

    if (coord_mode == Transform_mode::asymmetric && (nearest_mode == Nearest_mode::simple || nearest_mode == Nearest_mode::floor)) {
        return is_upsample;
    }

    if (is_upsample) {
        bool int_factor = scale_h == static_cast<int>(scale_h) && scale_w == static_cast<int>(scale_w);
        if (int_factor && coord_mode != Transform_mode::asymmetric &&
            (nearest_mode == Nearest_mode::round_prefer_ceil || nearest_mode == Nearest_mode::round_prefer_floor)) {
            return true;
        }
    } else if (scale_h < 1 && scale_w < 1) {
        float down_scale_h = inp_shape[2] / out_shape[2];
        float down_scale_w = inp_shape[3] / out_shape[3];
        bool int_factor = down_scale_h == static_cast<int>(down_scale_h) && down_scale_w == static_cast<int>(down_scale_w);

        if (int_factor && coord_mode != Transform_mode::align_corners && nearest_mode == Nearest_mode::simple) {
            return true;
        }

        if (int_factor && nearest_mode == Nearest_mode::round_prefer_ceil &&
            ((out_shape[2] > 1 && out_shape[3] > 1) || coord_mode != Transform_mode::half_pixel)) {
            return true;
        }
    }

    return false;
}


ArmPlugin::pass::ConvertInterpolate::ConvertInterpolate() {
    auto interp = ngraph::pattern::wrap_type<opset::Interpolate>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto interp = std::dynamic_pointer_cast<opset::Interpolate>(m.get_match_root());
        if (!interp) {
            return false;
        }

        if (interp->get_shape().size() != 4) {
            return false;
        }

        auto& attrs = interp->get_attrs();
        auto& pads_begin = attrs.pads_begin;
        auto& pads_end   = attrs.pads_end;

        if (!std::all_of(pads_begin.begin(), pads_begin.end(), [](int i){return i == 0;}) ||
            !std::all_of(pads_end.begin(), pads_end.end(), [](int i){return i == 0;})) {
            return false;
        }

        auto& nearest_mode = attrs.nearest_mode;
        auto& coord_mode   = attrs.coordinate_transformation_mode;
        if (attrs.antialias || coord_mode == Transform_mode::tf_half_pixel_for_nn || nearest_mode == Nearest_mode::ceil) {
            return false;
        }

        if (attrs.mode == opset::Interpolate::InterpolateMode::cubic) {
            return false;
        }

        if (attrs.mode == opset::Interpolate::InterpolateMode::nearest && !isSupportedConfiguration(*interp)) {
            return false;
        }

        if (coord_mode == Transform_mode::pytorch_half_pixel) {
            return false;
        }

        auto arm_interp = std::make_shared<opset::ArmInterpolate>(interp->input_value(0),
                                                                  interp->input_value(1),
                                                                  interp->input_value(2),
                                                                  interp->input_value(3),
                                                                  attrs);
        arm_interp->set_friendly_name(interp->get_friendly_name());
        ngraph::copy_runtime_info(interp, arm_interp);
        ngraph::replace_node(interp, arm_interp);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(interp, "ConvertInterpolate");
    register_matcher(m, callback);
}
