// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_fft_arm.hpp"
#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

static bool is_decomposable(unsigned int res) {
    const std::set<unsigned int> supported_factors{ 2, 3, 4, 5, 7, 8 };
    std::vector<unsigned int> stages;

    if (res < 2) { // Too small to decompose
        return false;
    }

    // Create reverse iterator (Start decomposing from the larger supported factors)
    for (auto rfactor_it = supported_factors.rbegin(); rfactor_it != supported_factors.rend(); ++rfactor_it) {
        const unsigned int factor = *rfactor_it;
        while ((res % factor) == 0) {
            res /= factor;
        }
    }

    if (res > 1) { // Couldn't decompose with given factors
        return false;
    }
    return true;
}

template <typename T>
static bool convert_fft(const std::shared_ptr<T>& fft, bool inverse) {
    if (!fft) {
        return false;
    }

    if (fft->get_input_element_type(0) != ngraph::element::f32) {
        return false;
    }

    if (fft->get_input_size() > 2) {
        return false;
    }

    auto axes = std::dynamic_pointer_cast<ArmPlugin::opset::Constant>(fft->input_value(1).get_node_shared_ptr());
    if (!axes) {
        return false;
    }
    std::vector<std::int64_t> axes_vals = axes->template cast_vector<std::int64_t>();

    if (axes_vals.size() > 2) {
        return false;
    }

    int axes_set = 0;
    auto &shape = fft->get_input_shape(0);
    for (auto& axis : axes_vals) {
        axis = axis < 0 ? - axis : shape.size() - 1 - axis;
        if (!axis || axis > 2) {
            return false;
        }
        auto axis_len = *(shape.rbegin() + axis);
        if (axis_len > 1) {
            if (!is_decomposable(axis_len)) {
                return false;
            }
            axes_set |= axis;
        }
    }

    if (!axes_set) {
        return false;
    }

    auto arm_fft = std::make_shared<ArmPlugin::opset::ArmFFT>(fft->input_value(0), axes_set & 1 ? ArmPlugin::opset::ArmFFT::Axis::axisX :
                                                                                                  ArmPlugin::opset::ArmFFT::Axis::axisY, inverse);
    ngraph::copy_runtime_info(fft, arm_fft);
    if (axes_set > 2) {
        arm_fft = std::make_shared<ArmPlugin::opset::ArmFFT>(arm_fft, ArmPlugin::opset::ArmFFT::Axis::axisY, inverse);
        ngraph::copy_runtime_info(fft, arm_fft);
    }

    arm_fft->set_friendly_name(fft->get_friendly_name());
    ngraph::replace_node(fft, arm_fft);
    return true;
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertDFT, "ConvertDFT", 0);
ArmPlugin::pass::ConvertDFT::ConvertDFT() {
    auto fft = ngraph::pattern::wrap_type<opset::DFT>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        return convert_fft(std::dynamic_pointer_cast<opset::DFT>(m.get_match_root()), false);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fft, "ConvertDFT");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertIDFT, "ConvertIDFT", 0);
ArmPlugin::pass::ConvertIDFT::ConvertIDFT() {
    auto fft = ngraph::pattern::wrap_type<opset::IDFT>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        return convert_fft(std::dynamic_pointer_cast<opset::IDFT>(m.get_match_root()), true);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fft, "ConvertIDFT");
    register_matcher(m, callback);
}
