// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_inputs_precision.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>

using namespace ArmPlugin;

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertPrecisionBase, "ConvertPrecisionBase", 0);
template <class T>
ngraph::matcher_pass_callback ArmPlugin::pass::ConvertPrecisionBase::convert_precision(const std::vector<int>& indices) {
    return [=](ngraph::pattern::Matcher& m) {
        auto node = m.get_match_root();

        if (!std::dynamic_pointer_cast<T>(node)) {
            return false;
        }

        bool is_modified = false;
        for (auto idx : indices) {
            if (idx >= node->get_input_size()) {
                IE_THROW() << "Index should be in range: [0, " << node->get_input_size() << ")";
            }
            if (node->get_input_element_type(0) != node->get_input_element_type(idx)) {
                auto convert = std::make_shared<opset::ConvertLike>(node->input_value(idx), node->input_value(0));
                node->set_argument(idx, convert);
                is_modified = true;
            }
        }
        return is_modified;
    };
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertPReluPrecision, "ConvertPReluPrecision", 0);
ArmPlugin::pass::ConvertPReluPrecision::ConvertPReluPrecision() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::PRelu>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                      ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                      ngraph::pattern::has_static_shape()), "ConvertPReluPrecision");
    register_matcher(m, convert_precision<opset::PRelu>({1}));
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertProposalPrecision, "ConvertProposalPrecision", 0);
ArmPlugin::pass::ConvertProposalPrecision::ConvertProposalPrecision() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Proposal>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                         ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                         ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                         ngraph::pattern::has_static_shape()), "ConvertProposalPrecision");
    register_matcher(m, convert_precision<opset::Proposal>({1, 2}));
}

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertInterpolatePrecision, "ConvertInterpolatePrecision", 0);
ArmPlugin::pass::ConvertInterpolatePrecision::ConvertInterpolatePrecision() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<opset::Interpolate>({ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                            ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                            ngraph::pattern::any_input(ngraph::pattern::has_static_shape()),
                                                            ngraph::pattern::any_input(ngraph::pattern::has_static_shape())},
                                                            ngraph::pattern::has_static_shape()), "ConvertInterpolatePrecision");
    register_matcher(m, convert_precision<opset::Interpolate>({2}));
}
