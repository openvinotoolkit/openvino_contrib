// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_batchnorm_v0_to_v5.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/rt_info.hpp"

using namespace ArmPlugin;

NGRAPH_RTTI_DEFINITION(pass::ConvertBatchNormInferenceV0toV5, "ConvertBatchNormInferenceV0toV5", 0);

pass::ConvertBatchNormInferenceV0toV5::ConvertBatchNormInferenceV0toV5() {
    register_matcher(
        std::make_shared<ngraph::pattern::Matcher>(
            std::make_shared<ngraph::op::v0::BatchNormInference>(
                ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(),
                ngraph::pattern::any_input(), ngraph::pattern::any_input(), 1.0),
                "ConvertBatchNormInferenceV0toV5"),
        [](ngraph::pattern::Matcher& m) {
            auto bnv0 = std::dynamic_pointer_cast<ngraph::op::v0::BatchNormInference>(m.get_match_root());
            if (!bnv0) {
                return false;
            }

            enum Input {Gamma, Beta, Features, Mean, Variance};
            auto bnv5 = std::make_shared<ngraph::op::v5::BatchNormInference>(
                                            bnv0->input_value(Input::Features),
                                            bnv0->input_value(Input::Gamma),
                                            bnv0->input_value(Input::Beta),
                                            bnv0->input_value(Input::Mean),
                                            bnv0->input_value(Input::Variance),
                                            bnv0->get_eps_value());

            bnv5->set_friendly_name(bnv0->get_friendly_name());
            ngraph::copy_runtime_info(bnv0, bnv5);
            ngraph::replace_node(bnv0, bnv5);
            return true;
        });
}

