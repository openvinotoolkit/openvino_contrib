// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_batchnorm_to_arm.hpp"
#include "ngraph/rt_info.hpp"
#include "opset/opset.hpp"
#include <openvino/pass/pattern/op/wrap_type.hpp>

using namespace ArmPlugin;

NGRAPH_RTTI_DEFINITION(pass::ConvertBatchNormInferenceToARM, "ConvertBatchNormInferenceToARM", 0);

pass::ConvertBatchNormInferenceToARM::ConvertBatchNormInferenceToARM() {
    auto root = ov::pass::pattern::wrap_type<ov::op::v5::BatchNormInference>(ov::pass::pattern::has_static_rank());
    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto bnv5 = std::dynamic_pointer_cast<ov::op::v5::BatchNormInference>(m.get_match_root());
        if (!bnv5) {
            return false;
        }

        enum ArmBatchNorm {Features, Gamma, Beta, Mean, Variance};
        auto bnv_arm = std::make_shared<opset::ArmBatchNormInference>(
                                        bnv5->input_value(ArmBatchNorm::Features),
                                        bnv5->input_value(ArmBatchNorm::Gamma),
                                        bnv5->input_value(ArmBatchNorm::Beta),
                                        bnv5->input_value(ArmBatchNorm::Mean),
                                        bnv5->input_value(ArmBatchNorm::Variance),
                                        bnv5->get_eps_value());

        bnv_arm->set_friendly_name(bnv5->get_friendly_name());
        ov::copy_runtime_info(bnv5, bnv_arm);
        ov::replace_node(bnv5, bnv_arm);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(root, "ConvertBatchNormInferenceToARM");
    register_matcher(m, callback);
}

