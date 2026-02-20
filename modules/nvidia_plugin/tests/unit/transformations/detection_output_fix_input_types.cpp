// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "cuda_operation_registry.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/detection_output.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformer/detection_output_fix_input_types_transformation.hpp"

using namespace ov;

namespace {

op::v0::DetectionOutput::Attributes get_attrs() {
    op::v0::DetectionOutput::Attributes attrs;

    attrs.background_label_id = 12;
    attrs.top_k = 75;
    attrs.variance_encoded_in_target = true;
    attrs.keep_top_k = {50};
    attrs.code_type = "caffe.PriorBoxParameter.CORNER";
    attrs.share_location = true;
    attrs.nms_threshold = 0.5f;
    attrs.confidence_threshold = 0.3f;
    attrs.clip_after_nms = true;
    attrs.clip_before_nms = true;
    attrs.decrease_label_id = true;
    attrs.normalized = true;
    attrs.input_height = 1ul;
    attrs.input_width = 1ul;
    attrs.objectness_score = 0.4f;
    attrs.num_classes = 11;

    return attrs;
}

ov::nvidia_gpu::OperationBase::Ptr createPluginOperation(const std::shared_ptr<ov::Node>& node) {
    return ov::nvidia_gpu::OperationRegistry::getInstance().createOperation(
        ov::nvidia_gpu::CreationContext{CUDA::Device{}, false},
        node,
        std::vector<ov::nvidia_gpu::TensorID>{ov::nvidia_gpu::TensorID{0u}},
        std::vector<ov::nvidia_gpu::TensorID>{ov::nvidia_gpu::TensorID{0u}});
}

template <typename T>
std::shared_ptr<T> find_op(const std::shared_ptr<Model> model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (const auto op = std::dynamic_pointer_cast<T>(node)) {
            return op;
        }
    }
    return nullptr;
}

}  // namespace

namespace testing {

TEST(detection_output_fix_input_types, three_params) {
    // Proposals type is different
    const ParameterVector param_vec{std::make_shared<op::v0::Parameter>(element::Type_t::f16, Shape{1, 60}),
                                    std::make_shared<op::v0::Parameter>(element::Type_t::f16, Shape{1, 165}),
                                    std::make_shared<op::v0::Parameter>(element::Type_t::f32, Shape{1, 1, 60})};

    std::shared_ptr<Model> model, model_ref;
    {
        const auto d_out =
            std::make_shared<op::v0::DetectionOutput>(param_vec[0], param_vec[1], param_vec[2], get_attrs());
        ASSERT_THROW(createPluginOperation(d_out), ov::AssertFailure);

        model = std::make_shared<Model>(d_out, param_vec);

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::InitNodeInfo>();
        pass_manager.register_pass<nvidia_gpu::pass::DetectionOutputFixInputTypesTransformation>();
        pass_manager.run_passes(model);

        ASSERT_EQ(count_ops_of_type<op::v0::DetectionOutput>(model), 1);
        ASSERT_EQ(model->get_results()[0]->input(0).get_element_type(), element::f16);

        const auto new_d_out = find_op<op::v0::DetectionOutput>(model);
        ASSERT_NE(new_d_out, nullptr);
        ASSERT_NO_THROW(createPluginOperation(new_d_out));
    }
    {
        const auto convert = std::make_shared<op::v0::Convert>(param_vec[2], element::Type_t::f16);

        const auto d_out = std::make_shared<op::v0::DetectionOutput>(param_vec[0], param_vec[1], convert, get_attrs());
        ASSERT_NO_THROW(createPluginOperation(d_out));

        model_ref = std::make_shared<Model>(d_out, param_vec);
        ASSERT_EQ(count_ops_of_type<op::v0::DetectionOutput>(model), 1);
    }
    const auto res = FunctionsComparator::with_default().compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(detection_output_fix_input_types, five_params) {
    // Proposals type is different
    const ParameterVector param_vec{std::make_shared<op::v0::Parameter>(element::Type_t::f32, Shape{1, 60}),
                                    std::make_shared<op::v0::Parameter>(element::Type_t::f32, Shape{1, 165}),
                                    std::make_shared<op::v0::Parameter>(element::Type_t::f16, Shape{1, 1, 60}),
                                    std::make_shared<op::v0::Parameter>(element::Type_t::f32, Shape{1, 30}),
                                    std::make_shared<op::v0::Parameter>(element::Type_t::f32, Shape{1, 60})};

    std::shared_ptr<Model> model, model_ref;
    {
        const auto d_out = std::make_shared<op::v0::DetectionOutput>(
            param_vec[0], param_vec[1], param_vec[2], param_vec[3], param_vec[4], get_attrs());
        ASSERT_THROW(createPluginOperation(d_out), ov::AssertFailure);

        model = std::make_shared<Model>(d_out, param_vec);

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::InitNodeInfo>();
        pass_manager.register_pass<nvidia_gpu::pass::DetectionOutputFixInputTypesTransformation>();
        pass_manager.run_passes(model);

        ASSERT_EQ(count_ops_of_type<op::v0::DetectionOutput>(model), 1);
        ASSERT_EQ(model->get_results()[0]->input(0).get_element_type(), element::f32);

        const auto new_d_out = find_op<op::v0::DetectionOutput>(model);
        ASSERT_NE(new_d_out, nullptr);
        ASSERT_NO_THROW(createPluginOperation(new_d_out));
    }
    {
        const auto convert = std::make_shared<op::v0::Convert>(param_vec[2], element::Type_t::f32);

        const auto d_out = std::make_shared<op::v0::DetectionOutput>(
            param_vec[0], param_vec[1], convert, param_vec[3], param_vec[4], get_attrs());
        ASSERT_NO_THROW(createPluginOperation(d_out));

        model_ref = std::make_shared<Model>(d_out, param_vec);
        ASSERT_EQ(count_ops_of_type<op::v0::DetectionOutput>(model), 1);
    }
    const auto res = FunctionsComparator::with_default().compare(model, model_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

}  // namespace testing
