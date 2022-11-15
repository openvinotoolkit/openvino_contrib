// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "convert_precision_fp16_to_fp32.hpp"
#include "convert_concat.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/pass/visualize_tree.hpp"

template<ov::element::Type_t T>
bool has_type(const std::shared_ptr<ov::Model>& m, const std::string& name_layer) {
    for (auto & node : m->get_ordered_ops()) {
        if (node->get_friendly_name() == name_layer) {
            for (auto & input : node->inputs()) {
                if (input.get_element_type() == ov::element::Type(T)) {
                    return true;
                }
            }
            for (auto & output : node->outputs()) {
                if (output.get_element_type() == ov::element::Type(T)) {
                    return true;
                }
            }
        }
    }
    return false;
}

TEST(TransformationConvertFP16ToFP32Tests, ConvertPrecision_Bucketize) {
    std::shared_ptr<ov::Model> m(nullptr);

    auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f16, ov::Shape{20});
    auto k = ov::opset8::Constant::create(ov::element::f16, ov::Shape{1}, {10});
    auto b = std::make_shared<ov::opset8::Bucketize>(input, k);
    m = std::make_shared<ov::Model>(ov::OutputVector{b}, ov::ParameterVector{input});
    {
        ov::pass::Manager manager;
        manager.register_pass<ArmPlugin::pass::ConvertPrecisionFP16ToFP32>();
        manager.run_passes(m);
    }
    ASSERT_FALSE(has_type<ov::element::Type_t::f16>(m, b->get_friendly_name()));
    ASSERT_TRUE(has_type<ov::element::Type_t::f32>(m, b->get_friendly_name()));
}

TEST(TransformationConvertFP16ToFP32Tests, ConvertPrecision_Bucketize_Mixed) {
    std::shared_ptr<ov::Model> m(nullptr);

    auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{20});
    auto k = ov::opset8::Constant::create(ov::element::f16, ov::Shape{1}, {10});
    auto b = std::make_shared<ov::opset8::Bucketize>(input, k);

    {
        m = std::make_shared<ov::Model>(ov::OutputVector{b}, ov::ParameterVector{input});

        ov::pass::Manager manager;
        manager.run_passes(m);

        ASSERT_TRUE(has_type<ov::element::Type_t::f16>(m, b->get_friendly_name()));
        ASSERT_TRUE(has_type<ov::element::Type_t::f32>(m, b->get_friendly_name()));
    }
    {
        m = std::make_shared<ov::Model>(ov::OutputVector{b}, ov::ParameterVector{input});

        ov::pass::Manager manager;
        manager.register_pass<ArmPlugin::pass::ConvertPrecisionFP16ToFP32>();
        manager.run_passes(m);

        ASSERT_FALSE(has_type<ov::element::Type_t::f16>(m, b->get_friendly_name()));
        ASSERT_TRUE(has_type<ov::element::Type_t::f32>(m, b->get_friendly_name()));
    }
}

TEST(TransformationConvertFP16ToFP32Tests, ConvertPrecision_Concat_Native) {
    const auto A = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f16, ov::Shape{1});
    const auto B = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f16, ov::Shape{2});
    const auto C = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f16, ov::Shape{3});
    auto concat = std::make_shared<ov::opset8::Concat>(ov::NodeVector{A, B, C}, 0);
    auto m = std::make_shared<ov::Model>(concat, ov::ParameterVector{A, B, C});

    {
        ov::pass::Manager manager;
        manager.register_pass<ArmPlugin::pass::ConvertConcat>();
        manager.register_pass<ArmPlugin::pass::ConvertPrecisionFP16ToFP32>();
        manager.run_passes(m);
    }

    ASSERT_TRUE(has_type<ov::element::Type_t::f16>(m, concat->get_friendly_name()));
    ASSERT_FALSE(has_type<ov::element::Type_t::f32>(m, concat->get_friendly_name()));

    ASSERT_TRUE(has_type<ov::element::Type_t::f16>(m, A->get_friendly_name()));
    ASSERT_TRUE(has_type<ov::element::Type_t::f16>(m, B->get_friendly_name()));
    ASSERT_TRUE(has_type<ov::element::Type_t::f16>(m, C->get_friendly_name()));

    for (auto&& output : m->outputs()) {
        ASSERT_FALSE(has_type<ov::element::Type_t::f16>(m, output.get_node()->get_friendly_name()));
        ASSERT_TRUE(has_type<ov::element::Type_t::f32>(m, output.get_node()->get_friendly_name()));
    }
}
