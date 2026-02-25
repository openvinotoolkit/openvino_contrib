// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformer/remove_redundant_convert_transformation.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace std;

TEST(remove_redundant_convert, remove_redundant_convert) {
    std::shared_ptr<Model> f;
    {
        Shape shape{};
        auto input_fp32 = make_shared<op::v0::Parameter>(element::f32, shape);
        auto convert_f32 = make_shared<op::v0::Convert>(input_fp32, element::f32);
        f = make_shared<Model>(make_shared<op::v0::Abs>(convert_f32), ParameterVector{input_fp32});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::RemoveRedundantConvertTransformation>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
}

TEST(remove_redundant_convert, remove_symmetric_converts) {
    std::shared_ptr<Model> f;
    {
        Shape shape{};
        auto input_fp32 = make_shared<op::v0::Parameter>(element::f32, shape);
        auto convert_f16 = make_shared<op::v0::Convert>(input_fp32, element::f16);
        auto convert_f32 = make_shared<op::v0::Convert>(convert_f16, element::f32);
        f = make_shared<Model>(make_shared<op::v0::Abs>(convert_f32), ParameterVector{input_fp32});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::RemoveRedundantConvertTransformation>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
}

TEST(remove_redundant_convert, remove_several_converts) {
    std::shared_ptr<Model> f;
    {
        Shape shape{};
        auto input_fp32 = make_shared<op::v0::Parameter>(element::f32, shape);
        auto convert_f16 = make_shared<op::v0::Convert>(input_fp32, element::f16);
        auto convert_f64 = make_shared<op::v0::Convert>(convert_f16, element::f64);
        auto convert_f32 = make_shared<op::v0::Convert>(convert_f64, element::f32);
        f = make_shared<Model>(make_shared<op::v0::Abs>(convert_f32), ParameterVector{input_fp32});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::RemoveRedundantConvertTransformation>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 0);
}

TEST(remove_redundant_convert, merge_converts_into_one) {
    std::shared_ptr<Model> f;
    {
        Shape shape{};
        auto input_fp32 = make_shared<op::v0::Parameter>(element::f32, shape);
        auto convert_f16 = make_shared<op::v0::Convert>(input_fp32, element::f16);
        auto convert_f64 = make_shared<op::v0::Convert>(convert_f16, element::f64);
        f = make_shared<Model>(make_shared<op::v0::Abs>(convert_f64), ParameterVector{input_fp32});
    }

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::RemoveRedundantConvertTransformation>();
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::v0::Convert>(f), 1);
    ASSERT_EQ(f->get_results()[0]->input(0).get_element_type(), element::f64);
}