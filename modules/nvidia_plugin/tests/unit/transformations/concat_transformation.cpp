// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformer/concat_transformation.hpp"

#include <gtest/gtest.h>

#include <tuple>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
#include "transformer/nodes/concat_optimized.hpp"

using ov::nvidia_gpu::nodes::ConcatOptimized;
using namespace ov;
using namespace std;

namespace testing {

TEST(concat_optimized, concat_2_inputs_axis_1) {
    shared_ptr<ov::Model> model, model_ref;
    int64_t axis = 1;
    {
        auto input0 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 512});
        auto input1 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 255, 512});
        auto concat = make_shared<op::v0::Concat>(NodeVector{input0, input1}, axis);
        model = make_shared<Model>(concat, ParameterVector{input0, input1});

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::InitNodeInfo>();
        pass_manager.register_pass<nvidia_gpu::pass::ConcatTransformation>();
        pass_manager.run_passes(model);
    }
    {
        auto input0 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 512});
        auto input1 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 255, 512});
        auto concat = make_shared<ConcatOptimized>(NodeVector{input0, input1}, axis);
        model_ref = make_shared<Model>(concat, ParameterVector{input0, input1});
    }

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(concat_optimized, concat_2_inputs_several_concats) {
    shared_ptr<ov::Model> model, model_ref;
    int64_t axis = 1;

    auto input0 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 512});
    auto input1 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 255, 512});
    auto input2 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 512});
    auto concat0 = make_shared<op::v0::Concat>(NodeVector{input0, input1}, axis);
    auto result0 = make_shared<op::v0::Result>(concat0);
    auto concat1 = make_shared<op::v0::Concat>(NodeVector{input1, input2}, axis);
    auto result1 = make_shared<op::v0::Result>(concat1);
    model = make_shared<Model>(ResultVector{result0, result1}, ParameterVector{input0, input1, input2});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ConcatTransformation>();
    pass_manager.run_passes(model);

    ASSERT_EQ(count_ops_of_type<ConcatOptimized>(model), 0);
}

TEST(concat_optimized, concat_3_inputs_axis_negative) {
    shared_ptr<ov::Model> model, model_ref;
    int64_t axis = -2;
    {
        auto input0 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 512});
        auto input1 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 255, 512});
        auto input2 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 256, 512});
        auto concat = make_shared<op::v0::Concat>(NodeVector{input0, input1, input2}, axis);
        model = make_shared<Model>(concat, ParameterVector{input0, input1, input2});

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::InitNodeInfo>();
        pass_manager.register_pass<nvidia_gpu::pass::ConcatTransformation>();
        pass_manager.run_passes(model);
    }
    {
        auto input0 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 512});
        auto input1 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 255, 512});
        auto input2 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 256, 512});
        auto concat = make_shared<ConcatOptimized>(NodeVector{input0, input1, input2}, axis);
        model_ref = make_shared<Model>(concat, ParameterVector{input0, input1, input2});
    }

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(concat_optimized, concat_with_constant_fail) {
    shared_ptr<ov::Model> model;
    int64_t axis = 1;

    auto input0 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1, 512});
    auto const_node = op::v0::Constant::create(element::f32, Shape{1, 255, 512}, {1});
    auto concat = make_shared<op::v0::Concat>(NodeVector{input0, const_node}, axis);
    model = make_shared<Model>(concat, ParameterVector{input0});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ConcatTransformation>();
    pass_manager.run_passes(model);

    ASSERT_EQ(count_ops_of_type<ConcatOptimized>(model), 0);
}

TEST(concat_optimized, concat_dynamic_fail) {
    shared_ptr<ov::Model> model;
    int64_t axis = 1;

    auto input0 = make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 1, 512});
    auto input1 = make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 255, 512});
    auto concat = make_shared<op::v0::Concat>(NodeVector{input0, input1}, axis);
    model = make_shared<Model>(concat, ParameterVector{input0, input1});

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ConcatTransformation>();
    pass_manager.run_passes(model);

    ASSERT_EQ(count_ops_of_type<ConcatOptimized>(model), 0);
}

}  // namespace testing
