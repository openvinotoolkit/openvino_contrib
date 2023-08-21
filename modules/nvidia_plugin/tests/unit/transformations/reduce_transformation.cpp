// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
#include "transformer/reduce_transformation.hpp"

using namespace ov;
using namespace std;

TEST(reduce_transformation, reduce_max_keep_dims_true) {
    shared_ptr<ov::Model> model, model_ref;
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{10, 20, 30, 40});
        auto axis = op::v0::Constant::create(element::i32, Shape{2}, {0, 3});
        auto reduce = make_shared<op::v1::ReduceMax>(input, axis, true);
        model = make_shared<Model>(reduce, ParameterVector{input});
        model_ref = model->clone();
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_transformation, reduce_max_keep_dims_false) {
    shared_ptr<ov::Model> model, model_ref;
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{10, 20, 30, 40});
        auto axis = op::v0::Constant::create(element::i32, Shape{2}, {0, 3});
        auto reduce = make_shared<op::v1::ReduceMax>(input, axis, false);
        model = make_shared<Model>(reduce, ParameterVector{input});
    }
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{10, 20, 30, 40});
        auto axis = op::v0::Constant::create(element::i32, Shape{2}, {0, 3});
        auto reduce = make_shared<op::v1::ReduceMax>(input, axis, true);
        auto reshape_const = op::v0::Constant::create(element::i32, Shape{2}, {20, 30});
        auto reshape = make_shared<op::v1::Reshape>(reduce, reshape_const, false);
        model_ref = make_shared<Model>(reshape, ParameterVector{input});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_transformation, reduce_mean_keep_dims_false) {
    shared_ptr<ov::Model> model, model_ref;
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{10, 20, 30, 40});
        auto axis = op::v0::Constant::create(element::i32, Shape{1}, {1});
        auto reduce = make_shared<op::v1::ReduceMean>(input, axis, false);
        model = make_shared<Model>(reduce, ParameterVector{input});
    }
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{10, 20, 30, 40});
        auto axis = op::v0::Constant::create(element::i32, Shape{1}, {1});
        auto reduce = make_shared<op::v1::ReduceMean>(input, axis, true);
        auto reshape_const = op::v0::Constant::create(element::i32, Shape{3}, {10, 30, 40});
        auto reshape = make_shared<op::v1::Reshape>(reduce, reshape_const, false);
        model_ref = make_shared<Model>(reshape, ParameterVector{input});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_transformation, reduce_min_keep_dims_false) {
    shared_ptr<ov::Model> model, model_ref;
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{10, 20, 30, 40});
        auto axis = op::v0::Constant::create(element::i32, Shape{4}, {0, 1, 2, 3});
        auto reduce = make_shared<op::v1::ReduceMin>(input, axis, false);
        model = make_shared<Model>(reduce, ParameterVector{input});
    }
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{10, 20, 30, 40});
        auto axis = op::v0::Constant::create(element::i32, Shape{4}, {0, 1, 2, 3});
        auto reduce = make_shared<op::v1::ReduceMin>(input, axis, true);
        auto reshape_const = op::v0::Constant::create(element::i32, Shape{0}, {});
        auto reshape = make_shared<op::v1::Reshape>(reduce, reshape_const, false);
        model_ref = make_shared<Model>(reshape, ParameterVector{input});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_transformation, reduce_prod_keep_dims_false) {
    shared_ptr<ov::Model> model, model_ref;
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{10, 20, 30, 40});
        auto axis = op::v0::Constant::create(element::i32, Shape{1}, {3});
        auto reduce = make_shared<op::v1::ReduceProd>(input, axis, false);
        model = make_shared<Model>(reduce, ParameterVector{input});
    }
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{10, 20, 30, 40});
        auto axis = op::v0::Constant::create(element::i32, Shape{1}, {3});
        auto reduce = make_shared<op::v1::ReduceProd>(input, axis, true);
        auto reshape_const = op::v0::Constant::create(element::i32, Shape{3}, {10, 20, 30});
        auto reshape = make_shared<op::v1::Reshape>(reduce, reshape_const, false);
        model_ref = make_shared<Model>(reshape, ParameterVector{input});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_transformation, reduce_sum_keep_dims_false) {
    shared_ptr<ov::Model> model, model_ref;
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{10, 20, 30, 40});
        auto axis = op::v0::Constant::create(element::i32, Shape{2}, {1, 2});
        auto reduce = make_shared<op::v1::ReduceSum>(input, axis, false);
        model = make_shared<Model>(reduce, ParameterVector{input});
    }
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{10, 20, 30, 40});
        auto axis = op::v0::Constant::create(element::i32, Shape{2}, {1, 2});
        auto reduce = make_shared<op::v1::ReduceSum>(input, axis, true);
        auto reshape_const = op::v0::Constant::create(element::i32, Shape{2}, {10, 40});
        auto reshape = make_shared<op::v1::Reshape>(reduce, reshape_const, false);
        model_ref = make_shared<Model>(reshape, ParameterVector{input});
    }
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}