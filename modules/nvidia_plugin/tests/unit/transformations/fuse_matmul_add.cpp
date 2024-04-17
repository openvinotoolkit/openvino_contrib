// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformer/fuse_matmul_add.hpp"

#include <gtest/gtest.h>

#include <tuple>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
#include "transformer/nodes/fully_connected.hpp"

using ov::nvidia_gpu::nodes::FullyConnected;
using namespace ov;
using namespace std;

namespace testing {

TEST(fuse_matmul_add, parameters_matmul_add_constant) {
    shared_ptr<ov::Model> model, model_ref;
    {
        auto input0 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 512});
        auto input1 = make_shared<op::v0::Parameter>(element::f32, Shape{1024, 512});
        auto matmul = make_shared<op::v0::MatMul>(input0, input1, false, true);
        auto const_node = op::v0::Constant::create(element::f32, Shape{1, 1024}, {1});
        auto add = make_shared<op::v1::Add>(matmul, const_node);
        model = make_shared<Model>(add, ParameterVector{input0, input1});

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::InitNodeInfo>();
        pass_manager.register_pass<nvidia_gpu::pass::FullyConnectedTransformation>();
        pass_manager.run_passes(model);

        ASSERT_EQ(count_ops_of_type<op::v0::MatMul>(model), 0);
    }
    {
        auto input0 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 512});
        auto input1 = make_shared<op::v0::Parameter>(element::f32, Shape{1024, 512});
        auto const_node = op::v0::Constant::create(element::f32, Shape{1, 1024}, {1});
        auto fc = make_shared<FullyConnected>(input0, input1, const_node, false, true);
        model_ref = make_shared<Model>(fc, ParameterVector{input0, input1});
    }

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(fuse_matmul_add, parameters_matmul_add_parameter_fail) {
    shared_ptr<ov::Model> model, model_ref;
    {
        auto input0 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 512});
        auto input1 = make_shared<op::v0::Parameter>(element::f32, Shape{512, 1024});
        auto matmul = make_shared<op::v0::MatMul>(input0, input1, false, false);
        auto input3 = make_shared<op::v0::Parameter>(element::f32, Shape{1, 1024});
        auto add = make_shared<op::v1::Add>(matmul, input3);
        model = make_shared<Model>(add, ParameterVector{input0, input1, input3});
    }
    model_ref = model->clone();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::FullyConnectedTransformation>();
    pass_manager.run_passes(model);

    ASSERT_EQ(count_ops_of_type<op::v0::MatMul>(model), 1);

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(fuse_matmul_add, parameter_constant_matmul_add_constant) {
    shared_ptr<ov::Model> model, model_ref;
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{512, 1});
        auto const_node0 = op::v0::Constant::create(element::f32, Shape{1024, 512}, {1});
        auto matmul = make_shared<op::v0::MatMul>(input, const_node0, true, true);
        auto const_node1 = op::v0::Constant::create(element::f32, Shape{1, 1024}, {2});
        auto add = make_shared<op::v1::Add>(matmul, const_node1);
        model = make_shared<Model>(add, ParameterVector{input});

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::InitNodeInfo>();
        pass_manager.register_pass<nvidia_gpu::pass::FullyConnectedTransformation>();
        pass_manager.run_passes(model);

        ASSERT_EQ(count_ops_of_type<op::v0::MatMul>(model), 0);
    }
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{512, 1});
        auto const_node0 = op::v0::Constant::create(element::f32, Shape{1024, 512}, {1});
        auto const_node1 = op::v0::Constant::create(element::f32, Shape{1, 1024}, {2});
        auto fc = make_shared<FullyConnected>(input, const_node0, const_node1, true, true);
        model_ref = make_shared<Model>(fc, ParameterVector{input});
    }

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(fuse_matmul_add, constant_parameter_matmul_add_constant) {
    shared_ptr<ov::Model> model, model_ref;
    {
        auto const_node0 = op::v0::Constant::create(element::f32, Shape{1024, 512}, {1});
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{1024, 1});
        auto matmul = make_shared<op::v0::MatMul>(const_node0, input, true, false);
        auto const_node1 = op::v0::Constant::create(element::f32, Shape{1, 512}, {2});
        auto add = make_shared<op::v1::Add>(matmul, const_node1);
        model = make_shared<Model>(add, ParameterVector{input});

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::InitNodeInfo>();
        pass_manager.register_pass<nvidia_gpu::pass::FullyConnectedTransformation>();
        pass_manager.run_passes(model);

        ASSERT_EQ(count_ops_of_type<op::v0::MatMul>(model), 0);
    }
    {
        auto const_node0 = op::v0::Constant::create(element::f32, Shape{1024, 512}, {1});
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{1024, 1});
        auto const_node1 = op::v0::Constant::create(element::f32, Shape{1, 512}, {2});
        auto fc = make_shared<FullyConnected>(const_node0, input, const_node1, true, false);
        model_ref = make_shared<Model>(fc, ParameterVector{input});
    }

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(fuse_matmul_add, parameter_variadic_split_matmul_add_constant) {
    shared_ptr<ov::Model> model, model_ref;
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{197, 128});
        auto split = make_shared<op::v1::VariadicSplit>(input,
                                                        op::v0::Constant::create(element::i32, {}, {0}),
                                                        op::v0::Constant::create(element::i32, Shape{2}, {196, 1}));
        auto const_node0 = op::v0::Constant::create(element::f32, Shape{128, 128}, {1});
        auto matmul = make_shared<op::v0::MatMul>(split->output(1), const_node0, false, true);
        auto const_node1 = op::v0::Constant::create(element::f32, Shape{1, 128}, {2});
        auto add = make_shared<op::v1::Add>(matmul, const_node1);
        auto concat = make_shared<op::v0::Concat>(OutputVector{split->output(0), add}, 0);
        model = make_shared<Model>(concat, ParameterVector{input});

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::InitNodeInfo>();
        pass_manager.register_pass<nvidia_gpu::pass::FullyConnectedTransformation>();
        pass_manager.run_passes(model);

        ASSERT_EQ(count_ops_of_type<op::v0::MatMul>(model), 0);
    }
    {
        auto input = make_shared<op::v0::Parameter>(element::f32, Shape{197, 128});
        auto split = make_shared<op::v1::VariadicSplit>(input,
                                                        op::v0::Constant::create(element::i32, {}, {0}),
                                                        op::v0::Constant::create(element::i32, Shape{2}, {196, 1}));
        auto const_node0 = op::v0::Constant::create(element::f32, Shape{128, 128}, {1});
        auto const_node1 = op::v0::Constant::create(element::f32, Shape{1, 128}, {2});
        auto fc = make_shared<FullyConnected>(split->output(1), const_node0, const_node1, false, true);
        auto concat = make_shared<op::v0::Concat>(OutputVector{split->output(0), fc}, 0);
        model_ref = make_shared<Model>(concat, ParameterVector{input});
    }

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(fuse_matmul_add, parameters_matmul_dynamic) {
    shared_ptr<ov::Model> model;
    {
        auto input0 = make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 512});
        auto input1 = make_shared<op::v0::Parameter>(element::f32, Shape{1024, 512});
        auto matmul = make_shared<op::v0::MatMul>(input0, input1, false, true);
        auto const_node = op::v0::Constant::create(element::f32, Shape{1, 1024}, {1});
        auto add = make_shared<op::v1::Add>(matmul, const_node);
        model = make_shared<Model>(add, ParameterVector{input0, input1});

        pass::Manager pass_manager;
        pass_manager.register_pass<pass::InitNodeInfo>();
        pass_manager.register_pass<nvidia_gpu::pass::FullyConnectedTransformation>();
        pass_manager.run_passes(model);

        ASSERT_EQ(count_ops_of_type<op::v0::MatMul>(model), 1);
    }
}

}  // namespace testing
