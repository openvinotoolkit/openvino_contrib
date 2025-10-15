// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformer/reduce_transformation.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/opsets/opset.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace std;

namespace {
Shape get_output_shape(const ov::Shape& input_shape, const std::vector<int32_t>& axis) {
    Shape output_shape;
    for (size_t i = 0; i < input_shape.size(); i++) {
        if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
            output_shape.push_back(input_shape.at(i));
        }
    }
    return output_shape;
}
template <typename T>
shared_ptr<Model> create_model(const Shape& input_shape, const vector<int32_t>& axis, bool keep_dims = false) {
    auto input = make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto axis_const = op::v0::Constant::create(element::i32, Shape{axis.size()}, axis);
    auto reduce = make_shared<T>(input, axis_const, keep_dims);
    return make_shared<Model>(reduce, ParameterVector{input});
}
template <typename T>
shared_ptr<Model> create_ref_model(const Shape& input_shape,
                                   const vector<int32_t>& axis,
                                   const Shape& output_shape,
                                   bool skip_reduce = false) {
    auto input = make_shared<op::v0::Parameter>(element::f32, input_shape);
    std::shared_ptr<ov::Node> last_node = input;
    if (!skip_reduce) {
        auto axis_const = op::v0::Constant::create(element::i32, Shape{axis.size()}, axis);
        last_node = make_shared<T>(input, axis_const, true);
    }
    auto reshape_const = op::v0::Constant::create(element::i64, Shape{output_shape.size()}, output_shape);
    auto reshape = make_shared<op::v1::Reshape>(last_node, reshape_const, false);
    return make_shared<Model>(reshape, ParameterVector{input});
}
}  // namespace

TEST(reduce_transformation, reduce_max_keep_dims_true) {
    const Shape input_shape{10, 20, 30, 40};
    const std::vector<int32_t> axis{0, 3};
    auto model = create_model<op::v1::ReduceMax>(input_shape, axis, true);
    auto model_ref = model->clone();

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_transformation, reduce_max_keep_dims_false) {
    const Shape input_shape{10, 20, 30, 40};
    const std::vector<int32_t> axis{0, 3};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceMax>(input_shape, axis);
    auto model_ref = create_ref_model<op::v1::ReduceMax>(input_shape, axis, output_shape);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_transformation, reduce_mean_keep_dims_false) {
    const Shape input_shape{10, 20, 30, 40};
    const std::vector<int32_t> axis{1};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceMean>(input_shape, axis);
    auto model_ref = create_ref_model<op::v1::ReduceMean>(input_shape, axis, output_shape);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_transformation, reduce_min_keep_dims_false) {
    const Shape input_shape{10, 20, 30, 40};
    const std::vector<int32_t> axis{0, 1, 2, 3};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceMin>(input_shape, axis);
    auto model_ref = create_ref_model<op::v1::ReduceMin>(input_shape, axis, output_shape);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_transformation, reduce_prod_keep_dims_false) {
    const Shape input_shape{10, 20, 30, 40};
    const std::vector<int32_t> axis{3};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceProd>(input_shape, axis);
    auto model_ref = create_ref_model<op::v1::ReduceProd>(input_shape, axis, output_shape);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_transformation, reduce_sum_keep_dims_false) {
    const Shape input_shape{10, 20, 30, 40};
    const std::vector<int32_t> axis{1, 2};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceSum>(input_shape, axis);
    auto model_ref = create_ref_model<op::v1::ReduceSum>(input_shape, axis, output_shape);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_transformation, reduce_sum_keep_all_matched) {
    const Shape input_shape{10, 20, 1, 40};
    const std::vector<int32_t> axis{2};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceSum>(input_shape, axis);
    auto model_ref = create_ref_model<op::v1::ReduceSum>(input_shape, axis, output_shape, true);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(reduce_transformation, reduce_prod_keep_all_matched) {
    const Shape input_shape{10, 1, 30, 40};
    const std::vector<int32_t> axis{1};
    const Shape output_shape = get_output_shape(input_shape, axis);
    auto model = create_model<op::v1::ReduceProd>(input_shape, axis);
    auto model_ref = create_ref_model<op::v1::ReduceProd>(input_shape, axis, output_shape, true);

    pass::Manager pass_manager;
    pass_manager.register_pass<pass::InitNodeInfo>();
    pass_manager.register_pass<nvidia_gpu::pass::ReduceTransformation>();
    pass_manager.run_passes(model);

    ASSERT_NO_THROW(check_rt_info(model));
    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}
