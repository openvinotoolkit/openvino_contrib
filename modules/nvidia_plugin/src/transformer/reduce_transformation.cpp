// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/pass/itt.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "transformations/op_conversions/convert_reduce_to_reshape.hpp"
#include "reduce_transformation.hpp"

using namespace ov::pass::pattern;

namespace ov::nvidia_gpu::pass {

namespace {

bool is_reduce_to_be_transformed(const ov::Output<ov::Node>& output) {
    auto node = std::dynamic_pointer_cast<ov::op::util::ArithmeticReductionKeepDims>(output.get_node_shared_ptr());
    if (!node) {
        return false;
    }
    if (node->is_dynamic()) {
        return false;
    }
    if (!node->get_keep_dims()) {
        return true;
    }
    const ov::Shape input_shape = node->input(0).get_shape();
    const ov::Shape output_shape = node->output(0).get_shape();
    return input_shape == output_shape;
}

template <class T>
bool transform_reduce(Matcher &m) {
    auto reduce = std::dynamic_pointer_cast<ov::op::util::ArithmeticReductionKeepDims>(m.get_match_root());
    const ov::Shape input_shape = reduce->input(0).get_shape();
    const ov::Shape output_shape = reduce->output(0).get_shape();
    auto consumers = reduce->get_output_target_inputs(0);

    auto new_reduce = std::make_shared<T>(reduce->input_value(0), reduce->input_value(1), true);
    new_reduce->set_friendly_name(reduce->get_friendly_name());

    ov::NodeVector new_ops;
    const ov::Shape new_output_shape = new_reduce->output(0).get_shape();
    auto reshape_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{output_shape.size()}, output_shape);
    auto reshape = std::make_shared<ov::op::v1::Reshape>(new_reduce, reshape_const, false);
    for (auto consumer : consumers) {
        consumer.replace_source_output(reshape);
    }
    new_ops = {new_reduce, reshape_const, reshape};

    ov::copy_runtime_info(reduce, new_ops);
    return true;
}
} // namespace

ReduceMaxTransformation::ReduceMaxTransformation() {
    MATCHER_SCOPE(ReduceMaxTransformation);
    auto reduce = wrap_type<ov::op::v1::ReduceMax>(is_reduce_to_be_transformed);
    matcher_pass_callback callback = [](Matcher &m) {
        return transform_reduce<ov::op::v1::ReduceMax>(m);
        };
    auto m = std::make_shared<Matcher>(reduce, matcher_name);
    register_matcher(m, callback);
}

ReduceMeanTransformation::ReduceMeanTransformation() {
    MATCHER_SCOPE(ReduceMeanTransformation);
    auto reduce = wrap_type<ov::op::v1::ReduceMean>(is_reduce_to_be_transformed);
    matcher_pass_callback callback = [](Matcher &m) {
        return transform_reduce<ov::op::v1::ReduceMean>(m);
        };
    auto m = std::make_shared<Matcher>(reduce, matcher_name);
    register_matcher(m, callback);
}

ReduceMinTransformation::ReduceMinTransformation() {
    MATCHER_SCOPE(ReduceMinTransformation);
    auto reduce = wrap_type<ov::op::v1::ReduceMin>(is_reduce_to_be_transformed);
    matcher_pass_callback callback = [](Matcher &m) {
        return transform_reduce<ov::op::v1::ReduceMin>(m);
        };
    auto m = std::make_shared<Matcher>(reduce, matcher_name);
    register_matcher(m, callback);
}

ReduceProdTransformation::ReduceProdTransformation() {
    MATCHER_SCOPE(ReduceProdTransformation);
    auto reduce = wrap_type<ov::op::v1::ReduceProd>(is_reduce_to_be_transformed);
    matcher_pass_callback callback = [](Matcher &m) {
        return transform_reduce<ov::op::v1::ReduceProd>(m);
        };
    auto m = std::make_shared<Matcher>(reduce, matcher_name);
    register_matcher(m, callback);
}

ReduceSumTransformation::ReduceSumTransformation() {
    MATCHER_SCOPE(ReduceSumTransformation);
    auto reduce = wrap_type<ov::op::v1::ReduceSum>(is_reduce_to_be_transformed);
    matcher_pass_callback callback = [](Matcher &m) {
        return transform_reduce<ov::op::v1::ReduceSum>(m);
        };
    auto m = std::make_shared<Matcher>(reduce, matcher_name);
    register_matcher(m, callback);
}

bool ov::nvidia_gpu::pass::ReduceTransformation::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(ReduceTransformation);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertReduceToReshape>();

    auto reduce_transformations = manager.register_pass<ov::pass::GraphRewrite>();
    ADD_MATCHER(reduce_transformations, ReduceMaxTransformation)
    ADD_MATCHER(reduce_transformations, ReduceMeanTransformation)
    ADD_MATCHER(reduce_transformations, ReduceMinTransformation)
    ADD_MATCHER(reduce_transformations, ReduceProdTransformation)
    ADD_MATCHER(reduce_transformations, ReduceSumTransformation)
    reduce_transformations->set_name("ov::nvidia_gpu::reduce_transformations");

    manager.run_passes(m);
    return false;
}

}  // namespace ov::nvidia_gpu::pass