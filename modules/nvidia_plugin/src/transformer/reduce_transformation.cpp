// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/pass/itt.hpp"

#include "exec_graph_info.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
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
    return !node->get_keep_dims();
}

bool transform_reduce(Matcher &m) {
    auto reduce = std::dynamic_pointer_cast<ov::op::util::ArithmeticReductionKeepDims>(m.get_match_root());
    const ov::Shape output_shape = reduce->output(0).get_shape();
    auto consumers = reduce->get_output_target_inputs(0);

    std::shared_ptr<ov::Node> new_reduce;
    if (ov::as_type_ptr<ov::op::v1::ReduceMax>(reduce)) {
        new_reduce = std::make_shared<ov::op::v1::ReduceMax>(reduce->input_value(0), reduce->input_value(1), true);
    } else if (ov::as_type_ptr<ov::op::v1::ReduceMean>(reduce)) {
        new_reduce = std::make_shared<ov::op::v1::ReduceMean>(reduce->input_value(0), reduce->input_value(1), true);
    } else if (ov::as_type_ptr<ov::op::v1::ReduceMin>(reduce)) {
        new_reduce = std::make_shared<ov::op::v1::ReduceMin>(reduce->input_value(0), reduce->input_value(1), true);
    } else if (ov::as_type_ptr<ov::op::v1::ReduceProd>(reduce)) {
        new_reduce = std::make_shared<ov::op::v1::ReduceProd>(reduce->input_value(0), reduce->input_value(1), true);
    } else if (ov::as_type_ptr<ov::op::v1::ReduceSum>(reduce)) {
        new_reduce = std::make_shared<ov::op::v1::ReduceSum>(reduce->input_value(0), reduce->input_value(1), true);
    } else {
        return false;
    }
    new_reduce->set_friendly_name(reduce->get_friendly_name());
    auto reshape_const = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{output_shape.size()}, output_shape);
    auto reshape = std::make_shared<ov::op::v1::Reshape>(new_reduce, reshape_const, false);
    for (auto consumer : consumers) {
        consumer.replace_source_output(reshape);
    }
    ov::NodeVector new_ops = {new_reduce, reshape_const, reshape};
    ov::copy_runtime_info(reduce, new_ops);
    for (auto& new_op : new_ops) {
        new_op->get_rt_info()[ExecGraphInfoSerialization::ORIGINAL_NAMES] = reduce->get_friendly_name();
    }
    return true;
}
} // namespace

ReduceTransformation::ReduceTransformation() {
    MATCHER_SCOPE(ReduceTransformation);
    auto reduce = wrap_type<ov::op::v1::ReduceMax,
                            ov::op::v1::ReduceMean,
                            ov::op::v1::ReduceMin,
                            ov::op::v1::ReduceProd,
                            ov::op::v1::ReduceSum>(is_reduce_to_be_transformed);
    matcher_pass_callback callback = [](Matcher &m) {
        return transform_reduce(m);
        };
    auto m = std::make_shared<Matcher>(reduce, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::nvidia_gpu::pass