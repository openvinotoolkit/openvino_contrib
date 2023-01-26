// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_layout.hpp"
#include "opset/opset.hpp"

#include <openvino/pass/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"
#include "transpose_utils.hpp"

using namespace ov;

namespace ArmPlugin {
namespace pass {

ConvertBatchNormLayout::ConvertBatchNormLayout() {
    enum BatchNormInput {Features, Gamma, Beta, Mean, Variance};
    auto root = ov::pass::pattern::wrap_type<opset::v5::ArmBatchNormInference>(ov::pass::pattern::has_static_rank());
    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (transformation_callback(node)) {
            return false;
        }
        auto batch_norm = ov::as_type_ptr<opset::v5::ArmBatchNormInference>(node);
        if (!batch_norm) {
            return false;
        }
        size_t rank = batch_norm->get_output_partial_shape(0).size();
        if (rank < 4 || rank > 5) {
            return false;
        }
        auto input_transpose = transpose_on_input(batch_norm->input_value(BatchNormInput::Features), rank);
        auto output_shape = transpose_output_shape(batch_norm, rank);
        auto new_batch_norm = std::make_shared<opset::v5::ArmBatchNormInference>(input_transpose,
                                                                               batch_norm->input_value(BatchNormInput::Gamma),
                                                                               batch_norm->input_value(BatchNormInput::Beta),
                                                                               batch_norm->input_value(BatchNormInput::Mean),
                                                                               batch_norm->input_value(BatchNormInput::Variance),
                                                                               batch_norm->get_eps_value(),
                                                                               output_shape);
        auto transpose = transpose_on_output(new_batch_norm, rank);
        transpose->set_friendly_name(batch_norm->get_friendly_name());
        copy_runtime_info(batch_norm, {new_batch_norm, input_transpose, transpose});
        replace_node(batch_norm, transpose);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(root, "ConvertBatchNormLayout");
    register_matcher(m, callback);
}

ConvertBatchToSpaceLayout::ConvertBatchToSpaceLayout() {
    enum BatchToSpace {Data, BlockShape, CropsBegin, CropsEnd};
    auto root = ov::pass::pattern::wrap_type<opset::v1::ArmBatchToSpace>(ov::pass::pattern::has_static_rank());
    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (transformation_callback(node)) {
            return false;
        }
        auto batch_to_space = ov::as_type_ptr<opset::v1::ArmBatchToSpace>(node);
        if (!batch_to_space) {
            return false;
        }
        size_t rank = batch_to_space->get_output_partial_shape(0).size();
        if (rank < 4 || rank > 5) {
            return false;
        }
        auto input_transpose = transpose_on_input(batch_to_space->input_value(BatchToSpace::Data), rank);
        auto output_shape = transpose_output_shape(batch_to_space, rank);
        auto new_batch_to_space = std::make_shared<opset::v1::ArmBatchToSpace>(input_transpose,
                                                                             batch_to_space->input_value(BatchToSpace::BlockShape),
                                                                             batch_to_space->input_value(BatchToSpace::CropsBegin),
                                                                             batch_to_space->input_value(BatchToSpace::CropsEnd),
                                                                             output_shape);
        auto transpose = transpose_on_output(new_batch_to_space, rank);
        transpose->set_friendly_name(transpose->get_friendly_name());
        copy_runtime_info(batch_to_space, {new_batch_to_space, input_transpose, transpose});
        replace_node(batch_to_space, transpose);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(root, "ConvertBatchToSpaceLayout");
    register_matcher(m, callback);
}

ConvertDepthToSpaceLayout::ConvertDepthToSpaceLayout() {
    auto root = ov::pass::pattern::wrap_type<opset::v0::ArmDepthToSpace>(ov::pass::pattern::has_static_rank());
    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        std::cout << node->get_friendly_name() << std::endl;
        if (transformation_callback(node)) {
            return false;
        }
        auto depth_to_space = ov::as_type_ptr<opset::v0::ArmDepthToSpace>(node);
        if (!depth_to_space) {
            return false;
        }
        size_t rank = depth_to_space->get_output_partial_shape(0).size();
        if (rank < 4 || rank > 5) {
            return false;
        }
        auto input_transpose = transpose_on_input(depth_to_space->input_value(0), rank);
        auto output_shape = transpose_output_shape(depth_to_space, rank);
        auto new_depth_to_space = std::make_shared<opset::v0::ArmDepthToSpace>(input_transpose,
                                                                             depth_to_space->get_mode(),
                                                                             depth_to_space->get_block_size(),
                                                                             output_shape);
        auto transpose = transpose_on_output(new_depth_to_space, rank);
        transpose->set_friendly_name(depth_to_space->get_friendly_name());
        copy_runtime_info(depth_to_space, {new_depth_to_space, input_transpose, transpose});
        replace_node(depth_to_space, transpose);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(root, "ConvertDepthToSpaceLayout");
    register_matcher(m, callback);
}

ConvertInterpolateLayout::ConvertInterpolateLayout() {
    auto root = ov::pass::pattern::wrap_type<opset::ArmInterpolate>(ov::pass::pattern::has_static_rank());
    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (transformation_callback(node)) {
            return false;
        }
        auto interpolate_op = ov::as_type_ptr<opset::ArmInterpolate>(node);
        if (!interpolate_op) {
            return false;
        }
        size_t rank = interpolate_op->get_output_partial_shape(0).size();
        if (rank < 4 || rank > 5) {
            return false;
        }
        enum InterpolateInput {data, sizes, scales, axes};
        auto input_transpose = transpose_on_input(interpolate_op->input_value(InterpolateInput::data), rank);
        auto output_shape = transpose_output_shape(interpolate_op, rank);
        auto new_interpolate_op = std::make_shared<opset::ArmInterpolate>(input_transpose,
                                                                          interpolate_op->input_value(InterpolateInput::sizes),
                                                                          interpolate_op->input_value(InterpolateInput::scales),
                                                                          interpolate_op->input_value(InterpolateInput::axes),
                                                                          interpolate_op->get_attrs());
        auto transpose = transpose_on_output(new_interpolate_op, rank);
        transpose->set_friendly_name(interpolate_op->get_friendly_name());
        copy_runtime_info(interpolate_op, {new_interpolate_op, input_transpose, transpose});
        replace_node(interpolate_op, transpose);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(root, "ConvertInterpolateLayout");
    register_matcher(m, callback);
}
} // pass
} // ArmPlugin

