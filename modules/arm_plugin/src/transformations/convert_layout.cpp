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
    auto root = ov::pass::pattern::wrap_type<ov::op::v1::BatchToSpace>(ov::pass::pattern::has_static_rank());
    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (transformation_callback(node)) {
            return false;
        }
        auto batch_to_space = ov::as_type_ptr<ov::op::v1::BatchToSpace>(node);
        if (!batch_to_space) {
            return false;
        }
        size_t rank = batch_to_space->get_output_partial_shape(0).size();
        if (rank < 4 || rank > 5) {
            return false;
        }
        std::cout << batch_to_space->get_friendly_name() << std::endl;
        auto input_transpose = transpose_on_input(batch_to_space->input_value(BatchToSpace::Data), rank);
        auto new_batch_to_space = std::make_shared<ov::op::v1::BatchToSpace>(input_transpose,
                                                                             batch_to_space->input_value(BatchToSpace::BlockShape),
                                                                             batch_to_space->input_value(BatchToSpace::CropsBegin),
                                                                             batch_to_space->input_value(BatchToSpace::CropsEnd));
        auto transpose = transpose_on_output(new_batch_to_space, rank);
        transpose->set_friendly_name(batch_to_space->get_friendly_name());
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
} // pass
} // ArmPlugin

