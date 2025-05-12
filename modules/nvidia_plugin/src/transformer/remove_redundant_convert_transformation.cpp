// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "remove_redundant_convert_transformation.hpp"
#include "transformations/common_optimizations/nop_elimination.hpp"

using namespace ov::pass::pattern;

namespace ov::nvidia_gpu::pass {
namespace {

bool remove_redundant_convert(Matcher& m) {
    auto last_convert = std::dynamic_pointer_cast<ov::op::v0::Convert>(m.get_match_root());
    OPENVINO_ASSERT(std::dynamic_pointer_cast<ov::op::v0::Convert>(last_convert) != nullptr);

    const auto& prev_convert = last_convert->input(0).get_source_output().get_node_shared_ptr();
    OPENVINO_ASSERT(std::dynamic_pointer_cast<ov::op::v0::Convert>(prev_convert) != nullptr);

    return ov::replace_output_update_name(prev_convert->output(0), prev_convert->input_value(0));
}

}  // namespace

bool RemoveRedundantConvertTransformation::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_FUNCTION_SCOPE(RemoveRedundantConvertTransformation);
    ov::pass::Manager manager;
    // Merge subsequent converts first
    manager.register_pass<MergeSubsequentConvertTransformation>();
    // Remove converts which are still present in graph after merge but doing nothing
    manager.register_pass<ov::pass::EliminateConvert>();
    manager.run_passes(m);
    return true;
}

MergeSubsequentConvertTransformation::MergeSubsequentConvertTransformation() {
    MATCHER_SCOPE(MergeSubsequentConvertTransformation);
    const auto convert0 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>(ov::pass::pattern::consumers_count(1));
    const auto convert1 = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({convert0});
    const auto m = std::make_shared<ov::pass::pattern::Matcher>(convert1, matcher_name);

    matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) { return remove_redundant_convert(m); };

    register_matcher(m, callback);
}

}  // namespace ov::nvidia_gpu::pass
