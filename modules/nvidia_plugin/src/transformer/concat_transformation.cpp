// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/pass/itt.hpp"
#include "concat_transformation.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "nodes/concat_optimized.hpp"

using namespace ov::pass::pattern;

namespace ov::nvidia_gpu::pass {
namespace {
bool change_concat_to_concat_optimized(Matcher& m) {
    using ov::nvidia_gpu::nodes::ConcatOptimized;

    auto concat = std::dynamic_pointer_cast<ov::op::v0::Concat>(m.get_match_root());
    for (auto& in : concat->inputs()) {
        auto source_output = in.get_source_output();
        if (dynamic_cast<ov::op::v0::Constant*>(source_output.get_node())) {
            return false;
        }
        unsigned num_concats = 0;
        for (auto& out : source_output.get_target_inputs()) {
            if (dynamic_cast<ov::op::v0::Concat*>(out.get_node())) {
                num_concats += 1;
            }
        }
        // NOTE: Apply ConcatOptimized node only for nodes with single output for Concat
        if (num_concats > 1) {
            return false;
        }
    }

    const auto& outputShape = concat->get_output_shape(0);
    int64_t axis = concat->get_axis();
    if (axis < 0) {
        axis += outputShape.size();
    }
    if (axis < 0 || axis >= outputShape.size()) {
        return false;
    }
    auto num_chunks =
        std::accumulate(outputShape.begin(), outputShape.begin() + axis + 1, 1, std::multiplies<size_t>());
    const size_t sizeAboveAxis = num_chunks / outputShape[axis];
    if (sizeAboveAxis != 1) {
        return false;
    }

    const auto& ins = concat->inputs();
    ov::OutputVector inOuts;
    std::transform(
        ins.begin(), ins.end(), std::back_inserter(inOuts), [](const auto& i) { return i.get_source_output(); });

    auto concat_optimized = std::make_shared<ConcatOptimized>(inOuts, concat->get_axis());
    concat_optimized->set_friendly_name(concat->get_friendly_name());
    ov::copy_runtime_info(concat, concat_optimized);
    ov::replace_node(concat, concat_optimized);

    return true;
}
} // namespace

ConcatTransformation::ConcatTransformation() {
    MATCHER_SCOPE(ConcatTransformation);
    auto concat = wrap_type<ov::op::v0::Concat>(has_static_shape());

    matcher_pass_callback callback = [](Matcher& m) { return change_concat_to_concat_optimized(m); };

    auto m = std::make_shared<Matcher>(concat, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::nvidia_gpu::pass
