// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/ngraph/itt.hpp"
#include "noop_broadcast_transformation.hpp"

#include <gsl/gsl_assert>
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include <openvino/op/broadcast.hpp>

using namespace ov::pass::pattern;

namespace ov::nvidia_gpu::pass {
namespace {

bool eliminate_noop_broadcast(Matcher &m) {
    auto node = std::dynamic_pointer_cast<ov::op::v3::Broadcast>(m.get_match_root());
    Expects(node);

    auto in_shape = node->get_input_shape(0);
    auto out_shape = node->get_output_shape(0);
    if (in_shape != out_shape) {
        return false;
    }

    return ov::replace_output_update_name(node->output(0), node->input_value(0));
}

}  // namespace

NoopBroadcastTransformation::NoopBroadcastTransformation() {
    MATCHER_SCOPE(NoopBroadcastTransformation);
    const auto op = wrap_type<ov::op::v3::Broadcast>();
    const auto m = std::make_shared<Matcher>(op, matcher_name);

    matcher_pass_callback callback = [](Matcher &m) { return eliminate_noop_broadcast(m); };

    register_matcher(m, callback);
}

}  // namespace ov::nvidia_gpu::pass
