// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "noop_broadcast_transformation.hpp"

#include <gsl/gsl_assert>
#include <openvino/op/broadcast.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

namespace ngraph::pass {

NGRAPH_RTTI_DEFINITION(ngraph::pass::NoopBroadcastTransformation, "NoopBroadcastTransformation", 0);

namespace {

bool eliminate_noop_broadcast(ngraph::pattern::Matcher &m) {
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
    const auto op = ngraph::pattern::wrap_type<ov::op::v3::Broadcast>();
    const auto m = std::make_shared<ngraph::pattern::Matcher>(op, "NoopBroadcastTransformation");

    matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) { return eliminate_noop_broadcast(m); };

    register_matcher(m, callback);
}

}  // namespace ngraph::pass
