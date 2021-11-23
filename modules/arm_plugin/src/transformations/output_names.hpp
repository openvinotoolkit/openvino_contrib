// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include <openvino/core/node.hpp>

std::string create_ie_output_name(const ngraph::Output<const ngraph::Node>& output) {
    std::string out_name;
    NGRAPH_SUPPRESS_DEPRECATED_START
    auto tensor_name = output.get_tensor().get_name();
    NGRAPH_SUPPRESS_DEPRECATED_END
    if (!tensor_name.empty()) {
        out_name = std::move(tensor_name);
    } else {
        const auto& prev_layer = output.get_node_shared_ptr();
        out_name = prev_layer->get_friendly_name();
        if (prev_layer->get_output_size() != 1) {
            out_name += "." + std::to_string(output.get_index());
        }
    }
    return out_name;
}

std::string create_ie_output_name(const ngraph::Output<ngraph::Node>& output) {
    return create_ie_output_name(ov::Output<const ngraph::Node>(output.get_node(), output.get_index()));
}
