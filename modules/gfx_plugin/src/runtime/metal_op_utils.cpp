// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/metal_op_utils.hpp"

#include <unordered_set>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
struct InputBinding {
    ov::Output<ov::Node> output;
    std::shared_ptr<ov::op::v0::Parameter> param;
};

InputBinding bind_input_or_param(const std::shared_ptr<const ov::Node>& src,
                                 const ov::PartialShape& pshape,
                                 const ov::element::Type& et) {
    InputBinding binding{};
    if (auto param = ov::as_type_ptr<const ov::op::v0::Parameter>(src)) {
        binding.output = std::const_pointer_cast<ov::op::v0::Parameter>(param)->output(0);
        binding.param = std::const_pointer_cast<ov::op::v0::Parameter>(param);
        return binding;
    }
    if (auto c = ov::as_type_ptr<const ov::op::v0::Constant>(src)) {
        binding.output = std::const_pointer_cast<ov::op::v0::Constant>(c)->output(0);
        return binding;
    }
    auto param = std::make_shared<ov::op::v0::Parameter>(et, pshape);
    binding.output = param->output(0);
    binding.param = param;
    return binding;
}
}  // namespace

std::shared_ptr<ov::Model> make_single_op_model(const std::shared_ptr<const ov::Node>& node) {
    OPENVINO_ASSERT(node, "make_single_op_model: node is null");
    ov::OutputVector inputs;
    inputs.reserve(node->get_input_size());
    ov::ParameterVector parameters;
    parameters.reserve(node->get_input_size());
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        auto src = node->input_value(i).get_node_shared_ptr();
        auto binding = bind_input_or_param(src,
                                           node->get_input_partial_shape(i),
                                           node->get_input_element_type(i));
        inputs.push_back(binding.output);
        if (binding.param) {
            parameters.push_back(binding.param);
        }
    }

    auto node_clone = node->clone_with_new_inputs(inputs);
    auto result = std::make_shared<ov::op::v0::Result>(node_clone->output(0));
    ov::ResultVector results{result};
    return std::make_shared<ov::Model>(results, parameters);
}

std::shared_ptr<ov::Model> make_single_op_model_all_outputs(const std::shared_ptr<const ov::Node>& node) {
    OPENVINO_ASSERT(node, "make_single_op_model_all_outputs: node is null");
    ov::OutputVector inputs;
    inputs.reserve(node->get_input_size());
    ov::ParameterVector parameters;
    parameters.reserve(node->get_input_size());
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        auto src = node->input_value(i).get_node_shared_ptr();
        auto binding = bind_input_or_param(src,
                                           node->get_input_partial_shape(i),
                                           node->get_input_element_type(i));
        inputs.push_back(binding.output);
        if (binding.param) {
            parameters.push_back(binding.param);
        }
    }

    auto node_clone = node->clone_with_new_inputs(inputs);
    ov::ResultVector results;
    results.reserve(node->get_output_size());
    for (size_t i = 0; i < node_clone->get_output_size(); ++i) {
        results.push_back(std::make_shared<ov::op::v0::Result>(node_clone->output(i)));
    }
    return std::make_shared<ov::Model>(results, parameters);
}

}  // namespace gfx_plugin
}  // namespace ov
