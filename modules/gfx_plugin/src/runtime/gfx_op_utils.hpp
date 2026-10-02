// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {

// Build a minimal ov::Model containing the provided node and its parameter inputs.
std::shared_ptr<ov::Model> make_single_op_model(const std::shared_ptr<const ov::Node>& node);
// Build a minimal ov::Model containing the provided node with all of its outputs.
std::shared_ptr<ov::Model> make_single_op_model_all_outputs(const std::shared_ptr<const ov::Node>& node);

}  // namespace gfx_plugin
}  // namespace ov
