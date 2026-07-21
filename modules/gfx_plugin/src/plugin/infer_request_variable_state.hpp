// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {

class Model;

namespace gfx_plugin {

struct BackendResources;
struct InferRequestState;

void initialize_variable_states(InferRequestState& state,
                                const std::shared_ptr<const ov::Model>& model,
                                const BackendResources& resources);

std::vector<ov::SoPtr<ov::IVariableState>>
query_variable_states(const InferRequestState& state);

}  // namespace gfx_plugin
}  // namespace ov
