// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/runtime/itensor.hpp"

namespace ov {
namespace gfx_plugin {

class CompiledModel;
struct InferRequestState;

void init_backend_infer_state(InferRequestState& state, const CompiledModel& cm);

ov::SoPtr<ov::ITensor> get_backend_tensor_override(const InferRequestState& state,
                                                   size_t idx,
                                                   const std::vector<ov::Output<const ov::Node>>& outputs);

}  // namespace gfx_plugin
}  // namespace ov
