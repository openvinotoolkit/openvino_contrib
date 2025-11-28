// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/model.hpp"

namespace ov {
namespace metal_plugin {
namespace transforms {

// Placeholder pipeline: currently returns the model unchanged.
std::shared_ptr<const ov::Model> run_pipeline(const std::shared_ptr<const ov::Model>& model);

}  // namespace transforms
}  // namespace metal_plugin
}  // namespace ov
