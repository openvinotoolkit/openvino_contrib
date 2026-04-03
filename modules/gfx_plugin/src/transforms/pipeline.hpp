// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/model.hpp"

namespace ov {
namespace gfx_plugin {
namespace transforms {

// Run GFX-specific transformation pipeline and return transformed clone.
std::shared_ptr<const ov::Model> run_pipeline(const std::shared_ptr<const ov::Model>& model);

}  // namespace transforms
}  // namespace gfx_plugin
}  // namespace ov
