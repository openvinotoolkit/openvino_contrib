// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "compiler/stage_placement.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

std::shared_ptr<const StagePlacementPolicy> make_metal_stage_placement_policy();

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
