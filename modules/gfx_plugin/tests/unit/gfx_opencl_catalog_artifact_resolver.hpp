// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include "backends/opencl/compiler/opencl_kernel_unit_catalog.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {
namespace test {

inline std::optional<GfxOpenClSourceArtifact>
resolve_opencl_catalog_source_artifact(
    const std::shared_ptr<const ov::Node> &node,
    std::string_view expected_source_id = {}) {
  for (const auto &family : compiler::opencl_artifact_family_entries()) {
    if (!family.matches(node)) {
      continue;
    }
    return family.make_source_artifact(node, expected_source_id);
  }
  return std::nullopt;
}

} // namespace test
} // namespace gfx_plugin
} // namespace ov
