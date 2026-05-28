// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {

struct ActivationOpenClArtifactCase {
  std::string name;
  std::function<std::shared_ptr<ov::Node>()> make_node;
  std::string expected_source_id;
  std::string expected_entry_point;
  GfxOpenClBaselineOp expected_op = GfxOpenClBaselineOp::Relu;
  std::vector<float> expected_static_f32_scalars{0.0f, 0.0f};
};

struct ActivationMslArtifactCase {
  std::string name;
  std::function<std::shared_ptr<ov::Node>()> make_node;
  std::string expected_source_snippet;
};

std::vector<ActivationOpenClArtifactCase> activation_opencl_artifact_cases();
std::vector<ActivationMslArtifactCase> activation_msl_artifact_cases();

} // namespace gfx_plugin
} // namespace ov
