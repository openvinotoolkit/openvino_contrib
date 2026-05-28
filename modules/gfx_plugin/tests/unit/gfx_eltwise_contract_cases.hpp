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

struct EltwiseOpenClArtifactCase {
  std::string name;
  std::function<std::shared_ptr<ov::Node>()> make_node;
  std::string expected_source_id;
  std::string expected_entry_point;
  GfxOpenClBaselineOp expected_op = GfxOpenClBaselineOp::Add;
  uint32_t expected_arg_count = 0;
  uint32_t expected_direct_input_count = 0;
  std::vector<size_t> expected_direct_inputs;
  std::vector<GfxOpenClSourceScalarArg> expected_scalar_args;
  std::vector<uint32_t> expected_static_u32_scalars;
  GfxOpenClBaselineInputMode expected_input_mode =
      GfxOpenClBaselineInputMode::Direct;
};

std::vector<EltwiseOpenClArtifactCase> eltwise_opencl_artifact_cases();

} // namespace gfx_plugin
} // namespace ov
