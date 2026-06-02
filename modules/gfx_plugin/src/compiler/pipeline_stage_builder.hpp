// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/gpu_backend.hpp"
#include "openvino/core/model.hpp"
#include "runtime/backend_stage_factory.hpp"
#include "runtime/pipeline_stage_desc.hpp"

namespace ov {
namespace gfx_plugin {

class GfxProfilingTrace;
struct RuntimeExecutableDescriptor;

namespace compiler {

struct PipelineStageBuildRequest {
  std::shared_ptr<const ov::Model> runtime_model;
  const BackendStageFactory *stage_factory = nullptr;
  const RuntimeExecutableDescriptor *runtime_descriptor = nullptr;
  GpuBackend backend = GpuBackend::Unknown;
  std::string backend_name;
  bool enable_fusion = true;
  bool diagnostic_f32_vendor_image = false;
  GfxProfilingTrace *compile_trace = nullptr;
};

struct PipelineStageBuildResult {
  std::vector<PipelineStageDesc> pipeline;
  std::unordered_map<const ov::Node *, size_t> node_to_stage;
  std::unordered_map<const ov::Node *, size_t> param_index;
};

PipelineStageBuildResult
build_pipeline_stage_descriptors(const PipelineStageBuildRequest &request);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
