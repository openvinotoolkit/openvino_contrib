// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/runtime_executable_descriptor_builder.hpp"

#include <sstream>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "compiler/pipeline_stage_runtime_descriptor_builder_detail.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

std::string join_diagnostics(const std::vector<std::string> &diagnostics) {
  std::ostringstream oss;
  for (size_t i = 0; i < diagnostics.size(); ++i) {
    if (i) {
      oss << "; ";
    }
    oss << diagnostics[i];
  }
  return oss.str();
}

} // namespace

RuntimeExecutableDescriptor RuntimeExecutableDescriptorBuilder::build_finalized(
    const RuntimeExecutableDescriptorBuildRequest &request) const {
  OPENVINO_ASSERT(request.executable,
                  "GFX: runtime executable descriptor build requires an "
                  "ExecutableBundle");
  OPENVINO_ASSERT(
      request.stage_graph_snapshot,
      "GFX: runtime executable descriptor build requires an explicit "
      "compiler-owned stage graph snapshot");
  OPENVINO_ASSERT(request.backend_registry,
                  "GFX: runtime executable descriptor build requires an "
                  "explicit BackendRegistry");
  OPENVINO_ASSERT(
      request.target.backend() != GpuBackend::Unknown,
      "GFX: runtime executable descriptor build requires concrete BackendTarget");

  auto descriptor_seed = build(*request.executable);
  detail::PipelineStageBuildRequest stage_request;
  stage_request.graph = *request.stage_graph_snapshot;
  stage_request.runtime_descriptor = &descriptor_seed;
  stage_request.backend_registry = request.backend_registry;
  stage_request.target = request.target;
  stage_request.backend_name =
      request.backend_name.empty() ? request.target.backend_id()
                                   : request.backend_name;
  stage_request.compile_trace = request.compile_trace;

  auto descriptor =
      detail::build_pipeline_stage_runtime_descriptor(stage_request);
  const auto descriptor_verification =
      verify_runtime_executable_descriptor(descriptor, *request.executable);
  OPENVINO_ASSERT(
      descriptor_verification.valid(),
      "GFX: compiler produced invalid runtime executable descriptor: ",
      join_diagnostics(descriptor_verification.diagnostics));
  const auto materialization_verification =
      verify_runtime_executable_descriptor_materialization(descriptor);
  OPENVINO_ASSERT(
      materialization_verification.valid(),
      "GFX: compiler produced invalid runtime descriptor materialization: ",
      join_diagnostics(materialization_verification.diagnostics));
  return descriptor;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
