// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/backend_module_provider.hpp"

#include "backends/metal/compiler/metal_backend_module.hpp"
#include "backends/opencl/compiler/opencl_backend_module.hpp"
#include "common/backend_config.hpp"
#include "common/gpu_device_profile.hpp"
#include "compiler/stage_compiler_policy.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

BackendRegistry::BackendRegistry()
    : BackendRegistry(make_default_backend_modules()) {}

const BackendRegistry &BackendRegistry::default_registry() {
  static const BackendRegistry registry;
  return registry;
}

StageCompilerPolicy resolve_stage_compiler_policy(const BackendTarget &target) {
  OPENVINO_ASSERT(target.backend() != GpuBackend::Unknown,
                  "GFX: stage compiler policy requires concrete BackendTarget");
  const auto backend_module =
      BackendRegistry::default_registry().resolve(target);
  OPENVINO_ASSERT(backend_module,
                  "GFX: stage compiler policy has no registered module for ",
                  target.debug_string());
  return make_stage_compiler_policy_from_capabilities(
      backend_module->capabilities());
}

std::vector<std::shared_ptr<const BackendModule>>
make_default_backend_modules() {
  std::vector<std::shared_ptr<const BackendModule>> modules;
  if (kGfxBackendMetalAvailable) {
    modules.push_back(make_metal_backend_module());
  }
  if (kGfxBackendOpenCLAvailable) {
    modules.push_back(make_opencl_backend_module());
    modules.push_back(make_opencl_backend_module(
        BackendTarget::from_backend_device_family(
            GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno)));
    modules.push_back(make_opencl_backend_module(
        BackendTarget::from_backend_device_family(
            GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D)));
  }
  return modules;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
