// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/gfx_device_info.hpp"

#include <string>
#include <vector>

#include "compiler/backend_registry.hpp"
#include "plugin/gfx_property_utils.hpp"
#include "common/gfx_backend_utils.hpp"

namespace ov {
namespace gfx_plugin {
void fill_metal_device_info(GfxDeviceInfo &info, const ov::AnyMap &properties);
} // namespace gfx_plugin
} // namespace ov

namespace ov {
namespace gfx_plugin {
namespace {

std::vector<std::string> device_capabilities_from_backend_capabilities(
    const compiler::BackendCapabilities &capabilities) {
  std::vector<std::string> device_capabilities;
  const auto &precision = capabilities.precision();
  if (precision.supports_fp32) {
    device_capabilities.push_back(ov::device::capability::FP32);
  }
  if (precision.supports_fp16) {
    device_capabilities.push_back(ov::device::capability::FP16);
  }
  if (precision.supports_int8) {
    device_capabilities.push_back(ov::device::capability::INT8);
  }
  if (capabilities.artifact_formats().supports_compiled_model_export_import) {
    device_capabilities.push_back(ov::device::capability::EXPORT_IMPORT);
  }
  return device_capabilities;
}

std::vector<std::string> query_device_capabilities(
    const compiler::BackendTarget &target) {
  const auto backend_module =
      compiler::BackendRegistry::default_registry().resolve(target);
  if (!backend_module ||
      !backend_module->target().is_compatible_with_fingerprint(
          target.fingerprint())) {
    return {};
  }
  return device_capabilities_from_backend_capabilities(
      backend_module->capabilities());
}

void finalize_device_info(GfxDeviceInfo &info,
                          const compiler::BackendTarget &target) {
  if (info.backend_name.empty()) {
    info.backend_name = backend_to_string(info.backend);
  }
  if (info.device_name.empty()) {
    info.device_name = "GFX";
  }
  if (info.full_name.empty()) {
    if (info.device_name == "GFX") {
      info.full_name = "GFX";
    } else {
      info.full_name = "GFX (" + info.device_name + ")";
    }
  }
  if (info.available_devices.empty()) {
    info.available_devices = {info.device_id.empty() ? std::string{"0"}
                                                     : info.device_id};
  }
  if (info.capabilities.empty()) {
    info.capabilities = query_device_capabilities(target);
  }
}

} // namespace

GfxDeviceInfo query_device_info(GpuBackend backend,
                                const ov::AnyMap &properties) {
  return query_device_info(compiler::BackendTarget::from_backend(backend),
                           properties);
}

GfxDeviceInfo query_device_info(const compiler::BackendTarget &target,
                                const ov::AnyMap &properties) {
  GfxDeviceInfo info;
  info.backend = target.backend();
  info.backend_name = target.backend_id();
  info.device_type = ov::device::Type::INTEGRATED;
  const int device_id = parse_device_id(properties);
  info.device_id =
      device_id >= 0 ? std::to_string(device_id) : std::string{"0"};
  info.capabilities = query_device_capabilities(target);

  switch (target.backend()) {
  case GpuBackend::Metal: {
    fill_metal_device_info(info, properties);
    break;
  }
  case GpuBackend::OpenCL: {
    info.device_name = target.device_name();
    info.full_name = "GFX (" + target.device_profile() + ")";
    break;
  }
  default:
    info.device_name = "GFX";
    info.full_name = "GFX";
    break;
  }

  finalize_device_info(info, target);
  return info;
}

GfxDeviceInfo query_device_info_from_properties(const ov::AnyMap &properties,
                                                bool log_fallback,
                                                const char *log_tag) {
  const auto target =
      resolve_backend_target_from_properties(properties, log_fallback, log_tag);
  return query_device_info(target, properties);
}

} // namespace gfx_plugin
} // namespace ov
