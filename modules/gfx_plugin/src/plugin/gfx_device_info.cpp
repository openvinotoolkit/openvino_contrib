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

std::vector<std::string> query_device_capabilities(GpuBackend backend) {
  const auto backend_module =
      compiler::BackendRegistry::default_registry().resolve(backend);
  if (!backend_module) {
    return {};
  }
  return device_capabilities_from_backend_capabilities(
      backend_module->capabilities());
}

void finalize_device_info(GfxDeviceInfo &info) {
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
    info.capabilities = query_device_capabilities(info.backend);
  }
}

} // namespace

GfxDeviceInfo query_device_info(GpuBackend backend,
                                const ov::AnyMap &properties) {
  GfxDeviceInfo info;
  info.backend = backend;
  info.backend_name = backend_to_string(backend);
  info.device_type = ov::device::Type::INTEGRATED;
  const int device_id = parse_device_id(properties);
  info.device_id =
      device_id >= 0 ? std::to_string(device_id) : std::string{"0"};
  info.capabilities = query_device_capabilities(backend);

  switch (backend) {
  case GpuBackend::Metal: {
    fill_metal_device_info(info, properties);
    break;
  }
  case GpuBackend::OpenCL: {
    info.device_name = "OpenCL";
    info.full_name = "GFX (OpenCL source kernels)";
    break;
  }
  default:
    info.device_name = "GFX";
    info.full_name = "GFX";
    break;
  }

  finalize_device_info(info);
  return info;
}

GfxDeviceInfo query_device_info_from_properties(const ov::AnyMap &properties,
                                                bool log_fallback,
                                                const char *log_tag) {
  const auto backend =
      resolve_backend_kind_from_properties(properties, log_fallback, log_tag);
  return query_device_info(backend, properties);
}

} // namespace gfx_plugin
} // namespace ov
