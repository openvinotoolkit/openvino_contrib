// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "kernel_ir/gfx_kernel_manifest.hpp"

namespace ov {
namespace gfx_plugin {

enum class KernelArtifactOrigin {
  Unknown,
  Common,
  Metadata,
  VendorPrimitive,
  Generated,
  HandwrittenException,
};

enum class KernelArtifactPayloadKind {
  None,
  VendorDescriptor,
  MslSource,
  OpenClSource,
};

inline std::string_view
kernel_buffer_role_descriptor_name(GfxKernelBufferRole role) noexcept {
  switch (role) {
  case GfxKernelBufferRole::TensorInput:
    return "tensor_input";
  case GfxKernelBufferRole::TensorOutput:
    return "tensor_output";
  case GfxKernelBufferRole::RuntimeParams:
    return "runtime_params";
  case GfxKernelBufferRole::ConstTensor:
    return "const_tensor";
  case GfxKernelBufferRole::ScalarParam:
    return "scalar_param";
  case GfxKernelBufferRole::Unknown:
  default:
    return "unknown";
  }
}

inline GfxKernelBufferRole
kernel_buffer_role_from_descriptor_name(std::string_view role) noexcept {
  if (role == "tensor_input") {
    return GfxKernelBufferRole::TensorInput;
  }
  if (role == "tensor_output") {
    return GfxKernelBufferRole::TensorOutput;
  }
  if (role == "runtime_params") {
    return GfxKernelBufferRole::RuntimeParams;
  }
  if (role == "const_tensor") {
    return GfxKernelBufferRole::ConstTensor;
  }
  if (role == "scalar_param") {
    return GfxKernelBufferRole::ScalarParam;
  }
  return GfxKernelBufferRole::Unknown;
}

struct KernelArtifactConstTensor {
  size_t source_input_index = 0;
  std::string logical_name;
  std::string element_type;
  std::vector<size_t> shape;
  std::vector<uint8_t> bytes;
};

struct KernelLaunchPlanDescriptor {
  bool valid = false;
  std::vector<std::string> buffer_roles;
  std::vector<size_t> direct_input_indices;
  std::vector<size_t> input_indices;
  size_t input_arg_count = 0;
  std::vector<int32_t> operand_kinds;
  std::vector<int32_t> operand_arg_indices;
  std::vector<int32_t> scalar_args;
  std::vector<uint32_t> scalar_arg_kinds;
};

class KernelArtifactPayload {
public:
  virtual ~KernelArtifactPayload() = default;

  virtual KernelArtifactPayloadKind payload_kind() const noexcept = 0;
  virtual std::string_view backend_domain() const noexcept = 0;
  virtual std::string_view source_id() const noexcept = 0;
  virtual std::string_view entry_point() const noexcept = 0;
  virtual bool valid() const noexcept = 0;
};

} // namespace gfx_plugin
} // namespace ov
