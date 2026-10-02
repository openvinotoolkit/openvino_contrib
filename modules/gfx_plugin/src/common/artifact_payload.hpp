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

enum class RuntimeParamDescriptorPayloadKind {
  None,
  BinaryBroadcast,
  Broadcast,
  Select,
  Tile,
  Interpolate,
  Softmax,
  Transpose,
  Reduce,
};

inline std::string_view runtime_param_descriptor_payload_kind_to_string(
    RuntimeParamDescriptorPayloadKind kind) noexcept {
  switch (kind) {
  case RuntimeParamDescriptorPayloadKind::None:
    return "none";
  case RuntimeParamDescriptorPayloadKind::BinaryBroadcast:
    return "binary_broadcast";
  case RuntimeParamDescriptorPayloadKind::Broadcast:
    return "broadcast";
  case RuntimeParamDescriptorPayloadKind::Select:
    return "select";
  case RuntimeParamDescriptorPayloadKind::Tile:
    return "tile";
  case RuntimeParamDescriptorPayloadKind::Interpolate:
    return "interpolate";
  case RuntimeParamDescriptorPayloadKind::Softmax:
    return "softmax";
  case RuntimeParamDescriptorPayloadKind::Transpose:
    return "transpose";
  case RuntimeParamDescriptorPayloadKind::Reduce:
    return "reduce";
  }
  return "none";
}

inline RuntimeParamDescriptorPayloadKind
runtime_param_descriptor_payload_kind_from_string(
    std::string_view kind) noexcept {
  if (kind == "binary_broadcast") {
    return RuntimeParamDescriptorPayloadKind::BinaryBroadcast;
  }
  if (kind == "broadcast") {
    return RuntimeParamDescriptorPayloadKind::Broadcast;
  }
  if (kind == "select") {
    return RuntimeParamDescriptorPayloadKind::Select;
  }
  if (kind == "tile") {
    return RuntimeParamDescriptorPayloadKind::Tile;
  }
  if (kind == "interpolate") {
    return RuntimeParamDescriptorPayloadKind::Interpolate;
  }
  if (kind == "softmax") {
    return RuntimeParamDescriptorPayloadKind::Softmax;
  }
  if (kind == "transpose") {
    return RuntimeParamDescriptorPayloadKind::Transpose;
  }
  if (kind == "reduce") {
    return RuntimeParamDescriptorPayloadKind::Reduce;
  }
  return RuntimeParamDescriptorPayloadKind::None;
}

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
