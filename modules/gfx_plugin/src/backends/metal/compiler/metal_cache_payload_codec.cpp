// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/metal_kernel_artifacts.hpp"

#include <exception>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

constexpr const char *kMetalVendorPayloadFormat =
    "gfx.metal.vendor_descriptor.v1";

void append_field(std::ostringstream &os, std::string_view value) {
  os << value.size() << ':' << value << ';';
}

void append_bool(std::ostringstream &os, bool value) {
  append_field(os, value ? "1" : "0");
}

void append_u32(std::ostringstream &os, uint32_t value) {
  append_field(os, std::to_string(value));
}

void append_u64(std::ostringstream &os, uint64_t value) {
  append_field(os, std::to_string(value));
}

void append_i64(std::ostringstream &os, int64_t value) {
  append_field(os, std::to_string(value));
}

void append_f32(std::ostringstream &os, float value) {
  std::ostringstream value_os;
  value_os << std::setprecision(std::numeric_limits<float>::max_digits10)
           << value;
  append_field(os, value_os.str());
}

class PayloadReader final {
public:
  explicit PayloadReader(std::string_view wire) : m_wire(wire) {}

  bool ok() const noexcept { return m_ok; }

  std::string field() {
    if (m_pos >= m_wire.size()) {
      m_ok = false;
      return {};
    }
    const auto colon = m_wire.find(':', m_pos);
    if (colon == std::string_view::npos) {
      m_ok = false;
      m_pos = m_wire.size();
      return {};
    }
    const auto length_text = m_wire.substr(m_pos, colon - m_pos);
    size_t length = 0;
    try {
      length = static_cast<size_t>(std::stoull(std::string(length_text)));
    } catch (const std::exception &) {
      m_ok = false;
      m_pos = m_wire.size();
      return {};
    }
    const auto value_begin = colon + 1;
    const auto value_end = value_begin + length;
    if (value_end >= m_wire.size() || m_wire[value_end] != ';') {
      m_ok = false;
      m_pos = m_wire.size();
      return {};
    }
    m_pos = value_end + 1;
    return std::string(m_wire.substr(value_begin, length));
  }

  bool boolean() { return field() == "1"; }

  uint32_t u32() {
    try {
      return static_cast<uint32_t>(std::stoul(field()));
    } catch (const std::exception &) {
      m_ok = false;
      return 0;
    }
  }

  uint64_t u64() {
    try {
      return static_cast<uint64_t>(std::stoull(field()));
    } catch (const std::exception &) {
      m_ok = false;
      return 0;
    }
  }

  int64_t i64() {
    try {
      return static_cast<int64_t>(std::stoll(field()));
    } catch (const std::exception &) {
      m_ok = false;
      return 0;
    }
  }

  float f32() {
    try {
      return std::stof(field());
    } catch (const std::exception &) {
      m_ok = false;
      return 0.0f;
    }
  }

private:
  std::string_view m_wire;
  size_t m_pos = 0;
  bool m_ok = true;
};

uint32_t
vendor_abi_arg_count(const GfxAppleMpsVendorPrimitiveContract &contract) {
  if (contract.external_buffer_abi.valid &&
      contract.external_buffer_abi.has_buffer_count) {
    return contract.external_buffer_abi.buffer_count;
  }
  return static_cast<uint32_t>(contract.input_descs.size() +
                               contract.output_descs.size());
}

uint32_t vendor_abi_output_arg_count(
    const GfxAppleMpsVendorPrimitiveContract &contract) {
  if (contract.external_buffer_abi.valid &&
      contract.external_buffer_abi.has_output_buffer_count) {
    return contract.external_buffer_abi.output_buffer_count;
  }
  return static_cast<uint32_t>(contract.output_descs.size());
}

void append_tensor_desc(std::ostringstream &os,
                        const GfxMpsrtTensorDesc &desc) {
  append_u32(os, desc.rank);
  for (const auto value : desc.dims) {
    append_u64(os, value);
  }
  for (const auto value : desc.strides) {
    append_i64(os, value);
  }
  append_u32(os, static_cast<uint32_t>(desc.dtype));
  append_u32(os, static_cast<uint32_t>(desc.storage));
  append_u32(os, static_cast<uint32_t>(desc.layout));
  append_u32(os, desc.flags);
  append_u64(os, desc.byte_offset);
  append_u64(os, desc.byte_length);
  append_u32(os, desc.image_width);
  append_u32(os, desc.image_height);
  append_u32(os, desc.image_feature_channels);
  append_u32(os, desc.image_batch);
  append_u32(os, desc.matrix_rows);
  append_u32(os, desc.matrix_columns);
  append_u32(os, desc.matrix_row_bytes);
  append_u32(os, desc.matrix_count);
  append_u32(os, desc.alias_of);
}

GfxMpsrtTensorDesc read_tensor_desc(PayloadReader &reader) {
  GfxMpsrtTensorDesc desc;
  desc.rank = reader.u32();
  for (auto &value : desc.dims) {
    value = reader.u64();
  }
  for (auto &value : desc.strides) {
    value = reader.i64();
  }
  desc.dtype = static_cast<GfxMpsrtDType>(reader.u32());
  desc.storage = static_cast<GfxMpsrtStorage>(reader.u32());
  desc.layout = static_cast<GfxMpsrtLayout>(reader.u32());
  desc.flags = reader.u32();
  desc.byte_offset = reader.u64();
  desc.byte_length = reader.u64();
  desc.image_width = reader.u32();
  desc.image_height = reader.u32();
  desc.image_feature_channels = reader.u32();
  desc.image_batch = reader.u32();
  desc.matrix_rows = reader.u32();
  desc.matrix_columns = reader.u32();
  desc.matrix_row_bytes = reader.u32();
  desc.matrix_count = reader.u32();
  desc.alias_of = reader.u32();
  return desc;
}

void append_gemm_desc(std::ostringstream &os, const GfxMpsrtGemmAbiDesc &desc) {
  append_u32(os, desc.transpose_lhs);
  append_u32(os, desc.transpose_rhs);
  append_u32(os, desc.accumulate_fp32);
  append_f32(os, desc.alpha);
  append_f32(os, desc.beta);
}

GfxMpsrtGemmAbiDesc read_gemm_desc(PayloadReader &reader) {
  GfxMpsrtGemmAbiDesc desc;
  desc.transpose_lhs = reader.u32();
  desc.transpose_rhs = reader.u32();
  desc.accumulate_fp32 = reader.u32();
  desc.alpha = reader.f32();
  desc.beta = reader.f32();
  return desc;
}

void append_conv2d_desc(std::ostringstream &os,
                        const GfxMpsrtConv2DAbiDesc &desc) {
  append_u32(os, desc.groups);
  append_u32(os, desc.strides[0]);
  append_u32(os, desc.strides[1]);
  append_u32(os, desc.dilations[0]);
  append_u32(os, desc.dilations[1]);
  for (const auto value : desc.pads) {
    append_u32(os, value);
  }
  append_u32(os, desc.fused_activation);
  append_u32(os, desc.accumulate_fp32);
}

GfxMpsrtConv2DAbiDesc read_conv2d_desc(PayloadReader &reader) {
  GfxMpsrtConv2DAbiDesc desc;
  desc.groups = reader.u32();
  desc.strides[0] = reader.u32();
  desc.strides[1] = reader.u32();
  desc.dilations[0] = reader.u32();
  desc.dilations[1] = reader.u32();
  for (auto &value : desc.pads) {
    value = reader.u32();
  }
  desc.fused_activation = reader.u32();
  desc.accumulate_fp32 = reader.u32();
  return desc;
}

void append_pool2d_desc(std::ostringstream &os,
                        const GfxMpsrtPool2DAbiDesc &desc) {
  append_u32(os, desc.is_avg);
  append_u32(os, desc.kernel[0]);
  append_u32(os, desc.kernel[1]);
  append_u32(os, desc.strides[0]);
  append_u32(os, desc.strides[1]);
  append_u32(os, desc.dilations[0]);
  append_u32(os, desc.dilations[1]);
  for (const auto value : desc.pads) {
    append_u32(os, value);
  }
  append_u32(os, desc.exclude_pad);
}

GfxMpsrtPool2DAbiDesc read_pool2d_desc(PayloadReader &reader) {
  GfxMpsrtPool2DAbiDesc desc;
  desc.is_avg = reader.u32();
  desc.kernel[0] = reader.u32();
  desc.kernel[1] = reader.u32();
  desc.strides[0] = reader.u32();
  desc.strides[1] = reader.u32();
  desc.dilations[0] = reader.u32();
  desc.dilations[1] = reader.u32();
  for (auto &value : desc.pads) {
    value = reader.u32();
  }
  desc.exclude_pad = reader.u32();
  return desc;
}

void append_resize2d_desc(std::ostringstream &os,
                          const GfxMpsrtResize2DAbiDesc &desc) {
  append_u32(os, desc.nearest);
  append_u32(os, desc.align_corners);
  append_u32(os, desc.half_pixel_centers);
}

GfxMpsrtResize2DAbiDesc read_resize2d_desc(PayloadReader &reader) {
  GfxMpsrtResize2DAbiDesc desc;
  desc.nearest = reader.u32();
  desc.align_corners = reader.u32();
  desc.half_pixel_centers = reader.u32();
  return desc;
}

void append_softmax_desc(std::ostringstream &os,
                         const GfxMpsrtSoftmaxAbiDesc &desc) {
  append_u32(os, desc.axis);
  append_u32(os, desc.log_softmax);
}

GfxMpsrtSoftmaxAbiDesc read_softmax_desc(PayloadReader &reader) {
  GfxMpsrtSoftmaxAbiDesc desc;
  desc.axis = reader.u32();
  desc.log_softmax = reader.u32();
  return desc;
}

void append_topk_desc(std::ostringstream &os, const GfxMpsrtTopKAbiDesc &desc) {
  append_u32(os, desc.axis);
  append_u32(os, desc.k);
  append_u32(os, desc.mode_max);
  append_u32(os, desc.sort_type);
}

GfxMpsrtTopKAbiDesc read_topk_desc(PayloadReader &reader) {
  GfxMpsrtTopKAbiDesc desc;
  desc.axis = reader.u32();
  desc.k = reader.u32();
  desc.mode_max = reader.u32();
  desc.sort_type = reader.u32();
  return desc;
}

void append_sdpa_desc(std::ostringstream &os, const GfxMpsrtSdpaAbiDesc &desc) {
  append_u32(os, desc.has_mask);
  append_u32(os, desc.causal);
  append_u32(os, desc.accumulate_fp32);
  append_u32(os, desc.layout);
  append_f32(os, desc.scale);
}

GfxMpsrtSdpaAbiDesc read_sdpa_desc(PayloadReader &reader) {
  GfxMpsrtSdpaAbiDesc desc;
  desc.has_mask = reader.u32();
  desc.causal = reader.u32();
  desc.accumulate_fp32 = reader.u32();
  desc.layout = reader.u32();
  desc.scale = reader.f32();
  return desc;
}

void append_descriptor(std::ostringstream &os,
                       const GfxAppleMpsVendorPrimitiveDescriptor &descriptor) {
  append_u32(os, static_cast<uint32_t>(descriptor.kind));
  append_gemm_desc(os, descriptor.gemm);
  append_conv2d_desc(os, descriptor.conv2d);
  append_pool2d_desc(os, descriptor.pool2d);
  append_resize2d_desc(os, descriptor.resize2d);
  append_softmax_desc(os, descriptor.softmax);
  append_topk_desc(os, descriptor.topk);
  append_sdpa_desc(os, descriptor.sdpa);
}

GfxAppleMpsVendorPrimitiveDescriptor read_descriptor(PayloadReader &reader) {
  GfxAppleMpsVendorPrimitiveDescriptor descriptor;
  descriptor.kind = static_cast<GfxAppleMpsVendorPrimitiveKind>(reader.u32());
  descriptor.gemm = read_gemm_desc(reader);
  descriptor.conv2d = read_conv2d_desc(reader);
  descriptor.pool2d = read_pool2d_desc(reader);
  descriptor.resize2d = read_resize2d_desc(reader);
  descriptor.softmax = read_softmax_desc(reader);
  descriptor.topk = read_topk_desc(reader);
  descriptor.sdpa = read_sdpa_desc(reader);
  return descriptor;
}

const char *external_buffer_role_name(GfxMpsrtExternalBufferRole role) {
  switch (role) {
  case GfxMpsrtExternalBufferRole::TensorInput:
    return "tensor_input";
  case GfxMpsrtExternalBufferRole::TensorOutput:
    return "tensor_output";
  case GfxMpsrtExternalBufferRole::ConstBuffer:
    return "const_buffer";
  case GfxMpsrtExternalBufferRole::RuntimeParams:
    return "runtime_params";
  case GfxMpsrtExternalBufferRole::Metadata:
    return "metadata";
  case GfxMpsrtExternalBufferRole::Unknown:
  default:
    return "unknown";
  }
}

GfxMpsrtExternalBufferRole
external_buffer_role_from_name(std::string_view name) {
  if (name == "tensor_input") {
    return GfxMpsrtExternalBufferRole::TensorInput;
  }
  if (name == "tensor_output") {
    return GfxMpsrtExternalBufferRole::TensorOutput;
  }
  if (name == "const_buffer") {
    return GfxMpsrtExternalBufferRole::ConstBuffer;
  }
  if (name == "runtime_params") {
    return GfxMpsrtExternalBufferRole::RuntimeParams;
  }
  if (name == "metadata") {
    return GfxMpsrtExternalBufferRole::Metadata;
  }
  return GfxMpsrtExternalBufferRole::Unknown;
}

void append_external_buffer_abi(std::ostringstream &os,
                                const GfxMpsrtExternalBufferAbiPlan &abi) {
  append_bool(os, abi.valid);
  append_bool(os, abi.has_buffer_count);
  append_bool(os, abi.has_output_buffer_count);
  append_bool(os, abi.has_buffer_roles);
  append_u32(os, abi.buffer_count);
  append_u32(os, abi.output_buffer_count);
  append_field(os, std::to_string(abi.buffer_roles.size()));
  for (const auto role : abi.buffer_roles) {
    append_field(os, external_buffer_role_name(role));
  }
}

GfxMpsrtExternalBufferAbiPlan read_external_buffer_abi(PayloadReader &reader) {
  GfxMpsrtExternalBufferAbiPlan abi;
  abi.valid = reader.boolean();
  abi.has_buffer_count = reader.boolean();
  abi.has_output_buffer_count = reader.boolean();
  abi.has_buffer_roles = reader.boolean();
  abi.buffer_count = reader.u32();
  abi.output_buffer_count = reader.u32();
  const auto count = reader.u32();
  abi.buffer_roles.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    abi.buffer_roles.push_back(external_buffer_role_from_name(reader.field()));
  }
  return abi;
}

void append_contract(std::ostringstream &os,
                     const GfxAppleMpsVendorPrimitiveContract &contract) {
  append_field(os, "GFX_METAL_VENDOR_DESCRIPTOR_PAYLOAD");
  append_field(os, "1");
  append_bool(os, contract.valid);
  append_descriptor(os, contract.descriptor);
  append_field(os, std::to_string(contract.semantic_input_roles.size()));
  for (const auto role : contract.semantic_input_roles) {
    append_field(os, kernel_buffer_role_descriptor_name(role));
  }
  append_external_buffer_abi(os, contract.external_buffer_abi);
  append_field(os, std::to_string(contract.input_descs.size()));
  for (const auto &desc : contract.input_descs) {
    append_tensor_desc(os, desc);
  }
  append_field(os, std::to_string(contract.output_descs.size()));
  for (const auto &desc : contract.output_descs) {
    append_tensor_desc(os, desc);
  }
}

GfxAppleMpsVendorPrimitiveContract read_contract(PayloadReader &reader) {
  GfxAppleMpsVendorPrimitiveContract contract;
  if (reader.field() != "GFX_METAL_VENDOR_DESCRIPTOR_PAYLOAD" ||
      reader.field() != "1") {
    return {};
  }
  contract.valid = reader.boolean();
  contract.descriptor = read_descriptor(reader);
  const auto semantic_role_count = reader.u32();
  contract.semantic_input_roles.reserve(semantic_role_count);
  for (uint32_t i = 0; i < semantic_role_count; ++i) {
    contract.semantic_input_roles.push_back(
        kernel_buffer_role_from_descriptor_name(reader.field()));
  }
  contract.external_buffer_abi = read_external_buffer_abi(reader);
  const auto input_count = reader.u32();
  contract.input_descs.reserve(input_count);
  for (uint32_t i = 0; i < input_count; ++i) {
    contract.input_descs.push_back(read_tensor_desc(reader));
  }
  const auto output_count = reader.u32();
  contract.output_descs.reserve(output_count);
  for (uint32_t i = 0; i < output_count; ++i) {
    contract.output_descs.push_back(read_tensor_desc(reader));
  }
  return contract;
}

bool descriptor_accepts_metal_vendor_contract(
    const KernelArtifactDescriptor &descriptor,
    const GfxAppleMpsVendorPrimitiveContract &contract) {
  return descriptor.kernel.backend_domain == "metal" &&
         descriptor.payload_kind ==
             KernelArtifactPayloadKind::VendorDescriptor &&
         descriptor.kernel.origin == KernelArtifactOrigin::VendorPrimitive &&
         !descriptor.kernel.kernel_id.empty() &&
         !descriptor.entry_point.empty() && contract.valid &&
         contract.descriptor.kind != GfxAppleMpsVendorPrimitiveKind::None &&
         descriptor.abi_arg_count == vendor_abi_arg_count(contract) &&
         descriptor.abi_output_arg_count ==
             vendor_abi_output_arg_count(contract);
}

} // namespace

CacheBackendPayloadEncoder make_metal_cache_payload_encoder() {
  return [](const KernelArtifactDescriptor &descriptor,
            const KernelArtifactPayloadRecord &payload_record) {
    CacheBackendPayloadRecord record;
    if (!payload_record.payload ||
        payload_record.payload->payload_kind() !=
            KernelArtifactPayloadKind::VendorDescriptor) {
      return record;
    }
    const auto *vendor_payload =
        dynamic_cast<const GfxMetalVendorPrimitiveArtifactPayload *>(
            payload_record.payload.get());
    if (!vendor_payload || !descriptor_accepts_metal_vendor_contract(
                               descriptor, vendor_payload->contract())) {
      return record;
    }
    std::ostringstream data;
    append_contract(data, vendor_payload->contract());
    record.payload_format = kMetalVendorPayloadFormat;
    record.payload_data = data.str();
    return record;
  };
}

CacheBackendPayloadDecoder make_metal_cache_payload_decoder() {
  return [](const CacheBackendPayloadRecord &payload,
            const KernelArtifactDescriptor &descriptor)
             -> std::shared_ptr<const KernelArtifactPayload> {
    if (payload.payload_format != kMetalVendorPayloadFormat ||
        payload.payload_kind !=
            kernel_artifact_payload_kind_to_string(
                KernelArtifactPayloadKind::VendorDescriptor) ||
        payload.backend_domain != "metal") {
      return {};
    }
    PayloadReader reader(payload.payload_data);
    auto contract = read_contract(reader);
    if (!reader.ok() ||
        !descriptor_accepts_metal_vendor_contract(descriptor, contract)) {
      return {};
    }
    return std::make_shared<GfxMetalVendorPrimitiveArtifactPayload>(
        descriptor.kernel.kernel_id, descriptor.kernel.backend_domain,
        descriptor.entry_point, std::move(contract));
  };
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
