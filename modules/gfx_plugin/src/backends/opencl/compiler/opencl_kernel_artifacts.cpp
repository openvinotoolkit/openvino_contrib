// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "backends/opencl/compiler/opencl_kernel_unit_catalog.hpp"
#include "common/runtime_param_descriptor.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

constexpr const char *kOpenClSourceArtifactPayloadFormat =
    "gfx.opencl.source_artifact.v1";

bool starts_with(std::string_view value, std::string_view prefix) noexcept {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

void append_field(std::ostringstream &os, std::string_view value) {
  os << value.size() << ":" << value << ";";
}

void append_bool(std::ostringstream &os, bool value) {
  append_field(os, value ? "1" : "0");
}

void append_u32(std::ostringstream &os, uint32_t value) {
  append_field(os, std::to_string(value));
}

void append_size(std::ostringstream &os, size_t value) {
  append_field(os, std::to_string(value));
}

void append_float(std::ostringstream &os, float value) {
  std::ostringstream value_os;
  value_os << std::setprecision(std::numeric_limits<float>::max_digits10)
           << value;
  append_field(os, value_os.str());
}

void append_string_vector(std::ostringstream &os,
                          const std::vector<std::string> &values) {
  append_size(os, values.size());
  for (const auto &value : values) {
    append_field(os, value);
  }
}

void append_size_vector(std::ostringstream &os,
                        const std::vector<size_t> &values) {
  append_size(os, values.size());
  for (const auto value : values) {
    append_size(os, value);
  }
}

void append_u32_vector(std::ostringstream &os,
                       const std::vector<uint32_t> &values) {
  append_size(os, values.size());
  for (const auto value : values) {
    append_u32(os, value);
  }
}

void append_float_vector(std::ostringstream &os,
                         const std::vector<float> &values) {
  append_size(os, values.size());
  for (const auto value : values) {
    append_float(os, value);
  }
}

void append_scalar_arg_vector(
    std::ostringstream &os,
    const std::vector<GfxOpenClSourceScalarArg> &values) {
  append_size(os, values.size());
  for (const auto value : values) {
    append_u32(os, static_cast<uint32_t>(value));
  }
}

class PayloadReader final {
public:
  explicit PayloadReader(std::string_view wire) : m_wire(wire) {}

  bool ok() const noexcept { return m_diagnostics.empty(); }

  std::string field(std::string_view name) {
    if (m_pos >= m_wire.size()) {
      m_diagnostics.push_back(std::string("OpenCL cache payload ended before ") +
                              std::string(name));
      return {};
    }
    const size_t colon = m_wire.find(':', m_pos);
    if (colon == std::string_view::npos) {
      m_diagnostics.push_back(std::string("OpenCL cache payload field ") +
                              std::string(name) +
                              " has no length separator");
      m_pos = m_wire.size();
      return {};
    }
    size_t size = 0;
    try {
      size = static_cast<size_t>(
          std::stoull(std::string(m_wire.substr(m_pos, colon - m_pos))));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("OpenCL cache payload field ") +
                              std::string(name) + " has invalid length");
      m_pos = m_wire.size();
      return {};
    }
    const size_t begin = colon + 1;
    const size_t end = begin + size;
    if (end >= m_wire.size() || m_wire[end] != ';') {
      m_diagnostics.push_back(std::string("OpenCL cache payload field ") +
                              std::string(name) + " is truncated");
      m_pos = m_wire.size();
      return {};
    }
    m_pos = end + 1;
    return std::string(m_wire.substr(begin, size));
  }

  bool boolean(std::string_view name) { return field(name) == "1"; }

  uint32_t u32(std::string_view name) {
    try {
      return static_cast<uint32_t>(std::stoul(field(name)));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("OpenCL cache payload field ") +
                              std::string(name) + " is not uint32");
      return 0;
    }
  }

  size_t size(std::string_view name) {
    try {
      return static_cast<size_t>(std::stoull(field(name)));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("OpenCL cache payload field ") +
                              std::string(name) + " is not size");
      return 0;
    }
  }

  float f32(std::string_view name) {
    try {
      return std::stof(field(name));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("OpenCL cache payload field ") +
                              std::string(name) + " is not float");
      return 0.0f;
    }
  }

  std::vector<std::string> string_vector(std::string_view name) {
    const auto count = size(name);
    std::vector<std::string> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(field(name));
    }
    return values;
  }

  std::vector<uint32_t> u32_vector(std::string_view name) {
    const auto count = size(name);
    std::vector<uint32_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(u32(name));
    }
    return values;
  }

  std::vector<size_t> size_vector(std::string_view name) {
    const auto count = size(name);
    std::vector<size_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(size(name));
    }
    return values;
  }

  std::vector<float> float_vector(std::string_view name) {
    const auto count = size(name);
    std::vector<float> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(f32(name));
    }
    return values;
  }

  std::vector<GfxOpenClSourceScalarArg> scalar_args(std::string_view name) {
    const auto raw = u32_vector(name);
    std::vector<GfxOpenClSourceScalarArg> values;
    values.reserve(raw.size());
    for (const auto value : raw) {
      values.push_back(static_cast<GfxOpenClSourceScalarArg>(value));
    }
    return values;
  }

private:
  std::string_view m_wire;
  size_t m_pos = 0;
  std::vector<std::string> m_diagnostics;
};

std::string serialize_opencl_source_artifact(
    const GfxOpenClSourceArtifact &artifact);

void append_chunk_vector(std::ostringstream &os,
                         const std::vector<GfxOpenClSourceChunkArtifact> &chunks) {
  append_size(os, chunks.size());
  for (const auto &chunk : chunks) {
    append_u32(os, chunk.binding_begin);
    append_u32(os, chunk.binding_count);
    append_u32(os, static_cast<uint32_t>(chunk.binding_role));
    append_u32(os, chunk.element_count_multiplier);
    append_u32(os, chunk.element_count_divisor);
    append_field(os, chunk.artifact ? serialize_opencl_source_artifact(
                                          *chunk.artifact)
                                    : std::string());
  }
}

std::string serialize_opencl_source_artifact(
    const GfxOpenClSourceArtifact &artifact) {
  std::ostringstream os;
  append_bool(os, artifact.valid);
  append_field(os, artifact.artifact_ref.source_id);
  append_field(os, artifact.artifact_ref.entry_point);
  append_field(os, artifact.source);
  append_string_vector(os, artifact.build_options);
  append_scalar_arg_vector(os, artifact.scalar_args);
  append_u32_vector(os, artifact.static_u32_scalars);
  append_float_vector(os, artifact.static_f32_scalars);
  append_u32_vector(os, artifact.source_static_u32_scalars);
  append_size_vector(os, artifact.direct_input_indices);
  append_u32(os, artifact.arg_count);
  append_u32(os, artifact.local_size_hint);
  append_u32(os, artifact.direct_input_count);
  append_u32(os, artifact.direct_output_count);
  append_u32(os, artifact.input_chunk_size);
  append_u32(os, artifact.output_chunk_size);
  append_u32(os, static_cast<uint32_t>(artifact.element_count_source));
  append_u32(os, static_cast<uint32_t>(artifact.op));
  append_u32(os, static_cast<uint32_t>(artifact.input_mode));
  append_float(os, artifact.scalar_constant_f32);
  append_chunk_vector(os, artifact.planned_chunks);
  return os.str();
}

std::vector<GfxOpenClSourceChunkArtifact> read_chunk_vector(
    PayloadReader &reader);

std::optional<GfxOpenClSourceArtifact> decode_opencl_source_artifact(
    const CacheBackendPayloadRecord &payload,
    const KernelArtifactDescriptor &descriptor,
    std::string_view payload_data) {
  PayloadReader reader(payload_data);
  GfxOpenClSourceArtifact artifact{};
  artifact.valid = reader.boolean("artifact valid");
  artifact.artifact_ref.valid = true;
  artifact.artifact_ref.kind = GfxKernelArtifactKind::OpenClSource;
  artifact.artifact_ref.backend_domain = GfxKernelBackendDomain::OpenCl;
  artifact.artifact_ref.source_id = reader.field("artifact source id");
  artifact.artifact_ref.entry_point = reader.field("artifact entry point");
  artifact.source = reader.field("artifact source");
  if (artifact.artifact_ref.source_id.empty()) {
    artifact.artifact_ref.source_id = payload.source_id;
  }
  if (artifact.artifact_ref.entry_point.empty()) {
    artifact.artifact_ref.entry_point = payload.entry_point;
  }
  if (artifact.source.empty()) {
    artifact.source = payload.source;
  }
  artifact.build_options = reader.string_vector("artifact build options");
  if (artifact.build_options.empty() && !descriptor.compile_options_key.empty()) {
    artifact.build_options = {descriptor.compile_options_key};
  }
  artifact.artifact_ref.build_options = artifact.build_options;
  artifact.scalar_args = reader.scalar_args("artifact scalar args");
  artifact.static_u32_scalars = reader.u32_vector("artifact static u32 scalars");
  artifact.static_f32_scalars =
      reader.float_vector("artifact static f32 scalars");
  artifact.source_static_u32_scalars =
      reader.u32_vector("artifact source static u32 scalars");
  artifact.direct_input_indices =
      reader.size_vector("artifact direct input indices");
  artifact.arg_count = reader.u32("artifact arg count");
  artifact.local_size_hint = reader.u32("artifact local size hint");
  artifact.direct_input_count = reader.u32("artifact direct input count");
  artifact.direct_output_count = reader.u32("artifact direct output count");
  artifact.input_chunk_size = reader.u32("artifact input chunk size");
  artifact.output_chunk_size = reader.u32("artifact output chunk size");
  artifact.element_count_source =
      static_cast<GfxOpenClSourceElementCountSource>(
          reader.u32("artifact element count source"));
  artifact.op =
      static_cast<GfxOpenClArtifactOp>(reader.u32("artifact op"));
  artifact.input_mode = static_cast<GfxOpenClArtifactInputMode>(
      reader.u32("artifact input mode"));
  artifact.scalar_constant_f32 = reader.f32("artifact scalar constant f32");
  artifact.planned_chunks = read_chunk_vector(reader);
  if (!reader.ok() || !artifact.valid || artifact.source.empty()) {
    return std::nullopt;
  }
  return artifact;
}

std::vector<GfxOpenClSourceChunkArtifact> read_chunk_vector(
    PayloadReader &reader) {
  const auto count = reader.size("artifact planned chunk count");
  std::vector<GfxOpenClSourceChunkArtifact> chunks;
  chunks.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    GfxOpenClSourceChunkArtifact chunk;
    chunk.binding_begin = reader.u32("chunk binding begin");
    chunk.binding_count = reader.u32("chunk binding count");
    chunk.binding_role = static_cast<GfxOpenClSourceChunkBindingRole>(
        reader.u32("chunk binding role"));
    chunk.element_count_multiplier = reader.u32("chunk element multiplier");
    chunk.element_count_divisor = reader.u32("chunk element divisor");
    const auto nested = reader.field("chunk nested artifact");
    if (!nested.empty()) {
      CacheBackendPayloadRecord nested_payload;
      nested_payload.source_id = "opencl/cache_chunk";
      nested_payload.entry_point = "opencl_cache_chunk";
      auto decoded =
          decode_opencl_source_artifact(nested_payload, KernelArtifactDescriptor{},
                                        nested);
      if (decoded) {
        chunk.artifact =
            std::make_shared<GfxOpenClSourceArtifact>(std::move(*decoded));
      }
    }
    chunks.push_back(std::move(chunk));
  }
  return chunks;
}

GfxOpenClSourceArtifact make_minimal_opencl_source_artifact(
    const CacheBackendPayloadRecord &payload,
    const KernelArtifactDescriptor &descriptor) {
  GfxOpenClSourceArtifact artifact{};
  artifact.valid = true;
  artifact.artifact_ref.valid = true;
  artifact.artifact_ref.kind = GfxKernelArtifactKind::OpenClSource;
  artifact.artifact_ref.backend_domain = GfxKernelBackendDomain::OpenCl;
  artifact.artifact_ref.source_id = payload.source_id;
  artifact.artifact_ref.entry_point = payload.entry_point;
  if (!descriptor.compile_options_key.empty()) {
    artifact.build_options = {descriptor.compile_options_key};
    artifact.artifact_ref.build_options = artifact.build_options;
  }
  artifact.source = payload.source;
  artifact.direct_input_indices = descriptor.launch_plan.direct_input_indices;
  artifact.arg_count = descriptor.abi_arg_count;
  artifact.direct_input_count = descriptor.launch_plan.input_arg_count;
  artifact.direct_output_count = descriptor.abi_output_arg_count;
  artifact.scalar_args.reserve(descriptor.launch_plan.scalar_arg_kinds.size());
  for (const auto scalar : descriptor.launch_plan.scalar_arg_kinds) {
    artifact.scalar_args.push_back(
        static_cast<GfxOpenClSourceScalarArg>(scalar));
  }
  return artifact;
}

bool source_artifact_matches_descriptor(
    const KernelArtifactDescriptor &descriptor,
    const GfxOpenClSourceArtifact &artifact) noexcept {
  return artifact.valid && artifact.artifact_ref.valid &&
         artifact.artifact_ref.kind == GfxKernelArtifactKind::OpenClSource &&
         artifact.artifact_ref.backend_domain ==
             GfxKernelBackendDomain::OpenCl &&
         descriptor.kernel.backend_domain == "opencl" &&
         descriptor.payload_kind == KernelArtifactPayloadKind::OpenClSource &&
         descriptor.kernel.kernel_id == artifact.artifact_ref.source_id &&
         descriptor.kernel.origin == classify_opencl_kernel_artifact_origin(
                                         artifact.artifact_ref.source_id);
}

uint32_t count_runtime_param_roles(const GfxKernelStageManifest &manifest) {
  if (!manifest.valid || !manifest.custom_kernel.valid ||
      !manifest.custom_kernel.external_buffer_abi.valid) {
    return 0;
  }
  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      manifest.custom_kernel.external_buffer_abi);
  return static_cast<uint32_t>(std::count(roles.begin(), roles.end(),
                                          GfxKernelBufferRole::RuntimeParams));
}

KernelLaunchPlanDescriptor
make_opencl_launch_plan_descriptor(const GfxOpenClSourceArtifact &artifact) {
  KernelLaunchPlanDescriptor descriptor;
  if (!artifact.stage_manifest.valid ||
      !artifact.stage_manifest.custom_kernel.valid ||
      !artifact.stage_manifest.custom_kernel.external_buffer_abi.valid) {
    return descriptor;
  }
  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      artifact.stage_manifest.custom_kernel.external_buffer_abi);
  if (roles.empty()) {
    return descriptor;
  }
  descriptor.valid = true;
  descriptor.buffer_roles.reserve(roles.size());
  for (const auto role : roles) {
    descriptor.buffer_roles.emplace_back(
        kernel_buffer_role_descriptor_name(role));
  }
  descriptor.direct_input_indices = artifact.direct_input_indices;
  descriptor.input_arg_count = artifact.direct_input_count;
  descriptor.scalar_arg_kinds.reserve(artifact.scalar_args.size());
  for (const auto scalar : artifact.scalar_args) {
    descriptor.scalar_arg_kinds.push_back(static_cast<uint32_t>(scalar));
  }
  return descriptor;
}

std::shared_ptr<const KernelArtifactPayload>
resolve_opencl_payload(const KernelArtifactDescriptor &descriptor,
                       const PlannedOperation &op) {
  if (descriptor.kernel.backend_domain != "opencl" ||
      descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
      !op.source_node) {
    return {};
  }

  for (const auto &family : opencl_artifact_family_entries()) {
    if (!family.matches(op.source_node)) {
      continue;
    }
    return family.make_payload(descriptor, op);
  }
  return {};
}

} // namespace

::ov::gfx_plugin::KernelArtifactOrigin classify_opencl_kernel_artifact_origin(
    std::string_view kernel_unit_id) noexcept {
  if (starts_with(kernel_unit_id, "opencl/generated/")) {
    return KernelArtifactOrigin::Generated;
  }
  if (starts_with(kernel_unit_id, "opencl/")) {
    return KernelArtifactOrigin::HandwrittenException;
  }
  return KernelArtifactOrigin::Unknown;
}

KernelArtifactPayloadResolver make_opencl_kernel_artifact_payload_resolver() {
  return [](const KernelArtifactDescriptor &descriptor,
            const PlannedOperation &op) {
    return resolve_opencl_payload(descriptor, op);
  };
}

CacheBackendPayloadEncoder make_opencl_cache_payload_encoder() {
  return [](const KernelArtifactDescriptor &descriptor,
            const KernelArtifactPayloadRecord &payload_record) {
    CacheBackendPayloadRecord record;
    if (descriptor.kernel.backend_domain != "opencl" ||
        descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
        !payload_record.payload) {
      return record;
    }
    const auto *payload =
        dynamic_cast<const GfxOpenClSourceArtifactPayload *>(
            payload_record.payload.get());
    if (!payload || !payload->valid()) {
      return record;
    }
    record.source_language = "opencl";
    record.source = payload->artifact().source;
    record.payload_format = kOpenClSourceArtifactPayloadFormat;
    record.payload_data = serialize_opencl_source_artifact(payload->artifact());
    return record;
  };
}

CacheBackendPayloadDecoder make_opencl_cache_payload_decoder() {
  return [](const CacheBackendPayloadRecord &payload,
            const KernelArtifactDescriptor &descriptor)
             -> std::shared_ptr<const KernelArtifactPayload> {
    if (descriptor.kernel.backend_domain != "opencl" ||
        descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
        payload.payload_kind != kernel_artifact_payload_kind_to_string(
                                    KernelArtifactPayloadKind::OpenClSource)) {
      return {};
    }
    if (payload.payload_format == kOpenClSourceArtifactPayloadFormat &&
        !payload.payload_data.empty()) {
      auto artifact = decode_opencl_source_artifact(payload, descriptor,
                                                    payload.payload_data);
      if (!artifact) {
        return {};
      }
      return std::make_shared<GfxOpenClSourceArtifactPayload>(
          std::move(*artifact));
    }
    if (!payload.source.empty()) {
      auto artifact = make_minimal_opencl_source_artifact(payload, descriptor);
      return std::make_shared<GfxOpenClSourceArtifactPayload>(
          std::move(artifact));
    }
    return {};
  };
}

bool finalize_opencl_kernel_artifact_descriptor_contract(
    KernelArtifactDescriptor &descriptor,
    const ::ov::gfx_plugin::GfxOpenClSourceArtifact &artifact) {
  if (!source_artifact_matches_descriptor(descriptor, artifact)) {
    return false;
  }
  descriptor.runtime_param_buffer_count =
      count_runtime_param_roles(artifact.stage_manifest);
  descriptor.runtime_param_payload_kind =
      runtime_param_descriptor_payload_kind_for_stage(
          descriptor.kernel.op_family, descriptor.runtime_param_buffer_count);
  descriptor.runtime_param_i64_metadata.clear();
  descriptor.runtime_param_reduce_keep_dims = false;
  descriptor.runtime_param_reduce_keep_dims_valid = false;
  descriptor.entry_point = artifact.artifact_ref.entry_point;
  descriptor.compile_options_key =
      gfx_opencl_source_artifact_build_options(artifact);
  descriptor.abi_arg_count = artifact.arg_count;
  descriptor.abi_output_arg_count = artifact.direct_output_count;
  descriptor.launch_plan = make_opencl_launch_plan_descriptor(artifact);
  finalize_kernel_artifact_descriptor_identity(descriptor);
  return true;
}

bool opencl_source_artifact_matches_descriptor_contract(
    const KernelArtifactDescriptor &descriptor,
    const ::ov::gfx_plugin::GfxOpenClSourceArtifact &artifact) {
  KernelArtifactDescriptor expected = descriptor;
  if (!finalize_opencl_kernel_artifact_descriptor_contract(expected,
                                                           artifact)) {
    return false;
  }
  return descriptor.entry_point == expected.entry_point &&
         descriptor.compile_options_key == expected.compile_options_key &&
         descriptor.abi_arg_count == expected.abi_arg_count &&
         descriptor.abi_output_arg_count == expected.abi_output_arg_count &&
         descriptor.runtime_param_buffer_count ==
             expected.runtime_param_buffer_count &&
         descriptor.runtime_param_payload_kind ==
             expected.runtime_param_payload_kind &&
         descriptor.runtime_param_i64_metadata ==
             expected.runtime_param_i64_metadata &&
         descriptor.runtime_param_reduce_keep_dims ==
             expected.runtime_param_reduce_keep_dims &&
         descriptor.runtime_param_reduce_keep_dims_valid ==
             expected.runtime_param_reduce_keep_dims_valid &&
         descriptor.launch_plan.valid == expected.launch_plan.valid &&
         descriptor.launch_plan.buffer_roles ==
             expected.launch_plan.buffer_roles &&
         descriptor.launch_plan.direct_input_indices ==
             expected.launch_plan.direct_input_indices &&
         descriptor.launch_plan.input_indices ==
             expected.launch_plan.input_indices &&
         descriptor.launch_plan.input_arg_count ==
             expected.launch_plan.input_arg_count &&
         descriptor.launch_plan.operand_kinds ==
             expected.launch_plan.operand_kinds &&
         descriptor.launch_plan.operand_arg_indices ==
             expected.launch_plan.operand_arg_indices &&
         descriptor.launch_plan.scalar_args ==
             expected.launch_plan.scalar_args &&
         descriptor.launch_plan.scalar_arg_kinds ==
             expected.launch_plan.scalar_arg_kinds &&
         descriptor.abi_fingerprint == expected.abi_fingerprint &&
         descriptor.manifest_ref == expected.manifest_ref &&
         descriptor.artifact_key == expected.artifact_key;
}

KernelArtifactDescriptorResolver
make_opencl_kernel_artifact_descriptor_resolver() {
  return [](KernelArtifactDescriptor &descriptor,
            const PlannedOperation &op) -> bool {
    if (descriptor.kernel.backend_domain != "opencl" ||
        descriptor.payload_kind != KernelArtifactPayloadKind::OpenClSource ||
        !op.source_node) {
      finalize_kernel_artifact_descriptor_identity(descriptor);
      return true;
    }

    for (const auto &family : opencl_artifact_family_entries()) {
      if (!family.matches(op.source_node)) {
        continue;
      }
      const auto artifact = family.make_source_artifact(
          op.source_node, descriptor.kernel.kernel_id);
      return artifact && artifact->valid &&
             finalize_opencl_kernel_artifact_descriptor_contract(descriptor,
                                                                 *artifact);
    }
    return false;
  };
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
