// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/cache_envelope.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

constexpr uint32_t kCacheEnvelopeSchemaVersion = 1;

uint64_t stable_hash64(std::string_view value) noexcept {
  uint64_t hash = 14695981039346656037ull;
  for (const unsigned char c : value) {
    hash ^= c;
    hash *= 1099511628211ull;
  }
  return hash;
}

void append_field(std::ostringstream &os, std::string_view value) {
  os << value.size() << ":" << value << ";";
}

void append_bool(std::ostringstream &os, bool value) {
  append_field(os, value ? "1" : "0");
}

template <typename T>
void append_number(std::ostringstream &os, T value) {
  append_field(os, std::to_string(value));
}

void append_number(std::ostringstream &os, float value) {
  std::ostringstream value_os;
  value_os << std::setprecision(std::numeric_limits<float>::max_digits10)
           << value;
  append_field(os, value_os.str());
}

void append_number(std::ostringstream &os, double value) {
  std::ostringstream value_os;
  value_os << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
  append_field(os, value_os.str());
}

template <typename T>
void append_vector(std::ostringstream &os, const std::vector<T> &values) {
  append_field(os, std::to_string(values.size()));
  for (const auto &value : values) {
    append_number(os, value);
  }
}

void append_vector(std::ostringstream &os,
                   const std::vector<std::string> &values) {
  append_field(os, std::to_string(values.size()));
  for (const auto &value : values) {
    append_field(os, value);
  }
}

template <typename T>
void append_integral_vector(std::ostringstream &os,
                            const std::vector<T> &values) {
  append_field(os, std::to_string(values.size()));
  for (const auto value : values) {
    append_field(os, std::to_string(value));
  }
}

void append_parallelism_band(std::ostringstream &os,
                             const GpuChunkDispatchBand &band) {
  append_field(os, std::to_string(band.min_work_per_elem));
  append_field(os, std::to_string(band.elems_per_dispatch));
  append_field(os, std::to_string(band.max_elems_per_dispatch));
  append_field(os, std::to_string(band.target_dispatches));
}

void append_parallelism_profile(std::ostringstream &os,
                                const GpuParallelismProfile &profile) {
  append_field(os, profile.profile_key);
  append_field(os, std::to_string(profile.preferred_simd_width));
  append_field(os, std::to_string(profile.subgroup_size));
  append_field(os, std::to_string(profile.max_total_threads_per_group));
  append_field(os, std::to_string(profile.max_threads_per_group[0]));
  append_field(os, std::to_string(profile.max_threads_per_group[1]));
  append_field(os, std::to_string(profile.max_threads_per_group[2]));
  append_bool(os, profile.supports_conv_output_channel_blocking);
  append_bool(os, profile.supports_conv_channel_block_spatial_tiling);
  append_bool(os, profile.sort_matmul_tiles_by_shape);
  append_bool(os, profile.enable_skinny_matmul_tiles);
  append_bool(os, profile.scale_conv_threads_for_large_spatial);
  append_bool(os, profile.scale_conv_threads_for_dense_reduction);
  append_bool(os, profile.scale_conv_threads_for_pointwise_reduction);
  append_bool(os, profile.conv_spatial_micro_tile_requires_large_output_area);
  append_field(
      os, std::to_string(profile.chunk_dispatch.small_total_elems_threshold));
  append_field(
      os, std::to_string(profile.chunk_dispatch.small_min_elems_per_dispatch));
  append_parallelism_band(os, profile.chunk_dispatch.light);
  append_parallelism_band(os, profile.chunk_dispatch.medium);
  append_parallelism_band(os, profile.chunk_dispatch.heavy);
  append_parallelism_band(os, profile.chunk_dispatch.very_heavy);
  append_bool(os, profile.chunk_dispatch.retune_threads_to_workload);
}

std::string hex64(uint64_t value) {
  std::ostringstream os;
  os << std::hex << std::setw(16) << std::setfill('0') << value;
  return os.str();
}

std::string shape_to_string(const ov::PartialShape &shape) {
  std::ostringstream os;
  os << shape;
  return os.str();
}

std::string hash_material(std::string_view material) {
  return hex64(stable_hash64(material));
}

std::string hash_bytes(const void *data, size_t size) {
  if (!data || size == 0) {
    return hash_material({});
  }
  return hash_material(
      std::string_view(static_cast<const char *>(data), size));
}

class ModelFingerprintAttributeVisitor final : public ov::AttributeVisitor {
public:
  explicit ModelFingerprintAttributeVisitor(std::ostringstream &material)
      : m_material(material) {}

  void on_adapter(const std::string &name,
                  ov::ValueAccessor<std::shared_ptr<ov::Model>> &adapter) override {
    append_attribute_header(name, "model_ref");
    const auto &model = adapter.get();
    append_field(m_material,
                 model ? model->get_friendly_name() : std::string("<null>"));
  }

  void on_adapter(const std::string &name,
                  ov::ValueAccessor<void> &adapter) override {
    append_attribute_header(name, adapter.get_type_info().name);
    if (auto element_type =
            ov::as_type<ov::AttributeAdapter<ov::element::Type>>(&adapter)) {
      append_field(m_material, element_type->get());
      return;
    }
    if (auto partial_shape =
            ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
      append_field(m_material, shape_to_string(partial_shape->get()));
      return;
    }
    if (auto dimension =
            ov::as_type<ov::AttributeAdapter<ov::Dimension>>(&adapter)) {
      std::ostringstream os;
      os << dimension->get();
      append_field(m_material, os.str());
      return;
    }
    if (auto element_types = ov::as_type<
            ov::AttributeAdapter<std::vector<ov::element::Type>>>(&adapter)) {
      append_field(m_material, std::to_string(element_types->get().size()));
      for (const auto &type : element_types->get()) {
        append_field(m_material, type.get_type_name());
      }
      return;
    }
    append_field(m_material, "<unsupported-attribute-adapter>");
  }

  void on_adapter(const std::string &name,
                  ov::ValueAccessor<void *> &adapter) override {
    append_attribute_header(name, "raw");
    append_field(m_material, std::to_string(adapter.size()));
    append_field(m_material, hash_bytes(adapter.get_ptr(), adapter.size()));
  }

#define GFX_CACHE_SCALAR_ATTR(TYPE)                                             \
  void on_adapter(const std::string &name,                                      \
                  ov::ValueAccessor<TYPE> &adapter) override {                  \
    append_attribute_header(name, #TYPE);                                       \
    append_number(m_material, adapter.get());                                   \
  }

  GFX_CACHE_SCALAR_ATTR(bool)
  GFX_CACHE_SCALAR_ATTR(int8_t)
  GFX_CACHE_SCALAR_ATTR(int16_t)
  GFX_CACHE_SCALAR_ATTR(int32_t)
  GFX_CACHE_SCALAR_ATTR(int64_t)
  GFX_CACHE_SCALAR_ATTR(uint8_t)
  GFX_CACHE_SCALAR_ATTR(uint16_t)
  GFX_CACHE_SCALAR_ATTR(uint32_t)
  GFX_CACHE_SCALAR_ATTR(uint64_t)
  GFX_CACHE_SCALAR_ATTR(float)
  GFX_CACHE_SCALAR_ATTR(double)

#undef GFX_CACHE_SCALAR_ATTR

  void on_adapter(const std::string &name,
                  ov::ValueAccessor<std::string> &adapter) override {
    append_attribute_header(name, "string");
    append_field(m_material, adapter.get());
  }

#define GFX_CACHE_VECTOR_ATTR(TYPE)                                             \
  void on_adapter(const std::string &name,                                      \
                  ov::ValueAccessor<std::vector<TYPE>> &adapter) override {     \
    append_attribute_header(name, "vector<" #TYPE ">");                       \
    append_vector(m_material, adapter.get());                                   \
  }

  GFX_CACHE_VECTOR_ATTR(int8_t)
  GFX_CACHE_VECTOR_ATTR(int16_t)
  GFX_CACHE_VECTOR_ATTR(int32_t)
  GFX_CACHE_VECTOR_ATTR(int64_t)
  GFX_CACHE_VECTOR_ATTR(uint8_t)
  GFX_CACHE_VECTOR_ATTR(uint16_t)
  GFX_CACHE_VECTOR_ATTR(uint32_t)
  GFX_CACHE_VECTOR_ATTR(uint64_t)
  GFX_CACHE_VECTOR_ATTR(float)
  GFX_CACHE_VECTOR_ATTR(double)
  GFX_CACHE_VECTOR_ATTR(std::string)

#undef GFX_CACHE_VECTOR_ATTR

private:
  void append_attribute_header(const std::string &name,
                               std::string_view type_name) {
    append_field(m_material, name);
    append_field(m_material, type_name);
  }

  std::ostringstream &m_material;
};

void append_constant_payload_fingerprint(std::ostringstream &material,
                                         const ov::op::v0::Constant &constant) {
  append_field(material, "constant_payload");
  append_field(material, constant.get_element_type().get_type_name());
  append_field(material, shape_to_string(constant.get_output_partial_shape(0)));
  append_field(material, std::to_string(constant.get_byte_size()));
  if (constant.get_element_type() == ov::element::string) {
    const auto values = constant.cast_vector<std::string>();
    append_vector(material, values);
    return;
  }
  append_field(material,
               hash_bytes(constant.get_data_ptr(), constant.get_byte_size()));
}

std::string make_cache_key_stable_key(const CacheKey &key) {
  std::ostringstream material;
  append_field(material, key.model_fingerprint);
  append_field(material, key.manifest_hash);
  append_field(material, key.target_fingerprint);
  append_field(material, key.backend_capabilities_fingerprint);
  append_field(material, key.compiler_revision);
  append_field(material, key.backend_compiler_revision);
  append_field(material, key.driver_identity);
  append_field(material, key.compile_options_hash);
  for (const auto &version : key.kernel_unit_versions) {
    append_field(material, version);
  }
  return hash_material(material.str());
}

bool has_artifact_key(const ExecutableBundle &executable,
                      std::string_view artifact_key) {
  return std::any_of(executable.artifact_descriptors.begin(),
                     executable.artifact_descriptors.end(),
                     [&](const KernelArtifactDescriptor &descriptor) {
                       return descriptor.artifact_key == artifact_key;
                     });
}

const KernelArtifactDescriptor *
find_artifact_descriptor(const ExecutableBundle &executable,
                         std::string_view artifact_key) {
  for (const auto &descriptor : executable.artifact_descriptors) {
    if (descriptor.artifact_key == artifact_key) {
      return &descriptor;
    }
  }
  return nullptr;
}

CacheBackendPayloadRecord
make_cache_payload_record(const ExecutableBundle &executable,
                          const KernelArtifactPayloadRecord &payload_record) {
  CacheBackendPayloadRecord record;
  record.artifact_key = payload_record.artifact_key;
  const auto *descriptor =
      find_artifact_descriptor(executable, payload_record.artifact_key);
  if (descriptor) {
    record.backend_domain = descriptor->kernel.backend_domain;
    record.payload_kind = std::string(
        kernel_artifact_payload_kind_to_string(descriptor->payload_kind));
  }
  if (payload_record.payload) {
    record.source_id = std::string(payload_record.payload->source_id());
    record.entry_point = std::string(payload_record.payload->entry_point());
    std::ostringstream identity;
    append_field(identity, kernel_artifact_payload_kind_to_string(
                               payload_record.payload->payload_kind()));
    append_field(identity, payload_record.payload->backend_domain());
    append_field(identity, payload_record.payload->source_id());
    append_field(identity, payload_record.payload->entry_point());
    append_field(identity, payload_record.artifact_key);
    record.payload_identity = hash_material(identity.str());
  }
  return record;
}

void require_nonempty(CacheEnvelopeVerificationResult &result,
                      std::string_view value, std::string diagnostic) {
  if (value.empty()) {
    result.diagnostics.push_back(std::move(diagnostic));
  }
}

LoweringRouteKind lowering_route_kind_from_string(std::string_view value) {
  if (value == lowering_route_kind_to_string(LoweringRouteKind::Common)) {
    return LoweringRouteKind::Common;
  }
  if (value == lowering_route_kind_to_string(LoweringRouteKind::Metadata)) {
    return LoweringRouteKind::Metadata;
  }
  if (value ==
      lowering_route_kind_to_string(LoweringRouteKind::VendorPrimitive)) {
    return LoweringRouteKind::VendorPrimitive;
  }
  if (value ==
      lowering_route_kind_to_string(LoweringRouteKind::GeneratedKernel)) {
    return LoweringRouteKind::GeneratedKernel;
  }
  if (value == lowering_route_kind_to_string(
                   LoweringRouteKind::HandwrittenKernelException)) {
    return LoweringRouteKind::HandwrittenKernelException;
  }
  return LoweringRouteKind::Unsupported;
}

TensorContractRole tensor_contract_role_from_string(std::string_view value) {
  if (value == tensor_contract_role_to_string(TensorContractRole::TensorOutput)) {
    return TensorContractRole::TensorOutput;
  }
  return TensorContractRole::TensorInput;
}

RuntimeParamKind runtime_param_kind_from_string(std::string_view value) {
  if (value == runtime_param_kind_to_string(RuntimeParamKind::Scalar)) {
    return RuntimeParamKind::Scalar;
  }
  return RuntimeParamKind::Shape;
}

StatefulEffectKind stateful_effect_kind_from_string(std::string_view value) {
  if (value == stateful_effect_kind_to_string(StatefulEffectKind::ReadValue)) {
    return StatefulEffectKind::ReadValue;
  }
  if (value == stateful_effect_kind_to_string(StatefulEffectKind::Assign)) {
    return StatefulEffectKind::Assign;
  }
  return StatefulEffectKind::None;
}

MemoryRegionKind memory_region_kind_from_string(std::string_view value) {
  if (value == memory_region_kind_to_string(MemoryRegionKind::ExternalTensor)) {
    return MemoryRegionKind::ExternalTensor;
  }
  if (value == memory_region_kind_to_string(MemoryRegionKind::ImmutableTensor)) {
    return MemoryRegionKind::ImmutableTensor;
  }
  return MemoryRegionKind::TransientTensor;
}

KernelArtifactOrigin kernel_artifact_origin_from_string(
    std::string_view value) {
  if (value == kernel_artifact_origin_to_string(KernelArtifactOrigin::Common)) {
    return KernelArtifactOrigin::Common;
  }
  if (value == kernel_artifact_origin_to_string(KernelArtifactOrigin::Metadata)) {
    return KernelArtifactOrigin::Metadata;
  }
  if (value == kernel_artifact_origin_to_string(
                   KernelArtifactOrigin::VendorPrimitive)) {
    return KernelArtifactOrigin::VendorPrimitive;
  }
  if (value == kernel_artifact_origin_to_string(KernelArtifactOrigin::Generated)) {
    return KernelArtifactOrigin::Generated;
  }
  if (value == kernel_artifact_origin_to_string(
                   KernelArtifactOrigin::HandwrittenException)) {
    return KernelArtifactOrigin::HandwrittenException;
  }
  return KernelArtifactOrigin::Unknown;
}

KernelArtifactPayloadKind kernel_artifact_payload_kind_from_string(
    std::string_view value) {
  if (value == kernel_artifact_payload_kind_to_string(
                   KernelArtifactPayloadKind::VendorDescriptor)) {
    return KernelArtifactPayloadKind::VendorDescriptor;
  }
  if (value ==
      kernel_artifact_payload_kind_to_string(KernelArtifactPayloadKind::MslSource)) {
    return KernelArtifactPayloadKind::MslSource;
  }
  if (value == kernel_artifact_payload_kind_to_string(
                   KernelArtifactPayloadKind::OpenClSource)) {
    return KernelArtifactPayloadKind::OpenClSource;
  }
  return KernelArtifactPayloadKind::None;
}

class WireReader final {
public:
  explicit WireReader(std::string_view wire) : m_wire(wire) {}

  bool ok() const noexcept { return m_diagnostics.empty(); }
  std::vector<std::string> take_diagnostics() { return std::move(m_diagnostics); }

  std::string string_field(std::string_view name) {
    if (m_pos >= m_wire.size()) {
      m_diagnostics.push_back(std::string("cache envelope wire ended before ") +
                              std::string(name));
      return {};
    }
    size_t colon = m_wire.find(':', m_pos);
    if (colon == std::string_view::npos) {
      m_diagnostics.push_back(std::string("cache envelope wire field ") +
                              std::string(name) + " has no length separator");
      m_pos = m_wire.size();
      return {};
    }
    const auto length_text = m_wire.substr(m_pos, colon - m_pos);
    size_t length = 0;
    try {
      length = static_cast<size_t>(std::stoull(std::string(length_text)));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("cache envelope wire field ") +
                              std::string(name) + " has invalid length");
      m_pos = m_wire.size();
      return {};
    }
    const size_t value_begin = colon + 1;
    const size_t value_end = value_begin + length;
    if (value_end >= m_wire.size() || m_wire[value_end] != ';') {
      m_diagnostics.push_back(std::string("cache envelope wire field ") +
                              std::string(name) + " is truncated");
      m_pos = m_wire.size();
      return {};
    }
    m_pos = value_end + 1;
    return std::string(m_wire.substr(value_begin, length));
  }

  uint32_t u32_field(std::string_view name) {
    return static_cast<uint32_t>(u64_field(name));
  }

  uint64_t u64_field(std::string_view name) {
    const auto value = string_field(name);
    try {
      return static_cast<uint64_t>(std::stoull(value));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("cache envelope wire field ") +
                              std::string(name) + " is not uint64");
      return 0;
    }
  }

  size_t size_field(std::string_view name) {
    return static_cast<size_t>(u64_field(name));
  }

  int64_t i64_field(std::string_view name) {
    const auto value = string_field(name);
    try {
      return static_cast<int64_t>(std::stoll(value));
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("cache envelope wire field ") +
                              std::string(name) + " is not int64");
      return 0;
    }
  }

  int32_t i32_field(std::string_view name) {
    return static_cast<int32_t>(i64_field(name));
  }

  double double_field(std::string_view name) {
    const auto value = string_field(name);
    try {
      return std::stod(value);
    } catch (const std::exception &) {
      m_diagnostics.push_back(std::string("cache envelope wire field ") +
                              std::string(name) + " is not double");
      return 0.0;
    }
  }

  bool bool_field(std::string_view name) {
    const auto value = string_field(name);
    if (value == "1") {
      return true;
    }
    if (value == "0") {
      return false;
    }
    m_diagnostics.push_back(std::string("cache envelope wire field ") +
                            std::string(name) + " is not bool");
    return false;
  }

  std::vector<std::string> string_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<std::string> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(string_field(name));
    }
    return values;
  }

  std::vector<int64_t> i64_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<int64_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(i64_field(name));
    }
    return values;
  }

  std::vector<int32_t> i32_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<int32_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(i32_field(name));
    }
    return values;
  }

  std::vector<uint32_t> u32_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<uint32_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(u32_field(name));
    }
    return values;
  }

  std::vector<size_t> size_vector(std::string_view name) {
    const auto count = size_field(name);
    std::vector<size_t> values;
    values.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      values.push_back(size_field(name));
    }
    return values;
  }

private:
  std::string_view m_wire;
  size_t m_pos = 0;
  std::vector<std::string> m_diagnostics;
};

void append_runtime_param_contract(std::ostringstream &os,
                                   const RuntimeParamContract &contract) {
  append_field(os, std::to_string(contract.scalar_param_count));
  append_field(os, std::to_string(contract.shape_param_count));
  append_field(os, std::to_string(contract.params.size()));
  for (const auto &param : contract.params) {
    append_field(os, param.logical_name);
    append_field(os, runtime_param_kind_to_string(param.kind));
    append_field(os, param.abi_type);
    append_field(os, param.source_tensor);
  }
  append_vector(os, contract.runtime_param_names);
}

RuntimeParamContract read_runtime_param_contract(WireReader &reader) {
  RuntimeParamContract contract;
  contract.scalar_param_count = reader.size_field("runtime scalar count");
  contract.shape_param_count = reader.size_field("runtime shape count");
  const auto count = reader.size_field("runtime param count");
  contract.params.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    RuntimeParamDescriptor param;
    param.logical_name = reader.string_field("runtime param logical name");
    param.kind =
        runtime_param_kind_from_string(reader.string_field("runtime param kind"));
    param.abi_type = reader.string_field("runtime param abi type");
    param.source_tensor = reader.string_field("runtime param source tensor");
    contract.params.push_back(std::move(param));
  }
  contract.runtime_param_names =
      reader.string_vector("runtime param names");
  return contract;
}

void append_tensor_contract(std::ostringstream &os,
                            const TensorContract &tensor) {
  append_field(os, tensor.logical_name);
  append_field(os, tensor.memory_region_id);
  append_field(os, tensor_contract_role_to_string(tensor.role));
  append_field(os, tensor.element_type);
  append_field(os, tensor.partial_shape);
  append_field(os, tensor.layout);
  append_field(os, tensor.storage_kind);
  append_field(os, tensor.lifetime_class);
  append_field(os, tensor.stateful_prebind_variable_id);
  append_field(os, tensor.stateful_prebind_shape_rule);
  append_field(os, std::to_string(tensor.stateful_prebind_shape_axis));
}

TensorContract read_tensor_contract(WireReader &reader) {
  TensorContract tensor;
  tensor.logical_name = reader.string_field("tensor logical name");
  tensor.memory_region_id = reader.string_field("tensor memory region id");
  tensor.role =
      tensor_contract_role_from_string(reader.string_field("tensor role"));
  tensor.element_type = reader.string_field("tensor element type");
  tensor.partial_shape = reader.string_field("tensor partial shape");
  tensor.layout = reader.string_field("tensor layout");
  tensor.storage_kind = reader.string_field("tensor storage kind");
  tensor.lifetime_class = reader.string_field("tensor lifetime class");
  tensor.stateful_prebind_variable_id =
      reader.string_field("tensor stateful variable id");
  tensor.stateful_prebind_shape_rule =
      reader.string_field("tensor stateful shape rule");
  tensor.stateful_prebind_shape_axis =
      reader.i64_field("tensor stateful shape axis");
  return tensor;
}

void append_memory_plan(std::ostringstream &os, const MemoryPlan &plan) {
  append_field(os, std::to_string(plan.schema_version));
  append_bool(os, plan.hidden_host_copies_allowed);
  append_field(os, std::to_string(plan.regions.size()));
  for (const auto &region : plan.regions) {
    append_field(os, region.region_id);
    append_field(os, region.logical_tensor_name);
    append_field(os, memory_region_kind_to_string(region.kind));
    append_field(os, region.element_type);
    append_field(os, region.partial_shape);
    append_field(os, region.layout);
    append_field(os, region.storage_kind);
    append_field(os, region.alias_group);
    append_field(os, std::to_string(region.lifetime.first_stage));
    append_field(os, std::to_string(region.lifetime.last_stage));
    append_bool(os, region.external_binding);
    append_bool(os, region.host_visible);
  }
  append_field(os, std::to_string(plan.alias_groups.size()));
  for (const auto &group : plan.alias_groups) {
    append_field(os, group.group_id);
    append_bool(os, group.output_aliasing);
    append_vector(os, group.region_ids);
  }
  append_field(os, std::to_string(plan.transient_arenas.size()));
  for (const auto &arena : plan.transient_arenas) {
    append_field(os, arena.arena_id);
    append_field(os, arena.storage_kind);
    append_vector(os, arena.region_ids);
  }
}

MemoryPlan read_memory_plan(WireReader &reader) {
  MemoryPlan plan;
  plan.schema_version = reader.u32_field("memory plan schema");
  plan.hidden_host_copies_allowed =
      reader.bool_field("memory plan hidden host copies");
  const auto region_count = reader.size_field("memory region count");
  plan.regions.reserve(region_count);
  for (size_t i = 0; i < region_count; ++i) {
    MemoryRegion region;
    region.region_id = reader.string_field("memory region id");
    region.logical_tensor_name =
        reader.string_field("memory region logical tensor name");
    region.kind =
        memory_region_kind_from_string(reader.string_field("memory region kind"));
    region.element_type = reader.string_field("memory region element type");
    region.partial_shape = reader.string_field("memory region partial shape");
    region.layout = reader.string_field("memory region layout");
    region.storage_kind = reader.string_field("memory region storage kind");
    region.alias_group = reader.string_field("memory region alias group");
    region.lifetime.first_stage =
        reader.size_field("memory region lifetime first");
    region.lifetime.last_stage =
        reader.size_field("memory region lifetime last");
    region.external_binding =
        reader.bool_field("memory region external binding");
    region.host_visible = reader.bool_field("memory region host visible");
    plan.regions.push_back(std::move(region));
  }
  const auto alias_count = reader.size_field("memory alias group count");
  plan.alias_groups.reserve(alias_count);
  for (size_t i = 0; i < alias_count; ++i) {
    AliasGroup group;
    group.group_id = reader.string_field("memory alias group id");
    group.output_aliasing =
        reader.bool_field("memory alias group output aliasing");
    group.region_ids = reader.string_vector("memory alias group regions");
    plan.alias_groups.push_back(std::move(group));
  }
  const auto arena_count = reader.size_field("memory transient arena count");
  plan.transient_arenas.reserve(arena_count);
  for (size_t i = 0; i < arena_count; ++i) {
    TransientArena arena;
    arena.arena_id = reader.string_field("memory transient arena id");
    arena.storage_kind =
        reader.string_field("memory transient arena storage kind");
    arena.region_ids = reader.string_vector("memory transient arena regions");
    plan.transient_arenas.push_back(std::move(arena));
  }
  return plan;
}

void append_stage_record(std::ostringstream &os, const StageRecord &stage) {
  append_field(os, std::to_string(stage.stage_id));
  append_field(os, std::to_string(stage.stable_record_key));
  append_field(os, stage.source_node_name);
  append_field(os, stage.normalized_op_family);
  append_field(os, lowering_route_kind_to_string(stage.execution_kind));
  append_field(os, stage.backend_domain);
  append_field(os, stage.kernel_unit_id);
  append_field(os, stage.kernel_unit_kind);
  append_bool(os, stage.requires_runtime_shape_args);
  append_field(os, std::to_string(stage.inputs.size()));
  for (const auto &input : stage.inputs) {
    append_tensor_contract(os, input);
  }
  append_field(os, std::to_string(stage.outputs.size()));
  for (const auto &output : stage.outputs) {
    append_tensor_contract(os, output);
  }
  append_runtime_param_contract(os, stage.runtime_params);
  append_field(os, stage.runtime_shape.rule);
  append_integral_vector(os, stage.runtime_shape.i64_metadata);
  append_field(os, stateful_effect_kind_to_string(stage.stateful_effect.kind));
  append_field(os, stage.stateful_effect.variable_id);
  append_field(os, lowering_route_kind_to_string(stage.dispatch.execution_kind));
  append_field(os, stage.dispatch.backend_domain);
  append_field(os, stage.dispatch.kernel_unit_id);
  append_field(os, stage.dispatch.kernel_unit_kind);
  append_field(os, stage.dispatch.dispatch_source);
  append_bool(os, stage.memory.hidden_host_copy_allowed);
  append_field(os, stage.memory.input_lifetime);
  append_field(os, stage.memory.output_lifetime);
  append_field(os, stage.memory.alias_group);
  append_field(os, std::to_string(stage.submission.stage_weight));
  append_field(os, std::to_string(stage.submission.macs_estimate));
  append_bool(os, stage.submission.dependency_extension_boundary);
  append_field(os, stage.handwritten_exception.ticket);
  append_field(os, stage.handwritten_exception.reason);
  append_field(os, stage.handwritten_exception.removal_condition);
  append_number(os, stage.profitability_score);
}

StageRecord read_stage_record(WireReader &reader) {
  StageRecord stage;
  stage.stage_id = reader.size_field("stage id");
  stage.stable_record_key = reader.u64_field("stage stable record key");
  stage.source_node_name = reader.string_field("stage source node name");
  stage.normalized_op_family =
      reader.string_field("stage normalized op family");
  stage.execution_kind =
      lowering_route_kind_from_string(reader.string_field("stage execution kind"));
  stage.backend_domain = reader.string_field("stage backend domain");
  stage.kernel_unit_id = reader.string_field("stage kernel unit id");
  stage.kernel_unit_kind = reader.string_field("stage kernel unit kind");
  stage.requires_runtime_shape_args =
      reader.bool_field("stage requires runtime shape args");
  const auto input_count = reader.size_field("stage input count");
  stage.inputs.reserve(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    stage.inputs.push_back(read_tensor_contract(reader));
  }
  const auto output_count = reader.size_field("stage output count");
  stage.outputs.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    stage.outputs.push_back(read_tensor_contract(reader));
  }
  stage.runtime_params = read_runtime_param_contract(reader);
  stage.runtime_shape.rule = reader.string_field("stage runtime shape rule");
  stage.runtime_shape.i64_metadata =
      reader.i64_vector("stage runtime shape metadata");
  stage.stateful_effect.kind = stateful_effect_kind_from_string(
      reader.string_field("stage stateful effect kind"));
  stage.stateful_effect.variable_id =
      reader.string_field("stage stateful effect variable id");
  stage.dispatch.execution_kind = lowering_route_kind_from_string(
      reader.string_field("stage dispatch execution kind"));
  stage.dispatch.backend_domain =
      reader.string_field("stage dispatch backend domain");
  stage.dispatch.kernel_unit_id =
      reader.string_field("stage dispatch kernel id");
  stage.dispatch.kernel_unit_kind =
      reader.string_field("stage dispatch kernel kind");
  stage.dispatch.dispatch_source =
      reader.string_field("stage dispatch source");
  stage.memory.hidden_host_copy_allowed =
      reader.bool_field("stage memory hidden host copy");
  stage.memory.input_lifetime = reader.string_field("stage memory input lifetime");
  stage.memory.output_lifetime =
      reader.string_field("stage memory output lifetime");
  stage.memory.alias_group = reader.string_field("stage memory alias group");
  stage.submission.stage_weight =
      reader.u32_field("stage submission weight");
  stage.submission.macs_estimate =
      reader.u64_field("stage submission macs estimate");
  stage.submission.dependency_extension_boundary =
      reader.bool_field("stage submission dependency boundary");
  stage.handwritten_exception.ticket =
      reader.string_field("stage handwritten ticket");
  stage.handwritten_exception.reason =
      reader.string_field("stage handwritten reason");
  stage.handwritten_exception.removal_condition =
      reader.string_field("stage handwritten removal condition");
  stage.profitability_score = reader.double_field("stage profitability score");
  return stage;
}

void append_manifest(std::ostringstream &os, const ManifestBundle &manifest) {
  append_field(os, std::to_string(manifest.schema_version));
  append_field(os, manifest.target_fingerprint);
  append_memory_plan(os, manifest.memory_plan);
  append_field(os, std::to_string(manifest.stages.size()));
  for (const auto &stage : manifest.stages) {
    append_stage_record(os, stage);
  }
}

ManifestBundle read_manifest(WireReader &reader) {
  ManifestBundle manifest;
  manifest.schema_version = reader.u32_field("manifest schema");
  manifest.target_fingerprint = reader.string_field("manifest target");
  manifest.memory_plan = read_memory_plan(reader);
  const auto stage_count = reader.size_field("manifest stage count");
  manifest.stages.reserve(stage_count);
  for (size_t i = 0; i < stage_count; ++i) {
    manifest.stages.push_back(read_stage_record(reader));
  }
  return manifest;
}

void append_launch_plan(std::ostringstream &os,
                        const KernelLaunchPlanDescriptor &plan) {
  append_bool(os, plan.valid);
  append_vector(os, plan.buffer_roles);
  append_integral_vector(os, plan.direct_input_indices);
  append_integral_vector(os, plan.input_indices);
  append_field(os, std::to_string(plan.input_arg_count));
  append_integral_vector(os, plan.operand_kinds);
  append_integral_vector(os, plan.operand_arg_indices);
  append_integral_vector(os, plan.scalar_args);
  append_integral_vector(os, plan.scalar_arg_kinds);
}

KernelLaunchPlanDescriptor read_launch_plan(WireReader &reader) {
  KernelLaunchPlanDescriptor plan;
  plan.valid = reader.bool_field("launch plan valid");
  plan.buffer_roles = reader.string_vector("launch plan buffer roles");
  plan.direct_input_indices =
      reader.size_vector("launch plan direct input indices");
  plan.input_indices = reader.size_vector("launch plan input indices");
  plan.input_arg_count = reader.size_field("launch plan input arg count");
  plan.operand_kinds = reader.i32_vector("launch plan operand kinds");
  plan.operand_arg_indices =
      reader.i32_vector("launch plan operand arg indices");
  plan.scalar_args = reader.i32_vector("launch plan scalar args");
  plan.scalar_arg_kinds =
      reader.u32_vector("launch plan scalar arg kinds");
  return plan;
}

void append_kernel_descriptor(std::ostringstream &os,
                              const KernelDescriptor &kernel) {
  append_field(os, kernel.kernel_id);
  append_field(os, kernel.op_family);
  append_field(os, kernel.backend_domain);
  append_field(os, kernel_artifact_origin_to_string(kernel.origin));
  append_vector(os, kernel.tensor_roles);
  append_vector(os, kernel.scalar_roles);
  append_field(os, kernel.layout_contract);
  append_field(os, kernel.precision_contract);
  append_field(os, kernel.dispatch_contract);
  append_field(os, kernel.runtime_shape_rule);
  append_integral_vector(os, kernel.runtime_shape_i64_metadata);
  append_bool(os, kernel.requires_runtime_shape_args);
  append_field(os, kernel.exception_ticket);
  append_field(os, kernel.exception_reason);
  append_field(os, kernel.exception_removal_condition);
}

KernelDescriptor read_kernel_descriptor(WireReader &reader) {
  KernelDescriptor kernel;
  kernel.kernel_id = reader.string_field("kernel id");
  kernel.op_family = reader.string_field("kernel op family");
  kernel.backend_domain = reader.string_field("kernel backend domain");
  kernel.origin = kernel_artifact_origin_from_string(
      reader.string_field("kernel origin"));
  kernel.tensor_roles = reader.string_vector("kernel tensor roles");
  kernel.scalar_roles = reader.string_vector("kernel scalar roles");
  kernel.layout_contract = reader.string_field("kernel layout contract");
  kernel.precision_contract = reader.string_field("kernel precision contract");
  kernel.dispatch_contract = reader.string_field("kernel dispatch contract");
  kernel.runtime_shape_rule = reader.string_field("kernel runtime shape rule");
  kernel.runtime_shape_i64_metadata =
      reader.i64_vector("kernel runtime shape metadata");
  kernel.requires_runtime_shape_args =
      reader.bool_field("kernel requires runtime shape args");
  kernel.exception_ticket = reader.string_field("kernel exception ticket");
  kernel.exception_reason = reader.string_field("kernel exception reason");
  kernel.exception_removal_condition =
      reader.string_field("kernel exception removal condition");
  return kernel;
}

void append_artifact_descriptor(std::ostringstream &os,
                                const KernelArtifactDescriptor &descriptor) {
  append_field(os, std::to_string(descriptor.stage_record_key));
  append_field(os, descriptor.manifest_ref);
  append_field(os, descriptor.abi_fingerprint);
  append_field(os, descriptor.artifact_key);
  append_kernel_descriptor(os, descriptor.kernel);
  append_field(os, kernel_artifact_payload_kind_to_string(
                        descriptor.payload_kind));
  append_field(os, descriptor.entry_point);
  append_field(os, descriptor.compile_options_key);
  append_field(os, std::to_string(descriptor.abi_arg_count));
  append_field(os, std::to_string(descriptor.abi_output_arg_count));
  append_field(os, std::to_string(descriptor.runtime_param_buffer_count));
  append_integral_vector(os, descriptor.runtime_param_i64_metadata);
  append_bool(os, descriptor.runtime_param_reduce_keep_dims);
  append_bool(os, descriptor.runtime_param_reduce_keep_dims_valid);
  append_launch_plan(os, descriptor.launch_plan);
  append_bool(os, descriptor.optional_cache_payload_allowed);
}

KernelArtifactDescriptor read_artifact_descriptor(WireReader &reader) {
  KernelArtifactDescriptor descriptor;
  descriptor.stage_record_key = reader.u64_field("artifact stage key");
  descriptor.manifest_ref = reader.string_field("artifact manifest ref");
  descriptor.abi_fingerprint = reader.string_field("artifact abi fingerprint");
  descriptor.artifact_key = reader.string_field("artifact key");
  descriptor.kernel = read_kernel_descriptor(reader);
  descriptor.payload_kind = kernel_artifact_payload_kind_from_string(
      reader.string_field("artifact payload kind"));
  descriptor.entry_point = reader.string_field("artifact entry point");
  descriptor.compile_options_key =
      reader.string_field("artifact compile options key");
  descriptor.abi_arg_count = reader.u32_field("artifact abi arg count");
  descriptor.abi_output_arg_count =
      reader.u32_field("artifact abi output arg count");
  descriptor.runtime_param_buffer_count =
      reader.u32_field("artifact runtime param buffer count");
  descriptor.runtime_param_i64_metadata =
      reader.i64_vector("artifact runtime param metadata");
  descriptor.runtime_param_reduce_keep_dims =
      reader.bool_field("artifact reduce keep dims");
  descriptor.runtime_param_reduce_keep_dims_valid =
      reader.bool_field("artifact reduce keep dims valid");
  descriptor.launch_plan = read_launch_plan(reader);
  descriptor.optional_cache_payload_allowed =
      reader.bool_field("artifact optional cache payload allowed");
  return descriptor;
}

void append_cache_key(std::ostringstream &os, const CacheKey &key) {
  append_field(os, key.model_fingerprint);
  append_field(os, key.manifest_hash);
  append_field(os, key.target_fingerprint);
  append_field(os, key.backend_capabilities_fingerprint);
  append_field(os, key.compiler_revision);
  append_field(os, key.backend_compiler_revision);
  append_field(os, key.driver_identity);
  append_field(os, key.compile_options_hash);
  append_vector(os, key.kernel_unit_versions);
  append_field(os, key.stable_key);
}

CacheKey read_cache_key(WireReader &reader) {
  CacheKey key;
  key.model_fingerprint = reader.string_field("cache key model fingerprint");
  key.manifest_hash = reader.string_field("cache key manifest hash");
  key.target_fingerprint = reader.string_field("cache key target fingerprint");
  key.backend_capabilities_fingerprint =
      reader.string_field("cache key capabilities fingerprint");
  key.compiler_revision = reader.string_field("cache key compiler revision");
  key.backend_compiler_revision =
      reader.string_field("cache key backend compiler revision");
  key.driver_identity = reader.string_field("cache key driver identity");
  key.compile_options_hash =
      reader.string_field("cache key compile options hash");
  key.kernel_unit_versions =
      reader.string_vector("cache key kernel unit versions");
  key.stable_key = reader.string_field("cache key stable key");
  return key;
}

void append_backend_payload(std::ostringstream &os,
                            const CacheBackendPayloadRecord &payload) {
  append_field(os, payload.artifact_key);
  append_field(os, payload.backend_domain);
  append_field(os, payload.payload_kind);
  append_field(os, payload.source_id);
  append_field(os, payload.entry_point);
  append_field(os, payload.payload_identity);
  append_bool(os, payload.optional);
}

CacheBackendPayloadRecord read_backend_payload(WireReader &reader) {
  CacheBackendPayloadRecord payload;
  payload.artifact_key = reader.string_field("payload artifact key");
  payload.backend_domain = reader.string_field("payload backend domain");
  payload.payload_kind = reader.string_field("payload kind");
  payload.source_id = reader.string_field("payload source id");
  payload.entry_point = reader.string_field("payload entry point");
  payload.payload_identity = reader.string_field("payload identity");
  payload.optional = reader.bool_field("payload optional");
  return payload;
}

std::string cache_envelope_file_name(std::string_view stable_key) {
  std::string sanitized;
  sanitized.reserve(stable_key.size());
  for (const char c : stable_key) {
    const bool allowed = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                         (c >= '0' && c <= '9') || c == '-' || c == '_';
    sanitized.push_back(allowed ? c : '_');
  }
  return sanitized + ".gfx-cache-envelope";
}

} // namespace

std::string make_model_cache_fingerprint(const ov::Model &model) {
  std::ostringstream material;
  append_field(material, model.get_friendly_name());
  append_field(material, std::to_string(model.inputs().size()));
  append_field(material, std::to_string(model.outputs().size()));
  const auto ordered_ops = model.get_ordered_ops();
  std::unordered_map<const ov::Node *, size_t> ordered_index;
  ordered_index.reserve(ordered_ops.size());
  for (size_t i = 0; i < ordered_ops.size(); ++i) {
    if (ordered_ops[i]) {
      ordered_index.emplace(ordered_ops[i].get(), i);
    }
  }
  append_field(material, std::to_string(ordered_ops.size()));
  for (const auto &node : ordered_ops) {
    if (!node) {
      append_field(material, "<null>");
      continue;
    }
    append_field(material, node->get_type_name());
    append_field(material, node->get_friendly_name());
    append_field(material, std::to_string(node->get_input_size()));
    append_field(material, std::to_string(node->get_output_size()));
    for (size_t i = 0; i < node->get_input_size(); ++i) {
      append_field(material, node->get_input_element_type(i).get_type_name());
      append_field(material, shape_to_string(node->get_input_partial_shape(i)));
      const auto source = node->input_value(i);
      const auto *source_node = source.get_node();
      const auto source_it = ordered_index.find(source_node);
      append_field(material, source_it == ordered_index.end()
                                 ? std::string("<external>")
                                 : std::to_string(source_it->second));
      append_field(material, std::to_string(source.get_index()));
      if (source_node) {
        append_field(material, source_node->get_type_name());
        append_field(material, source_node->get_friendly_name());
      }
    }
    for (size_t i = 0; i < node->get_output_size(); ++i) {
      append_field(material, node->get_output_element_type(i).get_type_name());
      append_field(material,
                   shape_to_string(node->get_output_partial_shape(i)));
    }
    if (const auto constant =
            ov::as_type_ptr<const ov::op::v0::Constant>(node)) {
      append_constant_payload_fingerprint(material, *constant);
    } else {
      ModelFingerprintAttributeVisitor visitor(material);
      try {
        append_field(material, "attributes");
        const bool visited = node->visit_attributes(visitor);
        append_bool(material, visited);
      } catch (const std::exception &ex) {
        append_field(material, "attribute_visit_failed");
        append_field(material, ex.what());
      }
    }
  }
  return hash_material(material.str());
}

std::string make_manifest_cache_hash(const ManifestBundle &manifest) {
  std::ostringstream material;
  append_field(material, std::to_string(manifest.schema_version));
  append_field(material, manifest.target_fingerprint);
  append_field(material, make_memory_plan_fingerprint(manifest.memory_plan));
  append_field(material, std::to_string(manifest.stages.size()));
  for (const auto &stage : manifest.stages) {
    append_field(material, std::to_string(stage.stage_id));
    append_field(material, std::to_string(stage.stable_record_key));
    append_field(material, stage.source_node_name);
    append_field(material, stage.normalized_op_family);
    append_field(material, lowering_route_kind_to_string(stage.execution_kind));
    append_field(material, stage.backend_domain);
    append_field(material, stage.kernel_unit_id);
    append_field(material, stage.kernel_unit_kind);
    append_field(material, stage.runtime_shape.rule);
    append_vector(material, stage.runtime_shape.i64_metadata);
    append_bool(material, stage.requires_runtime_shape_args);
    append_field(material, stage.dispatch.dispatch_source);
    append_field(material, stage.memory.alias_group);
    append_bool(material, stage.memory.hidden_host_copy_allowed);
    append_field(material, std::to_string(stage.submission.stage_weight));
    append_field(material, std::to_string(stage.submission.macs_estimate));
    append_bool(material, stage.submission.dependency_extension_boundary);
    append_field(material, std::to_string(stage.inputs.size()));
    for (const auto &tensor : stage.inputs) {
      append_field(material, tensor.logical_name);
      append_field(material, tensor.memory_region_id);
      append_field(material, tensor.element_type);
      append_field(material, tensor.partial_shape);
      append_field(material, tensor.layout);
      append_field(material, tensor.storage_kind);
      append_field(material, tensor.lifetime_class);
    }
    append_field(material, std::to_string(stage.outputs.size()));
    for (const auto &tensor : stage.outputs) {
      append_field(material, tensor.logical_name);
      append_field(material, tensor.memory_region_id);
      append_field(material, tensor.element_type);
      append_field(material, tensor.partial_shape);
      append_field(material, tensor.layout);
      append_field(material, tensor.storage_kind);
      append_field(material, tensor.lifetime_class);
    }
  }
  return hash_material(material.str());
}

std::string
make_executable_compile_options_hash(const ExecutableBundle &executable) {
  std::ostringstream material;
  append_field(material, executable.target_fingerprint);
  for (const auto &descriptor : executable.artifact_descriptors) {
    append_field(material, descriptor.artifact_key);
    append_field(material, descriptor.compile_options_key);
    append_field(material, descriptor.entry_point);
    append_field(material, descriptor.abi_fingerprint);
  }
  return hash_material(material.str());
}

std::vector<std::string>
make_kernel_unit_cache_versions(const ExecutableBundle &executable) {
  std::vector<std::string> versions;
  versions.reserve(executable.artifact_descriptors.size());
  for (const auto &descriptor : executable.artifact_descriptors) {
    std::ostringstream version;
    append_field(version, descriptor.kernel.kernel_id);
    append_field(version, descriptor.kernel.op_family);
    append_field(version, descriptor.kernel.backend_domain);
    append_field(version,
                 kernel_artifact_origin_to_string(descriptor.kernel.origin));
    append_field(version, kernel_artifact_payload_kind_to_string(
                              descriptor.payload_kind));
    append_field(version, descriptor.abi_fingerprint);
    versions.push_back(hash_material(version.str()));
  }
  return versions;
}

std::string
make_backend_capabilities_fingerprint(const BackendCapabilities &capabilities) {
  std::ostringstream material;
  append_field(material, capabilities.target().fingerprint());
  const auto &precision = capabilities.precision();
  append_bool(material, precision.supports_fp32);
  append_bool(material, precision.supports_fp16);
  append_bool(material, precision.supports_int8);
  const auto &artifact_formats = capabilities.artifact_formats();
  append_bool(material, artifact_formats.supports_compiled_model_export_import);
  const auto &fusion = capabilities.fusion();
  append_bool(material, fusion.enable_generic_attention_fusion);
  append_bool(material, fusion.supports_vendor_attention_stage);
  append_bool(material, fusion.enable_conv_activation_fusion);
  append_bool(material, fusion.enable_precision_sensitive_arithmetic_fusion);
  const auto &post_ops = capabilities.post_ops();
  append_bool(material, post_ops.enable_bias_fusion_for_convolution);
  append_bool(material, post_ops.enable_bias_fusion_for_group_convolution);
  append_bool(material, post_ops.enable_batchnorm_fusion_for_convolution);
  append_bool(material, post_ops.enable_batchnorm_fusion_for_group_convolution);
  append_bool(material, post_ops.enable_activation_fusion_for_convolution);
  append_bool(material,
              post_ops.enable_activation_fusion_for_group_convolution);
  append_bool(material, post_ops.enable_relu_activation_fusion);
  append_bool(material, post_ops.enable_sigmoid_activation_fusion);
  append_bool(material, post_ops.enable_tanh_activation_fusion);
  append_bool(material, post_ops.enable_elu_activation_fusion);
  append_bool(material, post_ops.enable_prelu_activation_fusion);
  append_bool(material, post_ops.enable_gelu_activation_fusion);
  append_bool(material, post_ops.enable_swish_activation_fusion);
  append_bool(material, post_ops.enable_hswish_activation_fusion);
  append_bool(material, post_ops.enable_hsigmoid_activation_fusion);
  append_bool(material, post_ops.enable_abs_activation_fusion);
  append_bool(material, post_ops.enable_sign_activation_fusion);
  const auto &execution = capabilities.execution();
  append_bool(material, execution.custom_kernel_dispatch_enabled);
  append_parallelism_profile(material,
                             execution.custom_kernel_dispatch_profile);
  append_bool(material, capabilities.stage_placement() != nullptr);
  return hash_material(material.str());
}

CacheEnvelopeVerificationResult
CacheEnvelope::verify(const ExecutableBundle &executable) const {
  CacheEnvelopeVerificationResult result;
  if (schema_version != kCacheEnvelopeSchemaVersion) {
    result.diagnostics.emplace_back("cache envelope schema version mismatch");
  }
  for (const auto &diagnostic : executable.verify().diagnostics) {
    result.diagnostics.push_back("executable: " + diagnostic);
  }
  for (const auto &diagnostic : manifest.verify().diagnostics) {
    result.diagnostics.push_back("manifest: " + diagnostic);
  }
  require_nonempty(result, key.model_fingerprint,
                   "cache key model fingerprint is empty");
  require_nonempty(result, key.manifest_hash,
                   "cache key manifest hash is empty");
  require_nonempty(result, key.target_fingerprint,
                   "cache key target fingerprint is empty");
  require_nonempty(result, key.backend_capabilities_fingerprint,
                   "cache key backend capabilities fingerprint is empty");
  require_nonempty(result, key.compiler_revision,
                   "cache key compiler revision is empty");
  require_nonempty(result, key.backend_compiler_revision,
                   "cache key backend compiler revision is empty");
  require_nonempty(result, key.driver_identity,
                   "cache key driver identity is empty");
  require_nonempty(result, key.compile_options_hash,
                   "cache key compile options hash is empty");
  require_nonempty(result, key.stable_key, "cache key stable key is empty");
  if (key.target_fingerprint != executable.target_fingerprint ||
      key.target_fingerprint != manifest.target_fingerprint) {
    result.diagnostics.emplace_back("cache key target fingerprint drift");
  }
  if (key.manifest_hash != make_manifest_cache_hash(executable.manifest) ||
      key.manifest_hash != make_manifest_cache_hash(manifest)) {
    result.diagnostics.emplace_back("cache key manifest hash drift");
  }
  if (key.compile_options_hash !=
      make_executable_compile_options_hash(executable)) {
    result.diagnostics.emplace_back("cache key compile options hash drift");
  }
  if (key.kernel_unit_versions != make_kernel_unit_cache_versions(executable)) {
    result.diagnostics.emplace_back("cache key kernel unit versions drift");
  }
  if (key.stable_key != make_cache_key_stable_key(key)) {
    result.diagnostics.emplace_back("cache key stable hash drift");
  }
  if (artifact_descriptors.size() != executable.artifact_descriptors.size()) {
    result.diagnostics.emplace_back("cache envelope artifact count drift");
  } else {
    for (size_t i = 0; i < artifact_descriptors.size(); ++i) {
      const auto &actual = artifact_descriptors[i];
      const auto &expected = executable.artifact_descriptors[i];
      if (actual.artifact_key != expected.artifact_key ||
          actual.abi_fingerprint != expected.abi_fingerprint ||
          actual.kernel.kernel_id != expected.kernel.kernel_id ||
          actual.payload_kind != expected.payload_kind) {
        result.diagnostics.push_back("cache envelope artifact drift at " +
                                     std::to_string(i));
      }
    }
  }
  for (size_t i = 0; i < backend_payloads.size(); ++i) {
    const auto &payload = backend_payloads[i];
    const auto *descriptor =
        find_artifact_descriptor(executable, payload.artifact_key);
    if (!descriptor) {
      result.diagnostics.push_back(
          "cache backend payload has unknown artifact at " + std::to_string(i));
      continue;
    }
    if (!has_artifact_key(executable, payload.artifact_key) ||
        payload.backend_domain != descriptor->kernel.backend_domain ||
        payload.payload_kind !=
            kernel_artifact_payload_kind_to_string(descriptor->payload_kind) ||
        payload.source_id.empty() || payload.entry_point.empty() ||
        payload.payload_identity.empty()) {
      result.diagnostics.push_back("cache backend payload identity drift at " +
                                   std::to_string(i));
    }
  }
  return result;
}

bool CacheEnvelope::valid(const ExecutableBundle &executable) const {
  return verify(executable).valid();
}

CacheEnvelope
CacheEnvelopeBuilder::build(const ExecutableBundle &executable,
                            const CacheEnvelopeBuildOptions &options) const {
  CacheEnvelope envelope;
  envelope.schema_version = kCacheEnvelopeSchemaVersion;
  envelope.manifest = executable.manifest;
  envelope.artifact_descriptors = executable.artifact_descriptors;
  envelope.key.model_fingerprint = options.model_fingerprint;
  envelope.key.manifest_hash = make_manifest_cache_hash(executable.manifest);
  envelope.key.target_fingerprint = executable.target_fingerprint;
  envelope.key.backend_capabilities_fingerprint =
      options.backend_capabilities_fingerprint;
  envelope.key.compiler_revision = options.compiler_revision;
  envelope.key.backend_compiler_revision = options.backend_compiler_revision;
  envelope.key.driver_identity = options.driver_identity;
  envelope.key.compile_options_hash =
      options.compile_options_hash.empty()
          ? make_executable_compile_options_hash(executable)
          : options.compile_options_hash;
  envelope.key.kernel_unit_versions =
      make_kernel_unit_cache_versions(executable);
  envelope.key.stable_key = make_cache_key_stable_key(envelope.key);

  if (options.include_optional_backend_payloads) {
    envelope.backend_payloads.reserve(executable.artifact_payloads.size());
    for (const auto &payload_record : executable.artifact_payloads) {
      envelope.backend_payloads.push_back(
          make_cache_payload_record(executable, payload_record));
    }
  }
  return envelope;
}

ArtifactCacheStore::ArtifactCacheStore(std::string cache_dir)
    : m_cache_dir(std::move(cache_dir)) {}

std::string ArtifactCacheStore::envelope_path(const CacheKey &key) const {
  if (m_cache_dir.empty() || key.stable_key.empty()) {
    return {};
  }
  return (std::filesystem::path(m_cache_dir) /
          cache_envelope_file_name(key.stable_key))
      .string();
}

ArtifactCacheStoreResult
ArtifactCacheStore::store(const CacheEnvelope &envelope) const {
  ArtifactCacheStoreResult result;
  result.cache_key = envelope.key.stable_key;
  if (!enabled()) {
    result.diagnostics.emplace_back("artifact cache store is disabled");
    return result;
  }
  if (envelope.key.stable_key.empty()) {
    result.diagnostics.emplace_back("cache envelope stable key is empty");
    return result;
  }

  result.location = envelope_path(envelope.key);
  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(m_cache_dir), ec);
  if (ec) {
    result.diagnostics.push_back("failed to create artifact cache directory: " +
                                 ec.message());
    return result;
  }

  std::ofstream out(result.location, std::ios::binary | std::ios::trunc);
  if (!out) {
    result.diagnostics.push_back("failed to open artifact cache envelope for write");
    return result;
  }
  const auto wire = serialize_cache_envelope(envelope);
  out.write(wire.data(), static_cast<std::streamsize>(wire.size()));
  if (!out) {
    result.diagnostics.push_back("failed to write artifact cache envelope");
    return result;
  }
  result.success = true;
  return result;
}

CacheEnvelopeWireResult ArtifactCacheStore::load(const CacheKey &key) const {
  CacheEnvelopeWireResult result;
  if (!enabled()) {
    result.diagnostics.emplace_back("artifact cache store is disabled");
    return result;
  }
  if (key.stable_key.empty()) {
    result.diagnostics.emplace_back("cache lookup stable key is empty");
    return result;
  }
  const auto path = envelope_path(key);
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    result.diagnostics.push_back("cache envelope miss: " + path);
    return result;
  }
  std::ostringstream wire;
  wire << in.rdbuf();
  result = deserialize_cache_envelope(wire.str());
  if (result.valid() && result.envelope.key.stable_key != key.stable_key) {
    result.diagnostics.emplace_back("cache envelope stable key mismatch");
  }
  return result;
}

std::string serialize_cache_envelope(const CacheEnvelope &envelope) {
  std::ostringstream os;
  append_field(os, "GFX_CACHE_ENVELOPE");
  append_field(os, "1");
  append_field(os, std::to_string(envelope.schema_version));
  append_cache_key(os, envelope.key);
  append_manifest(os, envelope.manifest);
  append_field(os, std::to_string(envelope.artifact_descriptors.size()));
  for (const auto &descriptor : envelope.artifact_descriptors) {
    append_artifact_descriptor(os, descriptor);
  }
  append_field(os, std::to_string(envelope.backend_payloads.size()));
  for (const auto &payload : envelope.backend_payloads) {
    append_backend_payload(os, payload);
  }
  return os.str();
}

CacheEnvelopeWireResult deserialize_cache_envelope(std::string_view wire) {
  CacheEnvelopeWireResult result;
  WireReader reader(wire);
  const auto magic = reader.string_field("cache envelope magic");
  const auto wire_version = reader.string_field("cache envelope wire version");
  if (magic != "GFX_CACHE_ENVELOPE") {
    result.diagnostics.emplace_back("cache envelope wire magic mismatch");
  }
  if (wire_version != "1") {
    result.diagnostics.emplace_back("cache envelope wire version mismatch");
  }
  result.envelope.schema_version =
      reader.u32_field("cache envelope schema version");
  result.envelope.key = read_cache_key(reader);
  result.envelope.manifest = read_manifest(reader);
  const auto artifact_count =
      reader.size_field("cache envelope artifact count");
  result.envelope.artifact_descriptors.reserve(artifact_count);
  for (size_t i = 0; i < artifact_count; ++i) {
    result.envelope.artifact_descriptors.push_back(
        read_artifact_descriptor(reader));
  }
  const auto payload_count = reader.size_field("cache envelope payload count");
  result.envelope.backend_payloads.reserve(payload_count);
  for (size_t i = 0; i < payload_count; ++i) {
    result.envelope.backend_payloads.push_back(read_backend_payload(reader));
  }
  auto read_diagnostics = reader.take_diagnostics();
  result.diagnostics.insert(result.diagnostics.end(),
                            read_diagnostics.begin(), read_diagnostics.end());
  return result;
}

ExecutableBundle
make_cache_envelope_executable_contract(const CacheEnvelope &envelope) {
  ExecutableBundle executable;
  executable.schema_version = 1;
  executable.target_fingerprint = envelope.key.target_fingerprint.empty()
                                      ? envelope.manifest.target_fingerprint
                                      : envelope.key.target_fingerprint;
  executable.manifest = envelope.manifest;
  executable.memory_plan = envelope.manifest.memory_plan;
  executable.artifact_descriptors = envelope.artifact_descriptors;
  executable.stages.reserve(envelope.manifest.stages.size());
  for (const auto &stage : envelope.manifest.stages) {
    ExecutableStageRecord record;
    record.stage_record_key = stage.stable_record_key;
    record.kernel_unit_id = stage.kernel_unit_id;
    record.kernel_unit_kind = stage.kernel_unit_kind;
    record.execution_kind = stage.execution_kind;
    auto it = std::find_if(
        executable.artifact_descriptors.begin(),
        executable.artifact_descriptors.end(),
        [&](const KernelArtifactDescriptor &descriptor) {
          return descriptor.stage_record_key == stage.stable_record_key;
        });
    record.artifact_descriptor_index =
        it == executable.artifact_descriptors.end()
            ? executable.artifact_descriptors.size()
            : static_cast<size_t>(
                  std::distance(executable.artifact_descriptors.begin(), it));
    executable.stages.push_back(std::move(record));
  }
  return executable;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
