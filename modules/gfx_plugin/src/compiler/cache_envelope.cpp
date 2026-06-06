// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler/cache_envelope.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
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

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
