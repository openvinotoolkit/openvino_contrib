// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "compiler/executable_bundle.hpp"
#include "compiler/operation_support.hpp"
#include "openvino/core/model.hpp"
#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

struct CacheKey {
  std::string model_fingerprint;
  std::string manifest_hash;
  std::string target_fingerprint;
  std::string backend_capabilities_fingerprint;
  std::string compiler_revision;
  std::string backend_compiler_revision;
  std::string driver_identity;
  std::string compile_options_hash;
  std::vector<std::string> kernel_unit_versions;
  std::string stable_key;
};

struct CacheBackendPayloadRecord {
  std::string artifact_key;
  std::string backend_domain;
  std::string payload_kind;
  std::string source_id;
  std::string entry_point;
  std::string payload_identity;
  std::string source_language;
  std::string source;
  std::string payload_format;
  std::string payload_data;
  bool optional = true;
};

using CacheBackendPayloadEncoder =
    std::function<CacheBackendPayloadRecord(
        const KernelArtifactDescriptor &descriptor,
        const KernelArtifactPayloadRecord &payload_record)>;

using CacheBackendPayloadDecoder =
    std::function<std::shared_ptr<const KernelArtifactPayload>(
        const CacheBackendPayloadRecord &payload,
        const KernelArtifactDescriptor &descriptor)>;

struct CacheEnvelopeVerificationResult {
  std::vector<std::string> diagnostics;

  bool valid() const noexcept { return diagnostics.empty(); }
};

struct CacheMaterializationContract {
  bool finalized = false;
  std::vector<PipelineStageMaterializationPlan> stages;
  std::vector<RuntimePublicOutputDescriptor> public_outputs;
  PipelineStageRuntimeOptionsPlan runtime_options;

  bool empty() const noexcept {
    return !finalized && stages.empty() && public_outputs.empty();
  }
};

struct ArtifactCacheStoreResult {
  bool success = false;
  std::string cache_key;
  std::string location;
  std::vector<std::string> diagnostics;
};

struct CacheEnvelope {
  uint32_t schema_version = 1;
  CacheKey key;
  ManifestBundle manifest;
  std::vector<KernelArtifactDescriptor> artifact_descriptors;
  std::vector<CacheBackendPayloadRecord> backend_payloads;
  CacheMaterializationContract materialization;

  CacheEnvelopeVerificationResult verify(const ExecutableBundle &executable) const;
  bool valid(const ExecutableBundle &executable) const;
};

struct CacheEnvelopeWireResult {
  CacheEnvelope envelope;
  std::vector<std::string> diagnostics;

  bool valid() const noexcept { return diagnostics.empty(); }
};

struct CacheEnvelopeBuildOptions {
  std::string model_fingerprint;
  std::string backend_capabilities_fingerprint;
  std::string compiler_revision = "gfx-compiler-cache-v1";
  std::string backend_compiler_revision;
  std::string driver_identity;
  std::string compile_options_hash;
  bool include_optional_backend_payloads = true;
  CacheBackendPayloadEncoder backend_payload_encoder = {};
};

class CacheEnvelopeBuilder final {
public:
  CacheEnvelope build(const ExecutableBundle &executable,
                      const RuntimeExecutableDescriptor &runtime_descriptor,
                      const CacheEnvelopeBuildOptions &options) const;
};

class ArtifactCacheStore final {
public:
  explicit ArtifactCacheStore(std::string cache_dir);

  bool enabled() const noexcept { return !m_cache_dir.empty(); }
  const std::string &cache_dir() const noexcept { return m_cache_dir; }

  ArtifactCacheStoreResult store(const CacheEnvelope &envelope) const;
  CacheEnvelopeWireResult load(const CacheKey &key) const;

private:
  std::string envelope_path(const CacheKey &key) const;

  std::string m_cache_dir;
};

std::string make_model_cache_fingerprint(const ov::Model &model);
std::string make_manifest_cache_hash(const ManifestBundle &manifest);
std::string make_executable_compile_options_hash(const ExecutableBundle &executable);
std::string make_backend_capabilities_fingerprint(const BackendCapabilities &capabilities);
std::vector<std::string> make_kernel_unit_cache_versions(const ExecutableBundle &executable);

std::string serialize_cache_envelope(const CacheEnvelope &envelope);
CacheEnvelopeWireResult deserialize_cache_envelope(std::string_view wire);
ExecutableBundle
make_cache_envelope_executable_contract(
    const CacheEnvelope &envelope,
    CacheBackendPayloadDecoder backend_payload_decoder = {});
RuntimeExecutableDescriptor
make_cache_envelope_runtime_descriptor_contract(
    const CacheEnvelope &envelope,
    const ExecutableBundle &executable);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
