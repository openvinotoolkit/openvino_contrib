// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "compiler/cache_envelope.hpp"
#include "compiler/executable_bundle.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/gfx_plugin/properties.hpp"
#include "plugin/compiled_model_cache_contract.hpp"
#include "plugin/gfx_property_lists.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool has_property(const std::vector<ov::PropertyName> &properties,
                  const std::string &name) {
  return std::any_of(properties.begin(), properties.end(),
                     [&](const ov::PropertyName &property) {
                       return static_cast<const std::string &>(property) == name;
                     });
}

std::string diagnostics_to_string(const std::vector<std::string> &diagnostics) {
  std::ostringstream os;
  for (const auto &diagnostic : diagnostics) {
    os << diagnostic << '\n';
  }
  return os.str();
}

class ScopedCacheContractDirectory {
public:
  ScopedCacheContractDirectory() {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::current_path() /
            ("ov_gfx_cache_contract_" + std::to_string(stamp));
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
    std::filesystem::create_directories(path_, ec);
    EXPECT_FALSE(ec) << ec.message();
  }

  ~ScopedCacheContractDirectory() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

compiler::TensorContract make_tensor_contract(std::string logical_name,
                                              std::string memory_region_id,
                                              compiler::TensorContractRole role) {
  compiler::TensorContract tensor;
  tensor.logical_name = std::move(logical_name);
  tensor.memory_region_id = std::move(memory_region_id);
  tensor.role = role;
  tensor.element_type = "f32";
  tensor.partial_shape = "{1,3,8,8}";
  tensor.layout = "nchw";
  tensor.storage_kind = "device_buffer";
  tensor.lifetime_class =
      role == compiler::TensorContractRole::TensorInput ? "external"
                                                        : "transient";
  return tensor;
}

compiler::MemoryRegion make_memory_region(std::string region_id,
                                           std::string logical_name,
                                           compiler::MemoryRegionKind kind,
                                           bool external) {
  compiler::MemoryRegion region;
  region.region_id = std::move(region_id);
  region.logical_tensor_name = std::move(logical_name);
  region.kind = kind;
  region.element_type = "f32";
  region.partial_shape = "{1,3,8,8}";
  region.layout = "nchw";
  region.storage_kind = "device_buffer";
  region.alias_group = region.region_id;
  region.lifetime = {0, 0};
  region.external_binding = external;
  region.host_visible = false;
  return region;
}

compiler::ManifestBundle make_cache_contract_manifest() {
  compiler::ManifestBundle manifest;
  manifest.schema_version = 2;
  manifest.target_fingerprint =
      "backend=metal;runtime=metal;family=apple;profile=apple_gpu;"
      "driver=metal-test;compiler=gfx-test;cache=cache-contract";

  auto input_region = make_memory_region(
      "stage_0.input_0", "input0", compiler::MemoryRegionKind::ExternalTensor,
      true);
  auto output_region =
      make_memory_region("stage_0.output_0", "output0",
                         compiler::MemoryRegionKind::TransientTensor, false);
  output_region.alias_group = "stage_0";
  manifest.memory_plan.regions = {input_region, output_region};

  compiler::AliasGroup input_alias;
  input_alias.group_id = "stage_0.input_0";
  input_alias.region_ids = {"stage_0.input_0"};
  compiler::AliasGroup output_alias;
  output_alias.group_id = "stage_0";
  output_alias.region_ids = {"stage_0.output_0"};
  manifest.memory_plan.alias_groups = {input_alias, output_alias};

  compiler::TransientArena arena;
  arena.arena_id = "transient_device_buffer_arena";
  arena.storage_kind = "device_buffer";
  arena.region_ids = {"stage_0.output_0"};
  manifest.memory_plan.transient_arenas = {arena};

  compiler::StageRecord stage;
  stage.stage_id = 0;
  stage.stable_record_key = 42;
  stage.source_node_name = "cache_contract_parameter";
  stage.normalized_op_family = "common_io";
  stage.execution_kind = compiler::LoweringRouteKind::Common;
  stage.backend_domain = "common";
  stage.kernel_unit_id = "common_io";
  stage.kernel_unit_kind = "common";
  stage.inputs = {make_tensor_contract(
      "input0", "stage_0.input_0", compiler::TensorContractRole::TensorInput)};
  stage.outputs = {make_tensor_contract(
      "output0", "stage_0.output_0",
      compiler::TensorContractRole::TensorOutput)};
  stage.dispatch.execution_kind = stage.execution_kind;
  stage.dispatch.backend_domain = stage.backend_domain;
  stage.dispatch.kernel_unit_id = stage.kernel_unit_id;
  stage.dispatch.kernel_unit_kind = stage.kernel_unit_kind;
  stage.dispatch.dispatch_source = "cache_contract";
  stage.memory.input_lifetime = "external";
  stage.memory.output_lifetime = "stage_output";
  stage.memory.alias_group = "stage_0";
  stage.submission.stage_weight = 1;
  manifest.stages = {stage};
  return manifest;
}

compiler::CacheEnvelope make_cache_contract_envelope(
    compiler::ExecutableBundle *executable_out = nullptr) {
  auto manifest = make_cache_contract_manifest();
  auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  EXPECT_TRUE(executable.verify().valid())
      << diagnostics_to_string(executable.verify().diagnostics);

  compiler::CacheEnvelopeBuildOptions options;
  options.model_fingerprint = "model-cache-contract";
  options.backend_capabilities_fingerprint = "capabilities-cache-contract";
  options.compiler_revision = "gfx-cache-contract-test";
  options.backend_compiler_revision = "backend-compiler-cache-contract";
  options.driver_identity = "driver-cache-contract";
  auto envelope = compiler::CacheEnvelopeBuilder{}.build(executable, options);
  EXPECT_TRUE(envelope.verify(executable).valid())
      << diagnostics_to_string(envelope.verify(executable).diagnostics);
  if (executable_out) {
    *executable_out = std::move(executable);
  }
  return envelope;
}

} // namespace

TEST(GfxCachePublicContractTest,
     OpenVinoCachePropertiesAreHiddenUntilSharedRoundTripExists) {
  ASSERT_FALSE(compiled_model_cache_roundtrip_supported());

  EXPECT_FALSE(has_property(gfx_plugin_supported_properties(),
                            ov::cache_dir.name()));
  EXPECT_FALSE(has_property(gfx_compiled_model_supported_properties(),
                            ov::cache_dir.name()));
  EXPECT_FALSE(has_property(gfx_internal_supported_properties(),
                            ov::internal::caching_properties.name()));
  EXPECT_FALSE(has_property(gfx_internal_supported_properties(),
                            ov::internal::cache_header_alignment.name()));
  EXPECT_TRUE(gfx_caching_properties().empty());
}

TEST(GfxCachePublicContractTest,
     CachePropertyContractDoesNotDependOnBackendAvailability) {
  const auto plugin_props = gfx_plugin_supported_properties();
  const auto compiled_props = gfx_compiled_model_supported_properties();
  const auto internal_props = gfx_internal_supported_properties();

  EXPECT_TRUE(has_property(plugin_props, ov::device::id.name()));
  EXPECT_TRUE(has_property(plugin_props, kGfxBackendProperty));
  EXPECT_TRUE(has_property(compiled_props, kGfxBackendProperty));
  EXPECT_TRUE(has_property(internal_props,
                           ov::internal::compiled_model_runtime_properties.name()));
  EXPECT_FALSE(has_property(plugin_props, ov::cache_dir.name()));
  EXPECT_FALSE(has_property(compiled_props, ov::cache_dir.name()));
}

TEST(GfxCachePublicContractTest,
     CacheEnvelopeWireRoundTripPreservesExecutableContract) {
  compiler::ExecutableBundle executable;
  const auto envelope = make_cache_contract_envelope(&executable);

  const auto wire = compiler::serialize_cache_envelope(envelope);
  const auto parsed = compiler::deserialize_cache_envelope(wire);
  ASSERT_TRUE(parsed.valid())
      << diagnostics_to_string(parsed.diagnostics);

  EXPECT_EQ(parsed.envelope.key.stable_key, envelope.key.stable_key);
  EXPECT_EQ(parsed.envelope.key.manifest_hash, envelope.key.manifest_hash);
  EXPECT_EQ(parsed.envelope.key.target_fingerprint,
            envelope.key.target_fingerprint);
  EXPECT_EQ(parsed.envelope.artifact_descriptors.size(),
            envelope.artifact_descriptors.size());
  EXPECT_EQ(compiler::serialize_cache_envelope(parsed.envelope), wire);

  const auto reconstructed =
      compiler::make_cache_envelope_executable_contract(parsed.envelope);
  EXPECT_TRUE(reconstructed.verify().valid())
      << diagnostics_to_string(reconstructed.verify().diagnostics);
  EXPECT_TRUE(parsed.envelope.verify(reconstructed).valid())
      << diagnostics_to_string(parsed.envelope.verify(reconstructed).diagnostics);
}

TEST(GfxCachePublicContractTest, CorruptCacheEnvelopeWireIsRejected) {
  const auto parsed =
      compiler::deserialize_cache_envelope("not-a-cache-envelope");
  EXPECT_FALSE(parsed.valid());
  EXPECT_FALSE(parsed.diagnostics.empty());
}

TEST(GfxCachePublicContractTest,
     ArtifactCacheStoreRoundTripUsesStableEnvelopeKey) {
  compiler::ExecutableBundle executable;
  const auto envelope = make_cache_contract_envelope(&executable);
  const ScopedCacheContractDirectory cache_dir;

  const compiler::ArtifactCacheStore store(cache_dir.path().string());
  const auto store_result = store.store(envelope);
  ASSERT_TRUE(store_result.success)
      << diagnostics_to_string(store_result.diagnostics);

  const auto loaded = store.load(envelope.key);
  ASSERT_TRUE(loaded.valid()) << diagnostics_to_string(loaded.diagnostics);
  EXPECT_EQ(loaded.envelope.key.stable_key, envelope.key.stable_key);
  const auto reconstructed =
      compiler::make_cache_envelope_executable_contract(loaded.envelope);
  EXPECT_TRUE(loaded.envelope.verify(reconstructed).valid())
      << diagnostics_to_string(loaded.envelope.verify(reconstructed).diagnostics);

  auto missing_key = envelope.key;
  missing_key.stable_key = "missing-cache-contract-key";
  const auto miss = store.load(missing_key);
  EXPECT_FALSE(miss.valid());
  EXPECT_FALSE(miss.diagnostics.empty());
}

} // namespace gfx_plugin
} // namespace ov
