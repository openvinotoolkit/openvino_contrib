// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "backends/metal/compiler/metal_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "compiler/cache_envelope.hpp"
#include "compiler/cache_import.hpp"
#include "compiler/cache_repository.hpp"
#include "compiler/executable_bundle.hpp"
#include "kernel_ir/gfx_kernel_source.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "openvino/gfx_plugin/properties.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "plugin/compiled_model_cache_contract.hpp"
#include "plugin/gfx_property_lists.hpp"
#include "runtime/executable_descriptor.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool has_property(const std::vector<ov::PropertyName> &properties,
                  const std::string &name) {
  return std::any_of(properties.begin(), properties.end(),
                     [&](const ov::PropertyName &property) {
                       return static_cast<const std::string &>(property) ==
                              name;
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

compiler::TensorContract
make_tensor_contract(std::string logical_name, std::string memory_region_id,
                     compiler::TensorContractRole role) {
  compiler::TensorContract tensor;
  tensor.logical_name = std::move(logical_name);
  tensor.memory_region_id = std::move(memory_region_id);
  tensor.role = role;
  tensor.element_type = "f32";
  tensor.partial_shape = "{1,3,8,8}";
  tensor.layout = "nchw";
  tensor.storage_kind = "device_buffer";
  tensor.lifetime_class = role == compiler::TensorContractRole::TensorInput
                              ? "external"
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

  auto input_region =
      make_memory_region("stage_0.input_0", "input0",
                         compiler::MemoryRegionKind::ExternalTensor, true);
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
  stage.outputs = {
      make_tensor_contract("output0", "stage_0.output_0",
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

std::shared_ptr<ov::Model> make_cache_contract_model(std::string name) {
  auto parameter = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::PartialShape{1, 3, 8, 8});
  parameter->set_friendly_name("input0");
  auto result = std::make_shared<ov::op::v0::Result>(parameter);
  result->set_friendly_name("output0");
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{parameter},
                                     std::move(name));
}

compiler::BackendCapabilities
make_cache_contract_capabilities(const compiler::BackendTarget &target) {
  compiler::ArtifactFormatCapabilities artifact_formats;
  artifact_formats.supports_compiled_model_export_import = true;
  return compiler::BackendCapabilities(target, {}, {}, {}, {}, {}, {},
                                       artifact_formats);
}

compiler::CacheEnvelope make_cache_source_contract_envelope_for_target(
    const compiler::BackendTarget &target,
    compiler::ExecutableBundle *executable_out = nullptr) {
  auto manifest = make_cache_contract_manifest();
  manifest.target_fingerprint = target.fingerprint();
  manifest.stages[0].execution_kind =
      compiler::LoweringRouteKind::GeneratedKernel;
  manifest.stages[0].backend_domain = target.backend_id();
  manifest.stages[0].kernel_unit_id = "cache_contract_kernel";
  manifest.stages[0].kernel_unit_kind = "generated";
  manifest.stages[0].dispatch.execution_kind =
      manifest.stages[0].execution_kind;
  manifest.stages[0].dispatch.backend_domain =
      manifest.stages[0].backend_domain;
  manifest.stages[0].dispatch.kernel_unit_id =
      manifest.stages[0].kernel_unit_id;
  manifest.stages[0].dispatch.kernel_unit_kind =
      manifest.stages[0].kernel_unit_kind;
  manifest.stages[0].dispatch.dispatch_source = "mlir_codegen";

  auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  EXPECT_EQ(executable.artifact_descriptors.size(), 1u);
  if (executable.artifact_descriptors.empty()) {
    return {};
  }
  executable.artifact_descriptors[0].launch_plan.valid = true;
  executable.artifact_descriptors[0].launch_plan.buffer_roles = {
      "tensor_input", "tensor_output"};
  compiler::KernelArtifactPayloadRecord payload;
  payload.artifact_descriptor_index = 0;
  payload.artifact_key = executable.artifact_descriptors[0].artifact_key;
  const auto source_language =
      target.backend() == GpuBackend::Metal
          ? GfxKernelSourceLanguage::MetalShadingLanguage
          : GfxKernelSourceLanguage::OpenCL;
  payload.payload = std::make_shared<GfxKernelSourcePayload>(
      "cache_contract_kernel", target.backend_id(), "cache_contract_kernel",
      source_language, "kernel void cache_contract_kernel() {}");
  executable.artifact_payloads.push_back(std::move(payload));

  EXPECT_TRUE(executable.verify().valid())
      << diagnostics_to_string(executable.verify().diagnostics);

  compiler::CacheEnvelopeBuildOptions options;
  options.model_fingerprint = "model-cache-source-contract";
  options.backend_capabilities_fingerprint =
      "capabilities-cache-source-contract";
  options.compiler_revision = "gfx-cache-source-contract-test";
  options.backend_compiler_revision = "backend-compiler-cache-source-contract";
  options.driver_identity = "driver-cache-source-contract";
  auto envelope = compiler::CacheEnvelopeBuilder{}.build(executable, options);
  EXPECT_TRUE(envelope.verify(executable).valid())
      << diagnostics_to_string(envelope.verify(executable).diagnostics);
  if (executable_out) {
    *executable_out = std::move(executable);
  }
  return envelope;
}

compiler::CacheEnvelope make_cache_source_contract_envelope_for_request(
    const compiler::BackendTarget &target, const ov::Model &model,
    const compiler::BackendCapabilities &capabilities,
    compiler::ExecutableBundle *executable_out = nullptr) {
  auto manifest = make_cache_contract_manifest();
  manifest.target_fingerprint = target.fingerprint();
  manifest.stages[0].execution_kind =
      compiler::LoweringRouteKind::GeneratedKernel;
  manifest.stages[0].backend_domain = target.backend_id();
  manifest.stages[0].kernel_unit_id = "cache_contract_kernel";
  manifest.stages[0].kernel_unit_kind = "generated";
  manifest.stages[0].dispatch.execution_kind =
      manifest.stages[0].execution_kind;
  manifest.stages[0].dispatch.backend_domain =
      manifest.stages[0].backend_domain;
  manifest.stages[0].dispatch.kernel_unit_id =
      manifest.stages[0].kernel_unit_id;
  manifest.stages[0].dispatch.kernel_unit_kind =
      manifest.stages[0].kernel_unit_kind;
  manifest.stages[0].dispatch.dispatch_source = "mlir_codegen";

  auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  EXPECT_EQ(executable.artifact_descriptors.size(), 1u);
  if (executable.artifact_descriptors.empty()) {
    return {};
  }
  executable.artifact_descriptors[0].launch_plan.valid = true;
  executable.artifact_descriptors[0].launch_plan.buffer_roles = {
      "tensor_input", "tensor_output"};
  compiler::KernelArtifactPayloadRecord payload;
  payload.artifact_descriptor_index = 0;
  payload.artifact_key = executable.artifact_descriptors[0].artifact_key;
  const auto source_language =
      target.backend() == GpuBackend::Metal
          ? GfxKernelSourceLanguage::MetalShadingLanguage
          : GfxKernelSourceLanguage::OpenCL;
  payload.payload = std::make_shared<GfxKernelSourcePayload>(
      "cache_contract_kernel", target.backend_id(), "cache_contract_kernel",
      source_language, "kernel void cache_contract_kernel() {}");
  executable.artifact_payloads.push_back(std::move(payload));

  EXPECT_TRUE(executable.verify().valid())
      << diagnostics_to_string(executable.verify().diagnostics);

  compiler::CacheEnvelopeBuildOptions options;
  options.model_fingerprint = compiler::make_model_cache_fingerprint(model);
  options.backend_capabilities_fingerprint =
      compiler::make_backend_capabilities_fingerprint(capabilities);
  options.backend_compiler_revision = target.compiler_id();
  options.driver_identity = target.driver_id();
  auto envelope = compiler::CacheEnvelopeBuilder{}.build(executable, options);
  EXPECT_TRUE(envelope.verify(executable).valid())
      << diagnostics_to_string(envelope.verify(executable).diagnostics);
  if (executable_out) {
    *executable_out = std::move(executable);
  }
  return envelope;
}

compiler::CacheEnvelope make_cache_source_contract_envelope(
    compiler::ExecutableBundle *executable_out = nullptr) {
  return make_cache_source_contract_envelope_for_target(
      compiler::BackendTarget::from_backend(GpuBackend::Metal), executable_out);
}

GfxMpsrtTensorDesc make_cache_mpsrt_tensor_desc(uint64_t rows,
                                                uint64_t columns,
                                                uint32_t flags) {
  GfxMpsrtTensorDesc desc;
  desc.rank = 2;
  desc.dims[0] = rows;
  desc.dims[1] = columns;
  desc.strides[0] = static_cast<int64_t>(columns);
  desc.strides[1] = 1;
  desc.dtype = GfxMpsrtDType::F32;
  desc.storage = GfxMpsrtStorage::Matrix;
  desc.layout = GfxMpsrtLayout::RowMajor;
  desc.flags = flags;
  desc.byte_length = rows * columns * sizeof(float);
  desc.matrix_rows = static_cast<uint32_t>(rows);
  desc.matrix_columns = static_cast<uint32_t>(columns);
  desc.matrix_row_bytes = static_cast<uint32_t>(columns * sizeof(float));
  desc.matrix_count = 1;
  return desc;
}

GfxAppleMpsVendorPrimitiveContract make_cache_mps_gemm_contract() {
  GfxAppleMpsVendorPrimitiveContract contract;
  contract.valid = true;
  contract.descriptor.kind = GfxAppleMpsVendorPrimitiveKind::Gemm;
  contract.descriptor.gemm.transpose_lhs = 0;
  contract.descriptor.gemm.transpose_rhs = 1;
  contract.descriptor.gemm.accumulate_fp32 = 1;
  contract.descriptor.gemm.alpha = 1.25f;
  contract.descriptor.gemm.beta = 0.0f;
  contract.semantic_input_roles = {GfxKernelBufferRole::TensorInput,
                                   GfxKernelBufferRole::TensorInput};
  contract.input_descs = {
      make_cache_mpsrt_tensor_desc(2, 4, GfxMpsrtTensorFlagExternalIo),
      make_cache_mpsrt_tensor_desc(3, 4, GfxMpsrtTensorFlagExternalIo)};
  contract.output_descs = {
      make_cache_mpsrt_tensor_desc(2, 3, GfxMpsrtTensorFlagExternalIo)};
  contract.external_buffer_abi = gfx_mpsrt_make_external_buffer_abi_from_roles(
      {GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorInput,
       GfxMpsrtExternalBufferRole::TensorOutput});
  return contract;
}

} // namespace

TEST(GfxCachePublicContractTest,
     OpenVinoCachePropertiesAreVisibleAfterSharedRoundTripExists) {
  ASSERT_TRUE(compiled_model_cache_roundtrip_supported());

  EXPECT_TRUE(
      has_property(gfx_plugin_supported_properties(), ov::cache_dir.name()));
  EXPECT_TRUE(has_property(gfx_compiled_model_supported_properties(),
                           ov::cache_dir.name()));
  EXPECT_TRUE(has_property(gfx_internal_supported_properties(),
                           ov::internal::caching_properties.name()));
  EXPECT_TRUE(has_property(gfx_internal_supported_properties(),
                           ov::internal::cache_header_alignment.name()));
  EXPECT_FALSE(gfx_caching_properties().empty());
}

TEST(GfxCachePublicContractTest,
     CachePropertyContractDoesNotDependOnBackendAvailability) {
  const auto plugin_props = gfx_plugin_supported_properties();
  const auto compiled_props = gfx_compiled_model_supported_properties();
  const auto internal_props = gfx_internal_supported_properties();

  EXPECT_TRUE(has_property(plugin_props, ov::device::id.name()));
  EXPECT_TRUE(has_property(plugin_props, kGfxBackendProperty));
  EXPECT_TRUE(has_property(compiled_props, kGfxBackendProperty));
  EXPECT_TRUE(has_property(
      internal_props, ov::internal::compiled_model_runtime_properties.name()));
  EXPECT_TRUE(has_property(plugin_props, ov::cache_dir.name()));
  EXPECT_TRUE(has_property(compiled_props, ov::cache_dir.name()));
}

TEST(GfxCachePublicContractTest,
     CacheEnvelopeWireRoundTripPreservesExecutableContract) {
  compiler::ExecutableBundle executable;
  const auto envelope = make_cache_contract_envelope(&executable);

  const auto wire = compiler::serialize_cache_envelope(envelope);
  const auto parsed = compiler::deserialize_cache_envelope(wire);
  ASSERT_TRUE(parsed.valid()) << diagnostics_to_string(parsed.diagnostics);

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
      << diagnostics_to_string(
             parsed.envelope.verify(reconstructed).diagnostics);
}

TEST(GfxCachePublicContractTest,
     CacheEnvelopeWireRoundTripPreservesSourcePayloadContract) {
  compiler::ExecutableBundle executable;
  const auto envelope = make_cache_source_contract_envelope(&executable);
  ASSERT_EQ(envelope.backend_payloads.size(), 1u);

  const auto wire = compiler::serialize_cache_envelope(envelope);
  const auto parsed = compiler::deserialize_cache_envelope(wire);
  ASSERT_TRUE(parsed.valid()) << diagnostics_to_string(parsed.diagnostics);
  ASSERT_EQ(parsed.envelope.backend_payloads.size(), 1u);
  EXPECT_EQ(parsed.envelope.backend_payloads[0].source_language,
            "metal_shading_language");
  EXPECT_EQ(parsed.envelope.backend_payloads[0].source,
            "kernel void cache_contract_kernel() {}");

  const auto reconstructed =
      compiler::make_cache_envelope_executable_contract(parsed.envelope);
  EXPECT_TRUE(reconstructed.verify().valid())
      << diagnostics_to_string(reconstructed.verify().diagnostics);
  ASSERT_EQ(reconstructed.artifact_payloads.size(), 1u);
  EXPECT_TRUE(reconstructed.artifact_payloads[0].payload);
  EXPECT_EQ(reconstructed.artifact_payloads[0].payload->payload_kind(),
            KernelArtifactPayloadKind::MslSource);
  EXPECT_TRUE(parsed.envelope.verify(reconstructed).valid())
      << diagnostics_to_string(
             parsed.envelope.verify(reconstructed).diagnostics);
}

TEST(GfxCachePublicContractTest,
     OpenClCachePayloadCodecKeepsBackendArtifactOutOfSharedEnvelope) {
  auto manifest = make_cache_contract_manifest();
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  manifest.target_fingerprint = target.fingerprint();
  manifest.stages[0].execution_kind =
      compiler::LoweringRouteKind::GeneratedKernel;
  manifest.stages[0].backend_domain = "opencl";
  manifest.stages[0].kernel_unit_id = "opencl/generated/eltwise";
  manifest.stages[0].kernel_unit_kind = "generated";
  manifest.stages[0].dispatch.execution_kind =
      manifest.stages[0].execution_kind;
  manifest.stages[0].dispatch.backend_domain = manifest.stages[0].backend_domain;
  manifest.stages[0].dispatch.kernel_unit_id = manifest.stages[0].kernel_unit_id;
  manifest.stages[0].dispatch.kernel_unit_kind =
      manifest.stages[0].kernel_unit_kind;
  manifest.stages[0].dispatch.dispatch_source = "opencl_cache_contract";

  auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_EQ(executable.artifact_descriptors.size(), 1u);

  auto artifact = make_opencl_source_artifact(
      make_opencl_source_manifest(GfxKernelStageFamily::Eltwise,
                                  "cache_contract",
                                  "gfx_opencl_generated_eltwise_identity_f32",
                                  /*direct_inputs=*/1,
                                  /*scalar_arg_count=*/2,
                                  /*direct_outputs=*/1),
      "opencl/generated/eltwise",
      "__kernel void gfx_opencl_generated_eltwise_identity_f32() {}",
      {GfxOpenClSourceScalarArg::ElementCount,
       GfxOpenClSourceScalarArg::StaticU32},
      {0}, GfxOpenClArtifactOp::Identity,
      GfxOpenClArtifactInputMode::Direct, 0.0f, {7});
  artifact.build_options = {"-cl-std=CL1.2"};
  artifact.artifact_ref.entry_point =
      "gfx_opencl_generated_eltwise_identity_f32";
  artifact.artifact_ref.build_options = artifact.build_options;
  ASSERT_TRUE(compiler::finalize_opencl_kernel_artifact_descriptor_contract(
      executable.artifact_descriptors[0], artifact));

  compiler::KernelArtifactPayloadRecord payload;
  payload.artifact_descriptor_index = 0;
  payload.artifact_key = executable.artifact_descriptors[0].artifact_key;
  payload.payload =
      std::make_shared<GfxOpenClSourceArtifactPayload>(std::move(artifact));
  executable.artifact_payloads.push_back(std::move(payload));
  ASSERT_TRUE(executable.verify().valid())
      << diagnostics_to_string(executable.verify().diagnostics);

  compiler::CacheEnvelopeBuildOptions options;
  options.model_fingerprint = "model-cache-opencl-codec";
  options.backend_capabilities_fingerprint = "capabilities-cache-opencl-codec";
  options.compiler_revision = "gfx-cache-opencl-codec-test";
  options.backend_compiler_revision = target.compiler_id();
  options.driver_identity = target.driver_id();
  options.backend_payload_encoder =
      compiler::make_opencl_cache_payload_encoder();
  const auto envelope = compiler::CacheEnvelopeBuilder{}.build(executable, options);
  ASSERT_TRUE(envelope.verify(executable).valid())
      << diagnostics_to_string(envelope.verify(executable).diagnostics);
  ASSERT_EQ(envelope.backend_payloads.size(), 1u);
  EXPECT_EQ(envelope.backend_payloads[0].payload_format,
            "gfx.opencl.source_artifact.v1");
  EXPECT_FALSE(envelope.backend_payloads[0].payload_data.empty());

  const auto wire = compiler::serialize_cache_envelope(envelope);
  const auto parsed = compiler::deserialize_cache_envelope(wire);
  ASSERT_TRUE(parsed.valid()) << diagnostics_to_string(parsed.diagnostics);
  const auto reconstructed = compiler::make_cache_envelope_executable_contract(
      parsed.envelope, compiler::make_opencl_cache_payload_decoder());
  ASSERT_TRUE(reconstructed.verify().valid())
      << diagnostics_to_string(reconstructed.verify().diagnostics);
  ASSERT_EQ(reconstructed.artifact_payloads.size(), 1u);
  const auto *opencl_payload =
      dynamic_cast<const GfxOpenClSourceArtifactPayload *>(
          reconstructed.artifact_payloads[0].payload.get());
  ASSERT_NE(opencl_payload, nullptr);
  EXPECT_EQ(opencl_payload->artifact().static_u32_scalars,
            std::vector<uint32_t>{7});
  EXPECT_EQ(opencl_payload->artifact().build_options,
            std::vector<std::string>{"-cl-std=CL1.2"});
}

TEST(GfxCachePublicContractTest,
     MetalCachePayloadCodecPreservesVendorDescriptorContract) {
  auto manifest = make_cache_contract_manifest();
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  manifest.target_fingerprint = target.fingerprint();
  manifest.stages[0].normalized_op_family = "MatMul";
  manifest.stages[0].execution_kind =
      compiler::LoweringRouteKind::VendorPrimitive;
  manifest.stages[0].backend_domain = "metal";
  manifest.stages[0].kernel_unit_id = "metal/vendor/mps_gemm";
  manifest.stages[0].kernel_unit_kind = "vendor";
  manifest.stages[0].dispatch.execution_kind =
      manifest.stages[0].execution_kind;
  manifest.stages[0].dispatch.backend_domain = manifest.stages[0].backend_domain;
  manifest.stages[0].dispatch.kernel_unit_id = manifest.stages[0].kernel_unit_id;
  manifest.stages[0].dispatch.kernel_unit_kind =
      manifest.stages[0].kernel_unit_kind;
  manifest.stages[0].dispatch.dispatch_source = "metal_vendor_descriptor";

  auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
  auto &descriptor = executable.artifact_descriptors[0];
  descriptor.entry_point = "mps_gemm";
  descriptor.compile_options_key = "metal_vendor_descriptor";
  descriptor.abi_arg_count = 3;
  descriptor.abi_output_arg_count = 1;
  compiler::finalize_kernel_artifact_descriptor_identity(descriptor);

  auto contract = make_cache_mps_gemm_contract();
  compiler::KernelArtifactPayloadRecord payload;
  payload.artifact_descriptor_index = 0;
  payload.artifact_key = descriptor.artifact_key;
  payload.payload = std::make_shared<GfxMetalVendorPrimitiveArtifactPayload>(
      descriptor.kernel.kernel_id, descriptor.kernel.backend_domain,
      descriptor.entry_point, contract);
  executable.artifact_payloads.push_back(std::move(payload));
  ASSERT_TRUE(executable.verify().valid())
      << diagnostics_to_string(executable.verify().diagnostics);

  compiler::CacheEnvelopeBuildOptions options;
  options.model_fingerprint = "model-cache-metal-vendor-codec";
  options.backend_capabilities_fingerprint = "capabilities-cache-metal-codec";
  options.compiler_revision = "gfx-cache-metal-codec-test";
  options.backend_compiler_revision = target.compiler_id();
  options.driver_identity = target.driver_id();
  options.backend_payload_encoder = compiler::make_metal_cache_payload_encoder();
  const auto envelope = compiler::CacheEnvelopeBuilder{}.build(executable, options);
  ASSERT_TRUE(envelope.verify(executable).valid())
      << diagnostics_to_string(envelope.verify(executable).diagnostics);
  ASSERT_EQ(envelope.backend_payloads.size(), 1u);
  EXPECT_EQ(envelope.backend_payloads[0].payload_format,
            "gfx.metal.vendor_descriptor.v1");
  EXPECT_FALSE(envelope.backend_payloads[0].payload_data.empty());

  const auto wire = compiler::serialize_cache_envelope(envelope);
  const auto parsed = compiler::deserialize_cache_envelope(wire);
  ASSERT_TRUE(parsed.valid()) << diagnostics_to_string(parsed.diagnostics);
  const auto reconstructed = compiler::make_cache_envelope_executable_contract(
      parsed.envelope, compiler::make_metal_cache_payload_decoder());
  ASSERT_TRUE(reconstructed.verify().valid())
      << diagnostics_to_string(reconstructed.verify().diagnostics);
  ASSERT_EQ(reconstructed.artifact_payloads.size(), 1u);
  const auto *metal_payload =
      dynamic_cast<const GfxMetalVendorPrimitiveArtifactPayload *>(
          reconstructed.artifact_payloads[0].payload.get());
  ASSERT_NE(metal_payload, nullptr);
  ASSERT_TRUE(metal_payload->valid());
  const auto &decoded_contract = metal_payload->contract();
  EXPECT_EQ(decoded_contract.descriptor.kind,
            GfxAppleMpsVendorPrimitiveKind::Gemm);
  EXPECT_EQ(decoded_contract.descriptor.gemm.transpose_rhs, 1u);
  EXPECT_FLOAT_EQ(decoded_contract.descriptor.gemm.alpha, 1.25f);
  EXPECT_EQ(decoded_contract.input_descs.size(), 2u);
  EXPECT_EQ(decoded_contract.output_descs.size(), 1u);
  EXPECT_EQ(decoded_contract.external_buffer_abi.buffer_roles.size(), 3u);
  EXPECT_EQ(decoded_contract.external_buffer_abi.buffer_roles[2],
            GfxMpsrtExternalBufferRole::TensorOutput);
}

TEST(GfxCachePublicContractTest,
     CacheImportContractMaterializesDescriptorOwnedPipeline) {
  const auto &registry = compiler::BackendRegistry::default_registry();
  const auto targets = registry.available_targets();
  if (targets.empty()) {
    const auto envelope = make_cache_source_contract_envelope_for_target(
        compiler::BackendTarget::from_backend(GpuBackend::Metal));
    const auto import_contract =
        compiler::make_cache_import_contract(envelope, registry);
    EXPECT_FALSE(import_contract.valid());
    EXPECT_FALSE(import_contract.diagnostics.empty());
    return;
  }

  const auto envelope =
      make_cache_source_contract_envelope_for_target(targets.front());
  const auto import_contract =
      compiler::make_cache_import_contract(envelope, registry);
  ASSERT_TRUE(import_contract.valid())
      << diagnostics_to_string(import_contract.diagnostics);
  ASSERT_TRUE(import_contract.runtime_descriptor);
  EXPECT_TRUE(import_contract.runtime_descriptor->materialization_finalized);
  EXPECT_EQ(import_contract.runtime_descriptor->materialization_stages.size(),
            1u);
  EXPECT_EQ(import_contract.runtime_descriptor->public_outputs.size(), 1u);
  ASSERT_TRUE(import_contract.runtime_model);
  EXPECT_EQ(import_contract.runtime_model->inputs().size(), 1u);
  EXPECT_EQ(import_contract.runtime_model->outputs().size(), 1u);
}

TEST(GfxCachePublicContractTest,
     ArtifactCacheRepositoryLoadsEnvelopeThroughRequestKey) {
  const std::vector<compiler::BackendTarget> targets{
      compiler::BackendTarget::from_backend(GpuBackend::Metal),
      compiler::BackendTarget::from_backend(GpuBackend::OpenCL),
      compiler::BackendTarget::from_backend_device_family(
          GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno),
      compiler::BackendTarget::from_backend_device_family(
          GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D),
  };
  for (const auto &target : targets) {
    const auto model = make_cache_contract_model("cache_repository_model");
    const auto capabilities = make_cache_contract_capabilities(target);
    compiler::ExecutableBundle executable;
    const auto envelope = make_cache_source_contract_envelope_for_request(
        target, *model, capabilities, &executable);
    const ScopedCacheContractDirectory cache_dir;
    const compiler::ArtifactCacheRepository repository(
        cache_dir.path().string());
    compiler::CacheLookupRequest request;
    request.model = model.get();
    request.target = &target;
    request.capabilities = &capabilities;
    request.enable_fusion = true;

    const auto before_store = repository.load(request);
    EXPECT_EQ(before_store.status, compiler::ArtifactCacheLookupStatus::Miss)
        << target.debug_string();

    const auto store_result = repository.store(request, envelope);
    ASSERT_TRUE(store_result.success)
        << target.debug_string() << ": "
        << diagnostics_to_string(store_result.diagnostics);

    const auto after_store = repository.load(request);
    ASSERT_TRUE(after_store.hit())
        << target.debug_string() << ": "
        << diagnostics_to_string(after_store.diagnostics);
    EXPECT_EQ(after_store.envelope.key.stable_key, envelope.key.stable_key);
    EXPECT_TRUE(after_store.envelope.verify(executable).valid())
        << diagnostics_to_string(
               after_store.envelope.verify(executable).diagnostics);

    auto fusion_off_request = request;
    fusion_off_request.enable_fusion = false;
    const auto fusion_miss = repository.load(fusion_off_request);
    EXPECT_EQ(fusion_miss.status, compiler::ArtifactCacheLookupStatus::Miss)
        << target.debug_string();
  }
}

TEST(GfxCachePublicContractTest,
     ArtifactCacheRepositoryRejectsEnvelopeRequestDrift) {
  const auto model = make_cache_contract_model("cache_repository_model");
  const auto other_model = make_cache_contract_model("cache_repository_other");
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  const auto capabilities = make_cache_contract_capabilities(target);
  const auto envelope = make_cache_source_contract_envelope_for_request(
      target, *model, capabilities);
  const ScopedCacheContractDirectory cache_dir;
  const compiler::ArtifactCacheRepository repository(cache_dir.path().string());
  compiler::CacheLookupRequest request;
  request.model = other_model.get();
  request.target = &target;
  request.capabilities = &capabilities;

  const auto store_result = repository.store(request, envelope);
  EXPECT_FALSE(store_result.success);
  EXPECT_FALSE(store_result.diagnostics.empty());
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
      << diagnostics_to_string(
             loaded.envelope.verify(reconstructed).diagnostics);

  auto missing_key = envelope.key;
  missing_key.stable_key = "missing-cache-contract-key";
  const auto miss = store.load(missing_key);
  EXPECT_FALSE(miss.valid());
  EXPECT_FALSE(miss.diagnostics.empty());
}

} // namespace gfx_plugin
} // namespace ov
