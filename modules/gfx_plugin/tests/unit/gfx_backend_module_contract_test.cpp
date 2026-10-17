// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backends/metal/compiler/metal_backend_module.hpp"
#include "backends/opencl/compiler/opencl_backend_module.hpp"
#include "common/gfx_activation.hpp"
#include "common/gpu_backend.hpp"
#include "common/gpu_device_profile.hpp"
#include "common/gpu_parallelism_profile.hpp"
#include "compiler/backend_config.hpp"
#include "compiler/gfx_compiler_service.hpp"
#include "compiler/pipeline_stage_fusion.hpp"
#include "compiler/pipeline_stage_graph_snapshot.hpp"
#include "compiler/runtime_executable_descriptor_builder.hpp"
#include "compiler/static_backend_module.hpp"
#include "openvino/core/except.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gpu_buffer.hpp"
#include "unit/gfx_backend_contracts.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

class GfxBackendModuleContractTest : public ::testing::Test {
protected:
  test::BackendContractCatalog backend_catalog;
  test::ModelContractFactory models;
};

bool target_has_feature(const compiler::BackendTarget &target,
                        std::string_view feature) {
  for (const auto &candidate : target.feature_bits()) {
    if (candidate == feature) {
      return true;
    }
  }
  return false;
}

TEST_F(GfxBackendModuleContractTest,
       KnownTargetsUseConcreteOopIdentityWithoutInverseBuckets) {
  std::unordered_set<std::string> fingerprints;
  bool saw_metal_apple = false;
  bool saw_opencl_generic = false;
  bool saw_opencl_adreno = false;
  bool saw_opencl_v3d = false;

  for (const auto &target_contract : backend_catalog.known_target_contracts()) {
    EXPECT_TRUE(target_contract.has_concrete_oop_identity())
        << target_contract.target().debug_string();
    EXPECT_TRUE(target_contract.has_profiled_cache_identity())
        << target_contract.target().fingerprint();
    EXPECT_TRUE(target_contract.avoids_inverse_apple_bucket())
        << target_contract.target().debug_string();
    EXPECT_TRUE(
        fingerprints.insert(target_contract.target().fingerprint()).second)
        << target_contract.target().debug_string();

    saw_metal_apple |=
        target_contract.target().device_profile() == "metal_apple";
    saw_opencl_generic |=
        target_contract.target().device_profile() == "opencl_generic";
    saw_opencl_adreno |=
        target_contract.target().device_profile() == "opencl_adreno";
    saw_opencl_v3d |=
        target_contract.target().device_profile() == "opencl_broadcom_v3d";
  }
  EXPECT_TRUE(saw_metal_apple);
  EXPECT_TRUE(saw_opencl_generic);
  EXPECT_TRUE(saw_opencl_adreno);
  EXPECT_TRUE(saw_opencl_v3d);
}

TEST_F(GfxBackendModuleContractTest,
       CompilerRegistryContainsAllProductionBackendModules) {
  const auto &registry = compiler::BackendRegistry::default_registry();
  const auto metal_target =
      compiler::BackendTarget::from_backend(GpuBackend::Metal);
  const auto opencl_target =
      compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const auto metal = registry.resolve(metal_target);
  const auto opencl = registry.resolve(opencl_target);
  const auto targets = registry.available_targets();
  const auto expected_target_count =
      static_cast<size_t>(kGfxBackendMetalAvailable) +
      (kGfxBackendOpenCLAvailable ? size_t{3} : size_t{0});

  EXPECT_EQ(static_cast<bool>(metal), kGfxBackendMetalAvailable);
  EXPECT_EQ(static_cast<bool>(opencl), kGfxBackendOpenCLAvailable);
  EXPECT_EQ(targets.size(), expected_target_count);

  if (metal) {
    EXPECT_TRUE(metal->capabilities().stage_placement());
    EXPECT_EQ(metal->target().fingerprint(), metal_target.fingerprint());
  }
  if (opencl) {
    EXPECT_TRUE(opencl->capabilities().stage_placement());
    EXPECT_EQ(opencl->target().fingerprint(), opencl_target.fingerprint());
    EXPECT_EQ(opencl->target().device_profile(), "opencl_generic");

    const auto adreno_target =
        compiler::BackendTarget::from_backend_device_family(
            GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno);
    const auto v3d_target = compiler::BackendTarget::from_backend_device_family(
        GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D);
    const auto adreno = registry.resolve(adreno_target);
    const auto v3d = registry.resolve(v3d_target);
    ASSERT_TRUE(adreno);
    ASSERT_TRUE(v3d);
    EXPECT_EQ(adreno->target().fingerprint(), adreno_target.fingerprint());
    EXPECT_EQ(v3d->target().fingerprint(), v3d_target.fingerprint());
    EXPECT_EQ(v3d->target().runtime_api(), kBackendOpenCL);
    EXPECT_EQ(v3d->target().driver_id(), "opencl-clvk");
    EXPECT_NE(opencl->target().fingerprint(), v3d->target().fingerprint());
  }
}

TEST_F(
    GfxBackendModuleContractTest,
    BackendCapabilitiesDoNotAdvertiseCompiledModelExportImportBeforeEnvelopeRoundTrip) {
  const auto module_contracts = backend_catalog.compiled_module_contracts();
  ASSERT_FALSE(module_contracts.empty());
  for (const auto &module_contract : module_contracts) {
    const auto &capabilities = module_contract.module().capabilities();
    EXPECT_FALSE(
        capabilities.artifact_formats().supports_compiled_model_export_import)
        << module_contract.target().debug_string();
  }
}

TEST_F(GfxBackendModuleContractTest,
       BackendModulesOwnVendorAttentionArtifactContracts) {
  compiler::PipelineVendorAttentionPlan plan;
  plan.name = "test_vendor_attention";
  plan.element_type = ov::element::f32;
  plan.query_shape = {1, 2, 3, 4};
  plan.key_shape = {1, 2, 3, 5};
  plan.value_shape = {1, 2, 6, 5};
  plan.output_shape = {1, 2, 6, 4};
  plan.scale = 0.5f;

  const auto &registry = compiler::BackendRegistry::default_registry();
  if (const auto metal = registry.resolve(
          compiler::BackendTarget::from_backend(GpuBackend::Metal))) {
    auto artifact = metal->materialize_vendor_attention_artifact(0x1234u, plan);
    ASSERT_TRUE(artifact.valid());
    EXPECT_EQ(artifact.descriptor.stage_record_key, 0x1234u);
    EXPECT_EQ(artifact.descriptor.kernel.kernel_id,
              "metal/vendor/mpsgraph_sdpa");
    EXPECT_EQ(artifact.descriptor.kernel.op_family, "VendorAttention");
    EXPECT_EQ(artifact.descriptor.kernel.backend_domain, "metal");
    EXPECT_EQ(artifact.descriptor.kernel.origin,
              compiler::KernelArtifactOrigin::VendorPrimitive);
    EXPECT_EQ(artifact.descriptor.payload_kind,
              compiler::KernelArtifactPayloadKind::VendorDescriptor);
    EXPECT_EQ(artifact.descriptor.entry_point, "mps_sdpa");
    EXPECT_EQ(artifact.descriptor.compile_options_key,
              "metal_vendor_descriptor");
    EXPECT_FALSE(artifact.descriptor.artifact_key.empty());
  }

  if (const auto opencl = registry.resolve(
          compiler::BackendTarget::from_backend(GpuBackend::OpenCL))) {
    EXPECT_FALSE(
        opencl->materialize_vendor_attention_artifact(0x1234u, plan).valid());
  }
}

TEST_F(GfxBackendModuleContractTest, SharedBackendValueObjectsDoNotDefaultToMetal) {
  EXPECT_EQ(static_cast<int>(GpuBackend::Metal), 0);
  EXPECT_EQ(static_cast<int>(GpuBackend::OpenCL), 1);
  EXPECT_NE(static_cast<int>(GpuBackend::Unknown),
            static_cast<int>(GpuBackend::Metal));

  compiler::BackendTarget target;
  EXPECT_EQ(target.backend(), GpuBackend::Unknown);
  EXPECT_TRUE(target.backend_id().empty());

  compiler::GfxCompileRequest compile_request;
  EXPECT_EQ(compile_request.target.backend(), GpuBackend::Unknown);

  compiler::RuntimeExecutableDescriptorBuildRequest descriptor_build_request;
  EXPECT_EQ(descriptor_build_request.target.backend(), GpuBackend::Unknown);
  EXPECT_EQ(descriptor_build_request.executable, nullptr);
  EXPECT_EQ(descriptor_build_request.stage_graph_snapshot, nullptr);
  EXPECT_EQ(descriptor_build_request.backend_registry, nullptr);
  EXPECT_TRUE(descriptor_build_request.backend_name.empty());
  EXPECT_EQ(descriptor_build_request.compile_trace, nullptr);
  EXPECT_FALSE(descriptor_build_request.valid());

  compiler::StageCompilerPolicy stage_compiler_policy;
  EXPECT_EQ(stage_compiler_policy.target.backend(), GpuBackend::Unknown);
  EXPECT_EQ(stage_compiler_policy.backend, GpuBackend::Unknown);

  compiler::LoweringPlan lowering_plan;
  EXPECT_EQ(lowering_plan.target.backend(), GpuBackend::Unknown);

  GpuBuffer buffer;
  EXPECT_EQ(buffer.backend, GpuBackend::Unknown);
}

TEST_F(GfxBackendModuleContractTest,
       BackendDeviceProfilesParticipateInCacheAndCapabilityIdentity) {
  const auto generic_target =
      compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const auto adreno_target =
      compiler::BackendTarget::from_backend_device_family(
          GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno);
  const auto v3d_target = compiler::BackendTarget::from_backend_device_family(
      GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D);

  EXPECT_EQ(generic_target.runtime_api(), kBackendOpenCL);
  EXPECT_EQ(adreno_target.runtime_api(), kBackendOpenCL);
  EXPECT_EQ(v3d_target.runtime_api(), kBackendOpenCL);
  EXPECT_NE(generic_target.fingerprint(), adreno_target.fingerprint());
  EXPECT_NE(generic_target.fingerprint(), v3d_target.fingerprint());
  EXPECT_NE(adreno_target.fingerprint(), v3d_target.fingerprint());
  EXPECT_EQ(adreno_target.device_profile(), "opencl_adreno");
  EXPECT_EQ(v3d_target.device_profile(), "opencl_broadcom_v3d");
  EXPECT_TRUE(target_has_feature(adreno_target, "adreno"));
  EXPECT_TRUE(target_has_feature(v3d_target, "clvk"));
  EXPECT_TRUE(target_has_feature(v3d_target, "clspv"));

  const auto generic_module =
      compiler::make_opencl_backend_module(generic_target);
  const auto adreno_module =
      compiler::make_opencl_backend_module(adreno_target);
  const auto v3d_module = compiler::make_opencl_backend_module(v3d_target);
  ASSERT_TRUE(generic_module);
  ASSERT_TRUE(adreno_module);
  ASSERT_TRUE(v3d_module);

  EXPECT_EQ(generic_module->target().fingerprint(),
            generic_target.fingerprint());
  EXPECT_EQ(adreno_module->target().fingerprint(), adreno_target.fingerprint());
  EXPECT_EQ(v3d_module->target().fingerprint(), v3d_target.fingerprint());

  const auto &generic_profile =
      generic_module->capabilities().execution().custom_kernel_dispatch_profile;
  const auto &adreno_profile =
      adreno_module->capabilities().execution().custom_kernel_dispatch_profile;
  const auto &v3d_profile =
      v3d_module->capabilities().execution().custom_kernel_dispatch_profile;

  EXPECT_EQ(generic_profile.profile_key, "opencl:generic");
  EXPECT_EQ(adreno_profile.profile_key, "opencl:adreno");
  EXPECT_EQ(v3d_profile.profile_key, "opencl:broadcom_v3d");
  EXPECT_TRUE(adreno_profile.supports_conv_output_channel_blocking);
  EXPECT_TRUE(adreno_profile.supports_conv_channel_block_spatial_tiling);
  EXPECT_TRUE(v3d_profile.enable_skinny_matmul_tiles);
  EXPECT_TRUE(v3d_profile.chunk_dispatch.retune_threads_to_workload);
  EXPECT_LT(v3d_profile.max_total_threads_per_group,
            adreno_profile.max_total_threads_per_group);

  EXPECT_NE(compiler::make_backend_capabilities_fingerprint(
                generic_module->capabilities()),
            compiler::make_backend_capabilities_fingerprint(
                adreno_module->capabilities()));
  EXPECT_NE(compiler::make_backend_capabilities_fingerprint(
                generic_module->capabilities()),
            compiler::make_backend_capabilities_fingerprint(
                v3d_module->capabilities()));
  EXPECT_NE(compiler::make_backend_capabilities_fingerprint(
                adreno_module->capabilities()),
            compiler::make_backend_capabilities_fingerprint(
                v3d_module->capabilities()));
}

TEST_F(GfxBackendModuleContractTest,
       OpenClParallelismProfileRejectsAppleDeviceFamily) {
  EXPECT_THROW(make_opencl_parallelism_profile(GpuDeviceFamily::Apple),
               ov::Exception);
}

TEST_F(GfxBackendModuleContractTest,
       RuntimeStagePlanningPreservesConcreteBackendTargetProfile) {
  const auto target = compiler::BackendTarget::from_backend_device_family(
      GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D);
  const compiler::BackendRegistry registry(
      {compiler::make_opencl_backend_module(target)});
  const auto module = registry.resolve(target);
  ASSERT_TRUE(module);
  EXPECT_EQ(module->capabilities()
                .execution()
                .custom_kernel_dispatch_profile.profile_key,
            "opencl:broadcom_v3d");

  const compiler::GfxCompilerService compiler_service(registry);
  const auto compile_result = compiler_service.compile({models.relu(), target});
  ASSERT_TRUE(compile_result.supported())
      << compile_result.unsupported_message();
  EXPECT_EQ(compile_result.target.fingerprint(), target.fingerprint());
  EXPECT_EQ(compile_result.executable.target_fingerprint, target.fingerprint());

  ASSERT_TRUE(compile_result.runtime_descriptor);
  EXPECT_EQ(compile_result.runtime_descriptor->target_fingerprint,
            target.fingerprint());
  EXPECT_FALSE(compile_result.runtime_descriptor->materialization_stages.empty());
  EXPECT_EQ(
      compile_result.runtime_descriptor->runtime_options
          .custom_kernel_dispatch_profile.profile_key,
      "opencl:broadcom_v3d");
  EXPECT_EQ(compile_result.runtime_descriptor->runtime_options
                .custom_kernel_dispatch_profile
                .max_total_threads_per_group,
            64u);
  EXPECT_TRUE(compile_result.runtime_descriptor->runtime_options
                  .custom_kernel_dispatch_profile
                  .chunk_dispatch.retune_threads_to_workload);

  const auto binding_ref_compile_result =
      compiler_service.compile({models.relu(), target});
  ASSERT_TRUE(binding_ref_compile_result.supported())
      << binding_ref_compile_result.unsupported_message();
  ASSERT_TRUE(binding_ref_compile_result.runtime_descriptor);
  bool checked_stage_input_ref = false;
  for (const auto &stage_plan :
       binding_ref_compile_result.runtime_descriptor->materialization_stages) {
    for (const auto &input : stage_plan.io_plan.inputs) {
      if (input.source_ref.kind == PipelineStageTensorRefKind::None) {
        continue;
      }
      EXPECT_TRUE(input.source_ref.valid());
      checked_stage_input_ref = true;
    }
  }
  EXPECT_TRUE(checked_stage_input_ref);
}

TEST_F(GfxBackendModuleContractTest,
       BackendRegistryRequiresExactConcreteBackendTargetProfiles) {
  const auto generic_target =
      compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  compiler::StaticBackendModuleConfig config;
  config.target = generic_target;
  config.kernel_registry =
      compiler::make_common_kernel_registry(generic_target);
  const compiler::BackendRegistry generic_only_registry(
      {compiler::make_static_backend_module(std::move(config))});
  const compiler::GfxCompilerService compiler_service(generic_only_registry);

  const auto v3d_target = compiler::BackendTarget::from_backend_device_family(
      GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D);
  ASSERT_TRUE(generic_only_registry.resolve(generic_target));
  EXPECT_FALSE(generic_only_registry.resolve(v3d_target));
  EXPECT_THROW((void)compiler_service.compile(
                   {models.passthrough(ov::PartialShape{1}), v3d_target}),
               ov::Exception);
}

TEST_F(GfxBackendModuleContractTest,
       RuntimeStagePlanningRequiresExplicitBackendRegistry) {
  const auto target = compiler::BackendTarget::from_backend_device_family(
      GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D);
  const compiler::BackendRegistry registry(
      {compiler::make_opencl_backend_module(target)});
  const compiler::GfxCompilerService compiler_service(registry);

  const auto compile_result =
      compiler_service.compile({models.static_range(), target});
  ASSERT_TRUE(compile_result.supported())
      << compile_result.unsupported_message();
  ASSERT_TRUE(compile_result.runtime_descriptor);

  const auto backend_module = registry.resolve(target);
  ASSERT_TRUE(backend_module);
  const auto compiler_stage_graph_snapshot =
      compiler::detail::make_pipeline_stage_graph_snapshot(
          compile_result.transformed_model,
          compiler::detail::make_pipeline_stage_fusion_config(
              backend_module->capabilities().fusion(),
              /*enable_fusion=*/true,
              /*debug_log_fusion_decisions=*/false));
  compiler::RuntimeExecutableDescriptorBuildRequest descriptor_request;
  descriptor_request.executable = &compile_result.executable;
  descriptor_request.stage_graph_snapshot = &compiler_stage_graph_snapshot;
  descriptor_request.target = target;

  EXPECT_THROW(
      (void)compiler::RuntimeExecutableDescriptorBuilder{}.build_finalized(
          descriptor_request),
      ov::Exception);

  descriptor_request.backend_registry = &registry;
  const auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build_finalized(
          descriptor_request);
  EXPECT_TRUE(compiler::runtime_executable_descriptor_materialization_valid(
      runtime_descriptor));
  EXPECT_EQ(
      runtime_descriptor.runtime_options.custom_kernel_dispatch_profile
          .profile_key,
      "opencl:broadcom_v3d");
}

TEST_F(GfxBackendModuleContractTest,
       RuntimeDescriptorFinalizationIsOwnedBySharedCompilerBuilder) {
  struct TargetCase {
    std::string name;
    compiler::BackendTarget target;
    std::shared_ptr<const compiler::BackendModule> module;
  };

  const std::vector<TargetCase> target_cases = {
      {"macos_metal",
       compiler::BackendTarget::from_backend(GpuBackend::Metal),
       compiler::make_metal_backend_module(
           compiler::BackendTarget::from_backend(GpuBackend::Metal))},
      {"android_opencl_adreno",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno),
       compiler::make_opencl_backend_module(
           compiler::BackendTarget::from_backend_device_family(
               GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno))},
      {"rpi4_v3d_opencl",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D),
       compiler::make_opencl_backend_module(
           compiler::BackendTarget::from_backend_device_family(
               GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D))},
      {"rpi5_v3d_opencl",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D),
       compiler::make_opencl_backend_module(
           compiler::BackendTarget::from_backend_device_family(
               GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D))},
  };

  std::vector<std::shared_ptr<const compiler::BackendModule>> modules;
  modules.reserve(target_cases.size());
  for (const auto &target_case : target_cases) {
    modules.push_back(target_case.module);
  }

  const compiler::BackendRegistry registry(std::move(modules));
  const compiler::GfxCompilerService compiler_service(registry);

  for (const auto &target_case : target_cases) {
    SCOPED_TRACE(target_case.name);

    const auto compile_result =
        compiler_service.compile({models.static_range(), target_case.target});
    ASSERT_TRUE(compile_result.supported())
        << compile_result.unsupported_message();
    const auto compiler_stage_graph_snapshot =
        compiler::detail::make_pipeline_stage_graph_snapshot(
            compile_result.transformed_model,
            compiler::detail::make_pipeline_stage_fusion_config(
                target_case.module->capabilities().fusion(),
                /*enable_fusion=*/true,
                /*debug_log_fusion_decisions=*/false));

    const auto seed_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(
            compile_result.executable);
    EXPECT_FALSE(seed_descriptor.materialization_finalized);
    EXPECT_FALSE(compiler::runtime_executable_descriptor_materialization_valid(
        seed_descriptor));

    compiler::RuntimeExecutableDescriptorBuildRequest descriptor_request;
    descriptor_request.executable = &compile_result.executable;
    descriptor_request.stage_graph_snapshot = &compiler_stage_graph_snapshot;
    descriptor_request.backend_registry = &registry;
    descriptor_request.target = target_case.target;
    descriptor_request.backend_name = target_case.target.backend_id();
    ASSERT_TRUE(descriptor_request.valid());

    descriptor_request.stage_graph_snapshot = nullptr;
    EXPECT_THROW(
        (void)compiler::RuntimeExecutableDescriptorBuilder{}.build_finalized(
            descriptor_request),
        ov::Exception);
    descriptor_request.stage_graph_snapshot = &compiler_stage_graph_snapshot;

    const auto finalized_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build_finalized(
            descriptor_request);
    EXPECT_TRUE(finalized_descriptor.materialization_finalized);
    EXPECT_TRUE(compiler::runtime_executable_descriptor_valid(
        finalized_descriptor, compile_result.executable));
    EXPECT_TRUE(compiler::runtime_executable_descriptor_materialization_valid(
        finalized_descriptor));
    EXPECT_EQ(finalized_descriptor.target_fingerprint,
              compile_result.executable.target_fingerprint);
  }
}

TEST_F(GfxBackendModuleContractTest,
       RegisteredBackendModulesShareTheSamePassthroughContract) {
  const auto contracts = backend_catalog.compiled_module_contracts();
  ASSERT_FALSE(contracts.empty());

  for (const auto &module_contract : contracts) {
    const auto model =
        models.passthrough(ov::PartialShape{ov::Dimension::dynamic(), 3});
    const auto compile_result =
        module_contract.compile_without_graph_pipeline(model);
    EXPECT_TRUE(
        module_contract.compile_result_obeys_manifest_contract(compile_result))
        << module_contract.target().debug_string() << ": "
        << compile_result.unsupported_message();
  }
}

TEST_F(GfxBackendModuleContractTest,
       RegisteredBackendModulesOwnGraphPipelineOptions) {
  const auto contracts = backend_catalog.compiled_module_contracts();
  ASSERT_FALSE(contracts.empty());

  for (const auto &module_contract : contracts) {
    const auto &module = module_contract.module();
    const auto &options = module.pipeline_options();
    EXPECT_EQ(&options, &module.pipeline_options())
        << module_contract.target().debug_string();

    if (module_contract.target().backend() == GpuBackend::Metal) {
      EXPECT_TRUE(options.preserve_scaled_dot_product_attention);
      EXPECT_TRUE(options.canonicalize_sigmoid_before_ranking);
      EXPECT_TRUE(options.enable_llm_attention_fusions);
    }
    if (module_contract.target().backend() == GpuBackend::OpenCL) {
      EXPECT_FALSE(options.preserve_scaled_dot_product_attention);
      EXPECT_FALSE(options.canonicalize_sigmoid_before_ranking);
      EXPECT_FALSE(options.enable_llm_attention_fusions);
    }
  }
}

TEST_F(GfxBackendModuleContractTest, RegisteredBackendModulesOwnFusionCapabilities) {
  const auto contracts = backend_catalog.compiled_module_contracts();
  ASSERT_FALSE(contracts.empty());

  for (const auto &module_contract : contracts) {
    const auto &fusion = module_contract.module().capabilities().fusion();
    if (module_contract.target().backend() == GpuBackend::Metal) {
      EXPECT_FALSE(fusion.enable_generic_attention_fusion);
      EXPECT_TRUE(fusion.supports_vendor_attention_stage);
      EXPECT_TRUE(fusion.enable_conv_activation_fusion);
      EXPECT_FALSE(fusion.enable_precision_sensitive_arithmetic_fusion);
    }
    if (module_contract.target().backend() == GpuBackend::OpenCL) {
      EXPECT_TRUE(fusion.enable_generic_attention_fusion);
      EXPECT_FALSE(fusion.supports_vendor_attention_stage);
      EXPECT_TRUE(fusion.enable_conv_activation_fusion);
      EXPECT_TRUE(fusion.enable_precision_sensitive_arithmetic_fusion);
    }
  }
}

TEST_F(GfxBackendModuleContractTest,
       RegisteredBackendModulesOwnPostOpFusionCapabilities) {
  const auto contracts = backend_catalog.compiled_module_contracts();
  ASSERT_FALSE(contracts.empty());

  for (const auto &module_contract : contracts) {
    const auto &capabilities = module_contract.module().capabilities();
    const auto &post_ops = capabilities.post_ops();
    if (module_contract.target().backend() == GpuBackend::Metal) {
      EXPECT_TRUE(post_ops.allow_stage_bias_fusion("Convolution"));
      EXPECT_TRUE(post_ops.allow_stage_bias_fusion("GroupConvolution"));
      EXPECT_TRUE(post_ops.allow_stage_batchnorm_fusion("Convolution"));
      EXPECT_TRUE(post_ops.allow_stage_batchnorm_fusion("GroupConvolution"));
      EXPECT_TRUE(capabilities.allow_stage_activation_fusion(
          "Convolution", ActivationKind::Relu));
      EXPECT_TRUE(capabilities.allow_stage_activation_fusion(
          "GroupConvolution", ActivationKind::Swish));
      EXPECT_FALSE(capabilities.allow_stage_activation_fusion(
          "Convolution", ActivationKind::Gelu));
    }
    if (module_contract.target().backend() == GpuBackend::OpenCL) {
      EXPECT_TRUE(post_ops.allow_stage_bias_fusion("Convolution"));
      EXPECT_FALSE(post_ops.allow_stage_bias_fusion("GroupConvolution"));
      EXPECT_TRUE(post_ops.allow_stage_batchnorm_fusion("Convolution"));
      EXPECT_FALSE(post_ops.allow_stage_batchnorm_fusion("GroupConvolution"));
      EXPECT_TRUE(capabilities.allow_stage_activation_fusion(
          "Convolution", ActivationKind::Relu));
      EXPECT_TRUE(capabilities.allow_stage_activation_fusion(
          "Convolution", ActivationKind::Swish));
      EXPECT_FALSE(capabilities.allow_stage_activation_fusion(
          "Convolution", ActivationKind::Sigmoid));
      EXPECT_FALSE(capabilities.allow_stage_activation_fusion(
          "GroupConvolution", ActivationKind::Relu));
    }
  }
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
