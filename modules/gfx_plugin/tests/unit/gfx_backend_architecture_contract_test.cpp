// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>

#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/gfx_compiler_service.hpp"
#include "compiler/tensor_layout.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "plugin/backend_state.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/execution_dispatcher.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_memory_ops.hpp"
#include "transforms/pipeline.hpp"
#include "unit/gfx_backend_contracts.hpp"

using ov::gfx_plugin::compiler::KernelUnitKind;
using ov::gfx_plugin::compiler::LoweringRouteKind;

namespace ov {
namespace gfx_plugin {
namespace {

class GfxBackendArchitectureContractTest : public ::testing::Test {
protected:
  test::BackendContractCatalog backend_catalog;
  test::ModelContractFactory models;
};

static_assert(
    std::is_same_v<decltype(&BackendState::create_stage),
                   std::unique_ptr<GpuStage> (BackendState::*)(
                       const std::shared_ptr<const ov::Node> &,
                       const RuntimeStageExecutableDescriptor *) const>,
    "BackendState must not expose descriptor-free stage materialization");

static_assert(
    std::is_same_v<
        decltype(&GpuStageFactory::create),
        std::unique_ptr<GpuStage> (*)(const std::shared_ptr<const ov::Node> &,
                                      const RuntimeStageExecutableDescriptor *,
                                      GpuBackend, void *, void *)>,
    "GpuStageFactory must require compiler-owned runtime descriptors");

static_assert(
    std::is_same_v<decltype(&transforms::run_pipeline),
                   std::shared_ptr<const ov::Model> (*)(
                       const std::shared_ptr<const ov::Model> &,
                       const transforms::PipelineOptions &)>,
    "Common graph pipeline must be configured by backend-owned options, "
    "not by a direct GpuBackend branch key");

static_assert(
    !std::is_invocable_v<decltype(&transforms::run_pipeline),
                         const std::shared_ptr<const ov::Model> &, GpuBackend>,
    "Common graph pipeline must not accept GpuBackend directly");

bool has_diagnostic_containing(const std::vector<std::string> &diagnostics,
                               std::string_view needle) {
  for (const auto &diagnostic : diagnostics) {
    if (diagnostic.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

compiler::TensorContract
make_tensor_contract(compiler::TensorContractRole role) {
  compiler::TensorContract contract;
  contract.logical_name =
      role == compiler::TensorContractRole::TensorInput ? "input0" : "output0";
  contract.role = role;
  contract.element_type = "f32";
  contract.partial_shape = "{1,3}";
  contract.layout = "logical";
  contract.storage_kind = "device_buffer";
  contract.lifetime_class = role == compiler::TensorContractRole::TensorInput
                                ? "producer_or_external"
                                : "stage_output";
  return contract;
}

compiler::ManifestBundle make_single_payload_route_manifest(
    LoweringRouteKind route_kind, std::string backend_domain,
    std::string kernel_unit_id, std::string kernel_unit_kind) {
  compiler::StageRecord stage;
  stage.stage_id = 0;
  stage.stable_record_key = 0x1234u;
  stage.source_node_name = "PayloadRoute";
  stage.normalized_op_family = "PayloadRoute";
  stage.execution_kind = route_kind;
  stage.backend_domain = std::move(backend_domain);
  stage.kernel_unit_id = std::move(kernel_unit_id);
  stage.kernel_unit_kind = std::move(kernel_unit_kind);
  stage.inputs.push_back(
      make_tensor_contract(compiler::TensorContractRole::TensorInput));
  stage.outputs.push_back(
      make_tensor_contract(compiler::TensorContractRole::TensorOutput));
  stage.dispatch.execution_kind = stage.execution_kind;
  stage.dispatch.backend_domain = stage.backend_domain;
  stage.dispatch.kernel_unit_id = stage.kernel_unit_id;
  stage.dispatch.kernel_unit_kind = stage.kernel_unit_kind;
  stage.dispatch.dispatch_source = "manifest";
  stage.memory.alias_group = "stage_0";

  compiler::ManifestBundle manifest;
  manifest.schema_version = 2;
  manifest.target_fingerprint = stage.backend_domain + ":test-target";
  manifest.stages.push_back(std::move(stage));
  return manifest;
}

std::shared_ptr<ov::Model> make_relu_model() {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{2, 3});
  auto relu = std::make_shared<ov::op::v0::Relu>(input);
  auto result = std::make_shared<ov::op::v0::Result>(relu);
  return std::make_shared<ov::Model>(ov::ResultVector{result},
                                     ov::ParameterVector{input});
}

compiler::PlannedOperation make_metadata_planned_operation(
    const std::shared_ptr<const ov::Node> &node,
    compiler::TensorLayoutPlan layout) {
  compiler::PlannedOperation op;
  op.source_node = node;
  op.node_name = node ? node->get_friendly_name() : "metadata";
  op.type_name = node ? node->get_type_name() : "Unknown";
  op.kernel_unit = compiler::KernelUnit::describe(
      LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
      "opencl", "metadata");
  op.layout = layout;
  op.profitability_score = 1.0;
  op.input_element_types = {"f32"};
  op.input_shapes = {"{1,2,3}"};
  op.output_element_types = {"f32"};
  op.output_shapes = {"{1,2,3}"};
  return op;
}

RuntimeStageExecutableDescriptor make_runtime_descriptor_for_layout(
    const std::shared_ptr<const ov::Node> &node,
    compiler::TensorLayoutPlan layout) {
  compiler::LoweringPlan plan;
  plan.target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  plan.operations.push_back(make_metadata_planned_operation(node, layout));
  const auto manifest = compiler::ManifestBuilder{}.build(plan);
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  const auto runtime_descriptor =
      RuntimeExecutableDescriptorBuilder{}.build(executable);
  EXPECT_EQ(runtime_descriptor.stages.size(), 1u);
  return runtime_descriptor.stages.front();
}

TEST_F(GfxBackendArchitectureContractTest,
       KnownTargetsUseConcreteOopIdentityWithoutInverseBuckets) {
  for (const auto &target_contract : backend_catalog.known_target_contracts()) {
    EXPECT_TRUE(target_contract.has_concrete_oop_identity())
        << target_contract.target().debug_string();
    EXPECT_TRUE(target_contract.avoids_inverse_apple_bucket())
        << target_contract.target().debug_string();
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerRegistryContainsAllProductionBackendModules) {
  const auto &registry = compiler::BackendRegistry::default_registry();
  const auto metal = registry.resolve(GpuBackend::Metal);
  const auto opencl = registry.resolve(GpuBackend::OpenCL);

  ASSERT_TRUE(metal);
  ASSERT_TRUE(opencl);
  EXPECT_TRUE(metal->capabilities().stage_placement());
  EXPECT_TRUE(opencl->capabilities().stage_placement());
  EXPECT_EQ(metal->target().backend(), GpuBackend::Metal);
  EXPECT_EQ(opencl->target().backend(), GpuBackend::OpenCL);
}

TEST_F(GfxBackendArchitectureContractTest,
       SharedBackendValueObjectsDoNotDefaultToMetal) {
  EXPECT_EQ(static_cast<int>(GpuBackend::Metal), 0);
  EXPECT_EQ(static_cast<int>(GpuBackend::OpenCL), 1);
  EXPECT_NE(static_cast<int>(GpuBackend::Unknown),
            static_cast<int>(GpuBackend::Metal));

  compiler::BackendTarget target;
  EXPECT_EQ(target.backend(), GpuBackend::Unknown);
  EXPECT_TRUE(target.backend_id().empty());

  compiler::GfxCompileRequest compile_request;
  EXPECT_EQ(compile_request.target.backend(), GpuBackend::Unknown);

  compiler::LoweringPlan lowering_plan;
  EXPECT_EQ(lowering_plan.target.backend(), GpuBackend::Unknown);

  GpuBuffer buffer;
  EXPECT_EQ(buffer.backend, GpuBackend::Unknown);
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerOwnsBackendNeutralTensorLayoutClassification) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 2, 3});
  auto identity_perm =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 2});
  auto identity_transpose =
      std::make_shared<ov::op::v1::Transpose>(input, identity_perm);
  const auto identity_plan = compiler::select_tensor_layout_plan(
      identity_transpose->get_type_name(), identity_transpose);
  EXPECT_TRUE(identity_plan.view_only);
  EXPECT_EQ(identity_plan.kind, GfxTensorLayoutKind::ViewOnly);

  auto materialized_perm =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
  auto materialized_transpose =
      std::make_shared<ov::op::v1::Transpose>(input, materialized_perm);
  const auto materialized_plan = compiler::select_tensor_layout_plan(
      materialized_transpose->get_type_name(), materialized_transpose);
  EXPECT_FALSE(materialized_plan.view_only);
  EXPECT_EQ(materialized_plan.kind, GfxTensorLayoutKind::Materialized);

  const auto reshape_plan =
      compiler::select_tensor_layout_plan("Reshape", nullptr);
  EXPECT_TRUE(reshape_plan.view_only);
  EXPECT_EQ(reshape_plan.kind, GfxTensorLayoutKind::ViewOnly);

  const auto view_descriptor =
      make_runtime_descriptor_for_layout(identity_transpose, identity_plan);
  EXPECT_EQ(view_descriptor.layout_contract, "view_only");
  EXPECT_TRUE(view_descriptor.tensor_view_only);

  const auto materialized_descriptor = make_runtime_descriptor_for_layout(
      materialized_transpose, materialized_plan);
  EXPECT_EQ(materialized_descriptor.layout_contract, "materialized");
  EXPECT_FALSE(materialized_descriptor.tensor_view_only);
}

TEST_F(GfxBackendArchitectureContractTest,
       UnknownBackendCannotMaterializeTargetOrStageFactory) {
  EXPECT_ANY_THROW(compiler::BackendTarget::from_backend(GpuBackend::Unknown));
  EXPECT_ANY_THROW(GpuStageFactory::factory_for_backend(GpuBackend::Unknown));
  EXPECT_ANY_THROW(memory_ops_for_backend(GpuBackend::Unknown));
}

TEST_F(GfxBackendArchitectureContractTest,
       KernelRegistriesRequireExplicitOpOwnedUnits) {
  const auto opencl_registry = test::KernelRegistryContract::for_opencl();
  ASSERT_TRUE(opencl_registry.audit_is_valid());

  EXPECT_TRUE(opencl_registry.rejects_unit(LoweringRouteKind::GeneratedKernel,
                                           "opencl_generated_kernel"));
  const auto matmul_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/matmul_f32");
  ASSERT_TRUE(matmul_unit.valid());
  EXPECT_EQ(matmul_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(matmul_unit.backend_domain(), "opencl");
  EXPECT_EQ(matmul_unit.op_family(), "MatMul");
  const auto shapeof_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/shapeof_i64");
  ASSERT_TRUE(shapeof_unit.valid());
  EXPECT_EQ(shapeof_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(shapeof_unit.backend_domain(), "opencl");
  EXPECT_EQ(shapeof_unit.op_family(), "ShapeOf");
  EXPECT_FALSE(shapeof_unit.exception_contract().valid());
  const auto tile_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/tile_f32");
  ASSERT_TRUE(tile_unit.valid());
  EXPECT_EQ(tile_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(tile_unit.backend_domain(), "opencl");
  EXPECT_EQ(tile_unit.op_family(), "Tile");
  EXPECT_FALSE(tile_unit.exception_contract().valid());
  const auto eltwise_unit =
      opencl_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
                                   "opencl/generated/eltwise_binary_f32");
  ASSERT_TRUE(eltwise_unit.valid());
  EXPECT_EQ(eltwise_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(eltwise_unit.backend_domain(), "opencl");
  EXPECT_EQ(eltwise_unit.op_family(), "Eltwise");
  const auto logical_bool_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel,
      "opencl/generated/eltwise_logical_binary_bool");
  ASSERT_TRUE(logical_bool_unit.valid());
  EXPECT_EQ(logical_bool_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(logical_bool_unit.backend_domain(), "opencl");
  EXPECT_EQ(logical_bool_unit.op_family(), "Eltwise");
  EXPECT_FALSE(logical_bool_unit.exception_contract().valid());
  const auto compare_unit =
      opencl_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
                                   "opencl/generated/eltwise_compare_f32");
  ASSERT_TRUE(compare_unit.valid());
  EXPECT_EQ(compare_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(compare_unit.backend_domain(), "opencl");
  EXPECT_EQ(compare_unit.op_family(), "Eltwise");
  EXPECT_FALSE(compare_unit.exception_contract().valid());
  const auto select_unit =
      opencl_registry.resolve_unit(LoweringRouteKind::GeneratedKernel,
                                   "opencl/generated/eltwise_select_f32");
  ASSERT_TRUE(select_unit.valid());
  EXPECT_EQ(select_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(select_unit.backend_domain(), "opencl");
  EXPECT_EQ(select_unit.op_family(), "Eltwise");
  EXPECT_FALSE(select_unit.exception_contract().valid());
  const auto activation_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/activation_f32");
  ASSERT_TRUE(activation_unit.valid());
  EXPECT_EQ(activation_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(activation_unit.backend_domain(), "opencl");
  EXPECT_EQ(activation_unit.op_family(), "Activation");
  const auto runtime_beta_activation_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel,
      "opencl/generated/activation_runtime_beta_f32");
  ASSERT_TRUE(runtime_beta_activation_unit.valid());
  EXPECT_EQ(runtime_beta_activation_unit.kind(),
            KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(runtime_beta_activation_unit.backend_domain(), "opencl");
  EXPECT_EQ(runtime_beta_activation_unit.op_family(), "Activation");
  const auto pool_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/pool2d_f32");
  ASSERT_TRUE(pool_unit.valid());
  EXPECT_EQ(pool_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(pool_unit.backend_domain(), "opencl");
  EXPECT_EQ(pool_unit.op_family(), "Pooling");
  const auto logical_reduce_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/reduction_bool");
  ASSERT_TRUE(logical_reduce_unit.valid());
  EXPECT_EQ(logical_reduce_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(logical_reduce_unit.backend_domain(), "opencl");
  EXPECT_EQ(logical_reduce_unit.op_family(), "Reduction");
  EXPECT_FALSE(logical_reduce_unit.exception_contract().valid());
  const auto transpose_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/transpose_f32");
  ASSERT_TRUE(transpose_unit.valid());
  EXPECT_EQ(transpose_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(transpose_unit.backend_domain(), "opencl");
  EXPECT_EQ(transpose_unit.op_family(), "Transpose");
  EXPECT_FALSE(transpose_unit.exception_contract().valid());
  const auto split_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/split3_f32");
  ASSERT_TRUE(split_unit.valid());
  EXPECT_EQ(split_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(split_unit.backend_domain(), "opencl");
  EXPECT_EQ(split_unit.op_family(), "Split");
  EXPECT_FALSE(split_unit.exception_contract().valid());
  const auto concat_unit = opencl_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "opencl/generated/concat2_f32");
  ASSERT_TRUE(concat_unit.valid());
  EXPECT_EQ(concat_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(concat_unit.backend_domain(), "opencl");
  EXPECT_EQ(concat_unit.op_family(), "Concat");
  EXPECT_FALSE(concat_unit.exception_contract().valid());

  const auto metal_registry = test::KernelRegistryContract::for_metal();
  ASSERT_TRUE(metal_registry.audit_is_valid());
  EXPECT_TRUE(metal_registry.rejects_unit(LoweringRouteKind::GeneratedKernel,
                                          "metal_lowering"));
  EXPECT_TRUE(metal_registry.rejects_unit(LoweringRouteKind::VendorPrimitive,
                                          "metal_lowering"));
  EXPECT_TRUE(metal_registry
                  .resolve_unit(LoweringRouteKind::VendorPrimitive,
                                "metal/vendor/mps_gemm")
                  .valid());
  EXPECT_TRUE(metal_registry
                  .resolve_unit(LoweringRouteKind::GeneratedKernel,
                                "metal/generated/slice")
                  .valid());
  const auto metal_eltwise_unit = metal_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "metal/generated/eltwise");
  ASSERT_TRUE(metal_eltwise_unit.valid());
  EXPECT_EQ(metal_eltwise_unit.op_family(), "Eltwise");
  const auto metal_activation_unit = metal_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "metal/generated/activation");
  ASSERT_TRUE(metal_activation_unit.valid());
  EXPECT_EQ(metal_activation_unit.op_family(), "Activation");
  const auto metal_transpose_unit = metal_registry.resolve_unit(
      LoweringRouteKind::GeneratedKernel, "metal/generated/transpose_f32");
  ASSERT_TRUE(metal_transpose_unit.valid());
  EXPECT_EQ(metal_transpose_unit.kind(), KernelUnitKind::GeneratedKernel);
  EXPECT_EQ(metal_transpose_unit.backend_domain(), "metal");
  EXPECT_EQ(metal_transpose_unit.op_family(), "Transpose");
  const auto metal_pool_unit = metal_registry.resolve_unit(
      LoweringRouteKind::VendorPrimitive, "metal/vendor/mps_pool2d");
  ASSERT_TRUE(metal_pool_unit.valid());
  EXPECT_EQ(metal_pool_unit.op_family(), "Pooling");
}

TEST_F(GfxBackendArchitectureContractTest,
       MetalUnsupportedCoverageNeverSelectsGenericKernelUnit) {
  const auto input = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::Shape{2, 3});
  const auto convert =
      std::make_shared<ov::op::v0::Convert>(input, ov::element::f16);
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_metal_operation_support_policy());

  const auto support = capabilities.query_operation({convert});
  EXPECT_FALSE(support.semantic_legal);
  EXPECT_EQ(support.preferred_route_kind, LoweringRouteKind::Unsupported);
  EXPECT_NE(support.preferred_route, "backend_lowering");
  EXPECT_TRUE(support.semantic_reason == "missing_metal_explicit_kernel_unit" ||
              support.semantic_reason == "unsupported_by_metal_capabilities");
}

TEST_F(GfxBackendArchitectureContractTest,
       KernelRegistryAuditRejectsUndocumentedHandwrittenExceptions) {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  compiler::KernelRegistry registry{
      target,
      {compiler::KernelUnit::describe(
          LoweringRouteKind::HandwrittenKernelException,
          KernelUnitKind::HandwrittenException, "opencl/baseline/undocumented",
          target.backend_id(), "Eltwise")}};

  const auto audit = registry.audit();
  EXPECT_FALSE(audit.valid());
  EXPECT_EQ(audit.handwritten_exception_count, 1u);
  bool found_contract_diagnostic = false;
  for (const auto &diagnostic : audit.diagnostics) {
    if (diagnostic.find("missing exception contract") != std::string::npos) {
      found_contract_diagnostic = true;
    }
  }
  EXPECT_TRUE(found_contract_diagnostic);
}

TEST_F(GfxBackendArchitectureContractTest,
       ExecutablePayloadRoutesRequireCompilerMaterializedPayloads) {
  struct RouteCase {
    LoweringRouteKind route_kind;
    std::string backend_domain;
    std::string kernel_unit_id;
    std::string kernel_unit_kind;
  };

  const std::vector<RouteCase> routes = {
      {LoweringRouteKind::GeneratedKernel, "metal", "metal/generated/test",
       "generated_kernel"},
      {LoweringRouteKind::GeneratedKernel, "opencl", "opencl/generated/test",
       "generated_kernel"},
      {LoweringRouteKind::VendorPrimitive, "metal", "metal/vendor/test",
       "vendor_primitive"},
  };

  for (const auto &route : routes) {
    const auto manifest = make_single_payload_route_manifest(
        route.route_kind, route.backend_domain, route.kernel_unit_id,
        route.kernel_unit_kind);
    ASSERT_TRUE(manifest.valid()) << route.kernel_unit_id;

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    const auto executable_result = executable.verify();
    EXPECT_FALSE(executable_result.valid()) << route.kernel_unit_id;
    EXPECT_TRUE(has_diagnostic_containing(executable_result.diagnostics,
                                          "requires a materialized payload"))
        << route.kernel_unit_id;

    const auto runtime_descriptor =
        RuntimeExecutableDescriptorBuilder{}.build(executable);
    const auto runtime_result = runtime_descriptor.verify(executable);
    EXPECT_FALSE(runtime_result.valid()) << route.kernel_unit_id;
    EXPECT_TRUE(has_diagnostic_containing(runtime_result.diagnostics,
                                          "requires a materialized payload"))
        << route.kernel_unit_id;
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       OpenClPayloadMaterializationIsOwnedByBackendModule) {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  const compiler::OperationLegalizer legalizer(capabilities);
  const compiler::LoweringPlanner planner(
      target, compiler::make_opencl_kernel_registry(target));

  const auto lowering_plan = planner.plan(make_relu_model(), legalizer);
  ASSERT_TRUE(lowering_plan.executable());
  const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
  ASSERT_TRUE(manifest.valid());

  const auto common_executable =
      compiler::ExecutableBundleBuilder{}.build(manifest, lowering_plan);
  EXPECT_TRUE(common_executable.artifact_payloads.empty());
  EXPECT_FALSE(common_executable.verify().valid());
  EXPECT_TRUE(has_diagnostic_containing(common_executable.verify().diagnostics,
                                        "requires a materialized payload"));

  const auto backend_executable =
      compiler::ExecutableBundleBuilder(
          compiler::make_opencl_kernel_artifact_payload_resolver())
          .build(manifest, lowering_plan);
  ASSERT_TRUE(backend_executable.verify().valid());
  ASSERT_EQ(backend_executable.artifact_payloads.size(), 1u);

  const auto descriptor_index =
      backend_executable.artifact_payloads.front().artifact_descriptor_index;
  ASSERT_LT(descriptor_index, backend_executable.artifact_descriptors.size());
  const auto &artifact =
      backend_executable.artifact_descriptors[descriptor_index];
  EXPECT_EQ(artifact.kernel.kernel_id, "opencl/generated/activation_f32");
  EXPECT_EQ(artifact.payload_kind,
            compiler::KernelArtifactPayloadKind::OpenClSource);
  EXPECT_EQ(artifact.entry_point, "gfx_opencl_generated_activation_f32");
}

TEST_F(GfxBackendArchitectureContractTest,
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

TEST_F(GfxBackendArchitectureContractTest,
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

TEST_F(GfxBackendArchitectureContractTest,
       RegisteredBackendModulesOwnFusionCapabilities) {
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

TEST_F(GfxBackendArchitectureContractTest,
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

} // namespace
} // namespace gfx_plugin
} // namespace ov
