// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "compiler/backend_config.hpp"
#include "compiler/cache_envelope.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/gfx_compiler_service.hpp"
#include "compiler/memory_plan.hpp"
#include "compiler/pipeline_stage_builder.hpp"
#include "compiler/pipeline_stage_fusion.hpp"
#include "compiler/pipeline_stage_plan.hpp"
#include "compiler/tensor_layout.hpp"
#include "mlir/gfx_apple_vendor_descriptors.hpp"
#include "openvino/op/add.hpp"
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
#include "runtime/backend_stage_factory.hpp"
#include "runtime/pipeline_stage_materializer.hpp"
#include "runtime/runtime_session.hpp"
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
    std::is_base_of_v<BackendStageFactory, BackendState>,
    "Plugin backend state must expose stage materialization through the "
    "runtime-facing BackendStageFactory interface");

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

static_assert(std::is_same_v<decltype(&RuntimeSession::prepare_stage),
                             PreparedKernelExecutable (RuntimeSession::*)(
                                 size_t, GpuStage &, GpuBufferManager *,
                                 ResourceBindingTable) const>,
              "RuntimeSession must own request-time executable preparation");

static_assert(
    std::is_same_v<decltype(&PipelineStageMaterializer::create_stage),
                   std::unique_ptr<GpuStage> (PipelineStageMaterializer::*)(
                       const std::shared_ptr<const ov::Node> &) const>,
    "PipelineStageMaterializer must own descriptor-backed stage prototype "
    "materialization for CompiledModel");

static_assert(
    std::is_same_v<decltype(&PipelineStageMaterializer::stage_index_for),
                   size_t (PipelineStageMaterializer::*)(
                       const std::shared_ptr<const ov::Node> &) const>,
    "PipelineStageMaterializer must expose descriptor-owned runtime stage "
    "indices without fallback repair");

static_assert(
    std::is_same_v<
        decltype(&PipelineStageMaterializer::fusion_contract_for),
        compiler::PipelineStageFusionContract (PipelineStageMaterializer::*)(
            const std::shared_ptr<const ov::Node> &) const>,
    "PipelineStageMaterializer must expose compiler-owned fusion contracts "
    "without probing runtime stages");

static_assert(
    std::is_same_v<
        decltype(&compiler::build_pipeline_stage_descriptors),
        compiler::PipelineStageBuildResult (*)(
            const compiler::PipelineStageBuildRequest &)>,
    "Compiler pipeline stage builder must own descriptor pipeline assembly "
    "outside CompiledModel and runtime common");

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

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream stream(path);
  if (!stream.good()) {
    throw std::runtime_error("GFX test: failed to read " + path.string());
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

std::filesystem::path find_gfx_module_root_for_source_contract() {
  auto source_path = std::filesystem::path(__FILE__);
  if (source_path.is_relative()) {
    source_path = std::filesystem::absolute(source_path);
  }
  for (auto dir = source_path.parent_path(); !dir.empty();) {
    if (std::filesystem::exists(dir / "src/runtime/gfx_stage_policy.cpp")) {
      return dir;
    }
    const auto parent = dir.parent_path();
    if (parent == dir) {
      break;
    }
    dir = parent;
  }

  const auto cwd = std::filesystem::current_path();
  for (const auto &candidate : {cwd, cwd / "modules/gfx_plugin"}) {
    if (std::filesystem::exists(candidate /
                                "src/runtime/gfx_stage_policy.cpp")) {
      return candidate;
    }
  }
  throw std::runtime_error("GFX test: failed to locate gfx_plugin module root");
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonRuntimeDoesNotOwnMpsrtContracts) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto runtime_root = module_root / "src/runtime";
  ASSERT_TRUE(std::filesystem::exists(runtime_root));
  const std::string old_mpsrt_include_prefix =
      std::string("runtime/") + "gfx_mpsrt_";

  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(runtime_root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto path = entry.path();
    EXPECT_EQ(path.filename().string().find("gfx_mpsrt_"), std::string::npos)
        << path;
    const auto extension = path.extension().string();
    if (extension == ".cpp" || extension == ".hpp" || extension == ".mm") {
      const auto source = read_text_file(path);
      EXPECT_EQ(source.find(old_mpsrt_include_prefix), std::string::npos)
          << path;
    }
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeStagePolicyDoesNotResolveCompilerBackendRegistry) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto stage_policy_source =
      read_text_file(module_root / "src/runtime/gfx_stage_policy.cpp");

  EXPECT_EQ(stage_policy_source.find("compiler/backend_registry.hpp"),
            std::string::npos);
  EXPECT_EQ(stage_policy_source.find("BackendRegistry::"), std::string::npos);
  EXPECT_EQ(stage_policy_source.find("make_post_op_fusion_capabilities("),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonKernelCodegenSourceDoesNotDependOnMetalMpsrtRuntime) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto source =
      read_text_file(module_root / "src/kernel_ir/gfx_codegen_backend.hpp");

  EXPECT_EQ(source.find("backends/metal/"), std::string::npos);
  EXPECT_EQ(source.find("gfx_mpsrt_"), std::string::npos);
  EXPECT_EQ(source.find("MpsrtConstTensorSource"), std::string::npos);
  EXPECT_NE(source.find("KernelConstTensorSource"), std::string::npos);
  EXPECT_NE(source.find("const_tensor_sources"), std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       SharedMlirCustomKernelContractsUseBackendDomainNotOpenClBoolSelector) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const std::vector<std::filesystem::path> shared_mlir_contract_files = {
      module_root / "src/mlir/gfx_backend_custom_kernel_adapter.hpp",
      module_root / "src/mlir/gfx_backend_custom_kernel_adapter.cpp",
      module_root / "src/mlir/gfx_stage_kernel_binding.hpp",
      module_root / "src/mlir/gfx_stage_runtime_values.hpp",
      module_root / "src/mlir/gfx_stage_runtime_values.cpp",
  };

  for (const auto &path : shared_mlir_contract_files) {
    const auto source = read_text_file(path);
    EXPECT_EQ(source.find("is_opencl_backend"), std::string::npos) << path;
    EXPECT_EQ(source.find("backend_domain_from_selector"), std::string::npos)
        << path;
    EXPECT_EQ(source.find("specialization_prefix_from_selector"),
              std::string::npos)
        << path;
  }

  const auto adapter_header = read_text_file(
      module_root / "src/mlir/gfx_backend_custom_kernel_adapter.hpp");
  EXPECT_NE(adapter_header.find("GfxKernelBackendDomain backend_domain"),
            std::string::npos);
  EXPECT_NE(adapter_header.find("backend_custom_kernel_specialization_prefix"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       SharedMlirStageRoutingDoesNotCarryConstantTemporaryBranches) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto mlir_stage_source =
      read_text_file(module_root / "src/mlir/mlir_stage.cpp");
  const std::vector<std::string> forbidden_patterns = {
      std::string("if ") + "(!false)",
      std::string("if ") + "(false)",
      std::string("false ") + "&&",
      std::string("!false ") + "&&",
      std::string("!false ") + "||",
      std::string("if ") + "(true)",
      std::string("!true"),
  };

  for (const auto &pattern : forbidden_patterns) {
    EXPECT_EQ(mlir_stage_source.find(pattern), std::string::npos) << pattern;
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeStagePolicyConsumesCompilerOwnedSourceDispatchPolicy) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto stage_policy_source =
      read_text_file(module_root / "src/runtime/gfx_stage_policy.cpp");

  EXPECT_EQ(stage_policy_source.find("backend == GpuBackend::OpenCL"),
            std::string::npos);
  EXPECT_EQ(stage_policy_source.find("backend == GpuBackend::Metal"),
            std::string::npos);
  EXPECT_NE(stage_policy_source.find("source_kernel_dispatch"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeStagesDoNotClassifyTensorViewOrLifetimeContracts) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto gpu_stage_header =
      read_text_file(module_root / "src/runtime/gpu_stage.hpp");
  const auto fused_sequence_source =
      read_text_file(module_root / "src/runtime/fused_sequence_stage.cpp");

  EXPECT_EQ(gpu_stage_header.find("is_view_only"), std::string::npos);
  EXPECT_EQ(gpu_stage_header.find("describe_output_lifetimes"),
            std::string::npos);
  EXPECT_EQ(gpu_stage_header.find("GpuStageOutputLifetime"), std::string::npos);
  EXPECT_EQ(fused_sequence_source.find("stage_may_alias_first_input"),
            std::string::npos);
  EXPECT_EQ(
      fused_sequence_source.find("stage_guarantees_first_input_storage_alias"),
      std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       PluginCompiledModelDoesNotOwnFusedOutputLifetimePlanning) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto compiled_model_source =
      read_text_file(module_root / "src/plugin/compiled_model.cpp");
  const auto planner_source = read_text_file(
      module_root / "src/runtime/fused_output_lifetime_plan.cpp");

  EXPECT_EQ(compiled_model_source.find("descriptor_outputs_may_alias_inputs"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find(
                "descriptor_outputs_share_first_input_storage"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("find_alias_group("), std::string::npos);
  EXPECT_EQ(compiled_model_source.find("build_fused_output_lifetime_plan"),
            std::string::npos);
  EXPECT_NE(planner_source.find("build_fused_output_lifetime_plan"),
            std::string::npos);
  EXPECT_NE(planner_source.find("RuntimeMemoryPlanDescriptor"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerOwnsPipelineStageIoPlanning) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto compiled_model_source =
      read_text_file(module_root / "src/plugin/compiled_model.cpp");
  const auto compiled_model_header = read_text_file(
      module_root / "include/openvino/gfx_plugin/compiled_model.hpp");
  const auto builder_header =
      read_text_file(module_root / "src/compiler/pipeline_stage_builder.hpp");
  const auto builder_source =
      read_text_file(module_root / "src/compiler/pipeline_stage_builder.cpp");
  const auto planner_header =
      read_text_file(module_root / "src/compiler/pipeline_stage_plan.hpp");
  const auto planner_source =
      read_text_file(module_root / "src/compiler/pipeline_stage_plan.cpp");

  EXPECT_EQ(compiled_model_source.find("compiler/pipeline_stage_plan.hpp"),
            std::string::npos);
  EXPECT_NE(compiled_model_source.find("compiler/pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("runtime/pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_NE(compiled_model_header.find("runtime/pipeline_stage_desc.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_header.find("struct PipelineStageDesc"),
            std::string::npos);
  EXPECT_EQ(builder_header.find("openvino/gfx_plugin/compiled_model.hpp"),
            std::string::npos);
  EXPECT_NE(builder_header.find("runtime/pipeline_stage_desc.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("plugin/pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("struct NodePortKey"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("struct InputKey"), std::string::npos);
  EXPECT_EQ(compiled_model_source.find("auto remap_input_link"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("auto merge_model_outputs"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("auto append_output_alias"),
            std::string::npos);

  EXPECT_NE(builder_header.find("PipelineStageBuildRequest"),
            std::string::npos);
  EXPECT_NE(builder_header.find("PipelineStageBuildResult"), std::string::npos);
  EXPECT_NE(builder_source.find("PipelineStagePlanBuilder"), std::string::npos);
  EXPECT_NE(builder_source.find("collect_model_output_ports"),
            std::string::npos);
  EXPECT_NE(builder_source.find("record_output_alias"), std::string::npos);
  EXPECT_NE(planner_header.find("PipelineStagePlanBuilder"), std::string::npos);
  EXPECT_NE(planner_source.find("collect_model_output_ports"),
            std::string::npos);
  EXPECT_NE(planner_source.find("record_output_alias"), std::string::npos);
  EXPECT_EQ(planner_header.find("runtime/gpu_stage"), std::string::npos);
  EXPECT_EQ(planner_header.find("runtime/executable_descriptor"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CompiledModelDelegatesStageMaterializationToSharedHelper) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto compiled_model_source =
      read_text_file(module_root / "src/plugin/compiled_model.cpp");
  const auto builder_header =
      read_text_file(module_root / "src/compiler/pipeline_stage_builder.hpp");
  const auto builder_source =
      read_text_file(module_root / "src/compiler/pipeline_stage_builder.cpp");
  const auto materializer_header = read_text_file(
      module_root / "src/runtime/pipeline_stage_materializer.hpp");
  const auto materializer_source = read_text_file(
      module_root / "src/runtime/pipeline_stage_materializer.cpp");

  EXPECT_NE(compiled_model_source.find("compiler/pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("runtime/pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("plugin/pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_EQ(
      compiled_model_source.find("plugin/pipeline_stage_materializer.hpp"),
      std::string::npos);
  EXPECT_EQ(
      compiled_model_source.find("runtime/pipeline_stage_materializer.hpp"),
      std::string::npos);
  EXPECT_EQ(compiled_model_source.find("PipelineStageMaterializer"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("PipelineStagePlanBuilder"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("get_ordered_ops()"), std::string::npos);
  EXPECT_NE(builder_header.find("build_pipeline_stage_descriptors"),
            std::string::npos);
  EXPECT_NE(builder_source.find("PipelineStageMaterializer"),
            std::string::npos);
  EXPECT_NE(builder_source.find("PipelineStagePlanBuilder"), std::string::npos);
  EXPECT_EQ(compiled_model_source.find("runtime_stage_descriptors"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("auto runtime_descriptor_for_node"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("auto create_backend_stage"),
            std::string::npos);
  EXPECT_NE(materializer_header.find("class PipelineStageMaterializer"),
            std::string::npos);
  EXPECT_NE(materializer_source.find("m_descriptors_by_node"),
            std::string::npos);
  EXPECT_NE(materializer_source.find("RuntimeExecutableDescriptor"),
            std::string::npos);
  EXPECT_NE(materializer_header.find("BackendStageFactory"), std::string::npos);
  EXPECT_NE(materializer_source.find("m_stage_factory"), std::string::npos);
  EXPECT_EQ(materializer_header.find("plugin/backend_state"), std::string::npos);
  EXPECT_EQ(materializer_source.find("BackendState"), std::string::npos);
  EXPECT_NE(materializer_header.find("stage_index_for"), std::string::npos);
  EXPECT_EQ(materializer_header.find("stage_index_or"), std::string::npos);
  EXPECT_EQ(compiled_model_source.find("FusedOutputLifetimeStage"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("FusedSequenceStage"),
            std::string::npos);
  EXPECT_NE(materializer_header.find("MaterializedFusedSequenceStage"),
            std::string::npos);
  EXPECT_NE(materializer_header.find("create_attention_sequence_stage"),
            std::string::npos);
  EXPECT_NE(materializer_source.find("FusedOutputLifetimeStage"),
            std::string::npos);
  EXPECT_NE(materializer_source.find("FusedSequenceStage"), std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStageBuilderPlansFusionFromCompilerContractWithoutStageProbes) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto compiled_model_source =
      read_text_file(module_root / "src/plugin/compiled_model.cpp");
  const auto builder_header =
      read_text_file(module_root / "src/compiler/pipeline_stage_builder.hpp");
  const auto builder_source =
      read_text_file(module_root / "src/compiler/pipeline_stage_builder.cpp");
  const auto fusion_header =
      read_text_file(module_root / "src/compiler/pipeline_stage_fusion.hpp");
  const auto fusion_source =
      read_text_file(module_root / "src/compiler/pipeline_stage_fusion.cpp");
  const auto materializer_header = read_text_file(
      module_root / "src/runtime/pipeline_stage_materializer.hpp");
  const auto materializer_source = read_text_file(
      module_root / "src/runtime/pipeline_stage_materializer.cpp");
  const auto backend_state_header =
      read_text_file(module_root / "src/plugin/backend_state.hpp");
  const auto runtime_descriptor_header =
      read_text_file(module_root / "src/runtime/executable_descriptor.hpp");
  const auto metal_state_header = read_text_file(
      module_root / "src/backends/metal/plugin/compiled_model_state.hpp");
  const auto metal_stage_factory_source = read_text_file(
      module_root / "src/backends/metal/runtime/stage_factory.cpp");
  const auto cmake_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");
  const auto backend_registry_header =
      read_text_file(module_root / "src/compiler/backend_registry.hpp");
  const auto backend_registry_source =
      read_text_file(module_root / "src/compiler/backend_registry.cpp");
  const auto metal_artifacts_header = read_text_file(
      module_root / "src/backends/metal/compiler/metal_kernel_artifacts.hpp");
  const auto metal_artifacts_source = read_text_file(
      module_root / "src/backends/metal/compiler/metal_kernel_artifacts.cpp");
  const auto apple_vendor_descriptors_header =
      read_text_file(module_root / "src/mlir/gfx_apple_vendor_descriptors.hpp");
  const auto apple_vendor_descriptors_source =
      read_text_file(module_root / "src/mlir/gfx_apple_vendor_descriptors.cpp");
  const auto mps_graph_attention_header =
      module_root / "src/backends/metal/runtime/mps_graph_attention_stage.hpp";
  const auto mps_graph_attention_source =
      module_root / "src/backends/metal/runtime/mps_graph_attention_stage.mm";

  EXPECT_EQ(compiled_model_source.find("compiler/pipeline_stage_fusion.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("compiler::plan_pipeline_fusions("),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("compiler::find_rms_residual_add("),
            std::string::npos);
  EXPECT_NE(builder_header.find("PipelineStageBuildRequest"),
            std::string::npos);
  EXPECT_NE(builder_header.find("BackendStageFactory"), std::string::npos);
  EXPECT_EQ(builder_source.find("plugin/backend_state.hpp"), std::string::npos);
  EXPECT_NE(builder_source.find("compiler::plan_pipeline_fusions("),
            std::string::npos);
  EXPECT_NE(builder_source.find("compiler::find_rms_residual_add("),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("FusionConfig"), std::string::npos);
  EXPECT_EQ(compiled_model_source.find("build_fusion_plan("),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("make_vendor_attention_subgraph_plan"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("read_uniform_scale_from_multiply"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("extract_scaled_tensor_input"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("shape_matches_without_broadcast"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("allow_stage_input_activation_fusion"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("allow_stage_residual_add_fusion"),
            std::string::npos);
  EXPECT_NE(builder_source.find("allow_stage_residual_add_fusion"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("probe_stage"), std::string::npos);
  EXPECT_EQ(compiled_model_source.find("vendor_attention_supported"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find(
                "backend_state->create_vendor_attention_stage"),
            std::string::npos);
  EXPECT_EQ(backend_state_header.find("VendorAttentionStageSpec"),
            std::string::npos);
  EXPECT_EQ(backend_state_header.find("create_vendor_attention_stage"),
            std::string::npos);
  EXPECT_NE(fusion_header.find("PipelineFusionSelectionPlan"),
            std::string::npos);
  EXPECT_NE(fusion_header.find("plan_pipeline_fusions"), std::string::npos);
  EXPECT_NE(fusion_header.find("plan_vendor_attention_subgraph"),
            std::string::npos);
  EXPECT_NE(fusion_header.find("find_rms_residual_add"), std::string::npos);
  EXPECT_EQ(fusion_header.find("vendor_attention_supported"),
            std::string::npos);
  EXPECT_NE(fusion_source.find("build_fusion_plan("), std::string::npos);
  EXPECT_NE(fusion_source.find("plan_vendor_attention_subgraph"),
            std::string::npos);
  EXPECT_NE(fusion_source.find("supports_vendor_attention_stage"),
            std::string::npos);
  EXPECT_EQ(fusion_source.find("create_vendor_attention_stage"),
            std::string::npos);
  EXPECT_EQ(fusion_header.find("plugin/backend_state"), std::string::npos);
  EXPECT_EQ(fusion_source.find("plugin/backend_state"), std::string::npos);
  EXPECT_NE(materializer_header.find("fusion_contract_for"), std::string::npos);
  EXPECT_NE(materializer_header.find("create_vendor_attention_stage"),
            std::string::npos);
  EXPECT_NE(materializer_source.find("create_vendor_attention_stage"),
            std::string::npos);
  EXPECT_NE(materializer_header.find("PipelineVendorAttentionArtifactResolver"),
            std::string::npos);
  EXPECT_EQ(materializer_header.find("PipelineVendorAttentionPayloadResolver"),
            std::string::npos);
  EXPECT_EQ(
      materializer_source.find("make_vendor_attention_artifact_descriptor"),
      std::string::npos);
  EXPECT_EQ(
      materializer_source.find("finalize_kernel_artifact_descriptor_identity"),
      std::string::npos);
  EXPECT_EQ(materializer_source.find("mps_sdpa"), std::string::npos);
  EXPECT_EQ(materializer_source.find("metal_vendor_descriptor"),
            std::string::npos);
  EXPECT_EQ(materializer_source.find("metal/vendor/mpsgraph_sdpa"),
            std::string::npos);
  EXPECT_EQ(materializer_source.find("m_vendor_attention_kernel_unit_id"),
            std::string::npos);
  EXPECT_EQ(materializer_source.find("m_vendor_attention_payload_resolver"),
            std::string::npos);
  EXPECT_NE(materializer_source.find("m_vendor_attention_artifact_resolver"),
            std::string::npos);
  EXPECT_EQ(materializer_source.find("RuntimeVendorAttentionDescriptor"),
            std::string::npos);
  EXPECT_EQ(materializer_source.find("payload.reset"), std::string::npos);
  EXPECT_EQ(materializer_source.find("vendor/attention/sdpa"),
            std::string::npos);
  EXPECT_EQ(materializer_source.find("vendor_attention_descriptor"),
            std::string::npos);
  EXPECT_NE(materializer_source.find("m_stage_factory.create_stage"),
            std::string::npos);
  EXPECT_EQ(runtime_descriptor_header.find("RuntimeVendorAttentionDescriptor"),
            std::string::npos);
  EXPECT_EQ(runtime_descriptor_header.find("vendor_attention"),
            std::string::npos);
  EXPECT_EQ(metal_state_header.find("mps_graph_attention_stage"),
            std::string::npos);
  EXPECT_EQ(metal_state_header.find("create_vendor_attention_stage"),
            std::string::npos);
  EXPECT_EQ(metal_stage_factory_source.find("mps_graph_attention_stage"),
            std::string::npos);
  EXPECT_EQ(metal_stage_factory_source.find("vendor_attention"),
            std::string::npos);
  EXPECT_EQ(cmake_sources.find("mps_graph_attention_stage"), std::string::npos);
  EXPECT_NE(cmake_sources.find("pipeline_stage_builder.cpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(mps_graph_attention_header));
  EXPECT_FALSE(std::filesystem::exists(mps_graph_attention_source));
  EXPECT_NE(
      backend_registry_header.find("materialize_vendor_attention_artifact"),
      std::string::npos);
  EXPECT_EQ(
      backend_registry_header.find("materialize_vendor_attention_payload"),
      std::string::npos);
  EXPECT_EQ(backend_registry_header.find("vendor_attention_kernel_unit_id"),
            std::string::npos);
  EXPECT_NE(
      backend_registry_source.find("PipelineVendorAttentionArtifactResolver"),
      std::string::npos);
  EXPECT_EQ(
      backend_registry_source.find("PipelineVendorAttentionPayloadResolver"),
      std::string::npos);
  EXPECT_NE(
      metal_artifacts_header.find("metal_mpsgraph_sdpa_vendor_kernel_unit_id"),
      std::string::npos);
  EXPECT_NE(metal_artifacts_source.find("metal/vendor/mpsgraph_sdpa"),
            std::string::npos);
  EXPECT_NE(metal_artifacts_header.find(
                "make_metal_vendor_attention_artifact_resolver"),
            std::string::npos);
  EXPECT_EQ(metal_artifacts_header.find(
                "make_metal_vendor_attention_payload_resolver"),
            std::string::npos);
  EXPECT_NE(metal_artifacts_source.find("mps_sdpa"), std::string::npos);
  EXPECT_NE(metal_artifacts_source.find("metal_vendor_descriptor"),
            std::string::npos);
  EXPECT_NE(metal_artifacts_source.find("PipelineVendorAttentionPlan"),
            std::string::npos);
  EXPECT_NE(metal_artifacts_source.find(
                "gfx_apple_make_mps_transposed_sdpa_contract"),
            std::string::npos);
  EXPECT_NE(apple_vendor_descriptors_header.find(
                "gfx_apple_make_mps_transposed_sdpa_contract"),
            std::string::npos);
  EXPECT_NE(
      apple_vendor_descriptors_source.find("GfxMpsrtSdpaLayoutTransposedBHDN"),
      std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStageFusionContractRejectsRuntimeProbeOnlyRoutes) {
  compiler::PipelineStageFusionContract msl_multiply;
  msl_multiply.op_family = "Multiply";
  msl_multiply.payload_kind = compiler::KernelArtifactPayloadKind::MslSource;
  msl_multiply.element_type = ov::element::f32;

  EXPECT_TRUE(compiler::allow_stage_input_activation_fusion(
      msl_multiply, 0, ActivationKind::Relu));
  EXPECT_TRUE(compiler::allow_stage_input_activation_fusion(
      msl_multiply, 1, ActivationKind::Swish));
  EXPECT_FALSE(compiler::allow_stage_input_activation_fusion(
      msl_multiply, 2, ActivationKind::Relu));
  EXPECT_FALSE(compiler::allow_stage_input_activation_fusion(
      msl_multiply, 0, ActivationKind::Abs));

  auto integral_multiply = msl_multiply;
  integral_multiply.element_type = ov::element::i32;
  EXPECT_FALSE(compiler::allow_stage_input_activation_fusion(
      integral_multiply, 0, ActivationKind::Relu));

  auto opencl_multiply = msl_multiply;
  opencl_multiply.payload_kind =
      compiler::KernelArtifactPayloadKind::OpenClSource;
  EXPECT_FALSE(compiler::allow_stage_input_activation_fusion(
      opencl_multiply, 0, ActivationKind::Relu));

  compiler::PipelineStageFusionContract msl_rms;
  msl_rms.op_family = "RMS";
  msl_rms.payload_kind = compiler::KernelArtifactPayloadKind::MslSource;
  EXPECT_TRUE(compiler::allow_stage_residual_add_fusion(msl_rms));

  auto opencl_rms = msl_rms;
  opencl_rms.payload_kind = compiler::KernelArtifactPayloadKind::OpenClSource;
  EXPECT_FALSE(compiler::allow_stage_residual_add_fusion(opencl_rms));
}

TEST_F(GfxBackendArchitectureContractTest,
       PipelineStagePlanBuilderMarksOutputsAndDeduplicatesAliases) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto relu = std::make_shared<ov::op::v0::Relu>(parameter);
  auto result = std::make_shared<ov::op::v0::Result>(relu);
  ov::Model model(ov::ResultVector{result}, ov::ParameterVector{parameter},
                  "pipeline_stage_plan_contract");

  const auto model_outputs = compiler::collect_model_output_ports(model);
  compiler::PipelineStagePlanBuilder builder(model_outputs);
  auto plan = builder.make_stage_plan(relu, 7);

  ASSERT_EQ(plan.outputs.size(), 1u);
  EXPECT_EQ(plan.runtime_stage_index, 7u);
  EXPECT_TRUE(plan.outputs[0].is_model_output);
  EXPECT_EQ(plan.outputs[0].source_node.get(), relu.get());
  EXPECT_EQ(plan.outputs[0].source_port, 0u);

  builder.append_output_alias(plan, relu, 0, 0);
  builder.append_output_alias(plan, relu, 0, 0);
  ASSERT_EQ(plan.output_aliases.size(), 1u);
  EXPECT_EQ(plan.output_aliases[0].node.get(), relu.get());

  compiler::PipelineOutputAliasMap aliases;
  builder.record_output_alias(aliases, relu.get(), 0, 3);
  const auto remapped = builder.remap_input_link(aliases, relu, 0);
  EXPECT_EQ(remapped.node.get(), relu.get());
  EXPECT_EQ(remapped.port, 3u);

  auto fused = builder.make_fused_stage_plan(relu, 2, 9);
  builder.describe_output(fused, 1, relu, 0);
  ASSERT_EQ(fused.outputs.size(), 2u);
  EXPECT_EQ(fused.runtime_stage_index, 9u);
  EXPECT_TRUE(fused.outputs[1].is_model_output);
  EXPECT_EQ(fused.outputs[1].source_node.get(), relu.get());
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerOwnsAbsorbedInputTransformPlanning) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto compiled_model_source =
      read_text_file(module_root / "src/plugin/compiled_model.cpp");
  const auto planner_header =
      read_text_file(module_root / "src/compiler/pipeline_stage_plan.hpp");
  const auto planner_source =
      read_text_file(module_root / "src/compiler/pipeline_stage_plan.cpp");

  EXPECT_EQ(compiled_model_source.find("evaluate_constant_i64"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("is_absorbable_transpose_candidate"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("is_supported_absorbing_consumer"),
            std::string::npos);
  EXPECT_NE(planner_header.find("PipelineAbsorbedInputTransformPlan"),
            std::string::npos);
  EXPECT_NE(planner_source.find("plan_absorbed_input_transforms"),
            std::string::npos);
  EXPECT_EQ(planner_header.find("runtime/gfx_input_transform"),
            std::string::npos);

  auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                           ov::Shape{1, 2, 3});
  auto permutation =
      ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 2, 1});
  auto transpose =
      std::make_shared<ov::op::v1::Transpose>(parameter, permutation);
  auto bias =
      ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
  auto add = std::make_shared<ov::op::v1::Add>(transpose, bias);
  auto result = std::make_shared<ov::op::v0::Result>(add);
  ov::Model model(ov::ResultVector{result}, ov::ParameterVector{parameter},
                  "absorbed_input_transform_contract");

  const auto ordered_ops = model.get_ordered_ops();
  const auto model_outputs = compiler::collect_model_output_ports(model);
  const std::unordered_set<const ov::Node *> fused_nodes;
  const auto transform_plan = compiler::plan_absorbed_input_transforms(
      ordered_ops, model_outputs, fused_nodes);

  EXPECT_NE(transform_plan.absorbed_nodes.find(transpose.get()),
            transform_plan.absorbed_nodes.end());
  auto transform_it = transform_plan.input_transforms.find(add.get());
  ASSERT_NE(transform_it, transform_plan.input_transforms.end());
  auto input_it = transform_it->second.find(0);
  ASSERT_NE(input_it, transform_it->second.end());
  EXPECT_EQ(input_it->second.source_shape, (ov::Shape{1, 2, 3}));
  EXPECT_EQ(input_it->second.transpose_permutation,
            (std::vector<int64_t>{0, 2, 1}));
  EXPECT_TRUE(input_it->second.has_transpose());
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerBackendIdentityContractsDoNotDependOnRuntimeHeaders) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto assert_no_runtime_backend_identity_include =
      [&](const char *relative_path) {
        const auto source = read_text_file(module_root / relative_path);
        EXPECT_EQ(source.find("runtime/gfx_backend_utils.hpp"),
                  std::string::npos)
            << relative_path;
        EXPECT_EQ(source.find("runtime/gpu_device_info.hpp"), std::string::npos)
            << relative_path;
      };

  assert_no_runtime_backend_identity_include("src/compiler/backend_target.hpp");
  assert_no_runtime_backend_identity_include(
      "src/compiler/stage_compiler_policy.hpp");
  assert_no_runtime_backend_identity_include(
      "src/compiler/stage_placement.hpp");

  const auto backend_target =
      read_text_file(module_root / "src/compiler/backend_target.hpp");
  EXPECT_NE(backend_target.find("common/gpu_backend.hpp"), std::string::npos);

  const auto stage_compiler_policy =
      read_text_file(module_root / "src/compiler/stage_compiler_policy.hpp");
  EXPECT_NE(stage_compiler_policy.find("common/gpu_backend.hpp"),
            std::string::npos);
  EXPECT_NE(stage_compiler_policy.find("common/gpu_device_profile.hpp"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       MlirSourcePlanBuildersUseCompilerOwnedStageCompilerPolicyResolver) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto resolver_source =
      read_text_file(module_root / "src/compiler/stage_compiler_policy.cpp");
  EXPECT_NE(resolver_source.find("compiler/backend_registry.hpp"),
            std::string::npos);
  EXPECT_NE(resolver_source.find("BackendRegistry::default_registry()"),
            std::string::npos);
  EXPECT_NE(
      resolver_source.find("make_stage_compiler_policy_from_capabilities("),
      std::string::npos);

  const auto assert_uses_shared_resolver = [&](const char *relative_path) {
    const auto source = read_text_file(module_root / relative_path);
    EXPECT_EQ(source.find("compiler/backend_registry.hpp"), std::string::npos)
        << relative_path;
    EXPECT_EQ(source.find("BackendRegistry::default_registry()"),
              std::string::npos)
        << relative_path;
    EXPECT_EQ(source.find("metal_stage_compiler_policy"), std::string::npos)
        << relative_path;
    EXPECT_EQ(source.find("mlir/gfx_stage_compiler_policy_resolver.hpp"),
              std::string::npos)
        << relative_path;
    EXPECT_NE(source.find("compiler/stage_compiler_policy.hpp"),
              std::string::npos)
        << relative_path;
    EXPECT_NE(source.find(
                  "compiler::resolve_stage_compiler_policy(GpuBackend::Metal)"),
              std::string::npos)
        << relative_path;
  };

  assert_uses_shared_resolver("src/mlir/msl_codegen_apple_mps.cpp");
  assert_uses_shared_resolver("src/mlir/msl_codegen_apple_msl_dispatch.cpp");
  assert_uses_shared_resolver("src/mlir/msl_codegen_matmul_metal.cpp");
}

TEST_F(GfxBackendArchitectureContractTest,
       CompiledModelUsesCompilerOwnedPrecisionSensitiveFusionPolicy) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto compiled_model_source =
      read_text_file(module_root / "src/plugin/compiled_model.cpp");
  const auto fusion_source =
      read_text_file(module_root / "src/compiler/pipeline_stage_fusion.cpp");

  EXPECT_EQ(compiled_model_source.find("runtime/gfx_stage_policy.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("select_stage_optimization_plan("),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("GpuBackend::Metal"), std::string::npos);
  EXPECT_EQ(compiled_model_source.find("AppleMps"), std::string::npos);
  EXPECT_EQ(compiled_model_source.find("fusion_precision_sensitive_mpsrt"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("PrecisionSensitiveFusionQuery"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find(
                "compiler::allow_precision_sensitive_arithmetic_fusion("),
            std::string::npos);
  EXPECT_NE(fusion_source.find(
                "compiler_allows_precision_sensitive_arithmetic_fusion_group"),
            std::string::npos);
  EXPECT_NE(fusion_source.find("allow_precision_sensitive_arithmetic_fusion("),
            std::string::npos);
}

compiler::TensorContract
make_tensor_contract(compiler::TensorContractRole role) {
  compiler::TensorContract contract;
  contract.logical_name =
      role == compiler::TensorContractRole::TensorInput ? "input0" : "output0";
  contract.memory_region_id = role == compiler::TensorContractRole::TensorInput
                                  ? "stage_0.input_0"
                                  : "stage_0.output_0";
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

compiler::MemoryRegion
make_memory_region_for_contract(const compiler::TensorContract &contract,
                                size_t stage_id) {
  compiler::MemoryRegion region;
  region.region_id = contract.memory_region_id;
  region.logical_tensor_name = contract.logical_name;
  region.kind = contract.role == compiler::TensorContractRole::TensorInput
                    ? compiler::MemoryRegionKind::ExternalTensor
                    : compiler::MemoryRegionKind::TransientTensor;
  region.element_type = contract.element_type;
  region.partial_shape = contract.partial_shape;
  region.layout = contract.layout;
  region.storage_kind = contract.storage_kind;
  region.alias_group =
      contract.role == compiler::TensorContractRole::TensorInput
          ? contract.memory_region_id
          : "stage_" + std::to_string(stage_id);
  region.lifetime = {0, stage_id};
  region.external_binding =
      contract.role == compiler::TensorContractRole::TensorInput;
  return region;
}

compiler::MemoryPlan
make_single_stage_memory_plan(const compiler::StageRecord &stage) {
  compiler::MemoryPlan plan;
  plan.schema_version = 1;
  for (const auto &input : stage.inputs) {
    auto region = make_memory_region_for_contract(input, stage.stage_id);
    compiler::AliasGroup group;
    group.group_id = region.alias_group;
    group.region_ids.push_back(region.region_id);
    plan.alias_groups.push_back(std::move(group));
    plan.regions.push_back(std::move(region));
  }
  compiler::TransientArena arena;
  arena.arena_id = "transient_device_buffer_arena";
  arena.storage_kind = "device_buffer";
  compiler::AliasGroup output_group;
  output_group.group_id = stage.memory.alias_group;
  for (const auto &output : stage.outputs) {
    auto region = make_memory_region_for_contract(output, stage.stage_id);
    output_group.region_ids.push_back(region.region_id);
    arena.region_ids.push_back(region.region_id);
    plan.regions.push_back(std::move(region));
  }
  if (!output_group.region_ids.empty()) {
    plan.alias_groups.push_back(std::move(output_group));
  }
  if (!arena.region_ids.empty()) {
    plan.transient_arenas.push_back(std::move(arena));
  }
  return plan;
}

compiler::ManifestBundle make_single_payload_route_manifest(
    LoweringRouteKind route_kind, std::string backend_domain,
    std::string kernel_unit_id, std::string kernel_unit_kind,
    std::string op_family = "PayloadRoute",
    bool requires_runtime_shape_args = false) {
  compiler::StageRecord stage;
  stage.stage_id = 0;
  stage.stable_record_key = 0x1234u;
  stage.source_node_name = op_family;
  stage.normalized_op_family = std::move(op_family);
  stage.execution_kind = route_kind;
  stage.backend_domain = std::move(backend_domain);
  stage.kernel_unit_id = std::move(kernel_unit_id);
  stage.kernel_unit_kind = std::move(kernel_unit_kind);
  stage.requires_runtime_shape_args = requires_runtime_shape_args;
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
  manifest.memory_plan = make_single_stage_memory_plan(stage);
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

compiler::CacheEnvelopeBuildOptions make_test_cache_options(
    const ov::Model &model,
    std::string backend_capabilities_fingerprint = "test-capabilities-v1",
    std::string backend_compiler_revision = "test-backend-compiler-v1",
    std::string driver_identity = "test-driver-v1") {
  compiler::CacheEnvelopeBuildOptions options;
  options.model_fingerprint = compiler::make_model_cache_fingerprint(model);
  options.backend_capabilities_fingerprint =
      std::move(backend_capabilities_fingerprint);
  options.backend_compiler_revision = std::move(backend_compiler_revision);
  options.driver_identity = std::move(driver_identity);
  return options;
}

compiler::PlannedOperation
make_metadata_planned_operation(const std::shared_ptr<const ov::Node> &node,
                                compiler::TensorLayoutPlan layout,
                                bool requires_runtime_shape_args = false) {
  compiler::PlannedOperation op;
  op.source_node = node;
  op.node_name = node ? node->get_friendly_name() : "metadata";
  op.type_name = node ? node->get_type_name() : "Unknown";
  op.kernel_unit = compiler::KernelUnit::describe(
      LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
      "opencl", "metadata", requires_runtime_shape_args);
  op.layout = layout;
  op.profitability_score = 1.0;
  op.input_element_types = {"f32"};
  op.input_shapes = {"{1,2,3}"};
  op.output_element_types = {"f32"};
  op.output_shapes = {"{1,2,3}"};
  return op;
}

RuntimeStageExecutableDescriptor
make_runtime_descriptor_for_layout(const std::shared_ptr<const ov::Node> &node,
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
       AppleTransposedSdpaVendorContractOwnsShapeAndTypeContract) {
  GfxAppleMpsVendorPrimitiveContract contract;
  EXPECT_TRUE(gfx_apple_make_mps_transposed_sdpa_contract(
      "attention", ov::element::f32, {1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 6, 5},
      {1, 2, 6, 4}, 0.5f, contract));
  EXPECT_TRUE(contract.valid);
  EXPECT_EQ(contract.descriptor.kind, GfxAppleMpsVendorPrimitiveKind::Sdpa);
  EXPECT_EQ(contract.descriptor.sdpa.layout, GfxMpsrtSdpaLayoutTransposedBHDN);
  EXPECT_FLOAT_EQ(contract.descriptor.sdpa.scale, 0.5f);
  ASSERT_TRUE(contract.external_buffer_abi.valid);
  EXPECT_EQ(contract.external_buffer_abi.buffer_count, 4u);
  EXPECT_EQ(contract.external_buffer_abi.output_buffer_count, 1u);

  EXPECT_FALSE(gfx_apple_make_mps_transposed_sdpa_contract(
      "attention", ov::element::i32, {1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 6, 5},
      {1, 2, 6, 4}, 0.5f, contract));
  EXPECT_FALSE(gfx_apple_make_mps_transposed_sdpa_contract(
      "attention", ov::element::f32, {1, 2, 3, 4}, {1, 2, 3, 5}, {1, 2, 6, 5},
      {1, 2, 7, 4}, 0.5f, contract));
  EXPECT_FALSE(gfx_apple_make_mps_transposed_sdpa_contract(
      "attention", ov::element::f32, {1, 2, 3, 4}, {1, 2, 3}, {1, 2, 6, 5},
      {1, 2, 6, 4}, 0.5f, contract));
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerRegistryContainsAllProductionBackendModules) {
  const auto &registry = compiler::BackendRegistry::default_registry();
  const auto metal = registry.resolve(GpuBackend::Metal);
  const auto opencl = registry.resolve(GpuBackend::OpenCL);
  const auto targets = registry.available_targets();
  const auto expected_target_count =
      static_cast<size_t>(kGfxBackendMetalAvailable) +
      static_cast<size_t>(kGfxBackendOpenCLAvailable);

  EXPECT_EQ(static_cast<bool>(metal), kGfxBackendMetalAvailable);
  EXPECT_EQ(static_cast<bool>(opencl), kGfxBackendOpenCLAvailable);
  EXPECT_EQ(targets.size(), expected_target_count);

  if (metal) {
    EXPECT_TRUE(metal->capabilities().stage_placement());
    EXPECT_EQ(metal->target().backend(), GpuBackend::Metal);
  }
  if (opencl) {
    EXPECT_TRUE(opencl->capabilities().stage_placement());
    EXPECT_EQ(opencl->target().backend(), GpuBackend::OpenCL);
  }
}

TEST_F(GfxBackendArchitectureContractTest,
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
  if (const auto metal = registry.resolve(GpuBackend::Metal)) {
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

  if (const auto opencl = registry.resolve(GpuBackend::OpenCL)) {
    EXPECT_FALSE(
        opencl->materialize_vendor_attention_artifact(0x1234u, plan).valid());
  }
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

  compiler::PipelineStageBuildRequest stage_build_request;
  EXPECT_EQ(stage_build_request.backend, GpuBackend::Unknown);

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
       CompilerOwnsMemoryPlanInManifestExecutableAndCacheIdentity) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                       ov::Shape{1, 2, 3});
  const auto layout = compiler::select_tensor_layout_plan("Relu", input);
  compiler::LoweringPlan lowering_plan;
  lowering_plan.target =
      compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  lowering_plan.operations.push_back(
      make_metadata_planned_operation(input, layout));

  const auto manifest = compiler::ManifestBuilder{}.build(lowering_plan);
  ASSERT_TRUE(manifest.verify().valid());
  ASSERT_TRUE(manifest.memory_plan.valid());
  ASSERT_EQ(manifest.stages.size(), 1u);
  const auto &stage = manifest.stages.front();
  ASSERT_FALSE(stage.inputs.empty());
  ASSERT_FALSE(stage.outputs.empty());
  EXPECT_TRUE(
      manifest.memory_plan.has_region(stage.inputs.front().memory_region_id));
  EXPECT_TRUE(
      manifest.memory_plan.has_region(stage.outputs.front().memory_region_id));
  EXPECT_TRUE(manifest.memory_plan.has_alias_group(stage.memory.alias_group));
  EXPECT_FALSE(manifest.memory_plan.hidden_host_copies_allowed);

  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.verify().valid());
  EXPECT_EQ(compiler::make_memory_plan_fingerprint(executable.memory_plan),
            compiler::make_memory_plan_fingerprint(manifest.memory_plan));

  auto stale_manifest = manifest;
  ASSERT_FALSE(stale_manifest.memory_plan.regions.empty());
  stale_manifest.memory_plan.regions.front().layout = "stale_layout";
  EXPECT_NE(compiler::make_manifest_cache_hash(stale_manifest),
            compiler::make_manifest_cache_hash(manifest));
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeExecutableDescriptorCarriesCompilerOwnedMemoryPlan) {
  for (const auto *backend_domain : {"opencl", "metal"}) {
    SCOPED_TRACE(backend_domain);
    const auto manifest = make_single_payload_route_manifest(
        LoweringRouteKind::Metadata, backend_domain, "metadata", "metadata");
    ASSERT_TRUE(manifest.valid());

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());
    ASSERT_FALSE(executable.memory_plan.regions.empty());
    ASSERT_FALSE(executable.memory_plan.alias_groups.empty());
    ASSERT_FALSE(executable.memory_plan.transient_arenas.empty());

    const auto runtime_descriptor =
        RuntimeExecutableDescriptorBuilder{}.build(executable);
    const auto verification = runtime_descriptor.verify(executable);
    ASSERT_TRUE(verification.valid())
        << (verification.diagnostics.empty()
                ? std::string{}
                : verification.diagnostics.front());

    EXPECT_EQ(runtime_descriptor.memory_plan.schema_version,
              executable.memory_plan.schema_version);
    EXPECT_EQ(runtime_descriptor.memory_plan.fingerprint,
              compiler::make_memory_plan_fingerprint(executable.memory_plan));
    EXPECT_EQ(runtime_descriptor.memory_plan.regions.size(),
              executable.memory_plan.regions.size());
    EXPECT_EQ(runtime_descriptor.memory_plan.alias_groups.size(),
              executable.memory_plan.alias_groups.size());
    EXPECT_EQ(runtime_descriptor.memory_plan.transient_arenas.size(),
              executable.memory_plan.transient_arenas.size());
    EXPECT_FALSE(runtime_descriptor.memory_plan.hidden_host_copies_allowed);
    EXPECT_TRUE(runtime_descriptor.memory_plan.has_region(
        executable.memory_plan.regions.front().region_id));
    EXPECT_TRUE(runtime_descriptor.memory_plan.has_alias_group(
        executable.memory_plan.alias_groups.front().group_id));
    EXPECT_EQ(runtime_descriptor.memory_plan.regions.front().layout,
              executable.memory_plan.regions.front().layout);
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    const auto &runtime_stage = runtime_descriptor.stages.front();
    ASSERT_EQ(runtime_stage.input_bindings.size(), 1u);
    ASSERT_EQ(runtime_stage.output_bindings.size(), 1u);
    EXPECT_EQ(
        runtime_stage.input_bindings.front().memory_region_id,
        executable.manifest.stages.front().inputs.front().memory_region_id);
    EXPECT_EQ(
        runtime_stage.output_bindings.front().memory_region_id,
        executable.manifest.stages.front().outputs.front().memory_region_id);
    EXPECT_EQ(runtime_stage.input_bindings.front().alias_group,
              executable.memory_plan.regions.front().alias_group);

    std::vector<GpuTensor *> input_slots(runtime_stage.input_bindings.size(),
                                         nullptr);
    std::vector<GpuTensor *> output_slots(runtime_stage.output_bindings.size(),
                                          nullptr);
    const auto binding_table = ResourceBindingTable::for_stage(
        input_slots, output_slots, runtime_stage);
    EXPECT_TRUE(binding_table.compatible_with(runtime_stage));
    EXPECT_EQ(binding_table.input_region_ids().front(),
              runtime_stage.input_bindings.front().memory_region_id);
    EXPECT_EQ(binding_table.output_region_ids().front(),
              runtime_stage.output_bindings.front().memory_region_id);

    output_slots.clear();
    const auto incomplete_binding_table = ResourceBindingTable::for_stage(
        input_slots, output_slots, runtime_stage);
    EXPECT_FALSE(incomplete_binding_table.compatible_with(runtime_stage));

    auto stale_region_descriptor = runtime_descriptor;
    stale_region_descriptor.memory_plan.regions.front().layout =
        "stale_runtime_layout";
    const auto stale_region_verification =
        stale_region_descriptor.verify(executable);
    EXPECT_FALSE(stale_region_verification.valid());
    EXPECT_TRUE(has_diagnostic_containing(stale_region_verification.diagnostics,
                                          "memory region drift"));

    auto stale_fingerprint_descriptor = runtime_descriptor;
    stale_fingerprint_descriptor.memory_plan.fingerprint =
        "stale-runtime-memory-plan";
    const auto stale_fingerprint_verification =
        stale_fingerprint_descriptor.verify(executable);
    EXPECT_FALSE(stale_fingerprint_verification.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_fingerprint_verification.diagnostics,
                                  "memory plan fingerprint drift"));

    auto stale_binding_descriptor = runtime_descriptor;
    stale_binding_descriptor.stages.front()
        .input_bindings.front()
        .memory_region_id = "stale-runtime-input-region";
    const auto stale_binding_verification =
        stale_binding_descriptor.verify(executable);
    EXPECT_FALSE(stale_binding_verification.valid());
    EXPECT_TRUE(has_diagnostic_containing(
        stale_binding_verification.diagnostics, "input binding drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CompiledModelDoesNotPrepareBackendKernelsDuringPipelineBuild) {
  const auto root = find_gfx_module_root_for_source_contract();
  const auto compiled_model_source =
      read_text_file(root / "src/plugin/compiled_model.cpp");
  EXPECT_EQ(compiled_model_source.find("stage.stage->compile("),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("stage->compile("), std::string::npos);

  const auto infer_pipeline_source =
      read_text_file(root / "src/plugin/infer_pipeline.cpp");
  EXPECT_NE(infer_pipeline_source.find("prepare_stage_runtime_executable"),
            std::string::npos);
  EXPECT_NE(infer_pipeline_source.find("RuntimeSession"), std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("stage.stage->compile("),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("stage->compile("), std::string::npos);

  const auto runtime_session_source =
      read_text_file(root / "src/runtime/runtime_session.cpp");
  EXPECT_NE(runtime_session_source.find("PreparedKernelExecutable::prepare"),
            std::string::npos);
  EXPECT_NE(runtime_session_source.find("stage.compile("), std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonInferPipelineDoesNotRepairMissingRuntimeStageIndex) {
  const auto root = find_gfx_module_root_for_source_contract();
  const auto infer_pipeline_source =
      read_text_file(root / "src/plugin/infer_pipeline.cpp");

  EXPECT_NE(infer_pipeline_source.find(
                "compiler-owned runtime stage index is required"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find(
                "desc.runtime_stage_index != PipelineStageDesc::npos\n"
                "                ? desc.runtime_stage_index"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find(": stage_id"), std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeSliceShapeArgsComeFromKernelUnitManifestAndDescriptor) {
  struct Case {
    GpuBackend backend;
    bool requires_runtime_shape_args;
  };

  for (const auto test_case :
       {Case{GpuBackend::OpenCL, true}, Case{GpuBackend::Metal, false}}) {
    SCOPED_TRACE(backend_to_string(test_case.backend));
    compiler::LoweringPlan plan;
    plan.target = compiler::BackendTarget::from_backend(test_case.backend);
    compiler::PlannedOperation op;
    op.node_name = "Slice";
    op.type_name = "Slice";
    op.kernel_unit = compiler::KernelUnit::describe(
        LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
        plan.target.backend_id(), "Slice",
        test_case.requires_runtime_shape_args);
    op.layout = compiler::TensorLayoutPlan{};
    op.profitability_score = 1.0;
    op.input_element_types = {"f32"};
    op.input_shapes = {"{1,2,3}"};
    op.output_element_types = {"f32"};
    op.output_shapes = {"{1,2,3}"};
    plan.operations.push_back(std::move(op));

    const auto manifest = compiler::ManifestBuilder{}.build(plan);
    ASSERT_TRUE(manifest.valid());
    ASSERT_EQ(manifest.stages.size(), 1u);
    EXPECT_EQ(manifest.stages.front().requires_runtime_shape_args,
              test_case.requires_runtime_shape_args);

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());
    ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
    EXPECT_EQ(executable.artifact_descriptors.front()
                  .kernel.requires_runtime_shape_args,
              test_case.requires_runtime_shape_args);

    const auto runtime_descriptor =
        RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(runtime_descriptor.valid(executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().requires_runtime_shape_args,
              test_case.requires_runtime_shape_args);

    auto stale_descriptor = runtime_descriptor;
    stale_descriptor.stages.front().requires_runtime_shape_args =
        !test_case.requires_runtime_shape_args;
    const auto stale_result = stale_descriptor.verify(executable);
    EXPECT_FALSE(stale_result.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_result.diagnostics, "artifact drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonInferPipelineUsesDescriptorOwnedSliceShapeArgPolicy) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto infer_pipeline_source =
      read_text_file(module_root / "src/plugin/infer_pipeline.cpp");
  const auto infer_pipeline_header =
      read_text_file(module_root / "src/plugin/infer_pipeline.hpp");

  EXPECT_EQ(infer_pipeline_source.find("backend == GpuBackend::OpenCL"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("backend == GpuBackend::Metal"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_header.find("GpuBackend backend"),
            std::string::npos);
  EXPECT_NE(infer_pipeline_source.find("requires_runtime_shape_args"),
            std::string::npos);
  EXPECT_NE(
      infer_pipeline_source.find("runtime_stage_descriptor_or_null(stage)"),
      std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonExecutableBundleDoesNotInferSliceShapeArgPolicy) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto bundle_source =
      read_text_file(module_root / "src/compiler/executable_bundle.cpp");

  EXPECT_EQ(bundle_source.find("requires_runtime_shape_args_for_stage"),
            std::string::npos);
  EXPECT_EQ(bundle_source.find("normalized_op_family != \"Slice\""),
            std::string::npos);
  EXPECT_NE(bundle_source.find("kernel.requires_runtime_shape_args =\n"
                               "      stage.requires_runtime_shape_args"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CacheEnvelopeContainsDocsRequiredCompilerOwnedIdentity) {
  const auto manifest = make_single_payload_route_manifest(
      LoweringRouteKind::Metadata, "opencl", "metadata", "metadata");
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.verify().valid());

  const auto model = make_relu_model();
  const auto envelope = compiler::CacheEnvelopeBuilder{}.build(
      executable, make_test_cache_options(*model));

  const auto verification = envelope.verify(executable);
  EXPECT_TRUE(verification.valid())
      << (verification.diagnostics.empty() ? std::string{}
                                           : verification.diagnostics.front());
  EXPECT_FALSE(envelope.key.model_fingerprint.empty());
  EXPECT_EQ(envelope.key.manifest_hash,
            compiler::make_manifest_cache_hash(manifest));
  EXPECT_EQ(envelope.key.target_fingerprint, manifest.target_fingerprint);
  EXPECT_EQ(envelope.key.compile_options_hash,
            compiler::make_executable_compile_options_hash(executable));
  EXPECT_FALSE(envelope.key.backend_capabilities_fingerprint.empty());
  EXPECT_FALSE(envelope.key.compiler_revision.empty());
  EXPECT_FALSE(envelope.key.backend_compiler_revision.empty());
  EXPECT_FALSE(envelope.key.driver_identity.empty());
  EXPECT_FALSE(envelope.key.kernel_unit_versions.empty());
  EXPECT_FALSE(envelope.key.stable_key.empty());
  EXPECT_EQ(envelope.manifest.target_fingerprint, manifest.target_fingerprint);
  EXPECT_EQ(envelope.artifact_descriptors.size(),
            executable.artifact_descriptors.size());
}

TEST_F(GfxBackendArchitectureContractTest,
       CacheEnvelopeRejectsStaleManifestDriverAndKernelIdentity) {
  const auto manifest = make_single_payload_route_manifest(
      LoweringRouteKind::Metadata, "metal", "metadata", "metadata");
  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.verify().valid());

  const auto model = make_relu_model();
  const auto envelope = compiler::CacheEnvelopeBuilder{}.build(
      executable, make_test_cache_options(*model));
  ASSERT_TRUE(envelope.verify(executable).valid());

  auto stale_manifest = envelope;
  stale_manifest.key.manifest_hash = "stale-manifest";
  EXPECT_FALSE(stale_manifest.verify(executable).valid());
  EXPECT_TRUE(has_diagnostic_containing(
      stale_manifest.verify(executable).diagnostics, "manifest hash drift"));

  auto missing_driver = envelope;
  missing_driver.key.driver_identity.clear();
  EXPECT_FALSE(missing_driver.verify(executable).valid());
  EXPECT_TRUE(has_diagnostic_containing(
      missing_driver.verify(executable).diagnostics, "driver identity"));

  auto stale_kernel_unit = envelope;
  stale_kernel_unit.key.kernel_unit_versions.front() = "stale-kernel-unit";
  EXPECT_FALSE(stale_kernel_unit.verify(executable).valid());
  EXPECT_TRUE(has_diagnostic_containing(
      stale_kernel_unit.verify(executable).diagnostics,
      "kernel unit versions drift"));
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
  const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                             ov::Shape{2, 3});
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
