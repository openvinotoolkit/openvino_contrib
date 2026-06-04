// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backends/metal/compiler/apple_vendor_descriptors.hpp"
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
#include "compiler/runtime_executable_descriptor_builder.hpp"
#include "compiler/tensor_layout.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "runtime/backend_request_state.hpp"
#include "runtime/backend_runtime.hpp"
#include "runtime/backend_runtime_provider.hpp"
#include "runtime/backend_stage_factory.hpp"
#include "runtime/executable_descriptor.hpp"
#include "runtime/execution_dispatcher.hpp"
#include "runtime/gpu_buffer.hpp"
#include "runtime/gpu_memory_ops.hpp"
#include "runtime/kernel_launch_plan.hpp"
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
    "Backend runtime state must expose stage materialization through the "
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
    std::is_same_v<decltype(PipelineStageRuntimeMaterializationRequest::
                                runtime_plan),
                   const PipelineStageRuntimePlan *>,
    "Runtime materializer must consume a runtime-facing stage plan instead of "
    "the compiler build result");

static_assert(std::is_same_v<decltype(&compiler::build_pipeline_stage_plan),
                             compiler::PipelineStageBuildResult (*)(
                                 const compiler::PipelineStageBuildRequest &)>,
              "Compiler pipeline stage builder must own stage planning outside "
              "CompiledModel without materializing runtime stages");

static_assert(
    std::is_same_v<decltype(&materialize_pipeline_stage_descriptors),
                   std::vector<PipelineStageDesc> (*)(
                       const PipelineStageRuntimeMaterializationRequest &)>,
    "Runtime materializer must be the only shared path that turns compiler "
    "stage plans into runtime PipelineStageDesc objects");

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

static_assert(
    std::is_same_v<decltype(BackendRuntimeProvider::execute_infer),
                   void (*)(InferRequest &,
                            const std::shared_ptr<const CompiledModel> &)>,
    "BackendRuntimeProvider must expose one narrow infer execution callback "
    "instead of requiring common InferRequest backend switches");

static_assert(
    std::is_same_v<decltype(&compiler::BackendCapabilities::precision),
                   const compiler::PrecisionCapabilities &(
                       compiler::BackendCapabilities::*)() const noexcept>,
    "Public precision capabilities must be owned by immutable "
    "compiler backend capabilities, not by runtime helpers");

static_assert(
    std::is_same_v<decltype(&compiler::BackendCapabilities::artifact_formats),
                   const compiler::ArtifactFormatCapabilities &(
                       compiler::BackendCapabilities::*)() const noexcept>,
    "Public artifact capabilities must be owned by immutable "
    "compiler backend capabilities, not by runtime helpers");

bool has_diagnostic_containing(const std::vector<std::string> &diagnostics,
                               std::string_view needle) {
  for (const auto &diagnostic : diagnostics) {
    if (diagnostic.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

GpuTensor make_test_launch_plan_tensor(uint64_t allocation_uid) {
  GpuTensor tensor;
  tensor.buf.buffer = reinterpret_cast<GpuBufferHandle>(
      static_cast<uintptr_t>(0x1000u + allocation_uid));
  tensor.buf.size = 64;
  tensor.buf.allocation_uid = allocation_uid;
  tensor.shape = ov::Shape{16};
  tensor.expected_type = ov::element::f32;
  return tensor;
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

std::vector<std::string> find_cmake_call_blocks(std::string_view source,
                                                std::string_view call_prefix) {
  std::vector<std::string> blocks;
  size_t pos = 0;
  while ((pos = source.find(call_prefix, pos)) != std::string_view::npos) {
    const auto end = source.find(')', pos);
    if (end == std::string_view::npos) {
      blocks.emplace_back(source.substr(pos));
      break;
    }
    blocks.emplace_back(source.substr(pos, end - pos + 1));
    pos = end + 1;
  }
  return blocks;
}

std::filesystem::path find_gfx_module_root_for_source_contract() {
  auto source_path = std::filesystem::path(__FILE__);
  if (source_path.is_relative()) {
    source_path = std::filesystem::absolute(source_path);
  }
  for (auto dir = source_path.parent_path(); !dir.empty();) {
    if (std::filesystem::exists(dir / "src/compiler/stage_policy.cpp")) {
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
    if (std::filesystem::exists(candidate / "src/compiler/stage_policy.cpp")) {
      return candidate;
    }
  }
  throw std::runtime_error("GFX test: failed to locate gfx_plugin module root");
}

TEST_F(GfxBackendArchitectureContractTest,
       KernelLaunchPlanBindsManifestRolesForConstAndRuntimeParams) {
  const std::vector<GfxKernelBufferRole> roles = {
      GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::ConstTensor,
      GfxKernelBufferRole::TensorOutput,
      GfxKernelBufferRole::ScalarParam,
      GfxKernelBufferRole::RuntimeParams,
      GfxKernelBufferRole::RuntimeParams,
  };

  auto input = make_test_launch_plan_tensor(1);
  auto const_tensor = make_test_launch_plan_tensor(2);
  auto output = make_test_launch_plan_tensor(3);
  auto runtime0 = make_test_launch_plan_tensor(4);
  auto runtime1 = make_test_launch_plan_tensor(5);

  const std::vector<size_t> direct_input_indices = {2};
  const std::vector<uint32_t> scalar_values = {42u};
  std::vector<GpuTensor *> outputs = {&output};
  std::vector<GpuTensor *> const_tensors = {&const_tensor};
  std::vector<GpuTensor> runtime_params = {runtime0, runtime1};

  size_t resolved_input_index = 0;
  auto plan = build_role_ordered_kernel_launch_plan<uint32_t>(
      roles, direct_input_indices, scalar_values, outputs, const_tensors,
      runtime_params,
      [&](size_t input_idx) -> GpuTensor * {
        resolved_input_index = input_idx;
        return &input;
      },
      "unit_launch_plan");

  ASSERT_EQ(plan.args.size(), roles.size());
  EXPECT_TRUE(kernel_args_dense(plan.args));
  EXPECT_EQ(resolved_input_index, 2u);

  EXPECT_EQ(plan.args[0].kind, KernelArg::Kind::Buffer);
  EXPECT_EQ(plan.args[0].index, 0u);
  EXPECT_EQ(plan.args[0].buffer.allocation_uid, 1u);
  EXPECT_EQ(plan.args[1].buffer.allocation_uid, 2u);
  EXPECT_EQ(plan.args[2].buffer.allocation_uid, 3u);
  EXPECT_EQ(plan.args[3].kind, KernelArg::Kind::Bytes);
  EXPECT_EQ(plan.args[3].index, 3u);
  ASSERT_EQ(plan.args[3].byte_size, sizeof(uint32_t));
  uint32_t scalar_value = 0;
  std::memcpy(&scalar_value, plan.args[3].bytes, sizeof(scalar_value));
  EXPECT_EQ(scalar_value, 42u);
  EXPECT_EQ(plan.args[4].buffer.allocation_uid, 4u);
  EXPECT_EQ(plan.args[5].buffer.allocation_uid, 5u);
}

TEST_F(GfxBackendArchitectureContractTest,
       KernelLaunchPlanRejectsUnconsumedConstAndRuntimeParamBuffers) {
  const std::vector<GfxKernelBufferRole> roles = {
      GfxKernelBufferRole::TensorInput,
      GfxKernelBufferRole::TensorOutput,
  };

  auto input = make_test_launch_plan_tensor(1);
  auto output = make_test_launch_plan_tensor(2);
  auto extra_const = make_test_launch_plan_tensor(3);
  auto extra_runtime = make_test_launch_plan_tensor(4);

  const std::vector<size_t> direct_input_indices = {0};
  const std::vector<uint32_t> scalar_values;
  std::vector<GpuTensor *> outputs = {&output};
  std::vector<GpuTensor *> const_tensors = {&extra_const};
  std::vector<GpuTensor> runtime_params = {extra_runtime};

  EXPECT_THROW(
      (void)build_role_ordered_kernel_launch_plan<uint32_t>(
          roles, direct_input_indices, scalar_values, outputs, const_tensors,
          {}, [&](size_t) -> GpuTensor * { return &input; },
          "unit_extra_const"),
      ov::Exception);
  EXPECT_THROW(
      (void)build_role_ordered_kernel_launch_plan<uint32_t>(
          roles, direct_input_indices, scalar_values, outputs, {}, runtime_params,
          [&](size_t) -> GpuTensor * { return &input; },
          "unit_extra_runtime"),
      ov::Exception);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeLaunchPlanOwnsRoleOrderedSourceArtifactBinding) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto cmake_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");
  const auto launch_plan_header =
      read_text_file(module_root / "src/runtime/kernel_launch_plan.hpp");
  const auto opencl_source_stage = read_text_file(
      module_root / "src/backends/opencl/runtime/opencl_source_stage.cpp");
  const auto metal_executor = read_text_file(
      module_root / "src/backends/metal/runtime/metal_executor.cpp");

  EXPECT_NE(cmake_sources.find("runtime/kernel_launch_plan.hpp"),
            std::string::npos);
  EXPECT_NE(launch_plan_header.find("struct KernelLaunchPlan"),
            std::string::npos);
  EXPECT_NE(launch_plan_header.find("GfxKernelBufferRole::ConstTensor"),
            std::string::npos);
  EXPECT_NE(launch_plan_header.find("GfxKernelBufferRole::RuntimeParams"),
            std::string::npos);
  EXPECT_NE(launch_plan_header.find("const tensor count mismatch"),
            std::string::npos);
  EXPECT_NE(launch_plan_header.find("runtime-parameter count mismatch"),
            std::string::npos);
  EXPECT_EQ(launch_plan_header.find("backends/opencl"), std::string::npos);
  EXPECT_EQ(launch_plan_header.find("backends/metal"), std::string::npos);

  EXPECT_NE(opencl_source_stage.find("runtime/kernel_launch_plan.hpp"),
            std::string::npos);
  EXPECT_NE(opencl_source_stage.find(
                "build_role_ordered_kernel_launch_plan<uint32_t>"),
            std::string::npos);
  EXPECT_EQ(opencl_source_stage.find(
                "unsupported role in source artifact ABI"),
            std::string::npos);

  EXPECT_NE(metal_executor.find("runtime/kernel_launch_plan.hpp"),
            std::string::npos);
  EXPECT_NE(metal_executor.find(
                "build_role_ordered_kernel_launch_plan<int32_t>"),
            std::string::npos);
  EXPECT_EQ(metal_executor.find(
                "descriptor-only source stage cannot materialize"),
            std::string::npos);
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
       MlirStageSupportTargetIsNotRuntimeNamed) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto gfx_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");
  const auto src_cmake = read_text_file(module_root / "src/CMakeLists.txt");
  const auto tests_cmake = read_text_file(module_root / "tests/CMakeLists.txt");
  const auto architecture_doc =
      read_text_file(module_root / "docs/ARCHITECTURE.md");
  const auto development_doc =
      read_text_file(module_root / "docs/DEVELOPMENT.md");

  EXPECT_EQ(gfx_sources.find("GFX_RUNTIME_MLIR"), std::string::npos);
  EXPECT_EQ(src_cmake.find("gfx_runtime_mlir"), std::string::npos);
  EXPECT_EQ(tests_cmake.find("gfx_runtime_mlir"), std::string::npos);
  EXPECT_EQ(architecture_doc.find("gfx_runtime_mlir"), std::string::npos);
  EXPECT_EQ(development_doc.find("gfx_runtime_mlir"), std::string::npos);
  EXPECT_NE(gfx_sources.find("GFX_MLIR_STAGE_SUPPORT_HEADERS"),
            std::string::npos);
  EXPECT_NE(gfx_sources.find("GFX_MLIR_STAGE_SUPPORT_SOURCES"),
            std::string::npos);
  EXPECT_NE(src_cmake.find("add_library(gfx_mlir_stage_support STATIC)"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeDoesNotOwnBackendCapabilitySurface) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto gfx_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");
  const auto device_info_source =
      read_text_file(module_root / "src/plugin/gfx_device_info.cpp");
  const auto operation_support_header =
      read_text_file(module_root / "src/compiler/operation_support.hpp");

  EXPECT_EQ(gfx_sources.find("runtime/gfx_backend_caps"), std::string::npos);
  EXPECT_EQ(device_info_source.find("runtime/gfx_backend_caps"),
            std::string::npos);
  EXPECT_NE(device_info_source.find("compiler/backend_registry.hpp"),
            std::string::npos);
  EXPECT_NE(device_info_source.find("BackendRegistry::default_registry()"),
            std::string::npos);
  EXPECT_NE(operation_support_header.find("struct PrecisionCapabilities"),
            std::string::npos);
  EXPECT_NE(operation_support_header.find("struct ArtifactFormatCapabilities"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeDoesNotOwnBackendIdentityConfig) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto gfx_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");
  const auto common_backend_config =
      read_text_file(module_root / "src/common/backend_config.hpp.in");
  const auto common_backend_utils_header =
      read_text_file(module_root / "src/common/gfx_backend_utils.hpp");
  const auto common_backend_utils_source =
      read_text_file(module_root / "src/common/gfx_backend_utils.cpp");
  const auto compiler_backend_config =
      read_text_file(module_root / "src/compiler/backend_config.hpp.in");
  const auto runtime_backend_utils =
      read_text_file(module_root / "src/runtime/gfx_backend_utils.hpp");
  const auto infer_pipeline_source =
      read_text_file(module_root / "src/runtime/infer_pipeline.cpp");

  EXPECT_NE(gfx_sources.find("common/backend_config.hpp.in"),
            std::string::npos);
  EXPECT_NE(gfx_sources.find("common/gfx_backend_utils.hpp"),
            std::string::npos);
  EXPECT_NE(gfx_sources.find("common/gfx_backend_utils.cpp"),
            std::string::npos);
  EXPECT_EQ(gfx_sources.find("runtime/gfx_backend_utils.cpp"),
            std::string::npos);
  EXPECT_EQ(gfx_sources.find("runtime/gfx_backend_utils.hpp"),
            std::string::npos);

  EXPECT_NE(common_backend_config.find("kGfxDefaultBackend"),
            std::string::npos);
  EXPECT_EQ(common_backend_config.find("compiler/"), std::string::npos);
  EXPECT_EQ(common_backend_config.find("runtime/"), std::string::npos);
  EXPECT_EQ(common_backend_utils_header.find("compiler/"), std::string::npos);
  EXPECT_EQ(common_backend_utils_header.find("runtime/"), std::string::npos);
  EXPECT_EQ(common_backend_utils_source.find("compiler/"), std::string::npos);
  EXPECT_EQ(common_backend_utils_source.find("runtime/"), std::string::npos);

  EXPECT_NE(compiler_backend_config.find("common/backend_config.hpp"),
            std::string::npos);
  EXPECT_NE(runtime_backend_utils.find("common/gfx_backend_utils.hpp"),
            std::string::npos);
  EXPECT_EQ(runtime_backend_utils.find("compiler/"), std::string::npos);
  EXPECT_EQ(runtime_backend_utils.find("compiler/backend_config.hpp"),
            std::string::npos);

  EXPECT_EQ(infer_pipeline_source.find("compiler/"), std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("stage_policy"), std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonRuntimeTargetDoesNotLinkMlirStageSupportTarget) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto src_cmake = read_text_file(module_root / "src/CMakeLists.txt");

  const auto runtime_common_target_begin =
      src_cmake.find("add_library(gfx_runtime_common STATIC)");
  const auto mlir_stage_support_target_begin =
      src_cmake.find("add_library(gfx_mlir_stage_support STATIC)");
  ASSERT_NE(runtime_common_target_begin, std::string::npos);
  ASSERT_NE(mlir_stage_support_target_begin, std::string::npos);
  ASSERT_LT(runtime_common_target_begin, mlir_stage_support_target_begin);

  const auto runtime_common_target_block = src_cmake.substr(
      runtime_common_target_begin,
      mlir_stage_support_target_begin - runtime_common_target_begin);
  EXPECT_EQ(runtime_common_target_block.find("GFX_MLIR_STAGE_SUPPORT"),
            std::string::npos);
  EXPECT_EQ(runtime_common_target_block.find("gfx_mlir_stage_support"),
            std::string::npos);

  const auto runtime_common_link_blocks = find_cmake_call_blocks(
      src_cmake, "target_link_libraries(gfx_runtime_common");
  ASSERT_FALSE(runtime_common_link_blocks.empty());
  for (const auto &block : runtime_common_link_blocks) {
    EXPECT_EQ(block.find("gfx_mlir_stage_support"), std::string::npos) << block;
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       BackendRuntimeTargetsDoNotLinkMlirStageSupportTarget) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto src_cmake = read_text_file(module_root / "src/CMakeLists.txt");

  const std::vector<std::string> runtime_targets = {
      "gfx_runtime_metal",
      "gfx_runtime_opencl",
  };
  for (const auto &target : runtime_targets) {
    const auto add_target =
        std::string("add_library(") + target + " STATIC)";
    const auto target_begin = src_cmake.find(add_target);
    ASSERT_NE(target_begin, std::string::npos) << target;

    const auto source_blocks =
        find_cmake_call_blocks(src_cmake, "target_sources(" + target);
    ASSERT_FALSE(source_blocks.empty()) << target;
    for (const auto &block : source_blocks) {
      EXPECT_EQ(block.find("GFX_MLIR_STAGE_SUPPORT"), std::string::npos)
          << block;
    }
    const auto link_blocks =
        find_cmake_call_blocks(src_cmake,
                               "target_link_libraries(" + target);
    for (const auto &block : link_blocks) {
      EXPECT_EQ(block.find("gfx_mlir_stage_support"), std::string::npos)
          << block;
    }
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonMlirDoesNotOwnAppleVendorPipelineContracts) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto cmake_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");

  struct MovedAppleCompilerContract {
    const char *old_mlir_path;
    const char *new_backend_path;
  };

  const std::vector<MovedAppleCompilerContract> moved_contracts = {
      {"src/mlir/gfx_apple_stage_pipeline.hpp",
       "src/backends/metal/compiler/apple_stage_pipeline.hpp"},
      {"src/mlir/gfx_apple_stage_pipeline.cpp",
       "src/backends/metal/compiler/apple_stage_pipeline.cpp"},
      {"src/mlir/gfx_apple_vendor_descriptors.hpp",
       "src/backends/metal/compiler/apple_vendor_descriptors.hpp"},
      {"src/mlir/gfx_apple_vendor_descriptors.cpp",
       "src/backends/metal/compiler/apple_vendor_descriptors.cpp"},
      {"src/mlir/gfx_mpsrt_const_tensor_sources.hpp",
       "src/backends/metal/compiler/apple_mpsrt_const_tensor_sources.hpp"},
      {"src/mlir/gfx_mpsrt_conv_metadata.hpp",
       "src/backends/metal/compiler/apple_mpsrt_conv_metadata.hpp"},
      {"src/mlir/gfx_mpsrt_matmul_metadata.hpp",
       "src/backends/metal/compiler/apple_mpsrt_matmul_metadata.hpp"},
      {"src/mlir/gfx_mpsrt_matmul_metadata.cpp",
       "src/backends/metal/compiler/apple_mpsrt_matmul_metadata.cpp"},
      {"src/mlir/gfx_mpsrt_source_plan.hpp",
       "src/backends/metal/compiler/apple_mpsrt_source_plan.hpp"},
  };

  for (const auto &contract : moved_contracts) {
    EXPECT_FALSE(std::filesystem::exists(module_root / contract.old_mlir_path))
        << contract.old_mlir_path;
    EXPECT_TRUE(
        std::filesystem::exists(module_root / contract.new_backend_path))
        << contract.new_backend_path;

    const auto old_cmake_ref =
        std::string(contract.old_mlir_path).substr(std::string("src/").size());
    const auto new_cmake_ref = std::string(contract.new_backend_path)
                                   .substr(std::string("src/").size());
    EXPECT_EQ(cmake_sources.find(old_cmake_ref), std::string::npos)
        << old_cmake_ref;
    EXPECT_NE(cmake_sources.find(new_cmake_ref), std::string::npos)
        << new_cmake_ref;
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonMlirDoesNotOwnAppleMslCodegenFiles) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto mlir_root = module_root / "src/mlir";
  const auto metal_compiler_root = module_root / "src/backends/metal/compiler";
  const auto cmake_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");

  ASSERT_TRUE(std::filesystem::exists(mlir_root));
  ASSERT_TRUE(std::filesystem::exists(metal_compiler_root));

  for (const auto &entry : std::filesystem::directory_iterator(mlir_root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto filename = entry.path().filename().string();
    EXPECT_NE(filename.rfind("msl_codegen", 0), 0) << entry.path();
  }

  EXPECT_EQ(cmake_sources.find("mlir/msl_codegen"), std::string::npos);
  EXPECT_EQ(cmake_sources.find("src/mlir/msl_codegen"), std::string::npos);

  const std::vector<std::string> backend_owned_codegen = {
      "src/backends/metal/compiler/msl_codegen.hpp",
      "src/backends/metal/compiler/msl_codegen_apple_mps.cpp",
      "src/backends/metal/compiler/msl_codegen_apple_msl_dispatch.cpp",
      "src/backends/metal/compiler/msl_codegen_matmul_metal.cpp",
      "src/backends/metal/compiler/msl_codegen_matmul_mpsrt.cpp",
  };

  for (const auto &relative_path : backend_owned_codegen) {
    EXPECT_TRUE(std::filesystem::exists(module_root / relative_path))
        << relative_path;
    const auto cmake_ref = relative_path.substr(std::string("src/").size());
    EXPECT_NE(cmake_sources.find(cmake_ref), std::string::npos) << cmake_ref;
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonMlirMpsrtOpsPublicApiDoesNotExposeMetalRuntimeTypes) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto ops_header =
      read_text_file(module_root / "src/mlir/gfx_mpsrt_ops.hpp");

  EXPECT_EQ(ops_header.find("backends/metal/runtime"), std::string::npos);
  EXPECT_EQ(ops_header.find("gfx_mpsrt_program.hpp"), std::string::npos);
  EXPECT_NE(ops_header.find("struct GfxMpsrtProgram;"), std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonMlirMpsrtMetadataUsesBackendLocalValueContracts) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto metadata_header =
      read_text_file(module_root / "src/mlir/gfx_mpsrt_metadata.hpp");
  const auto cmake_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");

  EXPECT_EQ(metadata_header.find("backends/metal/runtime"), std::string::npos);
  EXPECT_EQ(metadata_header.find("runtime/mpsrt/gfx_mpsrt_"),
            std::string::npos);
  EXPECT_EQ(
      cmake_sources.find("backends/metal/runtime/mpsrt/gfx_mpsrt_abi.hpp"),
      std::string::npos);
  EXPECT_EQ(
      cmake_sources.find("backends/metal/runtime/mpsrt/gfx_mpsrt_program.hpp"),
      std::string::npos);

  for (std::string_view required : {
           "backends/metal/common/mpsrt/gfx_mpsrt_abi.hpp",
           "backends/metal/common/mpsrt/gfx_mpsrt_builder_plan.hpp",
           "backends/metal/common/mpsrt/gfx_mpsrt_kernel_manifest_adapter.hpp",
           "backends/metal/common/mpsrt/gfx_mpsrt_plan.hpp",
           "backends/metal/common/mpsrt/gfx_mpsrt_program.hpp",
           "backends/metal/common/mpsrt/gfx_mpsrt_storage_bridge.hpp",
       }) {
    EXPECT_TRUE(std::filesystem::exists(module_root / "src" /
                                        std::filesystem::path(required)))
        << required;
    EXPECT_NE(cmake_sources.find(required), std::string::npos) << required;
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonMlirBackendHooksUseCommonValueContracts) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto hooks_header =
      read_text_file(module_root / "src/mlir/mlir_stage_backend_hooks.hpp");
  const auto parallelism_header =
      read_text_file(module_root / "src/runtime/gfx_parallelism.hpp");

  EXPECT_NE(hooks_header.find("common/gfx_bias.hpp"), std::string::npos);
  EXPECT_NE(hooks_header.find("common/gpu_parallelism_plan.hpp"),
            std::string::npos);
  EXPECT_EQ(hooks_header.find("runtime/gfx_bias.hpp"), std::string::npos);
  EXPECT_EQ(hooks_header.find("runtime/gfx_parallelism.hpp"),
            std::string::npos);
  EXPECT_FALSE(
      std::filesystem::exists(module_root / "src/runtime/gfx_bias.hpp"));

  EXPECT_NE(parallelism_header.find("common/gpu_parallelism_plan.hpp"),
            std::string::npos);
  EXPECT_EQ(parallelism_header.find("struct MatMulParallelismPlan"),
            std::string::npos);
  EXPECT_EQ(parallelism_header.find("struct ConvParallelismPlan"),
            std::string::npos);
  EXPECT_EQ(parallelism_header.find("struct ChunkDispatchPlan"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonMlirTargetDoesNotCompileMetalMslCompilerSources) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto gfx_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");
  const auto src_cmake = read_text_file(module_root / "src/CMakeLists.txt");

  EXPECT_EQ(gfx_sources.find("GFX_RUNTIME_METAL_MSL"), std::string::npos);
  EXPECT_EQ(src_cmake.find("GFX_RUNTIME_METAL_MSL"), std::string::npos);
  EXPECT_NE(gfx_sources.find("GFX_METAL_MSL_COMPILER_HEADERS"),
            std::string::npos);
  EXPECT_NE(gfx_sources.find("GFX_METAL_MSL_COMPILER_SOURCES"),
            std::string::npos);

  const auto mlir_target_begin =
      src_cmake.find("add_library(gfx_mlir_stage_support STATIC)");
  const auto metal_msl_target_begin =
      src_cmake.find("add_library(gfx_metal_msl_compiler STATIC)");
  ASSERT_NE(mlir_target_begin, std::string::npos);
  ASSERT_NE(metal_msl_target_begin, std::string::npos);
  ASSERT_LT(mlir_target_begin, metal_msl_target_begin);

  const auto mlir_target_block = src_cmake.substr(
      mlir_target_begin, metal_msl_target_begin - mlir_target_begin);
  EXPECT_EQ(mlir_target_block.find("GFX_METAL_MSL_COMPILER"),
            std::string::npos);
  EXPECT_EQ(mlir_target_block.find("msl_codegen"), std::string::npos);

  const auto metal_msl_target_block = src_cmake.substr(metal_msl_target_begin);
  EXPECT_NE(metal_msl_target_block.find("GFX_METAL_MSL_COMPILER_HEADERS"),
            std::string::npos);
  EXPECT_NE(metal_msl_target_block.find("GFX_METAL_MSL_COMPILER_SOURCES"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonMlirUsesBackendHooksForAppleStageMaterialization) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto common_mlir_stage =
      read_text_file(module_root / "src/mlir/mlir_stage.cpp");
  const auto metal_hook = read_text_file(
      module_root / "src/backends/metal/compiler/apple_mlir_stage_hooks.cpp");
  const auto metal_executor = read_text_file(
      module_root / "src/backends/metal/runtime/metal_executor.cpp");

  EXPECT_EQ(
      common_mlir_stage.find("backends/metal/compiler/apple_stage_pipeline"),
      std::string::npos);
  EXPECT_EQ(common_mlir_stage.find("backends/metal/compiler/apple_mpsrt_"),
            std::string::npos);
  EXPECT_EQ(common_mlir_stage.find("run_gfx_apple_stage_pipeline"),
            std::string::npos);
  EXPECT_EQ(common_mlir_stage.find("annotate_module_with_conv_mpsrt_plan"),
            std::string::npos);
  for (std::string_view forbidden : {
           "backends/metal/compiler/msl_codegen",
           "GfxMslGeneratedKernelSourcePlan",
           "CompressedMatMulInfo",
           "detect_compressed_matmul_weights",
           "make_shapeof_msl_kernel_source_plan",
           "make_concat_msl_kernel_source_plan",
           "make_causal_sdpa_msl_kernel_source_plan",
           "make_sdpa_msl_kernel_source_plan",
           "make_sdpa_msl_runtime_params_plan",
           "make_range_msl_kernel_source_plan",
           "make_tile_msl_kernel_source_plan",
           "make_activation_msl_kernel_source_plan",
           "make_apple_metal_runtime_matmul_kernel_source_plan",
           "make_direct_static_slice_msl_kernel_source_plan",
           "make_direct_split_msl_kernel_source_plan",
       }) {
    EXPECT_EQ(common_mlir_stage.find(forbidden), std::string::npos)
        << forbidden;
  }
  EXPECT_NE(common_mlir_stage.find("mlir/mlir_stage_backend_hooks.hpp"),
            std::string::npos);
  EXPECT_NE(common_mlir_stage.find("mlir_stage_backend_hooks_for"),
            std::string::npos);

  EXPECT_NE(metal_hook.find("run_gfx_apple_stage_pipeline"), std::string::npos);
  EXPECT_NE(metal_hook.find("annotate_module_with_conv_mpsrt_plan"),
            std::string::npos);
  EXPECT_NE(metal_hook.find("make_shapeof_msl_kernel_source_plan"),
            std::string::npos);
  EXPECT_NE(metal_hook.find("detect_compressed_matmul_weights"),
            std::string::npos);
  EXPECT_NE(metal_hook.find("make_sdpa_msl_runtime_params_plan"),
            std::string::npos);
  EXPECT_EQ(
      metal_executor.find("backends/metal/compiler/apple_mlir_stage_hooks"),
      std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       MetalRuntimeDoesNotIncludeAppleMpsrtCompilerSourcePlanHelpers) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto runtime_root = module_root / "src/backends/metal/runtime";
  ASSERT_TRUE(std::filesystem::exists(runtime_root));

  for (const auto &entry :
       std::filesystem::recursive_directory_iterator(runtime_root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto path = entry.path();
    const auto extension = path.extension().string();
    if (extension != ".cpp" && extension != ".hpp" && extension != ".mm") {
      continue;
    }

    const auto source = read_text_file(path);
    EXPECT_EQ(source.find("backends/metal/compiler/apple_mpsrt_"),
              std::string::npos)
        << path;
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeDoesNotOwnStagePolicyPlanner) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  EXPECT_FALSE(std::filesystem::exists(module_root /
                                       "src/runtime/gfx_stage_policy.cpp"));
  EXPECT_FALSE(std::filesystem::exists(module_root /
                                       "src/runtime/gfx_stage_policy.hpp"));

  const auto stage_policy_source =
      read_text_file(module_root / "src/compiler/stage_policy.cpp");
  const auto stage_compiler_policy_source =
      read_text_file(module_root / "src/compiler/stage_compiler_policy.cpp");

  EXPECT_EQ(stage_policy_source.find("compiler/backend_registry.hpp"),
            std::string::npos);
  EXPECT_EQ(stage_policy_source.find("BackendRegistry::"), std::string::npos);
  EXPECT_EQ(stage_policy_source.find("make_post_op_fusion_capabilities("),
            std::string::npos);
  EXPECT_EQ(stage_compiler_policy_source.find("switch (target.backend())"),
            std::string::npos);
  EXPECT_EQ(
      stage_compiler_policy_source.find("make_opencl_chunk_dispatch_profile"),
      std::string::npos);
  EXPECT_EQ(
      stage_compiler_policy_source.find("make_metal_chunk_dispatch_profile"),
      std::string::npos);
  EXPECT_NE(stage_compiler_policy_source.find("capabilities.execution()"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeGpuStageApiDoesNotExposeCompilerPolicyPointers) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto gpu_stage_header =
      read_text_file(module_root / "src/runtime/gpu_stage.hpp");
  const auto compiled_model_source =
      read_text_file(module_root / "src/plugin/compiled_model.cpp");

  EXPECT_EQ(gpu_stage_header.find("compiler::"), std::string::npos);
  EXPECT_EQ(gpu_stage_header.find("StagePlacementPolicy"), std::string::npos);
  EXPECT_EQ(gpu_stage_header.find("PostOpFusionCapabilities"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("stage_placement_policy"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("post_op_fusion_capabilities"),
            std::string::npos);
  EXPECT_NE(compiled_model_source.find("source_kernel_dispatch"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       OpenClArtifactsDoNotUseGenericBaselineSourceFallback) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto source = read_text_file(
      module_root / "src/kernel_ir/gfx_opencl_source_artifacts.cpp");

  EXPECT_EQ(source.find("kOpenClBaselineSource"), std::string::npos);
  EXPECT_EQ(source.find("artifact.source = kOpenClBaselineSource"),
            std::string::npos);
  EXPECT_NE(source.find("kOpenClConvertSource"), std::string::npos);
  EXPECT_NE(source.find("artifact.source = kOpenClConvertSource"),
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
       KernelDispatchHelpersUseCommonValueContractsNotRuntimeKernels) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto dispatch_config =
      read_text_file(module_root / "src/common/gpu_dispatch_config.hpp");
  const auto dispatch_helpers =
      read_text_file(module_root / "src/kernel_ir/gfx_kernel_dispatch.hpp");
  const auto runtime_kernel =
      read_text_file(module_root / "src/runtime/gpu_backend_base.hpp");

  EXPECT_NE(dispatch_config.find("struct KernelDispatch"), std::string::npos);
  EXPECT_NE(runtime_kernel.find("common/gpu_dispatch_config.hpp"),
            std::string::npos);
  EXPECT_NE(dispatch_helpers.find("common/gpu_dispatch_config.hpp"),
            std::string::npos);
  EXPECT_EQ(dispatch_helpers.find("runtime/gpu_backend_base.hpp"),
            std::string::npos);
  EXPECT_EQ(dispatch_helpers.find("ICompiledKernel"), std::string::npos);
  EXPECT_EQ(dispatch_helpers.find("clamp_threadgroup_size"), std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       SharedMlirCustomKernelContractsUseBackendDomainNotOpenClBoolSelector) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const std::vector<std::filesystem::path> shared_mlir_contract_files = {
      module_root / "src/mlir/gfx_backend_custom_kernel_adapter.hpp",
      module_root / "src/mlir/gfx_backend_custom_kernel_adapter.cpp",
      module_root / "src/mlir/gfx_stage_kernel_binding.hpp",
  };

  for (const auto &path : shared_mlir_contract_files) {
    const auto source = read_text_file(path);
    EXPECT_EQ(source.find("is_opencl_backend"), std::string::npos) << path;
    EXPECT_EQ(source.find("backend_domain_from_selector"), std::string::npos)
        << path;
    EXPECT_EQ(source.find("specialization_prefix_from_selector"),
              std::string::npos)
        << path;
    EXPECT_EQ(source.find("can_override_manifest"), std::string::npos) << path;
  }

  const auto adapter_header = read_text_file(
      module_root / "src/mlir/gfx_backend_custom_kernel_adapter.hpp");
  EXPECT_NE(adapter_header.find("GfxKernelBackendDomain backend_domain"),
            std::string::npos);
  EXPECT_NE(adapter_header.find("backend_custom_kernel_specialization_prefix"),
            std::string::npos);
  EXPECT_EQ(adapter_header.find("GfxKernelBackendDomain::AppleMsl"),
            std::string::npos);
  EXPECT_EQ(adapter_header.find("\"apple_msl:buffer:\""), std::string::npos);
  EXPECT_EQ(adapter_header.find(
                "std::vector<int32_t> scalar_args = {},\n"
                "    GfxKernelBackendDomain backend_domain"),
            std::string::npos);

  const auto family_header = read_text_file(
      module_root / "src/kernel_ir/gfx_custom_kernel_families.hpp");
  EXPECT_EQ(family_header.find("GfxKernelBackendDomain::AppleMsl"),
            std::string::npos);
  EXPECT_EQ(family_header.find("\"apple_msl:buffer:\""), std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeValuePlannerIsOwnedByRuntimeNotCommonMlir) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto cmake_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");
  const std::vector<std::filesystem::path> old_mlir_paths = {
      module_root / "src/mlir/gfx_stage_runtime_values.hpp",
      module_root / "src/mlir/gfx_stage_runtime_values.cpp",
  };
  const std::vector<std::filesystem::path> runtime_paths = {
      module_root / "src/runtime/gfx_stage_runtime_values.hpp",
      module_root / "src/runtime/gfx_stage_runtime_values.cpp",
  };

  for (const auto &path : old_mlir_paths) {
    EXPECT_FALSE(std::filesystem::exists(path)) << path;
  }
  for (const auto &path : runtime_paths) {
    ASSERT_TRUE(std::filesystem::exists(path)) << path;
    const auto source = read_text_file(path);
    EXPECT_EQ(source.find("is_opencl_backend"), std::string::npos) << path;
    EXPECT_EQ(source.find("backend_domain_from_selector"), std::string::npos)
        << path;
    EXPECT_EQ(source.find("specialization_prefix_from_selector"),
              std::string::npos)
        << path;
  }
  EXPECT_EQ(cmake_sources.find("mlir/gfx_stage_runtime_values"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/gfx_stage_runtime_values.hpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/gfx_stage_runtime_values.cpp"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeParamMaterializerIsOwnedByRuntimeNotKernelIrOrMlir) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto cmake_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");
  const auto runtime_params_path =
      module_root / "src/runtime/gfx_kernel_runtime_params.hpp";
  const std::vector<std::filesystem::path> old_paths = {
      module_root / "src/kernel_ir/gfx_kernel_runtime_params.hpp",
      module_root / "src/mlir/gfx_kernel_runtime_params.hpp",
  };

  ASSERT_TRUE(std::filesystem::exists(runtime_params_path))
      << runtime_params_path;
  for (const auto &path : old_paths) {
    EXPECT_FALSE(std::filesystem::exists(path)) << path;
  }

  const auto runtime_params = read_text_file(runtime_params_path);
  EXPECT_NE(runtime_params.find("runtime/gpu_buffer_manager.hpp"),
            std::string::npos);
  EXPECT_NE(runtime_params.find("runtime/gpu_tensor.hpp"), std::string::npos);
  EXPECT_EQ(cmake_sources.find("kernel_ir/gfx_kernel_runtime_params.hpp"),
            std::string::npos);
  EXPECT_EQ(cmake_sources.find("mlir/gfx_kernel_runtime_params.hpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/gfx_kernel_runtime_params.hpp"),
            std::string::npos);

  const auto mlir_stage_source =
      read_text_file(module_root / "src/mlir/mlir_stage.cpp");
  const auto metal_executor_source = read_text_file(
      module_root / "src/backends/metal/runtime/metal_executor.cpp");
  EXPECT_NE(mlir_stage_source.find("runtime/gfx_kernel_runtime_params.hpp"),
            std::string::npos);
  EXPECT_NE(metal_executor_source.find("runtime/gfx_kernel_runtime_params.hpp"),
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
       MlirRuntimeStagesRequireCompilerOwnedDescriptor) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto mlir_stage_header =
      read_text_file(module_root / "src/mlir/mlir_stage.hpp");
  const auto mlir_stage_source =
      read_text_file(module_root / "src/mlir/mlir_stage.cpp");
  const auto metal_executor_header = read_text_file(
      module_root / "src/backends/metal/runtime/metal_executor.hpp");
  const auto metal_stage_factory_source = read_text_file(
      module_root / "src/backends/metal/runtime/stage_factory.cpp");
  const auto opencl_stage_factory_source = read_text_file(
      module_root / "src/backends/opencl/runtime/stage_factory.cpp");

  EXPECT_EQ(mlir_stage_header.find("explicit MlirStage("), std::string::npos);
  EXPECT_EQ(mlir_stage_source.find("MlirStage(node, nullptr)"),
            std::string::npos);
  EXPECT_EQ(
      mlir_stage_source.find(
          ": compiler::select_tensor_layout_plan(m_type, m_node).view_only"),
      std::string::npos);
  EXPECT_NE(
      mlir_stage_source.find(
          "GFX MLIR stage requires a compiler-owned runtime executable"),
      std::string::npos);
  EXPECT_EQ(metal_executor_header.find("descriptor = nullptr"),
            std::string::npos);
  EXPECT_NE(metal_stage_factory_source.find("infer MLIR layout"),
            std::string::npos);
  EXPECT_NE(metal_stage_factory_source.find("create_view_only_stage"),
            std::string::npos);
  EXPECT_NE(opencl_stage_factory_source.find("create_view_only_stage"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CompilerStagePolicyConsumesCompilerOwnedSourceDispatchPolicy) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto stage_policy_source =
      read_text_file(module_root / "src/compiler/stage_policy.cpp");

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
  const auto runtime_plan_header =
      read_text_file(module_root / "src/runtime/pipeline_stage_plan.hpp");
  const auto runtime_desc_header =
      read_text_file(module_root / "src/runtime/pipeline_stage_desc.hpp");

  EXPECT_EQ(compiled_model_source.find("compiler/pipeline_stage_plan.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("compiler/pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("runtime/pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_NE(compiled_model_header.find("runtime/pipeline_stage_desc.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_header.find("struct PipelineStageDesc"),
            std::string::npos);
  EXPECT_EQ(builder_header.find("openvino/gfx_plugin/compiled_model.hpp"),
            std::string::npos);
  EXPECT_EQ(builder_header.find("runtime/pipeline_stage_desc.hpp"),
            std::string::npos);
  EXPECT_EQ(builder_header.find("runtime/backend_stage_factory.hpp"),
            std::string::npos);
  EXPECT_NE(builder_header.find("runtime/pipeline_stage_plan.hpp"),
            std::string::npos);
  EXPECT_NE(builder_header.find("PipelineStageMaterializationPlan"),
            std::string::npos);
  EXPECT_NE(builder_header.find("PipelineStageRuntimePlan"),
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
  EXPECT_NE(runtime_plan_header.find("PipelineStageRuntimePlan"),
            std::string::npos);
  EXPECT_NE(runtime_plan_header.find("PipelineStageMaterializationPlan"),
            std::string::npos);
  EXPECT_NE(runtime_plan_header.find("PipelineStageRuntimeOptionsPlan"),
            std::string::npos);
  EXPECT_NE(runtime_plan_header.find("node_to_stage"), std::string::npos);
  EXPECT_NE(runtime_plan_header.find("param_index"), std::string::npos);
  EXPECT_EQ(runtime_plan_header.find("compiler/"), std::string::npos);
  EXPECT_EQ(runtime_plan_header.find("compiler::"), std::string::npos);
  EXPECT_EQ(runtime_desc_header.find("compiler/"), std::string::npos);
  EXPECT_EQ(runtime_desc_header.find("compiler::"), std::string::npos);
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

  EXPECT_EQ(compiled_model_source.find("compiler/pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("runtime/pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("plugin/pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_EQ(
      compiled_model_source.find("plugin/pipeline_stage_materializer.hpp"),
      std::string::npos);
  EXPECT_NE(
      compiled_model_source.find("runtime/pipeline_stage_materializer.hpp"),
      std::string::npos);
  EXPECT_EQ(compiled_model_source.find("PipelineStageMaterializer"),
            std::string::npos);
  EXPECT_NE(
      compiled_model_source.find("materialize_pipeline_stage_descriptors"),
      std::string::npos);
  EXPECT_EQ(compiled_model_source.find("runtime_plan = &result.runtime_plan"),
            std::string::npos);
  EXPECT_NE(compiled_model_source.find("runtime_plan = &runtime_plan"),
            std::string::npos);
  EXPECT_NE(compiled_model_source.find("m_runtime_descriptor->stage_plan"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("build_result = &result"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("build_pipeline_stage_plan("),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("PipelineStagePlanBuilder"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("get_ordered_ops()"), std::string::npos);
  EXPECT_NE(builder_header.find("build_pipeline_stage_plan"),
            std::string::npos);
  EXPECT_EQ(builder_source.find("PipelineStageMaterializer"),
            std::string::npos);
  EXPECT_EQ(builder_source.find("create_stage("), std::string::npos);
  EXPECT_EQ(builder_source.find("fuse_activation("), std::string::npos);
  EXPECT_EQ(builder_source.find("fuse_batchnorm("), std::string::npos);
  EXPECT_EQ(builder_source.find("fuse_bias("), std::string::npos);
  EXPECT_NE(builder_source.find("PipelineStageMaterializationPlan"),
            std::string::npos);
  EXPECT_NE(builder_source.find("make_pipeline_stage_runtime_plan"),
            std::string::npos);
  EXPECT_NE(builder_source.find("PipelineStagePlanBuilder"), std::string::npos);
  const auto descriptor_builder_header = read_text_file(
      module_root / "src/compiler/runtime_executable_descriptor_builder.hpp");
  const auto descriptor_builder_source = read_text_file(
      module_root / "src/compiler/runtime_executable_descriptor_builder.cpp");
  EXPECT_NE(descriptor_builder_header.find(
                "RuntimeExecutableDescriptorBuildRequest"),
            std::string::npos);
  EXPECT_NE(descriptor_builder_source.find("build_pipeline_stage_plan("),
            std::string::npos);
  EXPECT_NE(descriptor_builder_source.find("descriptor.stage_plan"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("runtime_stage_descriptors"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("auto runtime_descriptor_for_node"),
            std::string::npos);
  EXPECT_EQ(compiled_model_source.find("auto create_backend_stage"),
            std::string::npos);
  EXPECT_NE(materializer_header.find("class PipelineStageMaterializer"),
            std::string::npos);
  EXPECT_NE(
      materializer_header.find("PipelineStageRuntimeMaterializationRequest"),
      std::string::npos);
  EXPECT_NE(materializer_header.find("PipelineStageRuntimePlan"),
            std::string::npos);
  EXPECT_NE(materializer_header.find("materialize_pipeline_stage_descriptors"),
            std::string::npos);
  EXPECT_EQ(materializer_header.find("compiler/"), std::string::npos);
  EXPECT_EQ(materializer_header.find("compiler::"), std::string::npos);
  EXPECT_EQ(materializer_header.find("PipelineStageBuildResult"),
            std::string::npos);
  EXPECT_EQ(materializer_header.find("PipelineVendorAttentionArtifact"),
            std::string::npos);
  EXPECT_EQ(materializer_source.find("compiler/"), std::string::npos);
  EXPECT_EQ(materializer_source.find("compiler::"), std::string::npos);
  EXPECT_EQ(materializer_source.find("PipelineStageBuildResult"),
            std::string::npos);
  EXPECT_EQ(materializer_source.find("PipelineVendorAttentionArtifact"),
            std::string::npos);
  EXPECT_NE(materializer_source.find("m_descriptors_by_node"),
            std::string::npos);
  EXPECT_NE(materializer_source.find("RuntimeExecutableDescriptor"),
            std::string::npos);
  EXPECT_NE(materializer_header.find("BackendStageFactory"), std::string::npos);
  EXPECT_NE(materializer_source.find("m_stage_factory"), std::string::npos);
  EXPECT_EQ(materializer_header.find("plugin/backend_state"),
            std::string::npos);
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
  const auto backend_state_header_path =
      module_root / "src/plugin/backend_state.hpp";
  const auto backend_runtime_header =
      read_text_file(module_root / "src/runtime/backend_runtime.hpp");
  const auto backend_request_state_header =
      read_text_file(module_root / "src/runtime/backend_request_state.hpp");
  const auto backend_runtime_provider_header =
      read_text_file(module_root / "src/runtime/backend_runtime_provider.hpp");
  const auto backend_runtime_provider_source =
      read_text_file(module_root / "src/runtime/backend_runtime_provider.cpp");
  const auto infer_request_state_header =
      read_text_file(module_root / "src/plugin/infer_request_state.hpp");
  const auto infer_pipeline_state_header =
      read_text_file(module_root / "src/runtime/infer_pipeline_state.hpp");
  const auto infer_pipeline_header =
      read_text_file(module_root / "src/runtime/infer_pipeline.hpp");
  const auto infer_executor_header =
      read_text_file(module_root / "src/runtime/infer_executor.hpp");
  const auto infer_executor_source =
      read_text_file(module_root / "src/runtime/infer_executor.cpp");
  const auto infer_io_utils_header =
      read_text_file(module_root / "src/plugin/infer_io_utils.hpp");
  const auto infer_io_utils_source =
      read_text_file(module_root / "src/plugin/infer_io_utils.cpp");
  const auto runtime_descriptor_header =
      read_text_file(module_root / "src/runtime/executable_descriptor.hpp");
  const auto runtime_descriptor_source =
      read_text_file(module_root / "src/runtime/executable_descriptor.cpp");
  const auto compiler_runtime_descriptor_builder_header = read_text_file(
      module_root / "src/compiler/runtime_executable_descriptor_builder.hpp");
  const auto compiler_runtime_descriptor_builder_source = read_text_file(
      module_root / "src/compiler/runtime_executable_descriptor_builder.cpp");
  const auto common_artifact_payload_header =
      read_text_file(module_root / "src/common/artifact_payload.hpp");
  const auto manifest_header =
      read_text_file(module_root / "src/compiler/manifest.hpp");
  const auto manifest_source =
      read_text_file(module_root / "src/compiler/manifest.cpp");
  const auto stateful_stage_header =
      read_text_file(module_root / "src/runtime/stateful_stage.hpp");
  const auto stateful_stage_source =
      read_text_file(module_root / "src/runtime/stateful_stage.cpp");
  const auto stateful_execution_header =
      read_text_file(module_root / "src/runtime/stateful_execution.hpp");
  const auto stateful_execution_source =
      read_text_file(module_root / "src/runtime/stateful_execution.cpp");
  const auto stateful_variable_state_header =
      read_text_file(module_root / "src/runtime/stateful_variable_state.hpp");
  const auto metal_infer_source = read_text_file(
      module_root / "src/backends/metal/plugin/infer_request.mm");
  const auto opencl_infer_source = read_text_file(
      module_root / "src/backends/opencl/plugin/infer_request.cpp");
  const auto metal_state_header = read_text_file(
      module_root / "src/backends/metal/plugin/compiled_model_state.hpp");
  const auto metal_stage_factory_source = read_text_file(
      module_root / "src/backends/metal/runtime/stage_factory.cpp");
  const auto opencl_stage_factory_source = read_text_file(
      module_root / "src/backends/opencl/runtime/stage_factory.cpp");
  const auto infer_request_header = read_text_file(
      module_root / "include/openvino/gfx_plugin/infer_request.hpp");
  const auto infer_request_common_source =
      read_text_file(module_root / "src/plugin/infer_request_common.cpp");
  const auto variable_state_source = read_text_file(
      module_root / "src/plugin/infer_request_variable_state.cpp");
  const auto cmake_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");
  const auto src_cmake = read_text_file(module_root / "src/CMakeLists.txt");
  const auto backend_registry_header =
      read_text_file(module_root / "src/compiler/backend_registry.hpp");
  const auto backend_registry_source =
      read_text_file(module_root / "src/compiler/backend_registry.cpp");
  const auto static_backend_module_source =
      read_text_file(module_root / "src/compiler/static_backend_module.cpp");
  const auto metal_artifacts_header = read_text_file(
      module_root / "src/backends/metal/compiler/metal_kernel_artifacts.hpp");
  const auto metal_artifacts_source = read_text_file(
      module_root / "src/backends/metal/compiler/metal_kernel_artifacts.cpp");
  const auto apple_vendor_descriptors_header = read_text_file(
      module_root / "src/backends/metal/compiler/apple_vendor_descriptors.hpp");
  const auto apple_vendor_descriptors_source = read_text_file(
      module_root / "src/backends/metal/compiler/apple_vendor_descriptors.cpp");
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
  EXPECT_EQ(builder_header.find("BackendStageFactory"), std::string::npos);
  EXPECT_FALSE(std::filesystem::exists(
      module_root / "src/runtime/pipeline_stage_builder.hpp"));
  EXPECT_FALSE(std::filesystem::exists(
      module_root / "src/runtime/pipeline_stage_builder.cpp"));
  EXPECT_EQ(builder_source.find("plugin/backend_state.hpp"), std::string::npos);
  EXPECT_EQ(builder_source.find("runtime/pipeline_stage_materializer"),
            std::string::npos);
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
  EXPECT_FALSE(std::filesystem::exists(backend_state_header_path));
  EXPECT_FALSE(
      std::filesystem::exists(module_root / "src/plugin/backend_factory.hpp"));
  EXPECT_FALSE(
      std::filesystem::exists(module_root / "src/plugin/backend_factory.cpp"));
  EXPECT_FALSE(
      std::filesystem::exists(module_root / "src/plugin/stateful_stage.hpp"));
  EXPECT_FALSE(
      std::filesystem::exists(module_root / "src/plugin/stateful_stage.cpp"));
  EXPECT_FALSE(std::filesystem::exists(module_root /
                                       "src/plugin/stateful_execution.hpp"));
  EXPECT_FALSE(std::filesystem::exists(module_root /
                                       "src/plugin/stateful_execution.cpp"));
  EXPECT_NE(
      backend_runtime_header.find("struct BackendState : BackendStageFactory"),
      std::string::npos);
  EXPECT_NE(backend_runtime_header.find("BackendRequestState&"),
            std::string::npos);
  EXPECT_EQ(backend_runtime_header.find("InferRequestState"),
            std::string::npos);
  EXPECT_NE(backend_request_state_header.find("struct BackendInferState"),
            std::string::npos);
  EXPECT_NE(backend_request_state_header.find("struct BackendRequestState"),
            std::string::npos);
  EXPECT_EQ(backend_request_state_header.find("plugin/"), std::string::npos);
  EXPECT_EQ(infer_pipeline_state_header.find("plugin/"), std::string::npos);
  EXPECT_NE(
      infer_pipeline_state_header.find("direct_stateful_assign_variable_ids"),
      std::string::npos);
  EXPECT_NE(infer_request_state_header.find("BackendRequestState runtime"),
            std::string::npos);
  EXPECT_NE(infer_request_state_header.find(
                "StatefulVariableStateMap variable_states"),
            std::string::npos);
  EXPECT_EQ(infer_request_state_header.find("struct VariableTensorState"),
            std::string::npos);
  EXPECT_EQ(
      infer_request_state_header.find("std::unique_ptr<BackendInferState>"),
      std::string::npos);
  EXPECT_EQ(infer_request_state_header.find("struct BackendInferState"),
            std::string::npos);
  EXPECT_NE(backend_runtime_provider_header.find("BackendRuntimeProvider"),
            std::string::npos);
  EXPECT_NE(backend_runtime_provider_header.find("execute_infer"),
            std::string::npos);
  EXPECT_NE(backend_runtime_provider_header.find("execute_backend_infer"),
            std::string::npos);
  EXPECT_NE(backend_runtime_provider_source.find("execute_backend_infer"),
            std::string::npos);
  EXPECT_EQ(backend_runtime_provider_header.find("plugin/"), std::string::npos);
  EXPECT_EQ(backend_runtime_provider_source.find("plugin/"), std::string::npos);
  EXPECT_EQ(backend_runtime_provider_header.find("compiler/"),
            std::string::npos);
  EXPECT_EQ(backend_runtime_provider_source.find("compiler/"),
            std::string::npos);
  EXPECT_EQ(metal_stage_factory_source.find("plugin/"), std::string::npos);
  EXPECT_EQ(opencl_stage_factory_source.find("plugin/"), std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/backend_runtime_provider.cpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/backend_runtime_provider.hpp"),
            std::string::npos);
  EXPECT_NE(src_cmake.find("$<LINK_LIBRARY:WHOLE_ARCHIVE,gfx_plugin_metal>"),
            std::string::npos);
  EXPECT_NE(src_cmake.find("$<LINK_LIBRARY:WHOLE_ARCHIVE,gfx_plugin_opencl>"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/backend_request_state.hpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/infer_pipeline_state.hpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/infer_executor.hpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/infer_executor.cpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/infer_pipeline.cpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/infer_pipeline.hpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/infer_submission.cpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/infer_submission.hpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("common/artifact_payload.hpp"),
            std::string::npos);
  EXPECT_EQ(cmake_sources.find("runtime/artifact_payload.hpp"),
            std::string::npos);
  const auto artifact_compiler_headers_begin =
      cmake_sources.find("set(GFX_COMPILER_COMMON_HEADERS");
  const auto artifact_plugin_sources_begin =
      cmake_sources.find("set(GFX_PLUGIN_SOURCES");
  ASSERT_NE(artifact_compiler_headers_begin, std::string::npos);
  ASSERT_NE(artifact_plugin_sources_begin, std::string::npos);
  const auto artifact_compiler_headers_block =
      cmake_sources.substr(artifact_compiler_headers_begin,
                           artifact_plugin_sources_begin -
                               artifact_compiler_headers_begin);
  EXPECT_NE(artifact_compiler_headers_block.find("common/artifact_payload.hpp"),
            std::string::npos);
  EXPECT_EQ(cmake_sources.find("plugin/infer_pipeline"), std::string::npos);
  EXPECT_EQ(cmake_sources.find("plugin/infer_submission"), std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/stateful_stage.cpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/stateful_stage.hpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/stateful_execution.cpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/stateful_execution.hpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/stateful_variable_state.hpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("plugin/infer_request_variable_state.cpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("plugin/infer_request_variable_state.hpp"),
            std::string::npos);
  EXPECT_EQ(cmake_sources.find("plugin/backend_factory"), std::string::npos);
  EXPECT_EQ(cmake_sources.find("plugin/stateful_stage"), std::string::npos);
  EXPECT_EQ(cmake_sources.find("plugin/stateful_execution"), std::string::npos);
  EXPECT_NE(manifest_header.find("StatefulEffectContract"), std::string::npos);
  EXPECT_NE(manifest_header.find("stateful_effect"), std::string::npos);
  EXPECT_NE(manifest_header.find("stateful_prebind_variable_id"),
            std::string::npos);
  EXPECT_NE(manifest_header.find("stateful_prebind_shape_rule"),
            std::string::npos);
  EXPECT_NE(manifest_header.find("stateful_prebind_shape_axis"),
            std::string::npos);
  EXPECT_NE(manifest_source.find("make_stateful_effect_contract"),
            std::string::npos);
  EXPECT_NE(manifest_source.find("sum_inputs_along_axis"), std::string::npos);
  EXPECT_NE(runtime_descriptor_header.find("stateful_effect"),
            std::string::npos);
  EXPECT_NE(runtime_descriptor_header.find("stateful_variable_id"),
            std::string::npos);
  EXPECT_NE(runtime_descriptor_header.find("stateful_prebind_variable_id"),
            std::string::npos);
  EXPECT_NE(runtime_descriptor_header.find("stateful_prebind_shape_rule"),
            std::string::npos);
  EXPECT_NE(runtime_descriptor_header.find("common/artifact_payload.hpp"),
            std::string::npos);
  EXPECT_EQ(runtime_descriptor_header.find("compiler/executable_bundle.hpp"),
            std::string::npos);
  EXPECT_EQ(runtime_descriptor_header.find("compiler::KernelArtifact"),
            std::string::npos);
  EXPECT_EQ(runtime_descriptor_source.find("compiler/"), std::string::npos);
  EXPECT_EQ(runtime_descriptor_source.find("ExecutableBundle"),
            std::string::npos);
  EXPECT_NE(compiler_runtime_descriptor_builder_header.find(
                "compiler/executable_bundle.hpp"),
            std::string::npos);
  EXPECT_NE(compiler_runtime_descriptor_builder_header.find(
                "runtime/executable_descriptor.hpp"),
            std::string::npos);
  EXPECT_NE(compiler_runtime_descriptor_builder_source.find(
                "verify_runtime_executable_descriptor"),
            std::string::npos);
  EXPECT_NE(common_artifact_payload_header.find("KernelArtifactPayloadKind"),
            std::string::npos);
  EXPECT_NE(common_artifact_payload_header.find("class KernelArtifactPayload"),
            std::string::npos);
  EXPECT_EQ(common_artifact_payload_header.find("compiler/"),
            std::string::npos);
  EXPECT_NE(stateful_stage_header.find(
                "const RuntimeStageExecutableDescriptor* descriptor"),
            std::string::npos);
  EXPECT_NE(
      stateful_stage_source.find("descriptor->stateful_effect == \"assign\""),
      std::string::npos);
  EXPECT_EQ(stateful_stage_source.find("AssignBase"), std::string::npos);
  EXPECT_EQ(stateful_stage_source.find("as_type_ptr"), std::string::npos);
  EXPECT_NE(stateful_execution_source.find(
                "descriptor->stateful_effect == \"read_value\""),
            std::string::npos);
  EXPECT_NE(stateful_execution_source.find(
                "descriptor->stateful_effect == \"assign\""),
            std::string::npos);
  EXPECT_NE(
      stateful_execution_source.find(
          "!slot.host_dirty && slot.initialized && slot.tensor.buf.valid()"),
      std::string::npos);
  EXPECT_NE(stateful_execution_header.find("sync_stateful_variable_host"),
            std::string::npos);
  EXPECT_NE(stateful_execution_source.find("stateful_variable_readback"),
            std::string::npos);
  EXPECT_NE(stateful_execution_source.find("slot.host_stale = true"),
            std::string::npos);
  EXPECT_NE(
      stateful_execution_source.find("direct_stateful_assign_variable_ids"),
      std::string::npos);
  EXPECT_NE(stateful_execution_source.find(
                "stateful_prebind_shape_rule == \"sum_inputs_along_axis\""),
            std::string::npos);
  EXPECT_EQ(stateful_execution_source.find("openvino/op/concat"),
            std::string::npos);
  EXPECT_EQ(stateful_execution_source.find("ov::op::v0::Concat"),
            std::string::npos);
  EXPECT_EQ(stateful_execution_source.find("normalize_axis"),
            std::string::npos);
  EXPECT_EQ(stateful_execution_header.find("is_stateful_read_value"),
            std::string::npos);
  EXPECT_EQ(stateful_execution_header.find("is_stateful_assign"),
            std::string::npos);
  EXPECT_EQ(stateful_execution_header.find("plugin/"), std::string::npos);
  EXPECT_EQ(stateful_execution_source.find("plugin/"), std::string::npos);
  EXPECT_EQ(stateful_execution_header.find("InferRequestState"),
            std::string::npos);
  EXPECT_EQ(stateful_execution_source.find("InferRequestState"),
            std::string::npos);
  EXPECT_NE(stateful_variable_state_header.find("StatefulVariableTensorState"),
            std::string::npos);
  EXPECT_NE(stateful_variable_state_header.find("StatefulVariableStateMap"),
            std::string::npos);
  EXPECT_NE(stateful_variable_state_header.find("host_stale"),
            std::string::npos);
  EXPECT_EQ(stateful_variable_state_header.find("plugin/"), std::string::npos);
  EXPECT_EQ(stateful_execution_source.find("ReadValueBase"), std::string::npos);
  EXPECT_EQ(stateful_execution_source.find("AssignBase"), std::string::npos);
  EXPECT_EQ(stateful_execution_source.find("dynamic_cast<const ov::op::util"),
            std::string::npos);
  EXPECT_NE(stateful_execution_header.find(
                "execute_infer_stage_with_stateful_contract"),
            std::string::npos);
  EXPECT_NE(stateful_execution_source.find(
                "execute_infer_stage_with_stateful_contract"),
            std::string::npos);
  EXPECT_NE(
      metal_infer_source.find("execute_infer_stage_with_stateful_contract"),
      std::string::npos);
  EXPECT_NE(
      opencl_infer_source.find("execute_infer_stage_with_stateful_contract"),
      std::string::npos);
  EXPECT_EQ(metal_infer_source.find("try_bind_direct_stateful_assign_output"),
            std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("try_bind_direct_stateful_assign_output"),
            std::string::npos);
  EXPECT_EQ(metal_infer_source.find("execute_stateful_stage"),
            std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("execute_stateful_stage"),
            std::string::npos);
  EXPECT_EQ(metal_infer_source.find("execute_pipeline_with_submission"),
            std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("execute_pipeline_with_submission"),
            std::string::npos);
  EXPECT_NE(infer_executor_header.find("InferRuntimeExecutionConfig"),
            std::string::npos);
  EXPECT_NE(infer_executor_header.find("runtime_input_tensors"),
            std::string::npos);
  EXPECT_NE(infer_pipeline_header.find("lookup_runtime_input_tensor"),
            std::string::npos);
  EXPECT_NE(metal_infer_source.find("lookup_runtime_input_tensor"),
            std::string::npos);
  EXPECT_NE(opencl_infer_source.find("lookup_runtime_input_tensor"),
            std::string::npos);
  EXPECT_EQ(metal_infer_source.find("get_input_device"), std::string::npos);
  EXPECT_EQ(metal_infer_source.find("has_input_device"), std::string::npos);
  EXPECT_EQ(metal_infer_source.find("bind_input_device"), std::string::npos);
  EXPECT_NE(
      infer_executor_header.find("prepare_reusable_infer_runtime_pipeline"),
      std::string::npos);
  EXPECT_NE(
      infer_executor_source.find("prepare_reusable_infer_runtime_pipeline"),
      std::string::npos);
  EXPECT_NE(infer_executor_source.find("assign_runtime_stage_output_shapes"),
            std::string::npos);
  EXPECT_NE(infer_executor_source.find("execute_pipeline_with_submission"),
            std::string::npos);
  EXPECT_EQ(
      infer_io_utils_header.find("prepare_reusable_pipeline_with_outputs"),
      std::string::npos);
  EXPECT_EQ(infer_io_utils_header.find("build_pipeline_with_outputs"),
            std::string::npos);
  EXPECT_EQ(infer_io_utils_source.find("reset_reusable_pipeline_outputs"),
            std::string::npos);
  EXPECT_EQ(infer_io_utils_source.find("prepare_stage_output_handles"),
            std::string::npos);
  EXPECT_NE(metal_infer_source.find("prepare_and_execute_infer_runtime"),
            std::string::npos);
  EXPECT_NE(opencl_infer_source.find("prepare_and_execute_infer_runtime"),
            std::string::npos);
  EXPECT_EQ(metal_infer_source.find("assign_runtime_stage_output_shapes"),
            std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("assign_runtime_stage_output_shapes"),
            std::string::npos);
  EXPECT_EQ(metal_infer_source.find("prepare_reusable_execution_plan"),
            std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("prepare_reusable_execution_plan"),
            std::string::npos);
  EXPECT_NE(opencl_infer_source.find("OpenClInferSubmissionSession"),
            std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("record_infer_submission_tuning_counters"),
            std::string::npos);
  EXPECT_NE(
      infer_executor_source.find("record_infer_submission_tuning_counters"),
      std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("execute_pipeline(\n"), std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("backend->context->finish();"),
            std::string::npos);
  EXPECT_NE(infer_request_header.find("query_state() const override;"),
            std::string::npos);
  EXPECT_EQ(
      infer_request_header.find("query_state() const override { return {}; }"),
      std::string::npos);
  EXPECT_EQ(infer_request_header.find("infer_metal_impl"), std::string::npos);
  EXPECT_EQ(infer_request_header.find("infer_opencl_impl"), std::string::npos);
  EXPECT_EQ(infer_request_common_source.find("switch (cm->backend())"),
            std::string::npos);
  EXPECT_NE(infer_request_common_source.find("execute_backend_infer"),
            std::string::npos);
  EXPECT_NE(metal_infer_source.find("execute_metal_infer_request"),
            std::string::npos);
  EXPECT_NE(opencl_infer_source.find("execute_opencl_infer_request"),
            std::string::npos);
  EXPECT_NE(infer_request_header.find("bind_outputs_after_infer"),
            std::string::npos);
  EXPECT_NE(infer_request_header.find("bind_inputs_before_infer"),
            std::string::npos);
  EXPECT_NE(infer_request_common_source.find("bind_inputs_before_infer"),
            std::string::npos);
  EXPECT_NE(infer_request_common_source.find("bind_inputs_for_infer"),
            std::string::npos);
  EXPECT_NE(infer_request_common_source.find(
                "ov::ISyncInferRequest::get_tensor(get_inputs().at(idx))"),
            std::string::npos);
  EXPECT_EQ(infer_request_common_source.find("return state.bound_inputs[idx]"),
            std::string::npos);
  EXPECT_NE(metal_infer_source.find("bind_inputs_before_infer"),
            std::string::npos);
  EXPECT_NE(opencl_infer_source.find("bind_inputs_before_infer"),
            std::string::npos);
  EXPECT_EQ(metal_infer_source.find("bind_inputs_for_infer"),
            std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("bind_inputs_for_infer"),
            std::string::npos);
  EXPECT_EQ(metal_infer_source.find("ensure_input_handles"), std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("ensure_input_handles"),
            std::string::npos);
  EXPECT_EQ(metal_infer_source.find("\"upload\""), std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("\"upload\""), std::string::npos);
  EXPECT_NE(infer_request_common_source.find("bind_outputs_after_infer"),
            std::string::npos);
  EXPECT_NE(infer_request_common_source.find("bind_outputs_for_infer"),
            std::string::npos);
  EXPECT_NE(infer_request_common_source.find("host_output_matches_public_port"),
            std::string::npos);
  EXPECT_NE(infer_request_common_source.find(
                "state.bound_output_hosts[idx] = host_output"),
            std::string::npos);
  EXPECT_NE(infer_request_common_source.find(
                "auto current = ov::ISyncInferRequest::get_tensor(port)"),
            std::string::npos);
  EXPECT_NE(metal_infer_source.find("bind_outputs_after_infer"),
            std::string::npos);
  EXPECT_NE(opencl_infer_source.find("bind_outputs_after_infer"),
            std::string::npos);
  EXPECT_EQ(metal_infer_source.find("bind_outputs_for_infer"),
            std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("bind_outputs_for_infer"),
            std::string::npos);
  EXPECT_NE(backend_request_state_header.find("output_tensors"),
            std::string::npos);
  EXPECT_NE(infer_request_common_source.find("output_tensors"),
            std::string::npos);
  EXPECT_NE(metal_infer_source.find("output_tensors"), std::string::npos);
  EXPECT_EQ(metal_infer_source.find("has_output_device"), std::string::npos);
  EXPECT_EQ(metal_infer_source.find("get_output_device"), std::string::npos);
  EXPECT_EQ(metal_infer_source.find("bind_output_device"), std::string::npos);
  EXPECT_EQ(metal_infer_source.find("MetalTensorMap"), std::string::npos);
  EXPECT_EQ(metal_infer_source.find("reusable_host_output_plan"),
            std::string::npos);
  EXPECT_EQ(opencl_infer_source.find("reusable_host_output_plan"),
            std::string::npos);
  EXPECT_NE(variable_state_source.find("class GfxVariableState"),
            std::string::npos);
  EXPECT_NE(variable_state_source.find("slot.host_dirty = true"),
            std::string::npos);
  EXPECT_NE(variable_state_source.find("sync_stateful_variable_host"),
            std::string::npos);
  EXPECT_NE(backend_runtime_header.find("struct BackendResources"),
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
  EXPECT_NE(builder_source.find("fusion_contract_for_node"), std::string::npos);
  EXPECT_EQ(materializer_header.find("fusion_contract_for"), std::string::npos);
  EXPECT_NE(materializer_header.find("create_vendor_attention_stage"),
            std::string::npos);
  EXPECT_NE(materializer_source.find("create_vendor_attention_stage"),
            std::string::npos);
  EXPECT_NE(builder_source.find("materialize_vendor_attention_artifact"),
            std::string::npos);
  EXPECT_NE(builder_header.find("PipelineVendorAttentionArtifact"),
            std::string::npos);
  EXPECT_EQ(materializer_header.find("PipelineVendorAttentionArtifactResolver"),
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
  EXPECT_EQ(materializer_source.find("m_vendor_attention_artifact_resolver"),
            std::string::npos);
  EXPECT_NE(materializer_source.find("valid runtime vendor"),
            std::string::npos);
  EXPECT_NE(materializer_source.find("attention stage plan"),
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
  EXPECT_NE(cmake_sources.find("compiler/pipeline_stage_builder.cpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("compiler/pipeline_stage_builder.hpp"),
            std::string::npos);
  EXPECT_NE(cmake_sources.find("runtime/pipeline_stage_plan.hpp"),
            std::string::npos);
  EXPECT_EQ(cmake_sources.find("runtime/pipeline_stage_builder.cpp"),
            std::string::npos);
  EXPECT_EQ(cmake_sources.find("runtime/pipeline_stage_builder.hpp"),
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
  EXPECT_NE(static_backend_module_source.find(
                "PipelineVendorAttentionArtifactResolver"),
            std::string::npos);
  EXPECT_EQ(
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
       PipelineStagePlanBuilderOnlyDirectAliasesStatefulAssignWhenExclusive) {
  auto parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto variable = std::make_shared<ov::op::util::Variable>(
      ov::op::util::VariableInfo{ov::Shape{1}, ov::element::f32, "variable0"});
  auto add = std::make_shared<ov::op::v1::Add>(parameter, parameter);
  auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
  auto unrelated_result = std::make_shared<ov::op::v0::Result>(parameter);
  ov::Model assign_only_model(
      ov::ResultVector{unrelated_result}, ov::SinkVector{assign},
      ov::ParameterVector{parameter}, "stateful_assign_direct_alias_safe");

  const auto assign_only_outputs =
      compiler::collect_model_output_ports(assign_only_model);
  compiler::PipelineStagePlanBuilder assign_only_builder(assign_only_outputs);
  const auto assign_only_plan = assign_only_builder.make_stage_plan(add, 0);
  ASSERT_EQ(assign_only_plan.outputs.size(), 1u);
  EXPECT_EQ(assign_only_plan.outputs[0].direct_stateful_assign_variable_id,
            "variable0");

  auto shared_parameter =
      std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
  auto shared_variable = std::make_shared<ov::op::util::Variable>(
      ov::op::util::VariableInfo{ov::Shape{1}, ov::element::f32, "variable0"});
  auto shared_add =
      std::make_shared<ov::op::v1::Add>(shared_parameter, shared_parameter);
  auto shared_assign =
      std::make_shared<ov::op::v6::Assign>(shared_add, shared_variable);
  auto shared_result = std::make_shared<ov::op::v0::Result>(shared_add);
  ov::Model shared_consumer_model(ov::ResultVector{shared_result},
                                  ov::SinkVector{shared_assign},
                                  ov::ParameterVector{shared_parameter},
                                  "stateful_assign_direct_alias_unsafe");

  const auto shared_outputs =
      compiler::collect_model_output_ports(shared_consumer_model);
  compiler::PipelineStagePlanBuilder shared_builder(shared_outputs);
  const auto shared_plan = shared_builder.make_stage_plan(shared_add, 0);
  ASSERT_EQ(shared_plan.outputs.size(), 1u);
  EXPECT_TRUE(
      shared_plan.outputs[0].direct_stateful_assign_variable_id.empty());
}

TEST_F(GfxBackendArchitectureContractTest,
       ManifestCarriesDirectStatefulAssignPrebindShapeContract) {
  auto lhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 2});
  auto rhs = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::PartialShape{ov::Dimension::dynamic(), 2});
  auto concat = std::make_shared<ov::op::v0::Concat>(
      ov::OutputVector{lhs->output(0), rhs->output(0)}, 0);
  auto variable = std::make_shared<ov::op::util::Variable>(
      ov::op::util::VariableInfo{ov::PartialShape{ov::Dimension::dynamic(), 2},
                                 ov::element::f32, "variable0"});
  auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
  ASSERT_NE(assign, nullptr);
  ASSERT_EQ(concat->output(0).get_target_inputs().size(), 1u);

  compiler::PlannedOperation op;
  op.source_node = concat;
  op.node_name = concat->get_friendly_name();
  op.type_name = concat->get_type_name();
  op.kernel_unit = compiler::KernelUnit::describe(
      LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
      "opencl", "metadata", true);
  op.layout = compiler::select_tensor_layout_plan("Concat", concat);
  op.profitability_score = 1.0;
  op.input_element_types = {"f32", "f32"};
  op.input_shapes = {"{?,2}", "{?,2}"};
  op.output_element_types = {"f32"};
  op.output_shapes = {"{?,2}"};

  compiler::LoweringPlan plan;
  plan.target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  plan.operations.push_back(std::move(op));

  const auto manifest = compiler::ManifestBuilder{}.build(plan);
  ASSERT_TRUE(manifest.valid());
  ASSERT_EQ(manifest.stages.size(), 1u);
  ASSERT_EQ(manifest.stages.front().outputs.size(), 1u);
  const auto &output = manifest.stages.front().outputs.front();
  EXPECT_EQ(output.stateful_prebind_variable_id, "variable0");
  EXPECT_EQ(output.stateful_prebind_shape_rule, "sum_inputs_along_axis");
  EXPECT_EQ(output.stateful_prebind_shape_axis, 0);

  const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
  ASSERT_TRUE(executable.valid());
  const auto runtime_descriptor =
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
  ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
  ASSERT_EQ(runtime_descriptor.stages.front().output_bindings.size(), 1u);
  const auto &runtime_output =
      runtime_descriptor.stages.front().output_bindings.front();
  EXPECT_EQ(runtime_output.stateful_prebind_variable_id, "variable0");
  EXPECT_EQ(runtime_output.stateful_prebind_shape_rule,
            "sum_inputs_along_axis");
  EXPECT_EQ(runtime_output.stateful_prebind_shape_axis, 0);
  EXPECT_TRUE(compiler::runtime_executable_descriptor_valid(runtime_descriptor,
                                                            executable));

  auto stale_runtime_descriptor = runtime_descriptor;
  stale_runtime_descriptor.stages.front()
      .output_bindings.front()
      .stateful_prebind_shape_rule = "stale_runtime_rule";
  const auto stale_verification =
      compiler::verify_runtime_executable_descriptor(stale_runtime_descriptor,
                                                     executable);
  EXPECT_FALSE(stale_verification.valid());
  EXPECT_TRUE(has_diagnostic_containing(stale_verification.diagnostics,
                                        "output binding drift"));
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
  EXPECT_NE(stage_compiler_policy.find("common/gpu_parallelism_profile.hpp"),
            std::string::npos);
  EXPECT_NE(stage_compiler_policy.find("BackendExecutionCapabilities"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       MlirSourcePlanBuildersUseCompilerOwnedStageCompilerPolicyResolver) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto policy_source =
      read_text_file(module_root / "src/compiler/stage_compiler_policy.cpp");
  const auto provider_source =
      read_text_file(module_root / "src/backends/backend_module_provider.cpp");
  EXPECT_EQ(policy_source.find("compiler/backend_registry.hpp"),
            std::string::npos);
  EXPECT_EQ(policy_source.find("BackendRegistry::default_registry()"),
            std::string::npos);
  EXPECT_NE(policy_source.find("make_stage_compiler_policy_from_capabilities("),
            std::string::npos);
  EXPECT_NE(provider_source.find("compiler/stage_compiler_policy.hpp"),
            std::string::npos);
  EXPECT_NE(provider_source.find("BackendRegistry::default_registry()"),
            std::string::npos);
  EXPECT_NE(
      provider_source.find("make_stage_compiler_policy_from_capabilities("),
      std::string::npos);
  EXPECT_NE(provider_source.find("resolve_stage_compiler_policy("),
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

  assert_uses_shared_resolver(
      "src/backends/metal/compiler/msl_codegen_apple_mps.cpp");
  assert_uses_shared_resolver(
      "src/backends/metal/compiler/msl_codegen_apple_msl_dispatch.cpp");
  assert_uses_shared_resolver(
      "src/backends/metal/compiler/msl_codegen_matmul_metal.cpp");
}

TEST_F(GfxBackendArchitectureContractTest,
       CompiledModelUsesCompilerOwnedPrecisionSensitiveFusionPolicy) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto compiled_model_source =
      read_text_file(module_root / "src/plugin/compiled_model.cpp");
  const auto fusion_source =
      read_text_file(module_root / "src/compiler/pipeline_stage_fusion.cpp");

  EXPECT_EQ(compiled_model_source.find("compiler/stage_policy.hpp"),
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
      compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
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
       SharedBackendRegistryDoesNotConstructConcreteBackends) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto backend_registry_source =
      read_text_file(module_root / "src/compiler/backend_registry.cpp");
  const auto backend_module_provider_source =
      read_text_file(module_root / "src/backends/backend_module_provider.cpp");
  const auto metal_backend_module_source = read_text_file(
      module_root / "src/backends/metal/compiler/metal_backend_module.cpp");
  const auto opencl_backend_module_source = read_text_file(
      module_root / "src/backends/opencl/compiler/opencl_backend_module.cpp");

  EXPECT_EQ(backend_registry_source.find("backends/"), std::string::npos);
  EXPECT_EQ(backend_registry_source.find("make_metal"), std::string::npos);
  EXPECT_EQ(backend_registry_source.find("make_opencl"), std::string::npos);
  EXPECT_EQ(backend_registry_source.find("StaticBackendModule"),
            std::string::npos);
  EXPECT_EQ(backend_registry_source.find("make_default_backend_modules"),
            std::string::npos);

  EXPECT_NE(backend_module_provider_source.find("make_default_backend_modules"),
            std::string::npos);
  EXPECT_NE(
      backend_module_provider_source.find("BackendRegistry::default_registry"),
      std::string::npos);
  EXPECT_NE(backend_module_provider_source.find("make_metal_backend_module"),
            std::string::npos);
  EXPECT_NE(backend_module_provider_source.find("make_opencl_backend_module"),
            std::string::npos);
  EXPECT_NE(metal_backend_module_source.find("make_static_backend_module"),
            std::string::npos);
  EXPECT_NE(opencl_backend_module_source.find("make_static_backend_module"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CMakeKeepsBackendCompilerTargetsOutsidePluginCore) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto gfx_sources =
      read_text_file(module_root / "cmake/GfxSources.cmake");
  const auto src_cmake = read_text_file(module_root / "src/CMakeLists.txt");
  const auto backend_registry_source =
      read_text_file(module_root / "src/compiler/backend_registry.cpp");
  const auto backend_module_provider_source =
      read_text_file(module_root / "src/backends/backend_module_provider.cpp");
  const auto stage_compiler_policy_source =
      read_text_file(module_root / "src/compiler/stage_compiler_policy.cpp");

  const auto compiler_sources_begin =
      gfx_sources.find("set(GFX_COMPILER_COMMON_SOURCES");
  const auto compiler_headers_begin =
      gfx_sources.find("set(GFX_COMPILER_COMMON_HEADERS");
  const auto plugin_sources_begin = gfx_sources.find("set(GFX_PLUGIN_SOURCES");
  const auto plugin_headers_begin = gfx_sources.find("set(GFX_PLUGIN_HEADERS");
  const auto provider_headers_begin =
      gfx_sources.find("set(GFX_BACKEND_MODULE_PROVIDER_HEADERS");
  ASSERT_NE(compiler_sources_begin, std::string::npos);
  ASSERT_NE(compiler_headers_begin, std::string::npos);
  ASSERT_NE(plugin_sources_begin, std::string::npos);
  ASSERT_NE(plugin_headers_begin, std::string::npos);
  ASSERT_NE(provider_headers_begin, std::string::npos);
  ASSERT_LT(compiler_sources_begin, compiler_headers_begin);
  ASSERT_LT(compiler_headers_begin, plugin_sources_begin);
  ASSERT_LT(plugin_sources_begin, plugin_headers_begin);
  ASSERT_LT(plugin_headers_begin, provider_headers_begin);

  const auto compiler_sources_block = gfx_sources.substr(
      compiler_sources_begin, compiler_headers_begin - compiler_sources_begin);
  const auto compiler_headers_block = gfx_sources.substr(
      compiler_headers_begin, plugin_sources_begin - compiler_headers_begin);
  const auto plugin_sources_block = gfx_sources.substr(
      plugin_sources_begin, plugin_headers_begin - plugin_sources_begin);
  const auto plugin_headers_block = gfx_sources.substr(
      plugin_headers_begin, provider_headers_begin - plugin_headers_begin);

  for (std::string_view required : {
           "compiler/backend_registry.cpp",
           "compiler/static_backend_module.cpp",
           "compiler/manifest.cpp",
           "compiler/stage_policy.cpp",
           "compiler/tensor_layout.cpp",
       }) {
    EXPECT_NE(compiler_sources_block.find(required), std::string::npos)
        << required;
  }

  for (std::string_view forbidden : {
           "plugin/",
           "runtime/",
           "transforms/",
           "backends/backend_module_provider.cpp",
           "backends/metal/compiler/",
           "backends/opencl/compiler/",
       }) {
    EXPECT_EQ(compiler_sources_block.find(forbidden), std::string::npos)
        << forbidden;
  }

  for (std::string_view required : {
           "compiler/backend_registry.hpp",
           "compiler/static_backend_module.hpp",
           "compiler/stage_compiler_policy.hpp",
       }) {
    EXPECT_NE(compiler_headers_block.find(required), std::string::npos)
        << required;
  }

  for (std::string_view forbidden : {
           "compiler/",
           "backends/backend_module_provider.cpp",
           "backends/metal/compiler/",
           "backends/opencl/compiler/",
       }) {
    EXPECT_EQ(plugin_sources_block.find(forbidden), std::string::npos)
        << forbidden;
  }

  for (std::string_view forbidden : {
           "compiler/",
           "backends/metal/compiler/",
           "backends/opencl/compiler/",
           "compiler/backend_module_provider.hpp",
       }) {
    EXPECT_EQ(plugin_headers_block.find(forbidden), std::string::npos)
        << forbidden;
  }

  for (std::string_view required : {
           "GFX_COMPILER_COMMON_SOURCES",
           "GFX_COMPILER_COMMON_HEADERS",
           "GFX_BACKEND_MODULE_PROVIDER_SOURCES",
           "GFX_METAL_BACKEND_COMPILER_SOURCES",
           "GFX_OPENCL_BACKEND_COMPILER_SOURCES",
       }) {
    EXPECT_NE(gfx_sources.find(required), std::string::npos) << required;
  }

  const auto compiler_common_target_begin =
      src_cmake.find("add_library(gfx_compiler_common STATIC)");
  const auto core_target_begin =
      src_cmake.find("add_library(gfx_plugin_core STATIC)");
  const auto metal_backend_target_begin =
      src_cmake.find("add_library(gfx_metal_backend_compiler STATIC)");
  const auto opencl_backend_target_begin =
      src_cmake.find("add_library(gfx_opencl_backend_compiler STATIC)");
  const auto provider_target_begin =
      src_cmake.find("add_library(gfx_backend_module_provider STATIC)");
  const auto backend_plugin_target_begin =
      src_cmake.find("add_library(gfx_plugin_metal STATIC)");
  ASSERT_NE(compiler_common_target_begin, std::string::npos);
  ASSERT_NE(core_target_begin, std::string::npos);
  ASSERT_NE(metal_backend_target_begin, std::string::npos);
  ASSERT_NE(opencl_backend_target_begin, std::string::npos);
  ASSERT_NE(provider_target_begin, std::string::npos);
  ASSERT_NE(backend_plugin_target_begin, std::string::npos);
  ASSERT_LT(compiler_common_target_begin, core_target_begin);
  ASSERT_LT(core_target_begin, metal_backend_target_begin);
  ASSERT_LT(metal_backend_target_begin, opencl_backend_target_begin);
  ASSERT_LT(opencl_backend_target_begin, provider_target_begin);
  ASSERT_LT(provider_target_begin, backend_plugin_target_begin);

  const auto compiler_common_target_block =
      src_cmake.substr(compiler_common_target_begin,
                       core_target_begin - compiler_common_target_begin);
  EXPECT_NE(compiler_common_target_block.find("GFX_COMPILER_COMMON_HEADERS"),
            std::string::npos);
  EXPECT_NE(compiler_common_target_block.find("GFX_COMPILER_COMMON_SOURCES"),
            std::string::npos);
  EXPECT_EQ(compiler_common_target_block.find("gfx_plugin_core"),
            std::string::npos);

  const auto core_target_block = src_cmake.substr(
      core_target_begin, metal_backend_target_begin - core_target_begin);
  EXPECT_NE(core_target_block.find("gfx_compiler_common"), std::string::npos);
  for (std::string_view forbidden : {
           "GFX_METAL_BACKEND_COMPILER",
           "GFX_OPENCL_BACKEND_COMPILER",
           "gfx_metal_msl_compiler",
           "gfx_opencl_kernel_artifacts",
           "backends/metal/compiler",
           "backends/opencl/compiler",
       }) {
    EXPECT_EQ(core_target_block.find(forbidden), std::string::npos)
        << forbidden;
  }

  const auto metal_backend_target_block = src_cmake.substr(
      metal_backend_target_begin,
      opencl_backend_target_begin - metal_backend_target_begin);
  EXPECT_NE(metal_backend_target_block.find("gfx_compiler_common"),
            std::string::npos);
  EXPECT_EQ(metal_backend_target_block.find("gfx_plugin_core"),
            std::string::npos);

  const auto opencl_backend_target_block =
      src_cmake.substr(opencl_backend_target_begin,
                       provider_target_begin - opencl_backend_target_begin);
  EXPECT_NE(opencl_backend_target_block.find("gfx_compiler_common"),
            std::string::npos);
  EXPECT_EQ(opencl_backend_target_block.find("gfx_plugin_core"),
            std::string::npos);

  const auto provider_target_block =
      src_cmake.substr(provider_target_begin,
                       backend_plugin_target_begin - provider_target_begin);
  EXPECT_NE(provider_target_block.find("GFX_BACKEND_MODULE_PROVIDER_HEADERS"),
            std::string::npos);
  EXPECT_NE(provider_target_block.find("GFX_BACKEND_MODULE_PROVIDER_SOURCES"),
            std::string::npos);
  EXPECT_NE(provider_target_block.find("gfx_compiler_common"),
            std::string::npos);
  EXPECT_NE(provider_target_block.find("gfx_metal_backend_compiler"),
            std::string::npos);
  EXPECT_NE(provider_target_block.find("gfx_opencl_backend_compiler"),
            std::string::npos);
  EXPECT_NE(src_cmake.find("gfx_backend_module_provider)"), std::string::npos);

  EXPECT_EQ(backend_registry_source.find("backend_module_provider"),
            std::string::npos);
  EXPECT_EQ(backend_registry_source.find("make_default_backend_modules"),
            std::string::npos);
  EXPECT_EQ(
      stage_compiler_policy_source.find("BackendRegistry::default_registry"),
      std::string::npos);
  EXPECT_NE(
      backend_module_provider_source.find("BackendRegistry::default_registry"),
      std::string::npos);
  EXPECT_NE(
      backend_module_provider_source.find("resolve_stage_compiler_policy"),
      std::string::npos);
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

  const auto read_value_plan =
      compiler::select_tensor_layout_plan("ReadValue", nullptr);
  EXPECT_FALSE(read_value_plan.view_only);
  EXPECT_EQ(read_value_plan.kind, GfxTensorLayoutKind::Materialized);

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
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    const auto verification =
        compiler::verify_runtime_executable_descriptor(runtime_descriptor,
                                                       executable);
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
        compiler::verify_runtime_executable_descriptor(stale_region_descriptor,
                                                       executable);
    EXPECT_FALSE(stale_region_verification.valid());
    EXPECT_TRUE(has_diagnostic_containing(stale_region_verification.diagnostics,
                                          "memory region drift"));

    auto stale_fingerprint_descriptor = runtime_descriptor;
    stale_fingerprint_descriptor.memory_plan.fingerprint =
        "stale-runtime-memory-plan";
    const auto stale_fingerprint_verification =
        compiler::verify_runtime_executable_descriptor(
            stale_fingerprint_descriptor, executable);
    EXPECT_FALSE(stale_fingerprint_verification.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_fingerprint_verification.diagnostics,
                                  "memory plan fingerprint drift"));

    auto stale_binding_descriptor = runtime_descriptor;
    stale_binding_descriptor.stages.front()
        .input_bindings.front()
        .memory_region_id = "stale-runtime-input-region";
    const auto stale_binding_verification =
        compiler::verify_runtime_executable_descriptor(stale_binding_descriptor,
                                                       executable);
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
      read_text_file(root / "src/runtime/infer_pipeline.cpp");
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
  EXPECT_EQ(runtime_session_source.find("stage.compile("), std::string::npos);
  EXPECT_NE(runtime_session_source.find("stage.prepare_runtime_handle("),
            std::string::npos);

  const auto gpu_stage_header =
      read_text_file(root / "src/runtime/gpu_stage.hpp");
  EXPECT_EQ(gpu_stage_header.find("virtual void compile("), std::string::npos);
  EXPECT_NE(gpu_stage_header.find("virtual void prepare_runtime_handle("),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonInferPipelineDoesNotRepairMissingRuntimeStageIndex) {
  const auto root = find_gfx_module_root_for_source_contract();
  const auto infer_pipeline_source =
      read_text_file(root / "src/runtime/infer_pipeline.cpp");

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
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().requires_runtime_shape_args,
              test_case.requires_runtime_shape_args);

    auto stale_descriptor = runtime_descriptor;
    stale_descriptor.stages.front().requires_runtime_shape_args =
        !test_case.requires_runtime_shape_args;
    const auto stale_result =
        compiler::verify_runtime_executable_descriptor(stale_descriptor,
                                                       executable);
    EXPECT_FALSE(stale_result.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_result.diagnostics, "artifact drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeShapeRuleComesFromCompilerManifestAndDescriptor) {
  struct Case {
    const char *op_type;
    const char *expected_rule;
  };

  for (const auto test_case :
       {Case{"Concat", "concat"}, Case{"Broadcast", "broadcast"},
        Case{"Select", "select"}, Case{"ShapeOf", "shape_of"},
        Case{"Slice", "slice"}, Case{"StridedSlice", "slice"},
        Case{"Range", "range"}, Case{"Tile", "tile"},
        Case{"Relu", "static_or_descriptor"}}) {
    SCOPED_TRACE(test_case.op_type);
    compiler::LoweringPlan plan;
    plan.target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    compiler::PlannedOperation op;
    op.node_name = test_case.op_type;
    op.type_name = test_case.op_type;
    op.kernel_unit = compiler::KernelUnit::describe(
        LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
        plan.target.backend_id(), test_case.op_type);
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
    EXPECT_EQ(manifest.stages.front().runtime_shape.rule,
              test_case.expected_rule);

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());
    ASSERT_EQ(executable.artifact_descriptors.size(), 1u);
    EXPECT_EQ(executable.artifact_descriptors.front().kernel.runtime_shape_rule,
              test_case.expected_rule);

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().runtime_shape_rule,
              test_case.expected_rule);

    auto stale_descriptor = runtime_descriptor;
    stale_descriptor.stages.front().runtime_shape_rule = "stale_runtime_rule";
    const auto stale_result =
        compiler::verify_runtime_executable_descriptor(stale_descriptor,
                                                       executable);
    EXPECT_FALSE(stale_result.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_result.diagnostics, "artifact drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       RuntimeSubmissionContractComesFromCompilerManifestAndDescriptor) {
  struct Case {
    const char *op_type;
    bool expected_dependency_boundary;
  };

  for (const auto test_case :
       {Case{"Concat", true}, Case{"Softmax", true}, Case{"Relu", false}}) {
    SCOPED_TRACE(test_case.op_type);
    compiler::LoweringPlan plan;
    plan.target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    compiler::PlannedOperation op;
    op.node_name = test_case.op_type;
    op.type_name = test_case.op_type;
    op.kernel_unit = compiler::KernelUnit::describe(
        LoweringRouteKind::Metadata, KernelUnitKind::Metadata, "metadata",
        plan.target.backend_id(), test_case.op_type);
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
    EXPECT_EQ(manifest.stages.front().submission.stage_weight, 1u);
    EXPECT_EQ(manifest.stages.front().submission.dependency_extension_boundary,
              test_case.expected_dependency_boundary);

    const auto executable = compiler::ExecutableBundleBuilder{}.build(manifest);
    ASSERT_TRUE(executable.valid());

    const auto runtime_descriptor =
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    ASSERT_TRUE(compiler::runtime_executable_descriptor_valid(
        runtime_descriptor, executable));
    ASSERT_EQ(runtime_descriptor.stages.size(), 1u);
    EXPECT_EQ(runtime_descriptor.stages.front().submission_stage_weight,
              manifest.stages.front().submission.stage_weight);
    EXPECT_EQ(runtime_descriptor.stages.front().submission_macs_estimate,
              manifest.stages.front().submission.macs_estimate);
    EXPECT_EQ(runtime_descriptor.stages.front().submission_dependency_boundary,
              test_case.expected_dependency_boundary);

    auto stale_descriptor = runtime_descriptor;
    stale_descriptor.stages.front().submission_dependency_boundary =
        !test_case.expected_dependency_boundary;
    const auto stale_result =
        compiler::verify_runtime_executable_descriptor(stale_descriptor,
                                                       executable);
    EXPECT_FALSE(stale_result.valid());
    EXPECT_TRUE(
        has_diagnostic_containing(stale_result.diagnostics, "artifact drift"));
  }
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonInferSubmissionUsesDescriptorOwnedWorkloadContract) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto infer_submission_source =
      read_text_file(module_root / "src/runtime/infer_submission.cpp");
  const auto infer_submission_header =
      read_text_file(module_root / "src/runtime/infer_submission.hpp");

  EXPECT_EQ(infer_submission_header.find("runtime/gpu_device_info.hpp"),
            std::string::npos);
  EXPECT_EQ(infer_submission_header.find("GpuBackend backend"),
            std::string::npos);
  EXPECT_EQ(infer_submission_header.find("GpuDeviceFamily device_family"),
            std::string::npos);
  EXPECT_EQ(infer_submission_source.find("openvino/op/convolution"),
            std::string::npos);
  EXPECT_EQ(infer_submission_source.find("openvino/op/group_conv"),
            std::string::npos);
  EXPECT_EQ(infer_submission_source.find("openvino/op/matmul"),
            std::string::npos);
  EXPECT_EQ(infer_submission_source.find("dynamic_cast<const ov::op"),
            std::string::npos);
  EXPECT_EQ(infer_submission_source.find("as_type_ptr<const ov::op"),
            std::string::npos);
  EXPECT_EQ(infer_submission_source.find("caps.backend"), std::string::npos);
  EXPECT_EQ(infer_submission_source.find("caps.device_family"),
            std::string::npos);
  EXPECT_EQ(infer_submission_source.find("GpuBackend::OpenCL"),
            std::string::npos);
  EXPECT_EQ(infer_submission_source.find("GpuBackend::Metal"),
            std::string::npos);
  EXPECT_EQ(infer_submission_source.find("GpuDeviceFamily::"),
            std::string::npos);
  EXPECT_NE(infer_submission_source.find("submission_macs_estimate"),
            std::string::npos);
  EXPECT_NE(infer_submission_source.find("submission_dependency_boundary"),
            std::string::npos);
  EXPECT_NE(infer_submission_source.find("submission_stage_weight"),
            std::string::npos);
  EXPECT_NE(
      infer_submission_source.find("runtime_stage_descriptor_or_null(stage)"),
      std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonParallelismUsesProfileDataNotBackendBranches) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto parallelism_source =
      read_text_file(module_root / "src/runtime/gfx_parallelism.cpp");
  const auto parallelism_header =
      read_text_file(module_root / "src/runtime/gfx_parallelism.hpp");

  EXPECT_EQ(parallelism_header.find("runtime/gpu_device_info.hpp"),
            std::string::npos);
  EXPECT_EQ(parallelism_header.find("GpuBackend backend"), std::string::npos);
  EXPECT_EQ(parallelism_header.find("GpuDeviceFamily device_family"),
            std::string::npos);
  EXPECT_EQ(parallelism_source.find("caps.backend"), std::string::npos);
  EXPECT_EQ(parallelism_source.find("caps.device_family"), std::string::npos);
  EXPECT_EQ(parallelism_source.find("GpuBackend::"), std::string::npos);
  EXPECT_EQ(parallelism_source.find("GpuDeviceFamily::"), std::string::npos);
  EXPECT_EQ(parallelism_source.find("BroadcomV3D"), std::string::npos);
  EXPECT_NE(parallelism_source.find("enable_skinny_matmul_tiles"),
            std::string::npos);
  EXPECT_NE(parallelism_source.find("chunk_dispatch"), std::string::npos);
  EXPECT_NE(parallelism_source.find("scale_conv_threads_for_dense_reduction"),
            std::string::npos);
}

TEST_F(GfxBackendArchitectureContractTest,
       CommonInferPipelineUsesDescriptorOwnedSliceShapeArgPolicy) {
  const auto module_root = find_gfx_module_root_for_source_contract();
  const auto infer_pipeline_source =
      read_text_file(module_root / "src/runtime/infer_pipeline.cpp");
  const auto infer_pipeline_header =
      read_text_file(module_root / "src/runtime/infer_pipeline.hpp");

  EXPECT_EQ(infer_pipeline_source.find("backend == GpuBackend::OpenCL"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("backend == GpuBackend::Metal"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_header.find("GpuBackend backend"),
            std::string::npos);
  EXPECT_NE(infer_pipeline_source.find("requires_runtime_shape_args"),
            std::string::npos);
  EXPECT_NE(infer_pipeline_source.find("descriptor->runtime_shape_rule"),
            std::string::npos);
  EXPECT_NE(
      infer_pipeline_source.find("runtime_stage_descriptor_or_null(stage)"),
      std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("openvino/op/broadcast"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("openvino/op/concat"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("openvino/op/range"), std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("openvino/op/select"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("openvino/op/shape_of"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("openvino/op/slice"), std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("openvino/op/strided_slice"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("openvino/op/tile"), std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("as_type_ptr<const ov::op::v0::Concat"),
            std::string::npos);
  EXPECT_EQ(
      infer_pipeline_source.find("as_type_ptr<const ov::op::v1::Broadcast"),
      std::string::npos);
  EXPECT_EQ(
      infer_pipeline_source.find("as_type_ptr<const ov::op::v3::Broadcast"),
      std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("as_type_ptr<const ov::op::v1::Select"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("as_type_ptr<const ov::op::v3::ShapeOf"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("as_type_ptr<const ov::op::v8::Slice"),
            std::string::npos);
  EXPECT_EQ(
      infer_pipeline_source.find("as_type_ptr<const ov::op::v1::StridedSlice"),
      std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("as_type_ptr<const ov::op::v4::Range"),
            std::string::npos);
  EXPECT_EQ(infer_pipeline_source.find("as_type_ptr<const ov::op::v0::Tile"),
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
        compiler::RuntimeExecutableDescriptorBuilder{}.build(executable);
    const auto runtime_result =
        compiler::verify_runtime_executable_descriptor(runtime_descriptor,
                                                       executable);
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
