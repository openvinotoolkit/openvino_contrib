// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backends/metal/compiler/metal_kernel_artifacts.hpp"
#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "common/gpu_device_profile.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/lowering_planner.hpp"
#include "compiler/manifest.hpp"
#include "compiler/operation_legalizer.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/opencl_kernels/conv2d_kernel.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

using compiler::ExecutableBundle;
using compiler::KernelArtifactDescriptor;
using compiler::KernelArtifactOrigin;
using compiler::KernelArtifactPayloadKind;
using compiler::LoweringPlan;
using compiler::LoweringRouteKind;
using compiler::ManifestBundle;

std::shared_ptr<ov::op::v0::Parameter> conv_param(const ov::element::Type &type,
                                                  ov::PartialShape shape) {
  return std::make_shared<ov::op::v0::Parameter>(type, std::move(shape));
}

std::shared_ptr<ov::op::v0::Constant> f32_const(ov::Shape shape) {
  std::vector<float> values(ov::shape_size(shape), 0.125f);
  return ov::op::v0::Constant::create(ov::element::f32, std::move(shape),
                                      values);
}

std::shared_ptr<ov::Node> conv2d_node() {
  auto data = conv_param(ov::element::f32, ov::Shape{1, 3, 8, 8});
  auto weights = f32_const(ov::Shape{4, 3, 3, 3});
  return std::make_shared<ov::op::v1::Convolution>(
      data, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
}

std::shared_ptr<ov::Node> conv2d_non_const_weights_node() {
  auto data = conv_param(ov::element::f32, ov::Shape{1, 3, 8, 8});
  auto weights = conv_param(ov::element::f32, ov::Shape{4, 3, 3, 3});
  return std::make_shared<ov::op::v1::Convolution>(
      data, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
}

std::shared_ptr<ov::Node> group_conv2d_node() {
  auto data = conv_param(ov::element::f32, ov::Shape{1, 4, 8, 8});
  auto weights = f32_const(ov::Shape{2, 2, 2, 3, 3});
  return std::make_shared<ov::op::v1::GroupConvolution>(
      data, weights, ov::Strides{1, 1}, ov::CoordinateDiff{1, 1},
      ov::CoordinateDiff{1, 1}, ov::Strides{1, 1});
}

std::shared_ptr<ov::Model>
model_from_node(const std::shared_ptr<ov::Node> &node) {
  ov::ParameterVector params;
  for (size_t i = 0; i < node->get_input_size(); ++i) {
    if (auto parameter = ov::as_type_ptr<ov::op::v0::Parameter>(
            node->input_value(i).get_node_shared_ptr())) {
      params.push_back(parameter);
    }
  }
  return std::make_shared<ov::Model>(
      ov::ResultVector{std::make_shared<ov::op::v0::Result>(node)},
      std::move(params));
}

struct ConvRouteCase {
  std::string name;
  std::function<std::shared_ptr<ov::Node>()> make_node;
  std::string expected_kernel_id;
  std::string expected_entry_point;
  std::string expected_op_family;
};

struct ConvCompiledContract {
  LoweringPlan plan;
  ManifestBundle manifest;
  ExecutableBundle executable;
};

class ConvRouteContract final {
public:
  explicit ConvRouteContract(ConvRouteCase route) : m_route(std::move(route)) {}

  ConvCompiledContract compile() const {
    const auto target =
        compiler::BackendTarget::from_backend(GpuBackend::Metal);
    const compiler::BackendCapabilities capabilities(
        target, compiler::make_metal_operation_support_policy());
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(
        target, compiler::make_metal_kernel_registry(target));
    ConvCompiledContract compiled;
    compiled.plan =
        planner.plan(model_from_node(m_route.make_node()), legalizer);
    compiled.manifest = compiler::ManifestBuilder{}.build(compiled.plan);
    compiled.executable =
        compiler::ExecutableBundleBuilder(
            compiler::make_metal_kernel_artifact_descriptor_resolver(),
            compiler::make_metal_kernel_artifact_payload_resolver())
            .build(compiled.manifest, compiled.plan);
    return compiled;
  }

  void verify(const ConvCompiledContract &compiled) const {
    ASSERT_TRUE(compiled.plan.executable());
    EXPECT_EQ(compiled.plan.route_count(LoweringRouteKind::VendorPrimitive),
              1u);
    ASSERT_TRUE(compiled.manifest.verify().valid());
    ASSERT_TRUE(compiled.executable.verify().valid());

    const auto &stage = find_conv_stage(compiled.manifest);
    EXPECT_EQ(stage.execution_kind, LoweringRouteKind::VendorPrimitive);
    EXPECT_EQ(stage.backend_domain, "metal");
    EXPECT_EQ(stage.kernel_unit_id, m_route.expected_kernel_id);
    EXPECT_EQ(stage.normalized_op_family, m_route.expected_op_family);
    EXPECT_FALSE(stage.memory.hidden_host_copy_allowed);

    const auto &artifact = find_conv_artifact(compiled.executable);
    EXPECT_EQ(artifact.kernel.origin, KernelArtifactOrigin::VendorPrimitive);
    EXPECT_EQ(artifact.payload_kind,
              KernelArtifactPayloadKind::VendorDescriptor);
    EXPECT_EQ(artifact.kernel.kernel_id, m_route.expected_kernel_id);
    EXPECT_EQ(artifact.entry_point, m_route.expected_entry_point);
    EXPECT_EQ(artifact.abi_arg_count, 3u);
    EXPECT_EQ(artifact.abi_output_arg_count, 1u);

    const auto payload =
        compiled.executable.find_artifact_payload(artifact.artifact_key);
    ASSERT_TRUE(payload);
    EXPECT_EQ(payload->payload_kind(),
              KernelArtifactPayloadKind::VendorDescriptor);
    EXPECT_EQ(payload->source_id(), m_route.expected_kernel_id);
    EXPECT_EQ(payload->entry_point(), m_route.expected_entry_point);

    const auto *vendor_payload =
        dynamic_cast<const compiler::GfxMetalVendorPrimitiveArtifactPayload *>(
            payload.get());
    ASSERT_NE(vendor_payload, nullptr);
    const auto &contract = vendor_payload->contract();
    ASSERT_TRUE(contract.valid);
    EXPECT_EQ(contract.descriptor.kind, GfxAppleMpsVendorPrimitiveKind::Conv2D);
    ASSERT_EQ(contract.input_descs.size(), 2u);
    ASSERT_EQ(contract.output_descs.size(), 1u);
    EXPECT_EQ(contract.external_buffer_abi.buffer_count, 3u);
    EXPECT_EQ(contract.external_buffer_abi.output_buffer_count, 1u);
  }

private:
  const compiler::StageRecord &
  find_conv_stage(const ManifestBundle &manifest) const {
    const auto it =
        std::find_if(manifest.stages.begin(), manifest.stages.end(),
                     [](const compiler::StageRecord &stage) {
                       return stage.normalized_op_family == "Convolution" ||
                              stage.normalized_op_family == "GroupConvolution";
                     });
    OPENVINO_ASSERT(it != manifest.stages.end(),
                    "Conv stage is missing from manifest");
    return *it;
  }

  const KernelArtifactDescriptor &
  find_conv_artifact(const ExecutableBundle &executable) const {
    const auto it = std::find_if(
        executable.artifact_descriptors.begin(),
        executable.artifact_descriptors.end(),
        [this](const KernelArtifactDescriptor &artifact) {
          return artifact.kernel.kernel_id == m_route.expected_kernel_id;
        });
    OPENVINO_ASSERT(it != executable.artifact_descriptors.end(),
                    "Conv artifact descriptor is missing");
    return *it;
  }

  ConvRouteCase m_route;
};

std::vector<ConvRouteCase> metal_conv_cases() {
  return {
      {"MetalMpsConv2DFirst", conv2d_node, "metal/vendor/mps_conv2d",
       "mps_conv2d", "Convolution"},
      {"MetalMpsGroupConv2DFirst", group_conv2d_node,
       "metal/vendor/mps_group_conv2d", "mps_group_conv2d", "GroupConvolution"},
  };
}

std::string
conv_route_case_name(const ::testing::TestParamInfo<ConvRouteCase> &info) {
  return info.param.name;
}

class ConvRouteContractTest : public ::testing::TestWithParam<ConvRouteCase> {};

TEST_P(ConvRouteContractTest, AppleCompilesThroughMpsVendorKernelUnit) {
  const ConvRouteContract contract(GetParam());
  contract.verify(contract.compile());
}

INSTANTIATE_TEST_SUITE_P(ConvolutionBackends, ConvRouteContractTest,
                         ::testing::ValuesIn(metal_conv_cases()),
                         conv_route_case_name);

struct OpenClConvRouteCase {
  std::string name;
  compiler::BackendTarget target;
  std::function<std::shared_ptr<ov::Node>()> make_node;
  std::string expected_kernel_id;
  std::string expected_entry_point;
  std::string expected_op_family;
  uint32_t expected_abi_arg_count = 0;
  uint32_t expected_scalar_arg_count = 0;
};

std::vector<OpenClConvRouteCase> opencl_conv_route_cases() {
  auto rpi4_v3d_target = compiler::BackendTarget::from_backend_profile(
      GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D,
      "raspberry-pi-4-v3d-opencl", "broadcom", "opencl-clvk");
  auto rpi5_v3d_target = compiler::BackendTarget::from_backend_profile(
      GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D,
      "raspberry-pi-5-v3d-opencl", "broadcom", "opencl-clvk");
  return {
      {"AndroidAdrenoConvUsesGeneratedKernelUnit",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno),
       conv2d_node, "opencl/generated/conv2d_f32",
       "gfx_opencl_generated_conv2d_f32", "Convolution", 19u, 16u},
      {"RaspberryPi4V3DConvUsesGeneratedKernelUnit", rpi4_v3d_target,
       conv2d_node, "opencl/generated/conv2d_f32",
       "gfx_opencl_generated_conv2d_f32", "Convolution", 19u, 16u},
      {"RaspberryPi5V3DConvUsesGeneratedKernelUnit", rpi5_v3d_target,
       conv2d_node, "opencl/generated/conv2d_f32",
       "gfx_opencl_generated_conv2d_f32", "Convolution", 19u, 16u},
      {"AndroidAdrenoGroupConvUsesGeneratedKernelUnit",
       compiler::BackendTarget::from_backend_device_family(
           GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno),
       group_conv2d_node, "opencl/generated/group_conv2d_f32",
       "gfx_opencl_generated_group_conv2d_f32", "GroupConvolution", 21u, 18u},
      {"RaspberryPi4V3DGroupConvUsesGeneratedKernelUnit", rpi4_v3d_target,
       group_conv2d_node, "opencl/generated/group_conv2d_f32",
       "gfx_opencl_generated_group_conv2d_f32", "GroupConvolution", 21u, 18u},
      {"RaspberryPi5V3DGroupConvUsesGeneratedKernelUnit", rpi5_v3d_target,
       group_conv2d_node, "opencl/generated/group_conv2d_f32",
       "gfx_opencl_generated_group_conv2d_f32", "GroupConvolution", 21u, 18u},
  };
}

std::string opencl_conv_route_case_name(
    const ::testing::TestParamInfo<OpenClConvRouteCase> &info) {
  return info.param.name;
}

class OpenClConvRouteContractTest
    : public ::testing::TestWithParam<OpenClConvRouteCase> {};

TEST_P(OpenClConvRouteContractTest,
       CompilesThroughGeneratedKernelUnitWithoutSourceArtifactFallback) {
  const auto &test_case = GetParam();
  const auto registry = compiler::make_opencl_kernel_registry(test_case.target);
  const compiler::BackendCapabilities capabilities(
      test_case.target,
      compiler::make_opencl_operation_support_policy(registry));
  const compiler::OperationLegalizer legalizer(capabilities);
  const compiler::LoweringPlanner planner(test_case.target, registry);

  const auto plan =
      planner.plan(model_from_node(test_case.make_node()), legalizer);
  ASSERT_TRUE(plan.executable());
  EXPECT_EQ(plan.route_count(LoweringRouteKind::GeneratedKernel), 1u);

  const auto planned_conv =
      std::find_if(plan.operations.begin(), plan.operations.end(),
                   [](const compiler::PlannedOperation &op) {
                     return op.type_name == "Convolution" ||
                            op.type_name == "GroupConvolution";
                   });
  ASSERT_NE(planned_conv, plan.operations.end());
  EXPECT_EQ(planned_conv->kernel_unit.id(), test_case.expected_kernel_id);

  const auto manifest = compiler::ManifestBuilder{}.build(plan);
  ASSERT_TRUE(manifest.verify().valid());
  const auto stage_it = std::find_if(
      manifest.stages.begin(), manifest.stages.end(),
      [&test_case](const compiler::StageRecord &stage) {
        return stage.kernel_unit_id == test_case.expected_kernel_id;
      });
  ASSERT_NE(stage_it, manifest.stages.end());
  const auto &stage = *stage_it;
  EXPECT_EQ(stage.execution_kind, LoweringRouteKind::GeneratedKernel);
  EXPECT_EQ(stage.backend_domain, "opencl");
  EXPECT_EQ(stage.kernel_unit_id, test_case.expected_kernel_id);
  EXPECT_EQ(stage.normalized_op_family, test_case.expected_op_family);
  EXPECT_FALSE(stage.memory.hidden_host_copy_allowed);

  const auto executable =
      compiler::ExecutableBundleBuilder(
          compiler::make_opencl_kernel_artifact_descriptor_resolver(),
          compiler::make_opencl_kernel_artifact_payload_resolver())
          .build(manifest, plan);
  ASSERT_TRUE(executable.verify().valid());
  const auto artifact_it = std::find_if(
      executable.artifact_descriptors.begin(),
      executable.artifact_descriptors.end(),
      [&test_case](const KernelArtifactDescriptor &artifact) {
        return artifact.kernel.kernel_id == test_case.expected_kernel_id;
      });
  ASSERT_NE(artifact_it, executable.artifact_descriptors.end());
  const auto &artifact = *artifact_it;
  EXPECT_EQ(artifact.kernel.origin, KernelArtifactOrigin::Generated);
  EXPECT_EQ(artifact.payload_kind, KernelArtifactPayloadKind::OpenClSource);
  EXPECT_EQ(artifact.kernel.kernel_id, test_case.expected_kernel_id);
  EXPECT_EQ(artifact.entry_point, test_case.expected_entry_point);
  EXPECT_EQ(artifact.abi_arg_count, test_case.expected_abi_arg_count);
  EXPECT_EQ(artifact.abi_output_arg_count, 1u);
  ASSERT_TRUE(artifact.launch_plan.valid);
  EXPECT_EQ(artifact.launch_plan.scalar_arg_kinds.size(),
            test_case.expected_scalar_arg_count);
  EXPECT_EQ(artifact.launch_plan.direct_input_indices, std::vector<size_t>{0u});
  EXPECT_EQ(std::count(artifact.launch_plan.buffer_roles.begin(),
                       artifact.launch_plan.buffer_roles.end(), "const_tensor"),
            1);

  const auto payload = executable.find_artifact_payload(artifact.artifact_key);
  ASSERT_TRUE(payload);
  EXPECT_EQ(payload->payload_kind(), KernelArtifactPayloadKind::OpenClSource);
  EXPECT_EQ(payload->backend_domain(), "opencl");
  EXPECT_EQ(payload->source_id(), test_case.expected_kernel_id);
  EXPECT_EQ(payload->entry_point(), test_case.expected_entry_point);

  const auto *opencl_payload =
      dynamic_cast<const GfxOpenClSourceArtifactPayload *>(payload.get());
  ASSERT_NE(opencl_payload, nullptr);
  const auto &source_artifact = opencl_payload->artifact();
  ASSERT_TRUE(source_artifact.valid);
  EXPECT_EQ(source_artifact.direct_input_indices, std::vector<size_t>{0u});
  EXPECT_EQ(source_artifact.direct_input_count, 1u);
  EXPECT_EQ(source_artifact.direct_output_count, 1u);
  EXPECT_EQ(source_artifact.scalar_args.size(),
            test_case.expected_scalar_arg_count);
  const auto *const_tensors =
      executable.find_artifact_const_tensors(artifact.artifact_key);
  ASSERT_NE(const_tensors, nullptr);
  ASSERT_EQ(const_tensors->size(), 1u);
  EXPECT_EQ((*const_tensors)[0].source_input_index, 1u);
  if (test_case.target.device_family() == "broadcom_v3d") {
    EXPECT_EQ(test_case.target.driver_id(), "opencl-clvk");
    EXPECT_EQ(test_case.target.compiler_id(), "gfx-mlir-opencl-clspv-v1");
  }
}

INSTANTIATE_TEST_SUITE_P(ConvolutionOpenClFamily, OpenClConvRouteContractTest,
                         ::testing::ValuesIn(opencl_conv_route_cases()),
                         opencl_conv_route_case_name);

TEST(OpenClConvUnsupportedContractTest,
     RejectsNonConstWeightsBeforeLegacySourceArtifactFallback) {
  const auto target = compiler::BackendTarget::from_backend_device_family(
      GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno);
  const auto registry = compiler::make_opencl_kernel_registry(target);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy(registry));
  const compiler::OperationLegalizer legalizer(capabilities);
  const compiler::LoweringPlanner planner(target, registry);
  const auto node = conv2d_non_const_weights_node();
  const auto support = legalizer.query(node);
  EXPECT_FALSE(support.semantic_legal);
  EXPECT_EQ(support.semantic_reason, "missing_opencl_convolution_kernel_unit");
  EXPECT_FALSE(make_opencl_conv2d_source_artifact(node));
  const auto plan = planner.plan(model_from_node(node), legalizer);
  EXPECT_FALSE(plan.executable());
}

TEST(ConvKernelRegistryContractTest, MetalRegistersMpsConvVendorUnits) {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
  const auto registry = compiler::make_metal_kernel_registry(target);

  const auto conv = registry.resolve(LoweringRouteKind::VendorPrimitive,
                                     "metal/vendor/mps_conv2d");
  ASSERT_TRUE(conv.valid());
  EXPECT_EQ(conv.kind(), compiler::KernelUnitKind::VendorPrimitive);
  EXPECT_EQ(conv.op_family(), "Convolution");

  const auto group_conv = registry.resolve(LoweringRouteKind::VendorPrimitive,
                                           "metal/vendor/mps_group_conv2d");
  ASSERT_TRUE(group_conv.valid());
  EXPECT_EQ(group_conv.kind(), compiler::KernelUnitKind::VendorPrimitive);
  EXPECT_EQ(group_conv.op_family(), "GroupConvolution");
  EXPECT_TRUE(registry.audit().valid());
}

TEST(ConvKernelRegistryContractTest,
     OpenClFamilyRegistersGeneratedConvUnitsForEveryProfile) {
  const std::vector<compiler::BackendTarget> targets = {
      compiler::BackendTarget::from_backend_device_family(
          GpuBackend::OpenCL, GpuDeviceFamily::QualcommAdreno),
      compiler::BackendTarget::from_backend_profile(
          GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D,
          "raspberry-pi-4-v3d-opencl", "broadcom", "opencl-clvk"),
      compiler::BackendTarget::from_backend_profile(
          GpuBackend::OpenCL, GpuDeviceFamily::BroadcomV3D,
          "raspberry-pi-5-v3d-opencl", "broadcom", "opencl-clvk"),
  };
  for (const auto &target : targets) {
    const auto registry = compiler::make_opencl_kernel_registry(target);
    EXPECT_TRUE(registry.audit().valid()) << target.debug_string();
    const auto conv = registry.resolve(LoweringRouteKind::GeneratedKernel,
                                       "opencl/generated/conv2d_f32");
    ASSERT_TRUE(conv.valid()) << target.debug_string();
    EXPECT_EQ(conv.kind(), compiler::KernelUnitKind::GeneratedKernel);
    EXPECT_EQ(conv.op_family(), "Convolution");
    const auto group_conv =
        registry.resolve(LoweringRouteKind::GeneratedKernel,
                         "opencl/generated/group_conv2d_f32");
    ASSERT_TRUE(group_conv.valid()) << target.debug_string();
    EXPECT_EQ(group_conv.kind(), compiler::KernelUnitKind::GeneratedKernel);
    EXPECT_EQ(group_conv.op_family(), "GroupConvolution");
    if (target.device_family() == "broadcom_v3d") {
      EXPECT_EQ(target.driver_id(), "opencl-clvk");
      EXPECT_EQ(target.compiler_id(), "gfx-mlir-opencl-clspv-v1");
    }
  }
}

} // namespace
} // namespace gfx_plugin
} // namespace ov
