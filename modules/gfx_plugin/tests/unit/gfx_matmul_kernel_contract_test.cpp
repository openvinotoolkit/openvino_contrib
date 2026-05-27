// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backends/metal/compiler/metal_kernel_artifacts.hpp"
#include "backends/metal/compiler/metal_operation_support.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "compiler/executable_bundle.hpp"
#include "compiler/kernel_registry.hpp"
#include "compiler/manifest.hpp"
#include "compiler/operation_legalizer.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "kernel_ir/opencl_kernels/matmul_f32_kernel.hpp"
#include "mlir/gfx_apple_vendor_descriptors.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "runtime/executable_descriptor.hpp"

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

struct MatMulRouteCase {
    std::string name;
    compiler::BackendTarget target;
    std::shared_ptr<const compiler::OperationSupportPolicy> policy;
    compiler::KernelRegistry kernel_registry;
    compiler::KernelArtifactPayloadResolver payload_resolver;
    LoweringRouteKind expected_route = LoweringRouteKind::Unsupported;
    KernelArtifactOrigin expected_origin = KernelArtifactOrigin::Unknown;
    KernelArtifactPayloadKind expected_payload =
        KernelArtifactPayloadKind::None;
    std::string expected_kernel_id;
    std::string expected_entry_point;
    uint32_t expected_abi_arg_count = 0;
    uint32_t expected_abi_output_arg_count = 0;
};

struct MatMulCompiledContract {
    LoweringPlan plan;
    ManifestBundle manifest;
    ExecutableBundle executable;
};

class MatMulModelFactory final {
public:
    std::shared_ptr<ov::Model> f32_2d() const {
        return make_model(ov::element::f32);
    }

    std::shared_ptr<ov::Model> f16_2d() const {
        return make_model(ov::element::f16);
    }

private:
    std::shared_ptr<ov::Model> make_model(const ov::element::Type& type) const {
        auto lhs =
            std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{2, 3});
        auto rhs =
            std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{3, 4});
        auto matmul =
            std::make_shared<ov::op::v0::MatMul>(lhs, rhs, false, false);
        auto result = std::make_shared<ov::op::v0::Result>(matmul);
        return std::make_shared<ov::Model>(ov::ResultVector{result},
                                           ov::ParameterVector{lhs, rhs});
    }
};

class MatMulRouteContract final {
public:
    explicit MatMulRouteContract(MatMulRouteCase route)
        : m_route(std::move(route)) {}

    MatMulCompiledContract compile(
        const std::shared_ptr<const ov::Model>& model) const {
        const compiler::BackendCapabilities capabilities(m_route.target,
                                                         m_route.policy);
        const compiler::OperationLegalizer legalizer(capabilities);
        const compiler::LoweringPlanner planner(m_route.target,
                                                m_route.kernel_registry);

        MatMulCompiledContract compiled;
        compiled.plan = planner.plan(model, legalizer);
        compiled.manifest = compiler::ManifestBuilder{}.build(compiled.plan);
        compiled.executable =
            compiler::ExecutableBundleBuilder(m_route.payload_resolver)
                .build(compiled.manifest, compiled.plan);
        return compiled;
    }

    void verify(const MatMulCompiledContract& compiled) const {
        ASSERT_TRUE(compiled.plan.executable());
        EXPECT_EQ(compiled.plan.route_count(m_route.expected_route), 1u);
        ASSERT_TRUE(compiled.manifest.verify().valid());
        ASSERT_TRUE(compiled.executable.verify().valid());

        const auto& stage = find_matmul_stage(compiled.manifest);
        EXPECT_EQ(stage.execution_kind, m_route.expected_route);
        EXPECT_EQ(stage.dispatch.execution_kind, m_route.expected_route);
        EXPECT_EQ(stage.backend_domain, m_route.target.backend_id());
        EXPECT_EQ(stage.dispatch.backend_domain, stage.backend_domain);
        EXPECT_EQ(stage.kernel_unit_id, m_route.expected_kernel_id);
        EXPECT_EQ(stage.dispatch.kernel_unit_id, m_route.expected_kernel_id);
        EXPECT_FALSE(stage.memory.hidden_host_copy_allowed);

        const auto& artifact = find_matmul_artifact(compiled.executable);
        EXPECT_EQ(artifact.kernel.origin, m_route.expected_origin);
        EXPECT_EQ(artifact.payload_kind, m_route.expected_payload);
        EXPECT_EQ(artifact.kernel.kernel_id, m_route.expected_kernel_id);
        EXPECT_EQ(artifact.kernel.backend_domain, m_route.target.backend_id());
        EXPECT_EQ(artifact.entry_point, m_route.expected_entry_point);
        EXPECT_EQ(artifact.abi_arg_count, m_route.expected_abi_arg_count);
        EXPECT_EQ(artifact.abi_output_arg_count,
                  m_route.expected_abi_output_arg_count);
        EXPECT_FALSE(artifact.manifest_ref.empty());
        EXPECT_FALSE(artifact.abi_fingerprint.empty());
        EXPECT_FALSE(artifact.artifact_key.empty());

        const auto payload =
            compiled.executable.find_artifact_payload(artifact.artifact_key);
        ASSERT_TRUE(payload);
        EXPECT_EQ(payload->payload_kind(), m_route.expected_payload);
        EXPECT_EQ(payload->backend_domain(), m_route.target.backend_id());
        EXPECT_EQ(payload->source_id(), m_route.expected_kernel_id);
        EXPECT_EQ(payload->entry_point(), m_route.expected_entry_point);

        verify_payload(payload.get());
        verify_runtime_descriptor(compiled.executable, payload);
    }

private:
    const compiler::StageRecord& find_matmul_stage(
        const ManifestBundle& manifest) const {
        const auto it = std::find_if(
            manifest.stages.begin(),
            manifest.stages.end(),
            [](const compiler::StageRecord& stage) {
                return stage.normalized_op_family == "MatMul";
            });
        OPENVINO_ASSERT(it != manifest.stages.end(),
                        "MatMul stage is missing from manifest");
        return *it;
    }

    const KernelArtifactDescriptor& find_matmul_artifact(
        const ExecutableBundle& executable) const {
        const auto it = std::find_if(
            executable.artifact_descriptors.begin(),
            executable.artifact_descriptors.end(),
            [](const KernelArtifactDescriptor& artifact) {
                return artifact.kernel.op_family == "MatMul";
            });
        OPENVINO_ASSERT(it != executable.artifact_descriptors.end(),
                        "MatMul artifact descriptor is missing");
        return *it;
    }

    void verify_payload(const compiler::KernelArtifactPayload* payload) const {
        if (m_route.expected_payload == KernelArtifactPayloadKind::OpenClSource) {
            const auto* source_payload =
                dynamic_cast<const GfxOpenClSourceArtifactPayload*>(payload);
            ASSERT_NE(source_payload, nullptr);
            ASSERT_TRUE(source_payload->artifact().valid);
            EXPECT_EQ(source_payload->artifact().source,
                      opencl_generated_matmul_f32_kernel_source().source);
            EXPECT_NE(source_payload->artifact().source.find(
                          "gfx_opencl_generated_matmul_f32"),
                      std::string::npos);
            return;
        }

        if (m_route.expected_payload ==
            KernelArtifactPayloadKind::VendorDescriptor) {
            const auto* vendor_payload =
                dynamic_cast<const compiler::GfxMetalVendorPrimitiveArtifactPayload*>(
                    payload);
            ASSERT_NE(vendor_payload, nullptr);
            const auto& contract = vendor_payload->contract();
            ASSERT_TRUE(contract.valid);
            EXPECT_EQ(contract.descriptor.kind,
                      GfxAppleMpsVendorPrimitiveKind::Gemm);
            EXPECT_TRUE(contract.external_buffer_abi.valid);
            EXPECT_EQ(contract.external_buffer_abi.buffer_count,
                      m_route.expected_abi_arg_count);
            EXPECT_EQ(contract.external_buffer_abi.output_buffer_count,
                      m_route.expected_abi_output_arg_count);
            EXPECT_EQ(contract.input_descs.size(), 2u);
            EXPECT_EQ(contract.output_descs.size(), 1u);
        }
    }

    void verify_runtime_descriptor(
        const ExecutableBundle& executable,
        const std::shared_ptr<const compiler::KernelArtifactPayload>& payload)
        const {
        const auto runtime_descriptor =
            RuntimeExecutableDescriptorBuilder{}.build(executable);
        ASSERT_TRUE(runtime_descriptor.verify(executable).valid());
        const auto stage_it = std::find_if(
            runtime_descriptor.stages.begin(),
            runtime_descriptor.stages.end(),
            [](const RuntimeStageExecutableDescriptor& stage) {
                return stage.op_family == "MatMul";
            });
        ASSERT_NE(stage_it, runtime_descriptor.stages.end());
        EXPECT_EQ(stage_it->payload, payload);
        EXPECT_EQ(stage_it->origin, m_route.expected_origin);
        EXPECT_EQ(stage_it->payload_kind, m_route.expected_payload);
        EXPECT_EQ(stage_it->backend_domain, m_route.target.backend_id());
        EXPECT_EQ(stage_it->kernel_id, m_route.expected_kernel_id);
        EXPECT_EQ(stage_it->entry_point, m_route.expected_entry_point);
        EXPECT_EQ(stage_it->abi_arg_count, m_route.expected_abi_arg_count);
        EXPECT_EQ(stage_it->abi_output_arg_count,
                  m_route.expected_abi_output_arg_count);
    }

    MatMulRouteCase m_route;
};

MatMulRouteCase opencl_generated_matmul_case() {
    const auto target =
        compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    return MatMulRouteCase{
        "OpenClGeneratedF32",
        target,
        compiler::make_opencl_operation_support_policy(),
        compiler::make_opencl_kernel_registry(target),
        {},
        LoweringRouteKind::GeneratedKernel,
        KernelArtifactOrigin::Generated,
        KernelArtifactPayloadKind::OpenClSource,
        "opencl/generated/matmul_f32",
        "gfx_opencl_generated_matmul_f32",
        13u,
        1u};
}

MatMulRouteCase apple_mps_gemm_matmul_case() {
    const auto target = compiler::BackendTarget::from_backend(GpuBackend::Metal);
    return MatMulRouteCase{
        "AppleMpsGemmF32",
        target,
        compiler::make_metal_operation_support_policy(),
        compiler::make_metal_kernel_registry(target),
        compiler::make_metal_kernel_artifact_payload_resolver(),
        LoweringRouteKind::VendorPrimitive,
        KernelArtifactOrigin::VendorPrimitive,
        KernelArtifactPayloadKind::VendorDescriptor,
        "metal/vendor/mps_gemm",
        "mps_gemm",
        3u,
        1u};
}

std::string matmul_route_case_name(
    const ::testing::TestParamInfo<MatMulRouteCase>& info) {
    return info.param.name;
}

class MatMulRouteContractTest
    : public ::testing::TestWithParam<MatMulRouteCase> {
protected:
    MatMulModelFactory models;
};

TEST_P(MatMulRouteContractTest, CompilesThroughExpectedKernelUnit) {
    const MatMulRouteContract contract(GetParam());
    contract.verify(contract.compile(models.f32_2d()));
}

INSTANTIATE_TEST_SUITE_P(MatMulBackends,
                         MatMulRouteContractTest,
                         ::testing::Values(opencl_generated_matmul_case(),
                                           apple_mps_gemm_matmul_case()),
                         matmul_route_case_name);

TEST(MatMulUnsupportedContractTest, OpenClF16RejectsMissingKernelUnit) {
    const auto target =
        compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    const compiler::BackendCapabilities capabilities(
        target,
        compiler::make_opencl_operation_support_policy());
    const compiler::OperationLegalizer legalizer(capabilities);
    const compiler::LoweringPlanner planner(
        target,
        compiler::make_opencl_kernel_registry(target));

    const MatMulModelFactory models;
    const auto model = models.f16_2d();
    const auto ordered_ops = model->get_ordered_ops();
    const auto matmul_it =
        std::find_if(ordered_ops.begin(), ordered_ops.end(), [](const auto& op) {
            return op->get_type_name() == std::string("MatMul");
        });
    ASSERT_NE(matmul_it, ordered_ops.end());

    const auto support = capabilities.query_operation({*matmul_it});
    EXPECT_FALSE(support.semantic_legal);
    EXPECT_EQ(support.semantic_reason, "missing_opencl_matmul_kernel_unit");

    const auto plan = planner.plan(model, legalizer);
    EXPECT_FALSE(plan.executable());
    const auto unsupported_it = std::find_if(
        plan.unsupported.type_counts.begin(),
        plan.unsupported.type_counts.end(),
        [](const auto& item) { return item.first == "MatMul"; });
    ASSERT_NE(unsupported_it, plan.unsupported.type_counts.end());
    EXPECT_EQ(unsupported_it->second, 1u);
}

}  // namespace
}  // namespace gfx_plugin
}  // namespace ov
