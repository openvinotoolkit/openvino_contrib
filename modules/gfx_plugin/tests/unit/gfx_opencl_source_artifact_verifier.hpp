// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backends/opencl/compiler/opencl_kernel_artifacts.hpp"
#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "compiler/kernel_registry.hpp"
#include "unit/gfx_opencl_catalog_artifact_resolver.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"

namespace ov {
namespace gfx_plugin {
namespace test {

inline bool
opencl_compiler_supports_node(const std::shared_ptr<const ov::Node> &node) {
  const auto target = compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
  const compiler::BackendCapabilities capabilities(
      target, compiler::make_opencl_operation_support_policy());
  return capabilities.query_operation({node}).semantic_legal;
}

inline compiler::LoweringRouteKind
opencl_artifact_route_kind(const GfxOpenClSourceArtifact &artifact) noexcept {
  const auto origin = compiler::classify_opencl_kernel_artifact_origin(
      artifact.artifact_ref.source_id);
  switch (origin) {
  case KernelArtifactOrigin::Generated:
    return compiler::LoweringRouteKind::GeneratedKernel;
  case KernelArtifactOrigin::HandwrittenException:
    return compiler::LoweringRouteKind::HandwrittenKernelException;
  default:
    return compiler::LoweringRouteKind::Unsupported;
  }
}

inline compiler::KernelUnit
resolve_opencl_artifact_kernel_unit(const compiler::KernelRegistry &registry,
                                    const GfxOpenClSourceArtifact &artifact) {
  const auto route_kind = opencl_artifact_route_kind(artifact);
  if (route_kind == compiler::LoweringRouteKind::Unsupported) {
    return {};
  }
  return registry.resolve(route_kind, artifact.artifact_ref.source_id);
}

class OpenClSourceArtifactVerifier final {
public:
  explicit OpenClSourceArtifactVerifier(std::shared_ptr<const ov::Node> node)
      : m_node(std::move(node)) {}

  OpenClSourceArtifactVerifier &
  expect_artifact(GfxKernelStageFamily family, const std::string &source_id,
                  const std::string &entry_point, uint32_t arg_count,
                  uint32_t direct_input_count,
                  std::vector<GfxOpenClSourceScalarArg> scalar_args =
                      {GfxOpenClSourceScalarArg::ElementCount,
                       GfxOpenClSourceScalarArg::OpCode},
                  std::vector<size_t> direct_input_indices = {},
                  std::vector<uint32_t> static_u32_scalars = {},
                  uint32_t direct_output_count = 1) {
    if (direct_input_indices.empty() && direct_input_count != 0) {
      for (size_t i = 0; i < direct_input_count; ++i) {
        direct_input_indices.push_back(i);
      }
    }

    auto artifact =
        resolve_opencl_catalog_source_artifact(m_node, source_id);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_TRUE(artifact->valid);
    EXPECT_EQ(artifact->stage_manifest.stage_family, family);
    EXPECT_EQ(artifact->stage_manifest.backend_domain,
              GfxKernelBackendDomain::OpenCl);
    EXPECT_EQ(artifact->stage_manifest.execution_kind,
              GfxKernelExecutionKind::CustomKernel);
    EXPECT_EQ(artifact->stage_manifest.storage, GfxKernelStorageKind::Buffer);
    EXPECT_TRUE(artifact->stage_manifest.custom_kernel.valid);
    EXPECT_EQ(artifact->stage_manifest.custom_kernel.entry_point, entry_point);
    EXPECT_EQ(artifact->artifact_ref.kind, GfxKernelArtifactKind::OpenClSource);
    EXPECT_EQ(artifact->artifact_ref.backend_domain,
              GfxKernelBackendDomain::OpenCl);
    EXPECT_EQ(artifact->artifact_ref.source_id, source_id);
    EXPECT_EQ(artifact->artifact_ref.entry_point, entry_point);
    EXPECT_EQ(artifact->arg_count, arg_count);
    EXPECT_EQ(artifact->direct_input_count, direct_input_count);
    EXPECT_EQ(artifact->direct_output_count, direct_output_count);
    EXPECT_EQ(artifact->direct_input_indices, direct_input_indices);
    EXPECT_EQ(artifact->local_size_hint, 64u);
    EXPECT_EQ(artifact->scalar_args, scalar_args);
    EXPECT_EQ(artifact->static_u32_scalars, static_u32_scalars);

    const auto roles =
        artifact->stage_manifest.custom_kernel.external_buffer_abi.roles;
    if (roles.size() != arg_count) {
      ADD_FAILURE() << "unexpected ABI role count";
      return *this;
    }
    for (size_t i = 0; i < direct_input_count; ++i) {
      EXPECT_EQ(roles[i], GfxKernelBufferRole::TensorInput);
    }
    for (size_t i = 0; i < direct_output_count; ++i) {
      EXPECT_EQ(roles[direct_input_count + i],
                GfxKernelBufferRole::TensorOutput);
    }
    for (size_t i = direct_input_count + direct_output_count; i < roles.size();
         ++i) {
      EXPECT_EQ(roles[i], GfxKernelBufferRole::ScalarParam);
    }

    return *this;
  }

  OpenClSourceArtifactVerifier &has_op(GfxOpenClArtifactOp op) {
    const auto artifact = resolve_opencl_catalog_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_EQ(artifact->op, op);
    return *this;
  }

  OpenClSourceArtifactVerifier &
  has_input_mode(GfxOpenClArtifactInputMode mode) {
    const auto artifact = resolve_opencl_catalog_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_EQ(artifact->input_mode, mode);
    return *this;
  }

  OpenClSourceArtifactVerifier &has_scalar_constant(float expected) {
    const auto artifact = resolve_opencl_catalog_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_FLOAT_EQ(artifact->scalar_constant_f32, expected);
    return *this;
  }

  OpenClSourceArtifactVerifier &supports_opencl_compiler() {
    const auto artifact = resolve_opencl_catalog_source_artifact(m_node);
    if (!artifact || !artifact->valid) {
      ADD_FAILURE()
          << "OpenCL compiler support requires a valid source artifact";
      return *this;
    }

    const auto target =
        compiler::BackendTarget::from_backend(GpuBackend::OpenCL);
    const auto registry = compiler::make_opencl_kernel_registry(target);
    const auto kernel_unit =
        resolve_opencl_artifact_kernel_unit(registry, *artifact);
    if (!kernel_unit.valid()) {
      ADD_FAILURE()
          << "OpenCL compiler support requires registered KernelUnit: "
          << artifact->artifact_ref.source_id;
      return *this;
    }

    const compiler::BackendCapabilities capabilities(
        target, compiler::make_opencl_operation_support_policy(registry));
    const auto support = capabilities.query_operation({m_node});
    if (!support.semantic_legal) {
      ADD_FAILURE() << "OpenCL operation support rejected node: "
                    << support.semantic_reason;
      return *this;
    }

    EXPECT_EQ(support.preferred_route_kind, kernel_unit.route_kind());
    EXPECT_EQ(support.preferred_route, kernel_unit.id());
    EXPECT_EQ(kernel_unit.backend_domain(), "opencl");
    return *this;
  }

private:
  std::shared_ptr<const ov::Node> m_node;
};

} // namespace test
} // namespace gfx_plugin
} // namespace ov
