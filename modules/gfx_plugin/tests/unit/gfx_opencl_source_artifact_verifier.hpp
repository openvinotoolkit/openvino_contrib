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

#include "backends/opencl/compiler/opencl_operation_support.hpp"
#include "kernel_ir/gfx_kernel_source.hpp"
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

    auto artifact = resolve_gfx_opencl_source_artifact(m_node);
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
    EXPECT_EQ(artifact->baseline_local_size, 64u);
    EXPECT_EQ(artifact->scalar_args, scalar_args);
    EXPECT_EQ(artifact->static_u32_scalars, static_u32_scalars);
    EXPECT_NE(artifact->source.find("__kernel void " + entry_point),
              std::string::npos);

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

  OpenClSourceArtifactVerifier &
  excludes(const std::vector<std::string> &needles) {
    const auto artifact = resolve_gfx_opencl_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    for (const auto &needle : needles) {
      EXPECT_EQ(artifact->source.find(needle), std::string::npos) << needle;
    }
    return *this;
  }

  OpenClSourceArtifactVerifier &has_op(GfxOpenClBaselineOp op) {
    const auto artifact = resolve_gfx_opencl_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_EQ(artifact->op, op);
    return *this;
  }

  OpenClSourceArtifactVerifier &
  has_input_mode(GfxOpenClBaselineInputMode mode) {
    const auto artifact = resolve_gfx_opencl_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_EQ(artifact->input_mode, mode);
    return *this;
  }

  OpenClSourceArtifactVerifier &has_scalar_constant(float expected) {
    const auto artifact = resolve_gfx_opencl_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_FLOAT_EQ(artifact->scalar_constant_f32, expected);
    return *this;
  }

  OpenClSourceArtifactVerifier &supports_opencl_compiler() {
    EXPECT_TRUE(opencl_compiler_supports_node(m_node));
    return *this;
  }

  OpenClSourceArtifactVerifier &uses_source(const GfxKernelSource &source) {
    const auto artifact = resolve_gfx_opencl_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_EQ(artifact->artifact_ref.source_id, source.kernel_id);
    EXPECT_EQ(artifact->artifact_ref.entry_point, source.entry_point);
    EXPECT_EQ(artifact->source, source.source);
    return *this;
  }

  OpenClSourceArtifactVerifier &contains_source(const std::string &needle) {
    const auto artifact = resolve_gfx_opencl_source_artifact(m_node);
    if (!artifact.has_value()) {
      ADD_FAILURE() << "missing OpenCL source artifact";
      return *this;
    }
    EXPECT_NE(artifact->source.find(needle), std::string::npos);
    return *this;
  }

private:
  std::shared_ptr<const ov::Node> m_node;
};

} // namespace test
} // namespace gfx_plugin
} // namespace ov
