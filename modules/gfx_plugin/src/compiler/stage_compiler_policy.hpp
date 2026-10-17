// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <optional>
#include <string>

#include "common/gpu_backend.hpp"
#include "common/gpu_parallelism_profile.hpp"
#include "compiler/backend_target.hpp"
#include "compiler/operation_support.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

using StageParallelismProfile = GpuParallelismProfile;

struct StageCustomKernelDispatchPolicy {
  bool enabled = false;
  StageParallelismProfile profile{};
};

struct StageCompilerPolicy {
  BackendTarget target;
  GpuBackend backend = GpuBackend::Unknown;
  const StagePlacementPolicy *placement = nullptr;
  const PostOpFusionCapabilities *post_ops = nullptr;
  StageCustomKernelDispatchPolicy custom_kernel_dispatch{};
};

struct PrecisionSensitiveFusionQuery {
  std::string group_kind;
  std::string stage_type;
  std::shared_ptr<const ov::Node> primary_node;
  ov::element::Type element_type = ov::element::dynamic;
  bool has_bias = false;
  std::optional<ActivationKind> activation;
  bool has_input_activation = false;
  bool has_batchnorm = false;
  GfxStageRuntimeTraits traits{};
};

StageCompilerPolicy make_stage_compiler_policy_from_capabilities(
    const BackendCapabilities &capabilities);

StageCustomKernelDispatchPolicy
make_stage_custom_kernel_dispatch_policy(
    const BackendExecutionCapabilities &execution);

StageCompilerPolicy resolve_stage_compiler_policy(const BackendTarget &target);

bool allow_precision_sensitive_arithmetic_fusion(
    const StageCompilerPolicy &policy,
    const PrecisionSensitiveFusionQuery &query);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
