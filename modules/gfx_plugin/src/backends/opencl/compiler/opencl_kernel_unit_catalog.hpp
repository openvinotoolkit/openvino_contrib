// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "common/artifact_payload.hpp"
#include "compiler/operation_support.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

struct KernelArtifactDescriptor;
class KernelRegistry;
struct PlannedOperation;

using OpenClNodePredicate =
    bool (*)(const std::shared_ptr<const ov::Node> &node);
using OpenClOperationSupportQueryFn = OperationSupportResult (*)(
    const std::shared_ptr<const ov::Node> &node,
    const KernelRegistry &kernel_registry);
using OpenClSourceArtifactFactory =
    std::optional<GfxOpenClSourceArtifact> (*)(
        const std::shared_ptr<const ov::Node> &node,
        std::string_view requested_kernel_unit_id);
using OpenClPayloadFactory =
    std::shared_ptr<const ::ov::gfx_plugin::KernelArtifactPayload> (*)(
        const KernelArtifactDescriptor &descriptor, const PlannedOperation &op);

struct OpenClGeneratedKernelUnitSpec {
  std::string_view kernel_unit_id;
  std::string_view op_family;
};

struct OpenClOperationSupportEntry {
  OpenClNodePredicate matches;
  OpenClOperationSupportQueryFn query_support;
  const char *unsupported_reason;
};

struct OpenClArtifactFamilyEntry {
  OpenClNodePredicate matches;
  OpenClSourceArtifactFactory make_source_artifact;
  OpenClPayloadFactory make_payload;
};

const std::vector<OpenClGeneratedKernelUnitSpec> &
opencl_generated_kernel_unit_specs();

const std::vector<OpenClOperationSupportEntry> &
opencl_operation_support_entries();

const std::vector<OpenClArtifactFamilyEntry> &
opencl_artifact_family_entries();

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
