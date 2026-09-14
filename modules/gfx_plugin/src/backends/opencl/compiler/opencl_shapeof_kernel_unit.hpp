// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <string_view>

#include "compiler/kernel_registry.hpp"
#include "compiler/kernel_unit.hpp"
#include "compiler/operation_support.hpp"
#include "kernel_ir/gfx_opencl_source_artifacts.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace gfx_plugin {

class KernelArtifactPayload;

std::optional<GfxOpenClSourceArtifact> make_opencl_shapeof_source_artifact(
    const std::shared_ptr<const ov::Node> &node,
    std::string_view requested_kernel_unit_id = {});

namespace compiler {

struct KernelArtifactDescriptor;
struct PlannedOperation;

bool is_opencl_shapeof_node(const std::shared_ptr<const ov::Node> &node);

KernelUnit
resolve_opencl_shapeof_kernel_unit(const std::shared_ptr<const ov::Node> &node,
                                   const KernelRegistry &registry);

OperationSupportResult
query_opencl_shapeof_operation(const std::shared_ptr<const ov::Node> &node,
                               const KernelRegistry &registry);

std::shared_ptr<const ::ov::gfx_plugin::KernelArtifactPayload>
build_opencl_shapeof_kernel_artifact_payload(
    const KernelArtifactDescriptor &descriptor, const PlannedOperation &op);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
