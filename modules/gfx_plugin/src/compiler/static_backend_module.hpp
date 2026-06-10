// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "compiler/backend_registry.hpp"
#include "transforms/pipeline.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

struct StaticBackendModuleConfig {
  BackendTarget target;
  std::shared_ptr<const OperationSupportPolicy> operation_policy;
  KernelRegistry kernel_registry;
  transforms::PipelineOptions pipeline_options = {};
  FusionCapabilities fusion_capabilities = {};
  PostOpFusionCapabilities post_op_fusion_capabilities = {};
  std::shared_ptr<const StagePlacementPolicy> stage_placement_policy = {};
  BackendExecutionCapabilities execution_capabilities = {};
  PrecisionCapabilities precision_capabilities = {};
  ArtifactFormatCapabilities artifact_format_capabilities = {};
  KernelArtifactDescriptorResolver artifact_descriptor_resolver = {};
  KernelArtifactPayloadResolver artifact_payload_resolver = {};
  CacheBackendPayloadEncoder cache_payload_encoder = {};
  CacheBackendPayloadDecoder cache_payload_decoder = {};
  PipelineVendorAttentionArtifactResolver vendor_attention_artifact_resolver =
      {};
};

std::shared_ptr<const BackendModule>
make_static_backend_module(StaticBackendModuleConfig config);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
