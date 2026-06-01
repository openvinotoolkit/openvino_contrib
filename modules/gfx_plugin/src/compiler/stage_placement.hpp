// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "runtime/gfx_backend_utils.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxStageRuntimeTraits {
  bool binary_chunked = false;
  bool unary_chunked = false;
  bool softmax_chunked = false;
  bool transpose_chunked = false;
  bool split_concat_chunked = false;
  bool convert_chunked = false;
  bool diagnostic_f32_vendor_image = false;
};

enum class GfxStageBackendDomain {
  Unknown,
  AppleMps,
  AppleMsl,
  OpenCl,
};

enum class GfxStageStorageKind {
  Unknown,
  Buffer,
  Image,
  Matrix,
  NDArray,
  Alias,
};

struct GfxStagePlacementPlan {
  GfxStageBackendDomain domain = GfxStageBackendDomain::Unknown;
  GfxStageStorageKind storage = GfxStageStorageKind::Unknown;
  bool uses_vendor_primitive = false;
  bool uses_custom_kernel = false;
  std::string specialization_key;
};

namespace compiler {

struct StagePlacementQuery {
  GpuBackend backend = GpuBackend::Unknown;
  std::string stage_type;
  std::shared_ptr<const ov::Node> node;
  ov::element::Type element_type = ov::element::dynamic;
  GfxStageRuntimeTraits traits{};
};

class StagePlacementPolicy {
public:
  virtual ~StagePlacementPolicy() = default;

  virtual GfxStagePlacementPlan
  select_placement(const StagePlacementQuery &query) const = 0;
};

std::string make_stage_placement_key(GfxStageBackendDomain domain,
                                     GfxStageStorageKind storage,
                                     std::string_view stage_type);

GfxStagePlacementPlan make_stage_placement(GfxStageBackendDomain domain,
                                           GfxStageStorageKind storage,
                                           std::string_view stage_type,
                                           bool vendor_primitive,
                                           bool custom_kernel);

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
