// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/opencl_kernels/pool2d_kernel.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "kernel_ir/opencl_kernels/pool2d_f16_kernel.hpp"
#include "kernel_ir/opencl_kernels/pool2d_f32_kernel.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/max_pool.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_f32_pool_type(const ov::element::Type &type) {
  return type == ov::element::f32;
}

bool is_f16_pool_type(const ov::element::Type &type) {
  return type == ov::element::f16;
}

const char *pool_type_suffix(const ov::element::Type &type) {
  return is_f16_pool_type(type) ? "f16" : "f32";
}

bool checked_u32(uint64_t value, uint32_t &out) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out = static_cast<uint32_t>(value);
  return true;
}

bool requested_unit_matches(std::string_view requested,
                            std::string_view actual) noexcept {
  return requested.empty() || requested == actual;
}

std::vector<GfxOpenClSourceScalarArg> pool2d_static_scalar_args() {
  std::vector<GfxOpenClSourceScalarArg> scalar_args = {
      GfxOpenClSourceScalarArg::ElementCount};
  scalar_args.insert(scalar_args.end(), 18,
                     GfxOpenClSourceScalarArg::StaticU32);
  return scalar_args;
}

std::optional<std::vector<uint32_t>>
pool2d_static_u32_scalars(const std::shared_ptr<const ov::Node> &node,
                          GfxOpenClArtifactOp &op) {
  if (!node || node->get_input_size() != 1 || node->get_output_size() != 1 ||
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static()) {
    return std::nullopt;
  }

  const auto element_type = node->get_input_element_type(0);
  if (element_type != node->get_output_element_type(0) ||
      (!is_f32_pool_type(element_type) && !is_f16_pool_type(element_type))) {
    return std::nullopt;
  }

  const auto &input_shape = node->get_input_shape(0);
  const auto &output_shape = node->get_output_shape(0);
  if (input_shape.size() != 4 || output_shape.size() != 4 ||
      ov::shape_size(input_shape) == 0 || ov::shape_size(output_shape) == 0) {
    return std::nullopt;
  }

  ov::Shape kernel;
  ov::Strides strides;
  ov::Strides dilations;
  ov::Shape pads_begin;
  ov::Shape pads_end;
  bool is_avg = false;
  bool exclude_pad = true;

  if (auto maxpool = ov::as_type_ptr<const ov::op::util::MaxPoolBase>(node)) {
    op = GfxOpenClArtifactOp::MaxPool;
    kernel = maxpool->get_kernel();
    strides = maxpool->get_strides();
    dilations = ov::Strides(kernel.size(), 1);
    if (auto p = ov::as_type_ptr<const ov::op::v8::MaxPool>(node)) {
      dilations = p->get_dilations();
    } else if (auto p = ov::as_type_ptr<const ov::op::v14::MaxPool>(node)) {
      dilations = p->get_dilations();
    }
    pads_begin = maxpool->get_pads_begin();
    pads_end = maxpool->get_pads_end();
  } else if (auto avgpool =
                 ov::as_type_ptr<const ov::op::util::AvgPoolBase>(node)) {
    op = GfxOpenClArtifactOp::AvgPool;
    is_avg = true;
    kernel = avgpool->get_kernel();
    strides = avgpool->get_strides();
    dilations = ov::Strides(kernel.size(), 1);
    if (auto p = ov::as_type_ptr<const ov::op::v16::AvgPool>(node)) {
      dilations = p->get_dilations();
    }
    pads_begin = avgpool->get_pads_begin();
    pads_end = avgpool->get_pads_end();
    exclude_pad = avgpool->get_exclude_pad();
  } else {
    return std::nullopt;
  }

  if (kernel.size() != 2 || strides.size() != 2 || dilations.size() != 2 ||
      pads_begin.size() != 2 || pads_end.size() != 2 || kernel[0] == 0 ||
      kernel[1] == 0 || strides[0] == 0 || strides[1] == 0 ||
      dilations[0] == 0 || dilations[1] == 0) {
    return std::nullopt;
  }

  std::vector<uint32_t> scalars;
  scalars.reserve(18);
  for (const auto value :
       {input_shape[0], input_shape[1], input_shape[2], input_shape[3],
        kernel[0], kernel[1], strides[0], strides[1], dilations[0],
        dilations[1], pads_begin[0], pads_begin[1], pads_end[0], pads_end[1],
        output_shape[2], output_shape[3]}) {
    uint32_t scalar = 0;
    if (!checked_u32(value, scalar)) {
      return std::nullopt;
    }
    scalars.push_back(scalar);
  }
  scalars.push_back(is_avg ? 1u : 0u);
  scalars.push_back(exclude_pad ? 1u : 0u);
  return scalars;
}

const GfxKernelSource *
pool2d_kernel_source(const ov::element::Type &type) noexcept {
  if (is_f32_pool_type(type)) {
    return &opencl_generated_pool2d_f32_kernel_source();
  }
  if (is_f16_pool_type(type)) {
    return &opencl_generated_pool2d_f16_kernel_source();
  }
  return nullptr;
}

GfxKernelStageManifest
make_opencl_pool2d_manifest(std::string specialization_key,
                            std::string entry_point,
                            uint32_t scalar_arg_count) {
  GfxKernelExternalBufferAbiSpec abi{};
  abi.valid = true;
  abi.roles.push_back(GfxKernelBufferRole::TensorInput);
  abi.roles.push_back(GfxKernelBufferRole::TensorOutput);
  abi.roles.insert(abi.roles.end(), scalar_arg_count,
                   GfxKernelBufferRole::ScalarParam);

  constexpr GfxKernelFamily kFamily = GfxKernelFamily::Pool2DWindow;
  auto custom = make_gfx_custom_kernel_manifest(
      gfx_kernel_family_name(kFamily), gfx_kernel_family_abi_id(kFamily),
      std::move(entry_point), std::move(abi),
      make_gfx_kernel_linear_dispatch_policy(
          /*threads_per_threadgroup=*/64,
          /*precompiled_binary_required=*/false));
  return make_gfx_custom_kernel_stage_manifest(
      GfxKernelStageFamily::Pooling, GfxKernelBackendDomain::OpenCl,
      GfxKernelStorageKind::Buffer, std::move(specialization_key),
      std::move(custom));
}

GfxOpenClSourceArtifact
make_pool2d_artifact(const GfxKernelSource &source,
                     std::string specialization_key, GfxOpenClArtifactOp op,
                     std::vector<uint32_t> static_u32_scalars) {
  GfxOpenClSourceArtifact artifact{};
  artifact.scalar_args = pool2d_static_scalar_args();
  artifact.stage_manifest = make_opencl_pool2d_manifest(
      std::move(specialization_key), source.entry_point,
      static_cast<uint32_t>(artifact.scalar_args.size()));
  artifact.valid = artifact.stage_manifest.valid;
  artifact.artifact_ref = make_gfx_kernel_artifact_ref(artifact.stage_manifest);
  artifact.artifact_ref.source_id = source.kernel_id;
  artifact.artifact_ref.entry_point = source.entry_point;
  artifact.source = source.source;
  artifact.static_u32_scalars = std::move(static_u32_scalars);
  artifact.direct_input_indices = {0};
  artifact.element_count_source = GfxOpenClSourceElementCountSource::Output0;
  artifact.op = op;
  artifact.input_mode = GfxOpenClArtifactInputMode::Direct;
  artifact.arg_count = static_cast<uint32_t>(
      materialize_gfx_kernel_external_buffer_roles(
          artifact.stage_manifest.custom_kernel.external_buffer_abi)
          .size());
  artifact.direct_input_count = 1;
  artifact.direct_output_count = 1;
  artifact.valid =
      artifact.valid && artifact.artifact_ref.valid &&
      artifact.artifact_ref.kind == GfxKernelArtifactKind::OpenClSource &&
      !artifact.source.empty();
  return artifact;
}

} // namespace

std::optional<GfxOpenClSourceArtifact>
make_opencl_pool2d_source_artifact(const std::shared_ptr<const ov::Node> &node,
                                   std::string_view requested_kernel_unit_id) {
  if (!node || node->get_input_size() != 1 || node->get_output_size() != 1) {
    return std::nullopt;
  }
  const auto element_type = node->get_output_element_type(0);
  const auto *source = pool2d_kernel_source(element_type);
  if (!source ||
      !requested_unit_matches(requested_kernel_unit_id, source->kernel_id)) {
    return std::nullopt;
  }

  GfxOpenClArtifactOp op = GfxOpenClArtifactOp::Identity;
  auto static_u32_scalars = pool2d_static_u32_scalars(node, op);
  if (!static_u32_scalars || op == GfxOpenClArtifactOp::Identity) {
    return std::nullopt;
  }

  return make_pool2d_artifact(
      *source,
      "opencl:generated:" + std::string(node->get_type_name()) + ":" +
          pool_type_suffix(element_type),
      op, std::move(*static_u32_scalars));
}

} // namespace gfx_plugin
} // namespace ov
