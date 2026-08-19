// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/opencl/compiler/opencl_conv_kernel_unit.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/gfx_custom_kernel_families.hpp"
#include "kernel_ir/opencl_kernels/conv2d_kernel.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

bool checked_u32(uint64_t value, uint32_t &out) {
  if (value > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out = static_cast<uint32_t>(value);
  return true;
}

bool checked_non_negative_u32(int64_t value, uint32_t &out) {
  if (value < 0) {
    return false;
  }
  return checked_u32(static_cast<uint64_t>(value), out);
}

bool constant_input(const ov::Node &node, size_t input_idx) {
  return input_idx < node.get_input_size() &&
         static_cast<bool>(ov::as_type_ptr<const ov::op::v0::Constant>(
             node.input_value(input_idx).get_node_shared_ptr()));
}

bool static_input_rank(const ov::Node &node, size_t input_idx, size_t rank) {
  return input_idx < node.get_input_size() &&
         node.get_input_partial_shape(input_idx).is_static() &&
         node.get_input_shape(input_idx).size() == rank;
}

bool static_output_rank4(const ov::Node &node) {
  return node.get_output_size() == 1 &&
         node.get_output_partial_shape(0).is_static() &&
         node.get_output_shape(0).size() == 4;
}

bool f32_conv_io(const ov::Node &node) {
  return node.get_input_size() == 2 && node.get_output_size() == 1 &&
         node.get_input_element_type(0) == ov::element::f32 &&
         node.get_input_element_type(1) == ov::element::f32 &&
         node.get_output_element_type(0) == ov::element::f32;
}

std::vector<GfxOpenClSourceScalarArg> static_u32_scalar_args(size_t count) {
  std::vector<GfxOpenClSourceScalarArg> scalar_args;
  scalar_args.reserve(count + 1);
  scalar_args.push_back(GfxOpenClSourceScalarArg::ElementCount);
  scalar_args.insert(scalar_args.end(), count,
                     GfxOpenClSourceScalarArg::StaticU32);
  return scalar_args;
}

bool requested_unit_matches(std::string_view requested,
                            std::string_view actual) noexcept {
  return requested.empty() || requested == actual;
}

std::optional<std::vector<uint32_t>>
conv2d_static_u32_scalars(const ov::op::v1::Convolution &conv) {
  if (!f32_conv_io(conv) || !static_input_rank(conv, 0, 4) ||
      !static_input_rank(conv, 1, 4) || !static_output_rank4(conv) ||
      !constant_input(conv, 1)) {
    return std::nullopt;
  }
  const auto data_shape = conv.get_input_shape(0);
  const auto weights_shape = conv.get_input_shape(1);
  const auto output_shape = conv.get_output_shape(0);
  if (data_shape[1] != weights_shape[1] ||
      output_shape[1] != weights_shape[0] ||
      ov::shape_size(output_shape) == 0) {
    return std::nullopt;
  }

  std::vector<uint32_t> scalars;
  scalars.reserve(15);
  for (const auto value :
       {data_shape[0], data_shape[1], data_shape[2], data_shape[3],
        weights_shape[0], weights_shape[2], weights_shape[3], output_shape[2],
        output_shape[3]}) {
    uint32_t out = 0;
    if (!checked_u32(value, out)) {
      return std::nullopt;
    }
    scalars.push_back(out);
  }
  const auto strides = conv.get_strides();
  const auto dilations = conv.get_dilations();
  const auto pads_begin = conv.get_pads_begin();
  if (strides.size() != 2 || dilations.size() != 2 || pads_begin.size() != 2) {
    return std::nullopt;
  }
  for (const auto value :
       {strides[0], strides[1], dilations[0], dilations[1]}) {
    uint32_t out = 0;
    if (!checked_u32(value, out) || out == 0) {
      return std::nullopt;
    }
    scalars.push_back(out);
  }
  for (const auto value : {pads_begin[0], pads_begin[1]}) {
    uint32_t out = 0;
    if (!checked_non_negative_u32(value, out)) {
      return std::nullopt;
    }
    scalars.push_back(out);
  }
  return scalars;
}

std::optional<std::vector<uint32_t>>
group_conv2d_static_u32_scalars(const ov::op::v1::GroupConvolution &conv) {
  if (!f32_conv_io(conv) || !static_input_rank(conv, 0, 4) ||
      !static_input_rank(conv, 1, 5) || !static_output_rank4(conv) ||
      !constant_input(conv, 1)) {
    return std::nullopt;
  }
  const auto data_shape = conv.get_input_shape(0);
  const auto weights_shape = conv.get_input_shape(1);
  const auto output_shape = conv.get_output_shape(0);
  const auto groups = weights_shape[0];
  const auto output_channels_per_group = weights_shape[1];
  const auto input_channels_per_group = weights_shape[2];
  if (groups == 0 || output_channels_per_group == 0 ||
      input_channels_per_group == 0 ||
      data_shape[1] != groups * input_channels_per_group ||
      output_shape[1] != groups * output_channels_per_group ||
      ov::shape_size(output_shape) == 0) {
    return std::nullopt;
  }

  std::vector<uint32_t> scalars;
  scalars.reserve(17);
  for (const auto value :
       {data_shape[0], data_shape[1], data_shape[2], data_shape[3], groups,
        output_channels_per_group, input_channels_per_group, weights_shape[3],
        weights_shape[4], output_shape[2], output_shape[3]}) {
    uint32_t out = 0;
    if (!checked_u32(value, out)) {
      return std::nullopt;
    }
    scalars.push_back(out);
  }
  const auto strides = conv.get_strides();
  const auto dilations = conv.get_dilations();
  const auto pads_begin = conv.get_pads_begin();
  if (strides.size() != 2 || dilations.size() != 2 || pads_begin.size() != 2) {
    return std::nullopt;
  }
  for (const auto value :
       {strides[0], strides[1], dilations[0], dilations[1]}) {
    uint32_t out = 0;
    if (!checked_u32(value, out) || out == 0) {
      return std::nullopt;
    }
    scalars.push_back(out);
  }
  for (const auto value : {pads_begin[0], pads_begin[1]}) {
    uint32_t out = 0;
    if (!checked_non_negative_u32(value, out)) {
      return std::nullopt;
    }
    scalars.push_back(out);
  }
  return scalars;
}

GfxKernelStageManifest make_opencl_conv2d_manifest(
    GfxKernelStageFamily stage_family, std::string specialization_key,
    std::string entry_point, uint32_t scalar_arg_count) {
  GfxKernelExternalBufferAbiSpec abi{};
  abi.valid = true;
  abi.roles = {GfxKernelBufferRole::TensorInput,
               GfxKernelBufferRole::ConstTensor,
               GfxKernelBufferRole::TensorOutput};
  abi.roles.insert(abi.roles.end(), scalar_arg_count,
                   GfxKernelBufferRole::ScalarParam);

  constexpr GfxKernelFamily kFamily = GfxKernelFamily::Conv2DDirect;
  auto custom = make_gfx_custom_kernel_manifest(
      gfx_kernel_family_name(kFamily), gfx_kernel_family_abi_id(kFamily),
      std::move(entry_point), std::move(abi),
      make_gfx_kernel_linear_dispatch_policy(
          /*threads_per_threadgroup=*/64,
          /*precompiled_binary_required=*/false));
  return make_gfx_custom_kernel_stage_manifest(
      stage_family, GfxKernelBackendDomain::OpenCl,
      GfxKernelStorageKind::Buffer, std::move(specialization_key),
      std::move(custom));
}

GfxOpenClSourceArtifact make_conv2d_artifact(
    const GfxKernelSource &source, GfxKernelStageFamily stage_family,
    std::string specialization_key, std::vector<uint32_t> static_u32_scalars) {
  GfxOpenClSourceArtifact artifact{};
  artifact.scalar_args = static_u32_scalar_args(static_u32_scalars.size());
  artifact.stage_manifest = make_opencl_conv2d_manifest(
      stage_family, std::move(specialization_key), source.entry_point,
      static_cast<uint32_t>(artifact.scalar_args.size()));
  artifact.valid = artifact.stage_manifest.valid;
  artifact.artifact_ref = make_gfx_kernel_artifact_ref(artifact.stage_manifest);
  artifact.artifact_ref.source_id = source.kernel_id;
  artifact.artifact_ref.entry_point = source.entry_point;
  artifact.source = source.source;
  artifact.static_u32_scalars = std::move(static_u32_scalars);
  artifact.direct_input_indices = {0};
  artifact.element_count_source = GfxOpenClSourceElementCountSource::Output0;
  artifact.op = GfxOpenClArtifactOp::Identity;
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
make_opencl_conv2d_source_artifact(const std::shared_ptr<const ov::Node> &node,
                                   std::string_view requested_kernel_unit_id) {
  if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
    const auto &source = opencl_generated_conv2d_f32_kernel_source();
    if (!requested_unit_matches(requested_kernel_unit_id, source.kernel_id)) {
      return std::nullopt;
    }
    auto static_u32_scalars = conv2d_static_u32_scalars(*conv);
    if (!static_u32_scalars) {
      return std::nullopt;
    }
    return make_conv2d_artifact(source, GfxKernelStageFamily::Convolution,
                                "opencl:generated:Convolution:f32",
                                std::move(*static_u32_scalars));
  }

  if (auto group_conv =
          ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
    const auto &source = opencl_generated_group_conv2d_f32_kernel_source();
    if (!requested_unit_matches(requested_kernel_unit_id, source.kernel_id)) {
      return std::nullopt;
    }
    auto static_u32_scalars = group_conv2d_static_u32_scalars(*group_conv);
    if (!static_u32_scalars) {
      return std::nullopt;
    }
    return make_conv2d_artifact(source, GfxKernelStageFamily::GroupConvolution,
                                "opencl:generated:GroupConvolution:f32",
                                std::move(*static_u32_scalars));
  }

  return std::nullopt;
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
