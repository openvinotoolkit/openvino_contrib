// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/msl_codegen_apple_msl_binding.hpp"

#include <memory>

#include "mlir/codegen_common.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_common.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

bool requires_scalar_fp32_convolution_path(
    const std::shared_ptr<const ov::Node> &node) {
  return node && node->get_output_size() > 0 &&
         node->get_output_element_type(0) == ov::element::f32 &&
         ov::fp16_compression_is_disabled(node);
}

void apply_convolution_tiling_policy(
    const std::shared_ptr<const ov::Node> &node, Conv2DCodegenDesc &desc) {
  if (requires_scalar_fp32_convolution_path(node)) {
    desc.output_channels_per_thread = 1;
    desc.output_width_per_thread = 1;
    return;
  }

  desc.output_channels_per_thread = gfx_conv2d_output_channel_block(desc);
  desc.output_width_per_thread = gfx_conv2d_output_width_block(desc);
}

} // namespace

std::optional<KernelSource> make_apple_metal_convolution_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }

  auto set_desc = [&](auto &&desc, const char *entry) {
    source.entry_point = entry;
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
  };

  if (auto conv =
          std::dynamic_pointer_cast<const ov::op::v1::Convolution>(node)) {
    const auto in_shape = conv->get_input_shape(0);
    if (in_shape.size() == 5) {
      Conv3DCodegenDesc desc{};
      const auto weights_shape = conv->get_input_shape(1);
      desc.element_type = conv->get_output_element_type(0);
      desc.N = static_cast<uint32_t>(in_shape.at(0));
      desc.C_in = static_cast<uint32_t>(in_shape.at(1));
      desc.D = static_cast<uint32_t>(in_shape.at(2));
      desc.H = static_cast<uint32_t>(in_shape.at(3));
      desc.W = static_cast<uint32_t>(in_shape.at(4));
      desc.C_out = static_cast<uint32_t>(weights_shape.at(0));
      desc.kD = static_cast<uint32_t>(weights_shape.at(2));
      desc.kH = static_cast<uint32_t>(weights_shape.at(3));
      desc.kW = static_cast<uint32_t>(weights_shape.at(4));
      desc.strideD = static_cast<uint32_t>(conv->get_strides().at(0));
      desc.strideH = static_cast<uint32_t>(conv->get_strides().at(1));
      desc.strideW = static_cast<uint32_t>(conv->get_strides().at(2));
      desc.dilationD = static_cast<uint32_t>(conv->get_dilations().at(0));
      desc.dilationH = static_cast<uint32_t>(conv->get_dilations().at(1));
      desc.dilationW = static_cast<uint32_t>(conv->get_dilations().at(2));
      desc.padFront = static_cast<uint32_t>(conv->get_pads_begin().at(0));
      desc.padTop = static_cast<uint32_t>(conv->get_pads_begin().at(1));
      desc.padLeft = static_cast<uint32_t>(conv->get_pads_begin().at(2));
      desc.padBack = static_cast<uint32_t>(conv->get_pads_end().at(0));
      desc.padBottom = static_cast<uint32_t>(conv->get_pads_end().at(1));
      desc.padRight = static_cast<uint32_t>(conv->get_pads_end().at(2));
      const auto out_shape = conv->get_output_shape(0);
      desc.outD = static_cast<uint32_t>(out_shape.at(2));
      desc.outH = static_cast<uint32_t>(out_shape.at(3));
      desc.outW = static_cast<uint32_t>(out_shape.at(4));
      set_desc(desc, "conv3d_kernel");
      require_apple_msl_generated_kernel_source_binding(source, "Convolution",
                                                        "conv3d_kernel");
      return source;
    }

    Conv2DCodegenDesc desc{};
    const auto weights_shape = conv->get_input_shape(1);
    desc.element_type = conv->get_output_element_type(0);
    desc.input_type = conv->get_input_element_type(0);
    desc.weight_type = conv->get_input_element_type(1);
    desc.output_type = conv->get_output_element_type(0);
    desc.N = static_cast<uint32_t>(in_shape.at(0));
    desc.C_in = static_cast<uint32_t>(in_shape.at(1));
    desc.H = static_cast<uint32_t>(in_shape.at(2));
    desc.W = static_cast<uint32_t>(in_shape.at(3));
    desc.C_out = static_cast<uint32_t>(weights_shape.at(0));
    const uint32_t cin_pg = static_cast<uint32_t>(weights_shape.at(1));
    desc.groups = (cin_pg && desc.C_in % cin_pg == 0) ? desc.C_in / cin_pg : 1;
    desc.C_in_pg = cin_pg;
    desc.C_out_pg = desc.groups ? desc.C_out / desc.groups : desc.C_out;
    desc.kH = static_cast<uint32_t>(weights_shape.at(2));
    desc.kW = static_cast<uint32_t>(weights_shape.at(3));
    desc.strideH = static_cast<uint32_t>(conv->get_strides().at(0));
    desc.strideW = static_cast<uint32_t>(conv->get_strides().at(1));
    desc.dilationH = static_cast<uint32_t>(conv->get_dilations().at(0));
    desc.dilationW = static_cast<uint32_t>(conv->get_dilations().at(1));
    desc.padTop = static_cast<uint32_t>(conv->get_pads_begin().at(0));
    desc.padLeft = static_cast<uint32_t>(conv->get_pads_begin().at(1));
    desc.padBottom = static_cast<uint32_t>(conv->get_pads_end().at(0));
    desc.padRight = static_cast<uint32_t>(conv->get_pads_end().at(1));
    const auto out_shape = conv->get_output_shape(0);
    desc.outH = static_cast<uint32_t>(out_shape.at(2));
    desc.outW = static_cast<uint32_t>(out_shape.at(3));
    apply_convolution_tiling_policy(conv, desc);
    set_desc(desc, "conv2d_kernel");
    require_apple_msl_generated_kernel_source_binding(source, "Convolution",
                                                      "conv2d_kernel");
    return source;
  }

  if (auto group_conv =
          std::dynamic_pointer_cast<const ov::op::v1::GroupConvolution>(node)) {
    Conv2DCodegenDesc desc{};
    const auto in_shape = group_conv->get_input_shape(0);
    const auto weights_shape = group_conv->get_input_shape(1);
    desc.element_type = group_conv->get_output_element_type(0);
    desc.input_type = group_conv->get_input_element_type(0);
    desc.weight_type = group_conv->get_input_element_type(1);
    desc.output_type = group_conv->get_output_element_type(0);
    desc.N = static_cast<uint32_t>(in_shape.at(0));
    desc.C_in = static_cast<uint32_t>(in_shape.at(1));
    desc.H = static_cast<uint32_t>(in_shape.at(2));
    desc.W = static_cast<uint32_t>(in_shape.at(3));
    desc.groups = static_cast<uint32_t>(weights_shape.at(0));
    desc.C_out_pg = static_cast<uint32_t>(weights_shape.at(1));
    desc.C_in_pg = static_cast<uint32_t>(weights_shape.at(2));
    desc.C_out = desc.groups * desc.C_out_pg;
    desc.kH = static_cast<uint32_t>(weights_shape.at(3));
    desc.kW = static_cast<uint32_t>(weights_shape.at(4));
    desc.strideH = static_cast<uint32_t>(group_conv->get_strides().at(0));
    desc.strideW = static_cast<uint32_t>(group_conv->get_strides().at(1));
    desc.dilationH = static_cast<uint32_t>(group_conv->get_dilations().at(0));
    desc.dilationW = static_cast<uint32_t>(group_conv->get_dilations().at(1));
    desc.padTop = static_cast<uint32_t>(group_conv->get_pads_begin().at(0));
    desc.padLeft = static_cast<uint32_t>(group_conv->get_pads_begin().at(1));
    desc.padBottom = static_cast<uint32_t>(group_conv->get_pads_end().at(0));
    desc.padRight = static_cast<uint32_t>(group_conv->get_pads_end().at(1));
    set_desc(desc, "conv2d_kernel");
    require_apple_msl_generated_kernel_source_binding(
        source, "GroupConvolution", "conv2d_kernel");
    return source;
  }

  return std::nullopt;
}

} // namespace gfx_plugin
} // namespace ov
