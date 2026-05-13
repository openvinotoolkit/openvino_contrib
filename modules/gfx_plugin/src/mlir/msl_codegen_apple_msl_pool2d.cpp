// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_binding.hpp"

#include <cstdint>
#include <memory>

#include "mlir/codegen_common.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/max_pool.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<KernelSource> make_apple_metal_pool2d_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }

  Pool2DCodegenDesc desc{};
  if (auto pool =
          std::dynamic_pointer_cast<const ov::op::util::MaxPoolBase>(node)) {
    const auto in = pool->get_input_shape(0);
    const auto out = pool->get_output_shape(0);
    ov::Strides dilations(pool->get_kernel().size(), 1);
    if (auto p = std::dynamic_pointer_cast<const ov::op::v8::MaxPool>(node)) {
      dilations = p->get_dilations();
    } else if (auto p = std::dynamic_pointer_cast<const ov::op::v14::MaxPool>(
                   node)) {
      dilations = p->get_dilations();
    }
    desc.element_type = pool->get_output_element_type(0);
    desc.N = static_cast<uint32_t>(in.at(0));
    desc.C = static_cast<uint32_t>(in.at(1));
    desc.H = static_cast<uint32_t>(in.at(2));
    desc.W = static_cast<uint32_t>(in.at(3));
    desc.kH = static_cast<uint32_t>(pool->get_kernel().at(0));
    desc.kW = static_cast<uint32_t>(pool->get_kernel().at(1));
    desc.strideH = static_cast<uint32_t>(pool->get_strides().at(0));
    desc.strideW = static_cast<uint32_t>(pool->get_strides().at(1));
    desc.dilationH = static_cast<uint32_t>(dilations.at(0));
    desc.dilationW = static_cast<uint32_t>(dilations.at(1));
    desc.padTop = static_cast<uint32_t>(pool->get_pads_begin().at(0));
    desc.padLeft = static_cast<uint32_t>(pool->get_pads_begin().at(1));
    desc.padBottom = static_cast<uint32_t>(pool->get_pads_end().at(0));
    desc.padRight = static_cast<uint32_t>(pool->get_pads_end().at(1));
    desc.outH = static_cast<uint32_t>(out.at(2));
    desc.outW = static_cast<uint32_t>(out.at(3));
    desc.is_avg = false;
    desc.exclude_pad = true;
  } else if (auto pool =
                 std::dynamic_pointer_cast<const ov::op::v1::AvgPool>(node)) {
    const auto in = pool->get_input_shape(0);
    const auto out = pool->get_output_shape(0);
    desc.element_type = pool->get_output_element_type(0);
    desc.N = static_cast<uint32_t>(in.at(0));
    desc.C = static_cast<uint32_t>(in.at(1));
    desc.H = static_cast<uint32_t>(in.at(2));
    desc.W = static_cast<uint32_t>(in.at(3));
    desc.kH = static_cast<uint32_t>(pool->get_kernel().at(0));
    desc.kW = static_cast<uint32_t>(pool->get_kernel().at(1));
    desc.strideH = static_cast<uint32_t>(pool->get_strides().at(0));
    desc.strideW = static_cast<uint32_t>(pool->get_strides().at(1));
    desc.padTop = static_cast<uint32_t>(pool->get_pads_begin().at(0));
    desc.padLeft = static_cast<uint32_t>(pool->get_pads_begin().at(1));
    desc.padBottom = static_cast<uint32_t>(pool->get_pads_end().at(0));
    desc.padRight = static_cast<uint32_t>(pool->get_pads_end().at(1));
    desc.outH = static_cast<uint32_t>(out.at(2));
    desc.outW = static_cast<uint32_t>(out.at(3));
    desc.is_avg = true;
    desc.exclude_pad = pool->get_exclude_pad();
  } else {
    return std::nullopt;
  }

  source.entry_point = "pool2d_kernel";
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  require_apple_msl_generated_kernel_source_binding(
      source, node->get_type_name(), "pool2d_kernel");
  return source;
}

} // namespace gfx_plugin
} // namespace ov
