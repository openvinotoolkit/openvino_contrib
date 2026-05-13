// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mlir/msl_codegen_apple_msl_binding.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "mlir/codegen_common.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/convert.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<KernelSource> make_apple_metal_convert_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  auto convert = std::dynamic_pointer_cast<const ov::op::v0::Convert>(node);
  if (!convert) {
    return std::nullopt;
  }

  ConvertCodegenDesc desc{};
  desc.src_type = convert->get_input_element_type(0);
  desc.dst_type = convert->get_output_element_type(0);
  desc.element_type =
      desc.dst_type == ov::element::dynamic ? ov::element::f32 : desc.dst_type;
  source.entry_point = "convert_kernel";
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  require_apple_msl_generated_kernel_source_binding(
      source, "Convert", "convert_kernel",
      {static_cast<int32_t>(ov::shape_size(convert->get_input_shape(0)))});
  return source;
}

} // namespace gfx_plugin
} // namespace ov
