// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "backends/metal/compiler/msl_codegen_apple_msl_binding.hpp"

#include <memory>

#include "mlir/codegen_common.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_slice_static.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<KernelSource> make_apple_metal_slice_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const ov::element::Type &storage_type, bool has_runtime_slice_params) {
  if (!node || (!ov::is_type<const ov::op::v8::Slice>(node) &&
                !ov::is_type<const ov::op::v1::StridedSlice>(node))) {
    return std::nullopt;
  }

  const ov::element::Type effective_type =
      storage_type == ov::element::dynamic ? node->get_output_element_type(0)
                                           : storage_type;
  ConvertCodegenDesc desc{};
  desc.element_type = effective_type;
  desc.dst_type = effective_type;
  source.entry_point = "slice_kernel";
  const bool dynamic_slice_shape =
      !node->get_input_partial_shape(0).is_static() ||
      !node->get_output_partial_shape(0).is_static();
  if (!has_runtime_slice_params && !dynamic_slice_shape) {
    source.msl_source = generate_static_msl_for_slice(node, desc.dst_type);
    source.msl_generator = {};
    source.module = {};
    return source;
  }

  source.msl_source.clear();
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_for_slice_generic(desc, module);
  };
  return source;
}

} // namespace gfx_plugin
} // namespace ov
