// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/msl_codegen_apple_msl_binding.hpp"

#include <cstdint>
#include <memory>

#include "mlir/codegen_common.hpp"
#include "openvino/op/topk.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<KernelSource> make_apple_metal_topk_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  auto topk = std::dynamic_pointer_cast<const ov::op::util::TopKBase>(node);
  if (!topk) {
    return std::nullopt;
  }

  TopKCodegenDesc desc{};
  const auto in = topk->get_input_shape(0);
  const int64_t axis_i64 = normalize_axis(topk->get_axis(), in.size(), "TopK");
  const size_t axis = static_cast<size_t>(axis_i64);
  desc.axis_len = static_cast<uint32_t>(in[axis]);
  desc.k = static_cast<uint32_t>(topk->get_k());
  uint32_t outer = 1;
  uint32_t inner = 1;
  for (size_t i = 0; i < axis; ++i) {
    outer *= static_cast<uint32_t>(in[i]);
  }
  for (size_t i = axis + 1; i < in.size(); ++i) {
    inner *= static_cast<uint32_t>(in[i]);
  }
  desc.outer = outer;
  desc.inner = inner;
  desc.mode_max = topk->get_mode() == ov::op::TopKMode::MAX;
  switch (topk->get_sort_type()) {
  case ov::op::TopKSortType::SORT_INDICES:
    desc.sort_type = TopKSortType::SortIndices;
    break;
  case ov::op::TopKSortType::NONE:
    desc.sort_type = TopKSortType::None;
    break;
  case ov::op::TopKSortType::SORT_VALUES:
  default:
    desc.sort_type = TopKSortType::SortValues;
    break;
  }
  desc.element_type = topk->get_output_element_type(0);
  desc.index_type = topk->get_output_element_type(1);
  source.entry_point = "topk_kernel";
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  return source;
}

} // namespace gfx_plugin
} // namespace ov
