// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mlir/msl_codegen_apple_msl_binding.hpp"

#include <memory>

#include "mlir/codegen_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/space_to_depth.hpp"
#include "openvino/op/transpose.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<KernelSource> make_apple_metal_layout_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }

  if (auto d2s =
          std::dynamic_pointer_cast<const ov::op::v0::DepthToSpace>(node)) {
    DepthToSpaceCodegenDesc desc{};
    const auto in = d2s->get_input_shape(0);
    const auto out = d2s->get_output_shape(0);
    desc.element_type = d2s->get_output_element_type(0);
    desc.N = in[0];
    desc.C = in[1];
    desc.H = in[2];
    desc.W = in[3];
    desc.block = static_cast<uint32_t>(d2s->get_block_size());
    desc.C_out = out[1];
    desc.H_out = out[2];
    desc.W_out = out[3];
    desc.mode = d2s->get_mode() ==
                        ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST
                    ? 0
                    : 1;
    desc.total = static_cast<uint32_t>(ov::shape_size(out));
    source.entry_point = "depth_to_space_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    return source;
  }

  if (auto s2d =
          std::dynamic_pointer_cast<const ov::op::v0::SpaceToDepth>(node)) {
    SpaceToDepthCodegenDesc desc{};
    const auto in = s2d->get_input_shape(0);
    const auto out = s2d->get_output_shape(0);
    desc.element_type = s2d->get_output_element_type(0);
    desc.N = in[0];
    desc.C = in[1];
    desc.H = in[2];
    desc.W = in[3];
    desc.block = static_cast<uint32_t>(s2d->get_block_size());
    desc.C_out = out[1];
    desc.H_out = out[2];
    desc.W_out = out[3];
    desc.mode = s2d->get_mode() ==
                        ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST
                    ? 0
                    : 1;
    desc.total = static_cast<uint32_t>(ov::shape_size(out));
    source.entry_point = "space_to_depth_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    return source;
  }

  if (auto transpose =
          std::dynamic_pointer_cast<const ov::op::v1::Transpose>(node)) {
    TransposeCodegenDesc desc{};
    desc.element_type = transpose->get_output_element_type(0);
    auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(
        transpose->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(perm_const, "Transpose perm must be constant");
    auto perm = perm_const->cast_vector<int64_t>();
    for (auto value : perm) {
      desc.perm.push_back(static_cast<uint32_t>(value));
    }
    source.entry_point = "transpose_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
      return generate_msl_from_mlir(module, desc);
    };
    require_apple_msl_generated_kernel_source_binding(source, "Transpose",
                                                      "transpose_kernel");
    return source;
  }

  return std::nullopt;
}

} // namespace gfx_plugin
} // namespace ov
