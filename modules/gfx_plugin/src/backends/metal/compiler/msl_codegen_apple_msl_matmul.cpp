// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/msl_codegen_apple_msl_binding.hpp"

#include <cstdint>
#include <memory>

#include "mlir/codegen_common.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<KernelSource> make_apple_metal_matmul_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return std::nullopt;
  }

  auto matmul = std::dynamic_pointer_cast<const ov::op::v0::MatMul>(node);
  if (!matmul) {
    return std::nullopt;
  }

  MatMulCodegenDesc desc{};
  const auto out_shape = output_shape_for_codegen(source.module, node);
  const auto a_shape =
      static_shape_or_placeholder(matmul->get_input_partial_shape(0));
  const auto b_shape =
      static_shape_or_placeholder(matmul->get_input_partial_shape(1));
  const size_t a_rank = a_shape.size();
  const size_t b_rank = b_shape.size();
  const size_t out_rank = out_shape.size();
  OPENVINO_ASSERT(a_rank >= 2 && b_rank >= 2 && out_rank >= 2,
                  "GFX Metal MatMul: ranks must be at least 2");
  desc.element_type = matmul->get_output_element_type(0);
  desc.input_a_type = matmul->get_input_element_type(0);
  desc.input_b_type = matmul->get_input_element_type(1);
  desc.output_type = matmul->get_output_element_type(0);
  desc.a_transpose = matmul->get_transpose_a();
  desc.b_transpose = matmul->get_transpose_b();
  desc.M = static_cast<int64_t>(out_shape[out_rank - 2]);
  desc.N = static_cast<int64_t>(out_shape[out_rank - 1]);
  desc.K = static_cast<int64_t>(desc.a_transpose ? a_shape[a_rank - 2]
                                                 : a_shape[a_rank - 1]);
  desc.batch_a = static_cast<int64_t>(ov::shape_size(a_shape) /
                                      static_cast<uint64_t>(desc.M * desc.K));
  desc.batch_b = static_cast<int64_t>(ov::shape_size(b_shape) /
                                      static_cast<uint64_t>(desc.K * desc.N));
  desc.b_is_nk_layout = desc.b_transpose;
  desc.batch =
      static_cast<int64_t>(ov::shape_size(out_shape) / (desc.M * desc.N));
  source.entry_point = "matmul_kernel";
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  require_apple_msl_generated_kernel_source_binding(source, "MatMul",
                                                    "matmul_kernel");
  return source;
}

} // namespace gfx_plugin
} // namespace ov
