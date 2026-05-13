// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_binding.hpp"

#include <memory>

#include "mlir/codegen_common.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/softmax.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<KernelSource> make_apple_metal_softmax_kernel_source(
    KernelSource source, const std::shared_ptr<const ov::Node> &node,
    const std::optional<ov::Shape> &runtime_input_shape) {
  if (!node) {
    return std::nullopt;
  }

  int64_t axis = -1;
  bool log_softmax = false;
  if (auto sm1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
    axis = sm1->get_axis();
  } else if (auto sm8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
    axis = sm8->get_axis();
  } else if (auto ls = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(node)) {
    axis = ls->get_axis();
    log_softmax = true;
  } else {
    return std::nullopt;
  }

  const ov::Shape input_shape =
      runtime_input_shape && !runtime_input_shape->empty()
          ? *runtime_input_shape
          : node->get_input_shape(0);
  OPENVINO_ASSERT(!input_shape.empty(),
                  "GFX Metal Softmax: input tensor shape is unknown");

  SoftmaxCodegenDesc desc{};
  desc.element_type = node->get_output_element_type(0);
  const auto dims = compute_softmax_dims(input_shape, axis, "GFX Metal");
  desc.rows = static_cast<int64_t>(dims.rows);
  desc.cols = static_cast<int64_t>(dims.axis_len);
  desc.inner = static_cast<int64_t>(dims.inner);
  desc.log_softmax = log_softmax;
  source.entry_point = "softmax_kernel";
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  return source;
}

} // namespace gfx_plugin
} // namespace ov
