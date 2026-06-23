// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/msl_codegen_apple_msl_binding.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>

#include "mlir/codegen_common.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace ov {
namespace gfx_plugin {

std::optional<KernelSource> make_apple_metal_llm_kernel_source(
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

  if (auto rms = std::dynamic_pointer_cast<const ov::op::internal::RMS>(node)) {
    const auto data_shape =
        static_shape_or_placeholder(rms->get_input_partial_shape(0));
    const auto gamma_shape =
        static_shape_or_placeholder(rms->get_input_partial_shape(1));
    OPENVINO_ASSERT(!data_shape.empty() && data_shape.back() > 0,
                    "GFX Metal RMS: hidden dimension must be static");
    RmsCodegenDesc desc{};
    desc.element_type = rms->get_output_element_type(0);
    desc.input_type = rms->get_input_element_type(0);
    desc.gamma_type = rms->get_input_element_type(1);
    desc.output_type = rms->get_output_element_type(0);
    desc.hidden = static_cast<uint32_t>(data_shape.back());
    desc.gamma_size = static_cast<uint32_t>(
        std::max<uint64_t>(1, ov::shape_size(gamma_shape)));
    desc.reduction_threads = gfx_rms_parallel_reduction_threads(desc.hidden);
    desc.epsilon = static_cast<float>(rms->get_epsilon());
    desc.has_residual_add =
        source.module && source.module->hasAttr("gfx.fused_residual_add");
    set_desc(desc, "rms_kernel");
    require_apple_msl_generated_kernel_source_binding(
        source, desc.has_residual_add ? "RMSResidual" : "RMS", "rms_kernel");
    return source;
  }

  if (auto rope =
          std::dynamic_pointer_cast<const ov::op::internal::RoPE>(node)) {
    const auto &cfg = rope->get_config();
    OPENVINO_ASSERT(!cfg.input_trans0213 && !cfg.output_trans0213,
                    "GFX Metal RoPE: transposed layouts are not supported yet");
    OPENVINO_ASSERT(
        !cfg.is_chatglm && !cfg.is_qwen,
        "GFX Metal RoPE: ChatGLM/Qwen-special layouts are not supported yet");
    OPENVINO_ASSERT(cfg.slice_start == 0 && cfg.slice_stop == 0,
                    "GFX Metal RoPE: sliced input layout is not supported yet");
    OPENVINO_ASSERT(
        rope->get_input_size() >= 3 && rope->get_input_size() <= 4,
        "GFX Metal RoPE: expected data, cos, sin and optional position inputs");
    OPENVINO_ASSERT(cfg.gather_position_arg_id == 0 ||
                        cfg.gather_position_arg_id == 3,
                    "GFX Metal RoPE: position gather must use input 3");
    const auto data_shape =
        static_shape_or_placeholder(rope->get_input_partial_shape(0));
    const auto cos_shape =
        static_shape_or_placeholder(rope->get_input_partial_shape(1));
    OPENVINO_ASSERT(data_shape.size() == 4 || data_shape.size() == 3,
                    "GFX Metal RoPE: expected rank-3 or rank-4 data tensor");
    OPENVINO_ASSERT(!data_shape.empty() && data_shape.back() > 0,
                    "GFX Metal RoPE: head size must be static");
    OPENVINO_ASSERT(cos_shape.size() >= 2 && cos_shape.size() <= 4,
                    "GFX Metal RoPE: expected rank-2/3/4 cos/sin tensors");
    RopeCodegenDesc desc{};
    desc.element_type = rope->get_output_element_type(0);
    desc.input_type = rope->get_input_element_type(0);
    desc.cos_type = rope->get_input_element_type(1);
    desc.sin_type = rope->get_input_element_type(2);
    desc.output_type = rope->get_output_element_type(0);
    desc.position_type = rope->get_input_size() > 3
                             ? rope->get_input_element_type(3)
                             : ov::element::dynamic;
    desc.rank = static_cast<uint32_t>(data_shape.size());
    desc.batch =
        static_cast<uint32_t>(data_shape.size() == 4 ? data_shape[0] : 1);
    desc.heads = static_cast<uint32_t>(data_shape.size() == 4 ? data_shape[1]
                                                              : data_shape[1]);
    desc.head_size = static_cast<uint32_t>(data_shape.back());
    desc.rotary_dims = static_cast<uint32_t>(cfg.rotary_ndims ? cfg.rotary_ndims
                                                              : desc.head_size);
    desc.cos_sin_dims = static_cast<uint32_t>(
        cfg.cos_sin_ndims ? cfg.cos_sin_ndims : desc.rotary_dims);
    desc.cos_rank = static_cast<uint32_t>(cos_shape.size());
    const auto cos_pshape = rope->get_input_partial_shape(1);
    auto mark_dynamic = [&](size_t logical_dim, size_t source_dim) {
      if (source_dim < static_cast<size_t>(cos_pshape.rank().get_length()) &&
          cos_pshape[source_dim].is_dynamic()) {
        desc.cos_dynamic_mask |= (1u << logical_dim);
      }
    };
    if (cos_shape.size() == 2) {
      desc.cos_dims = {{1, 1, static_cast<uint32_t>(cos_shape[0]),
                        static_cast<uint32_t>(cos_shape[1])}};
      mark_dynamic(2, 0);
      mark_dynamic(3, 1);
    } else if (cos_shape.size() == 3) {
      desc.cos_dims = {{1, static_cast<uint32_t>(cos_shape[0]),
                        static_cast<uint32_t>(cos_shape[1]),
                        static_cast<uint32_t>(cos_shape[2])}};
      mark_dynamic(1, 0);
      mark_dynamic(2, 1);
      mark_dynamic(3, 2);
    } else {
      desc.cos_dims = {{static_cast<uint32_t>(cos_shape[0]),
                        static_cast<uint32_t>(cos_shape[1]),
                        static_cast<uint32_t>(cos_shape[2]),
                        static_cast<uint32_t>(cos_shape[3])}};
      mark_dynamic(0, 0);
      mark_dynamic(1, 1);
      mark_dynamic(2, 2);
      mark_dynamic(3, 3);
    }
    desc.is_interleaved = cfg.is_interleaved;
    desc.input_trans0213 = cfg.input_trans0213;
    desc.output_trans0213 = cfg.output_trans0213;
    desc.has_position =
        cfg.gather_position_arg_id == 3 && rope->get_input_size() > 3;
    set_desc(desc, "rope_kernel");
    require_apple_msl_generated_kernel_source_binding(
        source, desc.has_position ? "RoPEWithPosition" : "RoPE", "rope_kernel");
    return source;
  }

  return std::nullopt;
}

} // namespace gfx_plugin
} // namespace ov
