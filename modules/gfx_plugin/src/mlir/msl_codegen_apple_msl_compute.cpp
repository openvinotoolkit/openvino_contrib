// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_msl_ops.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "mlir/msl_codegen.hpp"
#include "mlir/msl_codegen_apple_msl.hpp"
#include "mlir/msl_codegen_apple_msl_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/softmax.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

bool configure_apple_metal_compute_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return false;
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
      if (source.module) {
        require_apple_msl_custom_kernel_binding(source.module, "Convolution",
                                                "conv3d_kernel");
      }
      return true;
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
    desc.output_channels_per_thread = gfx_conv2d_output_channel_block(desc);
    desc.output_width_per_thread = gfx_conv2d_output_width_block(desc);
    set_desc(desc, "conv2d_kernel");
    if (source.module) {
      require_apple_msl_custom_kernel_binding(source.module, "Convolution",
                                              "conv2d_kernel");
    }
    return true;
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
    if (source.module) {
      require_apple_msl_custom_kernel_binding(source.module, "GroupConvolution",
                                              "conv2d_kernel");
    }
    return true;
  }

  if (auto matmul = std::dynamic_pointer_cast<const ov::op::v0::MatMul>(node)) {
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
    set_desc(desc, "matmul_kernel");
    if (source.module) {
      require_apple_msl_custom_kernel_binding(source.module, "MatMul",
                                              "matmul_kernel");
    }
    return true;
  }

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
    if (source.module) {
      require_apple_msl_custom_kernel_binding(
          source.module, desc.has_residual_add ? "RMSResidual" : "RMS",
          "rms_kernel");
    }
    return true;
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
    if (source.module) {
      require_apple_msl_custom_kernel_binding(
          source.module, desc.has_position ? "RoPEWithPosition" : "RoPE",
          "rope_kernel");
    }
    return true;
  }

  return false;
}

bool configure_apple_metal_softmax_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node,
    const std::optional<ov::Shape> &runtime_input_shape) {
  if (!node) {
    return false;
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
    return false;
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
  return true;
}

bool configure_apple_metal_pool2d_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return false;
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
    return false;
  }

  source.entry_point = "pool2d_kernel";
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  if (source.module) {
    require_apple_msl_custom_kernel_binding(
        source.module, node->get_type_name(), "pool2d_kernel");
  }
  return true;
}

bool configure_apple_metal_unary_kernel_source(
    KernelSource &source, const std::shared_ptr<const ov::Node> &node) {
  if (!node) {
    return false;
  }

  const auto activation = unary_activation_kind_from_node(*node);
  if (!activation) {
    return false;
  }

  UnaryCodegenDesc desc{};
  desc.element_type = node->get_output_element_type(0);
  desc.activation = *activation;
  desc.alpha = 0.0f;
  if (auto elu = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
    desc.alpha = static_cast<float>(elu->get_alpha());
  }
  if (auto clamp = ov::as_type_ptr<const ov::op::v0::Clamp>(node)) {
    desc.clamp_min = clamp->get_min();
    desc.clamp_max = clamp->get_max();
  }

  source.entry_point = "unary_kernel";
  source.msl_generator = [desc](mlir::ModuleOp module) mutable {
    return generate_msl_from_mlir(module, desc);
  };
  const auto out_shape = output_shape_for_codegen(source.module, node);
  const int32_t num_elements = static_cast<int32_t>(ov::shape_size(out_shape));
  if (source.module) {
    require_apple_msl_custom_kernel_binding(
        source.module, node->get_type_name(), "unary_kernel", {num_elements});
  }
  return true;
}

} // namespace gfx_plugin
} // namespace ov
