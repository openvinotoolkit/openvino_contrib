// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "backends/metal/common/mpsrt/gfx_mpsrt_abi.hpp"
#include "backends/metal/common/mpsrt/gfx_mpsrt_program.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "common/gfx_activation.hpp"
#include "common/gfx_bias.hpp"
#include "compiler/stage_policy.hpp"

namespace ov {
class Node;

namespace gfx_plugin {

enum class GfxAppleMpsVendorPrimitiveKind {
  None,
  Gemm,
  Conv2D,
  Pool2D,
  Resize2D,
  Softmax,
  TopK,
  Sdpa,
};

struct GfxAppleMpsVendorPrimitiveDescriptor {
  GfxAppleMpsVendorPrimitiveKind kind = GfxAppleMpsVendorPrimitiveKind::None;
  GfxMpsrtGemmAbiDesc gemm{};
  GfxMpsrtConv2DAbiDesc conv2d{};
  GfxMpsrtPool2DAbiDesc pool2d{};
  GfxMpsrtResize2DAbiDesc resize2d{};
  GfxMpsrtSoftmaxAbiDesc softmax{};
  GfxMpsrtTopKAbiDesc topk{};
  GfxMpsrtSdpaAbiDesc sdpa{};
};

struct GfxAppleMpsVendorPrimitiveContract {
  bool valid = false;
  GfxAppleMpsVendorPrimitiveDescriptor descriptor;
  std::vector<GfxKernelBufferRole> semantic_input_roles;
  GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
  std::vector<GfxMpsrtTensorDesc> input_descs;
  std::vector<GfxMpsrtTensorDesc> output_descs;
};

uint32_t gfx_apple_mps_conv_fused_activation_code(ActivationKind kind);
bool gfx_apple_mps_conv_supports_fused_activation(ActivationKind kind);
std::string gfx_apple_mps_canonical_conv_stage_type(
    const std::shared_ptr<const ov::Node> &node,
    std::string_view fallback_stage_type);

bool gfx_apple_make_mps_conv2d_desc(
    const std::shared_ptr<const ov::Node> &node, GfxMpsrtConv2DAbiDesc &desc,
    bool has_activation = false,
    ActivationKind activation = ActivationKind::Identity);
bool gfx_apple_make_mps_conv2d_contract(
    const std::shared_ptr<const ov::Node> &node, bool has_bias,
    const BiasParams *bias_params, bool has_activation,
    ActivationKind activation, GfxAppleMpsVendorPrimitiveContract &contract);
bool gfx_apple_make_mps_gemm_contract(
    const GfxMpsrtGemmAbiDesc &desc, const GfxMpsrtTensorDesc &lhs,
    const GfxMpsrtTensorDesc &rhs, const GfxMpsrtTensorDesc &output,
    GfxAppleMpsVendorPrimitiveContract &contract);
bool gfx_apple_make_mps_gemm_contract(
    const std::shared_ptr<const ov::Node> &node,
    GfxAppleMpsVendorPrimitiveContract &contract);
bool gfx_apple_make_mps_pool2d_desc(const std::shared_ptr<const ov::Node> &node,
                                    GfxMpsrtPool2DAbiDesc &desc);
bool gfx_apple_make_mps_resize2d_desc(
    const std::shared_ptr<const ov::Node> &node, GfxMpsrtResize2DAbiDesc &desc);
bool gfx_apple_make_mps_softmax_desc(
    const std::shared_ptr<const ov::Node> &node, GfxMpsrtSoftmaxAbiDesc &desc);
bool gfx_apple_make_mps_topk_desc(const std::shared_ptr<const ov::Node> &node,
                                  GfxMpsrtTopKAbiDesc &desc);
bool gfx_apple_make_mps_sdpa_desc(const std::shared_ptr<const ov::Node> &node,
                                  GfxMpsrtSdpaAbiDesc &desc);
bool gfx_apple_make_mps_io_tensor_descs_for_node(
    const std::shared_ptr<const ov::Node> &node, GfxStageStorageKind storage,
    std::vector<GfxMpsrtTensorDesc> &inputs,
    std::vector<GfxMpsrtTensorDesc> &outputs);
bool gfx_apple_make_mps_pool2d_contract(
    const std::shared_ptr<const ov::Node> &node,
    const GfxMpsrtPool2DAbiDesc &desc,
    GfxAppleMpsVendorPrimitiveContract &contract);
bool gfx_apple_make_mps_resize2d_contract(
    const std::shared_ptr<const ov::Node> &node,
    const GfxMpsrtResize2DAbiDesc &desc,
    GfxAppleMpsVendorPrimitiveContract &contract);
bool gfx_apple_make_mps_softmax_contract(
    const std::shared_ptr<const ov::Node> &node,
    const GfxMpsrtSoftmaxAbiDesc &desc,
    GfxAppleMpsVendorPrimitiveContract &contract);
bool gfx_apple_make_mps_topk_contract(
    const std::shared_ptr<const ov::Node> &node,
    const GfxMpsrtTopKAbiDesc &desc,
    GfxAppleMpsVendorPrimitiveContract &contract);
bool gfx_apple_make_mps_sdpa_contract(
    const std::shared_ptr<const ov::Node> &node,
    const GfxMpsrtSdpaAbiDesc &desc,
    GfxAppleMpsVendorPrimitiveContract &contract);
bool gfx_apple_make_mps_transposed_sdpa_contract(
    std::string_view name, const ov::element::Type &element_type,
    const ov::Shape &query_shape, const ov::Shape &key_shape,
    const ov::Shape &value_shape, const ov::Shape &output_shape, float scale,
    GfxAppleMpsVendorPrimitiveContract &contract);

} // namespace gfx_plugin
} // namespace ov
