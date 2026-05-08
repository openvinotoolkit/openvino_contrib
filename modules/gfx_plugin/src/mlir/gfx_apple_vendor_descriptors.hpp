// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "runtime/gfx_activation.hpp"
#include "runtime/gfx_mpsrt_abi.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
class Node;

namespace gfx_plugin {

uint32_t gfx_apple_mps_conv_fused_activation_code(ActivationKind kind);
bool gfx_apple_mps_conv_supports_fused_activation(ActivationKind kind);
std::string gfx_apple_mps_canonical_conv_stage_type(const std::shared_ptr<const ov::Node>& node,
                                                    std::string_view fallback_stage_type);

bool gfx_apple_make_mps_conv2d_desc(const std::shared_ptr<const ov::Node>& node,
                                    GfxMpsrtConv2DAbiDesc& desc,
                                    bool has_activation = false,
                                    ActivationKind activation = ActivationKind::Identity);
bool gfx_apple_make_mps_pool2d_desc(const std::shared_ptr<const ov::Node>& node,
                                    GfxMpsrtPool2DAbiDesc& desc);
bool gfx_apple_make_mps_resize2d_desc(const std::shared_ptr<const ov::Node>& node,
                                      GfxMpsrtResize2DAbiDesc& desc);
bool gfx_apple_make_mps_softmax_desc(const std::shared_ptr<const ov::Node>& node,
                                     GfxMpsrtSoftmaxAbiDesc& desc);
bool gfx_apple_make_mps_topk_desc(const std::shared_ptr<const ov::Node>& node,
                                  GfxMpsrtTopKAbiDesc& desc);
bool gfx_apple_make_mps_io_tensor_descs_for_node(const std::shared_ptr<const ov::Node>& node,
                                                 GfxStageStorageKind storage,
                                                 std::vector<GfxMpsrtTensorDesc>& inputs,
                                                 std::vector<GfxMpsrtTensorDesc>& outputs);

}  // namespace gfx_plugin
}  // namespace ov
