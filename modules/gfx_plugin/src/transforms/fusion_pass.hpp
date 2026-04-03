// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "runtime/gfx_activation.hpp"
#include "runtime/gfx_batchnorm.hpp"
#include "runtime/gfx_bias.hpp"

namespace ov {
namespace gfx_plugin {

struct FusionConfig {
    bool enable_fusion = true;
    bool debug_dump_ir = false;
};

struct FusionGroup {
    std::vector<size_t> node_indices;
    std::optional<ActivationKind> activation;
    float activation_alpha = 0.0f;
    std::optional<BiasParams> bias;
    std::optional<BatchNormParams> batchnorm;
    std::string kind;  // e.g. "ConvRelu"
};

struct FusionPlan {
    std::vector<FusionGroup> groups;
};

FusionPlan build_fusion_plan(const std::shared_ptr<const ov::Model>& model,
                             const FusionConfig& config);

}  // namespace gfx_plugin
}  // namespace ov
