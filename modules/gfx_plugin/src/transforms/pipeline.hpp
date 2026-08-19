// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/model.hpp"

namespace ov {
namespace gfx_plugin {
namespace transforms {

struct PipelineOptions {
    bool preserve_scaled_dot_product_attention = false;
    bool canonicalize_sigmoid_before_ranking = false;
    bool enable_llm_attention_fusions = false;
};

// Run GFX-specific transformation pipeline and return transformed clone.
std::shared_ptr<const ov::Model> run_pipeline(const std::shared_ptr<const ov::Model>& model,
                                              const PipelineOptions& options = {});

}  // namespace transforms
}  // namespace gfx_plugin
}  // namespace ov
