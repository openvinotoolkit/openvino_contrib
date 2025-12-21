// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

namespace ov {
namespace gfx_plugin {

// Build a minimal ov::Model containing the provided node and its parameter inputs.
std::shared_ptr<ov::Model> make_single_op_model(const std::shared_ptr<const ov::Node>& node);
// Build a minimal ov::Model containing the provided node with all of its outputs.
std::shared_ptr<ov::Model> make_single_op_model_all_outputs(const std::shared_ptr<const ov::Node>& node);

// Clamp threadgroup size to the pipeline's maximum to avoid invalid dispatch.
inline size_t metal_clamp_tg_size(void* pipeline, size_t desired) {
    if (desired == 0)
        desired = 1;
#ifdef __OBJC__
    if (!pipeline)
        return desired;
    id<MTLComputePipelineState> p = (id<MTLComputePipelineState>)pipeline;
    const NSUInteger max_threads = [p maxTotalThreadsPerThreadgroup];
    if (max_threads == 0)
        return desired;
    return static_cast<size_t>(std::min<NSUInteger>(static_cast<NSUInteger>(desired), max_threads));
#else
    (void)pipeline;
    return desired;
#endif
}

}  // namespace gfx_plugin
}  // namespace ov
