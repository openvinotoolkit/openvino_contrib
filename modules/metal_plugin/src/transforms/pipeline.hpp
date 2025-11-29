// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/control_flow/unroll_if.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"
#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "transforms/conv_relu_fusion.hpp"

namespace ov {
namespace metal_plugin {
namespace transforms {

// Run METAL-specific transformation pipeline and return transformed clone.
inline std::shared_ptr<const ov::Model> run_pipeline(const std::shared_ptr<const ov::Model>& model) {
    OPENVINO_ASSERT(model, "run_pipeline: model is null");
    // Work on a clone to preserve the original model passed by the caller.
    auto cloned = model->clone();

    ov::pass::Manager manager("Plugin:METAL");
    // Common optimizations from OpenVINO transformations library
    manager.register_pass<ov::pass::CommonOptimizations>();

    // Align behaviour with template plugin: disable a few transformations that can be harmful for backend mapping
    auto pass_config = manager.get_pass_config();
    pass_config->disable<ov::pass::UnrollIf>();
    pass_config->disable<ov::pass::ConvertMaxPool14ToMaxPool8>();
    pass_config->disable<ov::pass::ConvertAvgPool14ToAvgPool1>();
    pass_config->disable<ov::pass::ConvertReduceSumToPooling>();

    // Allow FP16 converts to be folded and decompression converts to be upgraded back to FP32
    pass_config->disable<ov::pass::DisableDecompressionConvertConstantFolding>();
    pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

    // Keep SDPA as-is (METAL has its own lowering path later)
    pass_config->disable<ov::pass::ScaledDotProductAttentionDecomposition>();

    // METAL-specific hinting pass
    manager.register_pass<ConvReluFusion>();
    manager.run_passes(cloned);
    return cloned;
}

}  // namespace transforms
}  // namespace metal_plugin
}  // namespace ov
