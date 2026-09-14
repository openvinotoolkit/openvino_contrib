// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "mlir/IR/PatternMatch.h"
#include "transforms/fusion_pass.hpp"

namespace ov {
namespace gfx_plugin {

void add_conv_activation_fusion_patterns(mlir::RewritePatternSet& patterns,
                                         const FusionConfig& config);
void add_conv_swish_fusion_patterns(mlir::RewritePatternSet& patterns,
                                    const FusionConfig& config);
void add_conv_batchnorm_fusion_patterns(mlir::RewritePatternSet& patterns,
                                        const FusionConfig& config);
void add_conv_batchnorm_act_fusion_patterns(mlir::RewritePatternSet& patterns,
                                            const FusionConfig& config);
void add_conv_batchnorm_swish_fusion_patterns(mlir::RewritePatternSet& patterns,
                                              const FusionConfig& config);
void add_eltwise_activation_fusion_patterns(mlir::RewritePatternSet& patterns,
                                            const FusionConfig& config);
void add_eltwise_input_activation_fusion_patterns(mlir::RewritePatternSet& patterns,
                                                  const FusionConfig& config);
void add_eltwise_bias_activation_fusion_patterns(mlir::RewritePatternSet& patterns,
                                                 const FusionConfig& config);
void add_eltwise_bias_fusion_patterns(mlir::RewritePatternSet& patterns,
                                      const FusionConfig& config);
void add_matmul_activation_fusion_patterns(mlir::RewritePatternSet& patterns,
                                           const FusionConfig& config);
void add_matmul_swish_fusion_patterns(mlir::RewritePatternSet& patterns,
                                      const FusionConfig& config);
void add_matmul_bias_activation_fusion_patterns(mlir::RewritePatternSet& patterns,
                                                const FusionConfig& config);
void add_matmul_bias_swish_fusion_patterns(mlir::RewritePatternSet& patterns,
                                           const FusionConfig& config);
void add_matmul_bias_fusion_patterns(mlir::RewritePatternSet& patterns,
                                     const FusionConfig& config);
void add_conv_bias_activation_fusion_patterns(mlir::RewritePatternSet& patterns,
                                              const FusionConfig& config);
void add_conv_bias_swish_fusion_patterns(mlir::RewritePatternSet& patterns,
                                         const FusionConfig& config);
void add_conv_bias_fusion_patterns(mlir::RewritePatternSet& patterns,
                                   const FusionConfig& config);
void add_conv_scale_activation_fusion_patterns(mlir::RewritePatternSet& patterns,
                                               const FusionConfig& config);
void add_conv_scale_fusion_patterns(mlir::RewritePatternSet& patterns,
                                    const FusionConfig& config);
void add_attention_fusion_patterns(mlir::RewritePatternSet& patterns,
                                   const FusionConfig& config);
void add_attention_scale_mask_fusion_patterns(mlir::RewritePatternSet& patterns,
                                               const FusionConfig& config);

}  // namespace gfx_plugin
}  // namespace ov
