// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <string>
#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "runtime/gfx_activation.hpp"
#include "runtime/gfx_backend_utils.hpp"
#include "runtime/gpu_stage.hpp"

namespace ov {
namespace gfx_plugin {

struct GfxStageRuntimeTraits {
    bool binary_chunked = false;
    bool binary_same_shape = false;
    bool binary_bias_add = false;
    bool unary_chunked = false;
    bool softmax_chunked = false;
    bool conv2d_1x1_direct = false;
    bool conv2d_3x3_direct = false;
    bool conv2d_chunked = false;
    bool group_conv2d_chunked = false;
    bool transpose_chunked = false;
    bool split_concat_chunked = false;
    bool convert_chunked = false;
};

struct GfxStageFusionPolicy {
    bool allow_bias = false;
    bool allow_activation = false;
    bool allow_batchnorm = false;
};

struct GfxStagePostOpSupport {
    bool bias = false;
    bool activation = false;
    bool batchnorm = false;
};

enum class GfxTensorLayoutKind {
    Unknown,
    Materialized,
    ViewOnly,
};

struct GfxTensorLayoutPlan {
    GfxTensorLayoutKind kind = GfxTensorLayoutKind::Unknown;
    bool view_only = false;
};

struct GfxStageExecutionPolicy {
    GfxStageFusionPolicy fusion{};
    GpuStageSubmitPolicy submit{};
};

enum class GfxStageArchetype {
    Unknown,
    Convolution,
    GroupConvolution,
    MatMul,
    UnaryElementwise,
    BinaryElementwise,
    Reduction,
    Layout,
    Convert,
    SplitConcat,
};

enum class GfxConvRouteKind {
    None,
    Direct1x1,
    Direct3x3,
    Chunked,
    GroupChunked,
};

enum class GfxConvFamily {
    Unknown,
    Pointwise1x1,
    Spatial3x3,
    Depthwise,
    Grouped,
    General,
};

enum class GfxConvAlgorithmKind {
    None,
    Direct1x1,
    Direct3x3Stride1,
    Direct3x3Stride2,
    DepthwiseDirect,
    ChunkedDirect,
    Im2ColMatMul,
    Indirect,
};

struct GfxConvAlgorithmPlan {
    GfxConvAlgorithmKind kind = GfxConvAlgorithmKind::None;
    std::string variant;
};

struct GfxConvRoutePlan {
    GfxConvRouteKind kind = GfxConvRouteKind::None;
    GfxConvFamily family = GfxConvFamily::Unknown;
    GfxConvAlgorithmPlan algorithm{};
};

struct GfxStageOptimizationPlan {
    GfxStageArchetype archetype = GfxStageArchetype::Unknown;
    GfxTensorLayoutPlan layout{};
    GfxStagePostOpSupport post_ops{};
    GfxStageExecutionPolicy execution{};
    GfxConvRoutePlan conv{};
};

GfxTensorLayoutPlan select_tensor_layout_plan(const std::string& stage_type,
                                              const std::shared_ptr<const ov::Node>& node);

GfxStageOptimizationPlan select_stage_optimization_plan(const GpuBufferManager* buffer_manager,
                                                        GpuBackend backend,
                                                        const std::string& stage_type,
                                                        const std::shared_ptr<const ov::Node>& node,
                                                        const ov::element::Type& element_type,
                                                        bool has_bias,
                                                        bool has_activation,
                                                        bool has_batchnorm,
                                                        const GfxStageRuntimeTraits& traits);

GfxStageExecutionPolicy select_stage_execution_policy(const GpuBufferManager* buffer_manager,
                                                      GpuBackend backend,
                                                      const std::string& stage_type,
                                                      const GfxStageRuntimeTraits& traits);
GfxConvRoutePlan select_conv_route_plan(const GpuBufferManager* buffer_manager,
                                        GpuBackend backend,
                                        const std::shared_ptr<const ov::Node>& node,
                                        const ov::element::Type& element_type,
                                        bool has_bias,
                                        bool has_activation,
                                        bool has_batchnorm);

bool allow_stage_bias_fusion(GpuBackend backend, const std::string& stage_type);
bool allow_stage_batchnorm_fusion(GpuBackend backend, const std::string& stage_type);
bool allow_stage_activation_fusion(GpuBackend backend,
                                   const std::string& stage_type,
                                   ActivationKind kind);

}  // namespace gfx_plugin
}  // namespace ov
