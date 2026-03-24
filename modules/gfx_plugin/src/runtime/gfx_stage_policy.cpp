// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "runtime/gfx_stage_policy.hpp"

#include <string_view>

#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/transpose.hpp"
#include "runtime/gfx_parallelism.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool is_conv_like(std::string_view stage_type) {
    return stage_type == "Convolution" || stage_type == "GroupConvolution";
}

GfxStageArchetype classify_stage_archetype(const std::string& stage_type,
                                           const std::shared_ptr<const ov::Node>& node,
                                           const GfxStageRuntimeTraits& traits) {
    if (traits.conv2d_1x1_direct || traits.conv2d_3x3_direct || traits.conv2d_chunked) {
        return GfxStageArchetype::Convolution;
    }
    if (traits.group_conv2d_chunked) {
        return GfxStageArchetype::GroupConvolution;
    }
    if (stage_type == "Convolution") {
        return GfxStageArchetype::Convolution;
    }
    if (stage_type == "GroupConvolution") {
        return GfxStageArchetype::GroupConvolution;
    }
    if (stage_type == "MatMul") {
        return GfxStageArchetype::MatMul;
    }
    if (traits.unary_chunked) {
        return GfxStageArchetype::UnaryElementwise;
    }
    if (traits.binary_chunked || traits.binary_same_shape || traits.binary_bias_add) {
        return GfxStageArchetype::BinaryElementwise;
    }
    if (traits.softmax_chunked || stage_type == "Softmax" || stage_type == "LogSoftmax") {
        return GfxStageArchetype::Reduction;
    }
    if (traits.transpose_chunked || stage_type == "Transpose" || stage_type == "Reshape") {
        return GfxStageArchetype::Layout;
    }
    if (traits.split_concat_chunked || stage_type == "Split" || stage_type == "VariadicSplit" || stage_type == "Concat") {
        return GfxStageArchetype::SplitConcat;
    }
    if (traits.convert_chunked || stage_type == "Convert") {
        return GfxStageArchetype::Convert;
    }
    if (node && ov::is_type<ov::op::v1::Convolution>(node)) {
        return GfxStageArchetype::Convolution;
    }
    if (node && ov::is_type<ov::op::v1::GroupConvolution>(node)) {
        return GfxStageArchetype::GroupConvolution;
    }
    return GfxStageArchetype::Unknown;
}

bool is_safe_shared_activation(ActivationKind kind) {
    switch (kind) {
        case ActivationKind::Relu:
        case ActivationKind::Sigmoid:
        case ActivationKind::Tanh:
        case ActivationKind::Elu:
        case ActivationKind::Prelu:
        case ActivationKind::Gelu:
        case ActivationKind::Abs:
        case ActivationKind::Sign:
            return true;
        default:
            return false;
    }
}

bool is_identity_permutation(const std::shared_ptr<const ov::Node>& node) {
    auto transpose = ov::as_type_ptr<const ov::op::v1::Transpose>(node);
    if (!transpose || transpose->get_input_size() != 2) {
        return false;
    }
    auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
    if (!perm_const) {
        return false;
    }
    const auto perm = perm_const->cast_vector<int64_t>();
    const auto& in_shape = transpose->get_input_shape(0);
    const auto& out_shape = transpose->get_output_shape(0);
    if (perm.size() != in_shape.size() || perm.size() != out_shape.size()) {
        return false;
    }
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] != static_cast<int64_t>(i)) {
            return false;
        }
    }
    return in_shape == out_shape;
}

uint64_t shape_elements(const ov::Shape& shape) {
    uint64_t total = 1;
    for (const auto dim : shape) {
        total *= std::max<uint64_t>(1, static_cast<uint64_t>(dim));
    }
    return total;
}

bool is_compile_safe_im2col_spatial3x3_bucket(const ov::Shape& in_shape,
                                               const ov::Shape& w_shape,
                                               const ov::Shape& out_shape,
                                               bool stride2) {
    if (stride2 || in_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4) {
        return false;
    }
    if (in_shape[0] != 1) {
        return false;
    }

    const uint64_t kernel_work = static_cast<uint64_t>(in_shape[1]) * 9ull;
    if (kernel_work >= 1152 && w_shape[0] >= 128) {
        return true;
    }
    return false;
}

GfxParallelismCaps query_stage_caps(const GpuBufferManager* buffer_manager, GpuBackend backend) {
    if (buffer_manager) {
        return query_parallelism_caps(buffer_manager);
    }
    GfxParallelismCaps caps{};
    caps.backend = backend;
    if (backend == GpuBackend::Vulkan) {
        caps.preferred_simd_width = 32;
        caps.subgroup_size = 32;
        caps.max_total_threads_per_group = 128;
        caps.max_threads_per_group = {128, 128, 64};
    } else {
        caps.preferred_simd_width = 32;
        caps.subgroup_size = 32;
        caps.max_total_threads_per_group = 256;
        caps.max_threads_per_group = {256, 256, 64};
    }
    return caps;
}

}  // namespace

bool allow_stage_bias_fusion(GpuBackend backend, const std::string& stage_type) {
    if (backend == GpuBackend::Vulkan) {
        return false;
    }
    return is_conv_like(stage_type);
}

bool allow_stage_batchnorm_fusion(GpuBackend /*backend*/, const std::string& stage_type) {
    return is_conv_like(stage_type);
}

bool allow_stage_activation_fusion(GpuBackend backend,
                                   const std::string& stage_type,
                                   ActivationKind kind) {
    if (!is_safe_shared_activation(kind)) {
        return false;
    }
    if (backend == GpuBackend::Vulkan) {
        return false;
    }
    return is_conv_like(stage_type);
}

GfxStagePostOpSupport select_stage_post_op_support(GpuBackend backend,
                                                   GfxStageArchetype archetype,
                                                   const std::string& stage_type) {
    GfxStagePostOpSupport support{};
    switch (archetype) {
        case GfxStageArchetype::Convolution:
        case GfxStageArchetype::GroupConvolution:
            support.batchnorm = allow_stage_batchnorm_fusion(backend, stage_type);
            support.bias = allow_stage_bias_fusion(backend, stage_type);
            support.activation = allow_stage_activation_fusion(backend, stage_type, ActivationKind::Relu);
            break;
        default:
            break;
    }
    return support;
}

GfxTensorLayoutPlan select_tensor_layout_plan(const std::string& stage_type,
                                              const std::shared_ptr<const ov::Node>& node) {
    GfxTensorLayoutPlan plan{};
    if (stage_type == "Reshape" || stage_type == "Squeeze" || stage_type == "Unsqueeze") {
        plan.kind = GfxTensorLayoutKind::ViewOnly;
        plan.view_only = true;
        return plan;
    }
    if (stage_type == "Transpose" && is_identity_permutation(node)) {
        plan.kind = GfxTensorLayoutKind::ViewOnly;
        plan.view_only = true;
        return plan;
    }
    if (stage_type == "Transpose" || stage_type == "Reshape" || stage_type == "Squeeze" || stage_type == "Unsqueeze") {
        plan.kind = GfxTensorLayoutKind::Materialized;
    }
    return plan;
}

GfxStageOptimizationPlan select_stage_optimization_plan(const GpuBufferManager* buffer_manager,
                                                        GpuBackend backend,
                                                        const std::string& stage_type,
                                                        const std::shared_ptr<const ov::Node>& node,
                                                        const ov::element::Type& element_type,
                                                        bool has_bias,
                                                        bool has_activation,
                                                        bool has_batchnorm,
                                                        const GfxStageRuntimeTraits& traits) {
    GfxStageOptimizationPlan plan{};
    plan.archetype = classify_stage_archetype(stage_type, node, traits);
    plan.layout = select_tensor_layout_plan(stage_type, node);
    plan.post_ops = select_stage_post_op_support(backend, plan.archetype, stage_type);
    plan.execution.fusion.allow_bias = plan.post_ops.bias;
    plan.execution.fusion.allow_batchnorm = plan.post_ops.batchnorm;
    plan.execution.fusion.allow_activation = plan.post_ops.activation;
    if (node) {
        plan.conv = select_conv_route_plan(buffer_manager,
                                           backend,
                                           node,
                                           element_type,
                                           has_bias,
                                           has_activation,
                                           has_batchnorm);
    }

    if (backend != GpuBackend::Vulkan) {
        return plan;
    }

    const auto caps = query_stage_caps(buffer_manager, backend);
    const uint32_t wave = std::max<uint32_t>(1u, std::max(caps.subgroup_size, caps.preferred_simd_width));

    if (traits.binary_chunked || traits.binary_same_shape || traits.binary_bias_add) {
        plan.execution.submit.weight = 8;
        return plan;
    }
    if (traits.unary_chunked || traits.softmax_chunked) {
        plan.execution.submit.weight = 6;
        return plan;
    }
    if (plan.conv.kind == GfxConvRouteKind::Direct1x1 || plan.conv.kind == GfxConvRouteKind::Direct3x3 ||
        plan.conv.kind == GfxConvRouteKind::Chunked || plan.conv.kind == GfxConvRouteKind::GroupChunked) {
        plan.execution.submit.weight = wave >= 64 ? 10 : 8;
        return plan;
    }
    if (traits.transpose_chunked || traits.split_concat_chunked) {
        plan.execution.submit.weight = 4;
        return plan;
    }
    if (traits.convert_chunked) {
        plan.execution.submit.weight = 6;
        return plan;
    }
    return plan;
}

GfxStageExecutionPolicy select_stage_execution_policy(const GpuBufferManager* buffer_manager,
                                                      GpuBackend backend,
                                                      const std::string& stage_type,
                                                      const GfxStageRuntimeTraits& traits) {
    return select_stage_optimization_plan(buffer_manager,
                                          backend,
                                          stage_type,
                                          nullptr,
                                          ov::element::dynamic,
                                          false,
                                          false,
                                          false,
                                          traits)
        .execution;
}

GfxConvRoutePlan select_conv_route_plan(const GpuBufferManager* /*buffer_manager*/,
                                        GpuBackend backend,
                                        const std::shared_ptr<const ov::Node>& node,
                                        const ov::element::Type& element_type,
                                        bool has_bias,
                                        bool has_activation,
                                        bool has_batchnorm) {
    GfxConvRoutePlan plan{};
    if (backend != GpuBackend::Vulkan || !node || has_bias || has_activation || has_batchnorm) {
        return plan;
    }
    if (element_type != ov::element::f16 && element_type != ov::element::f32) {
        return plan;
    }

    if (auto gconv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
        const auto& in_shape = gconv->get_input_shape(0);
        const auto& w_shape = gconv->get_input_shape(1);
        const auto& out_shape = gconv->get_output_shape(0);
        if (in_shape.size() == 4 && out_shape.size() == 4 && w_shape.size() == 5 &&
            w_shape[0] == in_shape[1] && w_shape[0] == out_shape[1] && w_shape[1] == 1 && w_shape[2] == 1) {
            plan.kind = GfxConvRouteKind::GroupChunked;
            plan.family = GfxConvFamily::Depthwise;
            plan.algorithm.kind = GfxConvAlgorithmKind::DepthwiseDirect;
            plan.algorithm.variant = "depthwise_chunked";
        } else {
            plan.family = GfxConvFamily::Grouped;
            plan.algorithm.kind = GfxConvAlgorithmKind::ChunkedDirect;
        }
        return plan;
    }

    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node);
    if (!conv || conv->get_input_size() != 2 || conv->get_output_size() != 1) {
        return plan;
    }
    const auto& in_shape = conv->get_input_shape(0);
    const auto& w_shape = conv->get_input_shape(1);
    const auto& out_shape = conv->get_output_shape(0);
    if (in_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4) {
        return plan;
    }

    const uint64_t out_elems = shape_elements(out_shape);
    const bool is_1x1_s1 = w_shape[2] == 1 && w_shape[3] == 1 &&
                           conv->get_strides().at(0) == 1 && conv->get_strides().at(1) == 1 &&
                           conv->get_dilations().at(0) == 1 && conv->get_dilations().at(1) == 1 &&
                           conv->get_pads_begin().at(0) == 0 && conv->get_pads_begin().at(1) == 0 &&
                           conv->get_pads_end().at(0) == 0 && conv->get_pads_end().at(1) == 0 &&
                           in_shape[2] == out_shape[2] && in_shape[3] == out_shape[3];
    if (is_1x1_s1 && out_elems >= 16384) {
        plan.kind = GfxConvRouteKind::Direct1x1;
        plan.family = GfxConvFamily::Pointwise1x1;
        plan.algorithm.kind = GfxConvAlgorithmKind::Direct1x1;
        plan.algorithm.variant = "direct_1x1";
        return plan;
    }

    const bool is_3x3 = w_shape[2] == 3 && w_shape[3] == 3 &&
                        conv->get_dilations().at(0) == 1 && conv->get_dilations().at(1) == 1;
    if (is_3x3 && out_elems >= 16384) {
        const bool stride2 = conv->get_strides().at(0) == 2 && conv->get_strides().at(1) == 2;
        const uint64_t kernel_work = static_cast<uint64_t>(in_shape.at(1)) * 9ull;
        const bool can_use_im2col = backend == GpuBackend::Vulkan &&
                                    is_compile_safe_im2col_spatial3x3_bucket(in_shape, w_shape, out_shape, stride2);
        if (can_use_im2col) {
            plan.kind = GfxConvRouteKind::Chunked;
            plan.family = GfxConvFamily::Spatial3x3;
            plan.algorithm.kind = GfxConvAlgorithmKind::Im2ColMatMul;
            plan.algorithm.variant = "im2col_matmul";
            return plan;
        }
        const bool prefer_direct = stride2 || kernel_work >= 288 || w_shape[0] >= 64;
        if (prefer_direct) {
            plan.kind = GfxConvRouteKind::Direct3x3;
            plan.family = GfxConvFamily::Spatial3x3;
            plan.algorithm.kind = stride2 ? GfxConvAlgorithmKind::Direct3x3Stride2
                                          : GfxConvAlgorithmKind::Direct3x3Stride1;
            plan.algorithm.variant = stride2 ? "direct_3x3_s2" : "direct_3x3";
            return plan;
        }
    }

    const uint64_t macs = out_elems *
                          static_cast<uint64_t>(in_shape.at(1)) *
                          static_cast<uint64_t>(w_shape.at(2)) *
                          static_cast<uint64_t>(w_shape.at(3));
    if (!is_1x1_s1 && (out_elems >= 16384 || macs >= (1ull << 20))) {
        plan.kind = GfxConvRouteKind::Chunked;
        plan.family = is_3x3 ? GfxConvFamily::Spatial3x3 : GfxConvFamily::General;
        const bool can_use_im2col = false;
        if (can_use_im2col) {
            plan.algorithm.kind = GfxConvAlgorithmKind::Im2ColMatMul;
            plan.algorithm.variant = "im2col_matmul";
        } else {
            plan.algorithm.kind = GfxConvAlgorithmKind::ChunkedDirect;
            plan.algorithm.variant = "general_chunked";
        }
    }
    return plan;
}

}  // namespace gfx_plugin
}  // namespace ov
