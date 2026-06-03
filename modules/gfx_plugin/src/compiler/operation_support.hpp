// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/gfx_activation.hpp"
#include "common/gpu_parallelism_profile.hpp"
#include "compiler/backend_target.hpp"
#include "compiler/stage_placement.hpp"
#include "openvino/core/model.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {

struct UnsupportedSummary {
    std::vector<std::string> node_names;
    std::vector<std::pair<std::string, size_t>> type_counts;
};

struct OperationSupportQuery {
    std::shared_ptr<const ov::Node> node;
};

enum class LoweringRouteKind {
    Unsupported,
    Common,
    Metadata,
    VendorPrimitive,
    GeneratedKernel,
    HandwrittenKernelException,
};

struct OperationSupportResult {
    bool semantic_legal = false;
    std::string semantic_reason;
    std::string preferred_route;
    LoweringRouteKind preferred_route_kind = LoweringRouteKind::Unsupported;
    std::vector<LoweringRouteKind> alternative_route_kinds;
    double profitability_score = 0.0;
};

struct FusionCapabilities {
    bool enable_generic_attention_fusion = true;
    bool supports_vendor_attention_stage = false;
    bool enable_conv_activation_fusion = true;
    bool enable_precision_sensitive_arithmetic_fusion = true;
};

struct BackendExecutionCapabilities {
    bool source_kernel_dispatch_enabled = false;
    GpuParallelismProfile fallback_parallelism{};
};

struct PostOpFusionCapabilities {
    bool enable_bias_fusion_for_convolution = true;
    bool enable_bias_fusion_for_group_convolution = true;
    bool enable_batchnorm_fusion_for_convolution = true;
    bool enable_batchnorm_fusion_for_group_convolution = true;
    bool enable_activation_fusion_for_convolution = true;
    bool enable_activation_fusion_for_group_convolution = true;
    bool enable_relu_activation_fusion = true;
    bool enable_sigmoid_activation_fusion = true;
    bool enable_tanh_activation_fusion = true;
    bool enable_elu_activation_fusion = true;
    bool enable_prelu_activation_fusion = true;
    bool enable_gelu_activation_fusion = true;
    bool enable_swish_activation_fusion = true;
    bool enable_hswish_activation_fusion = true;
    bool enable_hsigmoid_activation_fusion = true;
    bool enable_abs_activation_fusion = true;
    bool enable_sign_activation_fusion = true;

    bool allow_stage_bias_fusion(std::string_view stage_type) const;
    bool allow_stage_batchnorm_fusion(std::string_view stage_type) const;
    bool allow_stage_activation_fusion(std::string_view stage_type,
                                       ActivationKind kind) const;
};

std::string_view lowering_route_kind_to_string(LoweringRouteKind kind) noexcept;
OperationSupportResult make_supported_operation(std::string semantic_reason,
                                                LoweringRouteKind route_kind,
                                                double profitability_score,
                                                std::string preferred_route = {});
OperationSupportResult make_unsupported_operation(std::string semantic_reason);

class OperationSupportPolicy {
public:
    virtual ~OperationSupportPolicy() = default;

    virtual OperationSupportResult query_operation(const OperationSupportQuery& query) const = 0;
};

class BackendCapabilities final {
public:
    BackendCapabilities(BackendTarget target,
                        std::shared_ptr<const OperationSupportPolicy> operation_policy,
                        FusionCapabilities fusion_capabilities = {},
                        PostOpFusionCapabilities post_op_fusion_capabilities = {},
                        std::shared_ptr<const StagePlacementPolicy> stage_placement_policy = {},
                        BackendExecutionCapabilities execution_capabilities = {});

    const BackendTarget& target() const noexcept {
        return m_target;
    }

    GpuBackend backend() const noexcept {
        return m_target.backend();
    }

    const FusionCapabilities& fusion() const noexcept {
        return m_fusion_capabilities;
    }

    const PostOpFusionCapabilities& post_ops() const noexcept {
        return m_post_op_fusion_capabilities;
    }

    const StagePlacementPolicy* stage_placement() const noexcept {
        return m_stage_placement_policy.get();
    }

    const BackendExecutionCapabilities& execution() const noexcept {
        return m_execution_capabilities;
    }

    OperationSupportResult query_operation(const OperationSupportQuery& query) const;
    bool supports_node(const std::shared_ptr<const ov::Node>& node) const;
    bool allow_stage_bias_fusion(std::string_view stage_type) const;
    bool allow_stage_batchnorm_fusion(std::string_view stage_type) const;
    bool allow_stage_activation_fusion(std::string_view stage_type,
                                       ActivationKind kind) const;

private:
    BackendTarget m_target;
    std::shared_ptr<const OperationSupportPolicy> m_operation_policy;
    FusionCapabilities m_fusion_capabilities;
    PostOpFusionCapabilities m_post_op_fusion_capabilities;
    std::shared_ptr<const StagePlacementPolicy> m_stage_placement_policy;
    BackendExecutionCapabilities m_execution_capabilities;
};

bool is_supported_node(const std::shared_ptr<const ov::Node>& node, GpuBackend backend);
bool model_supported_by_backend(const std::shared_ptr<const ov::Model>& model, GpuBackend backend);
UnsupportedSummary collect_unsupported(const std::shared_ptr<const ov::Model>& model,
                                       GpuBackend backend,
                                       size_t max_nodes = 8);

}  // namespace compiler
}  // namespace gfx_plugin
}  // namespace ov
