// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen_apple_mps.hpp"

#include <string>

#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "mlir/gfx_apple_stage_pipeline.hpp"
#include "mlir/gfx_apple_vendor_descriptors.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mpsrt_const_tensor_sources.hpp"
#include "mlir/gfx_mpsrt_conv_metadata.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/topk.hpp"
#include "runtime/gfx_stage_policy.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

bool has_apple_msl_custom_kernel_manifest(mlir::ModuleOp module) {
    GfxKernelStageManifest manifest{};
    return module && detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest) &&
           manifest.valid && manifest.backend_domain == GfxKernelBackendDomain::AppleMsl &&
           manifest.execution_kind == GfxKernelExecutionKind::CustomKernel;
}

GfxMpsrtKernelSourcePlan try_configure_apple_mps_vendor_kernel_source_plan_for_node(
    KernelSource source, const std::shared_ptr<const ov::Node>& node,
    const GpuBufferManager* buffer_manager, std::string_view stage_type, bool has_bias,
    bool has_activation, bool has_batchnorm, ActivationKind activation) {
    if (!source.module || !node) {
        return {};
    }
    if (has_apple_msl_custom_kernel_manifest(source.module)) {
        return {};
    }

    const bool conv_candidate =
        (ov::is_type<const ov::op::v1::Convolution>(node) ||
         ov::is_type<const ov::op::v1::GroupConvolution>(node)) &&
        !has_bias && !has_batchnorm &&
        (!has_activation || gfx_mpsrt_conv_supports_fused_activation(activation));
    if (conv_candidate) {
        const bool group_conv = ov::is_type<const ov::op::v1::GroupConvolution>(node);
        const char* canonical_stage_type = group_conv ? "GroupConvolution" : "Convolution";
        const char* fallback_stage_type = group_conv ? "GroupConv2D" : "Convolution";
        const auto plan =
            select_stage_optimization_plan(buffer_manager, GpuBackend::Metal, canonical_stage_type,
                                           node, node->get_output_element_type(0),
                                           /*has_bias=*/false,
                                           /*has_activation=*/false,
                                           /*has_batchnorm=*/false, GfxStageRuntimeTraits{});
        const auto lowering = annotate_module_with_conv_mpsrt_plan(
            source.module, plan, node, fallback_stage_type, has_activation, activation);
        if (lowering == GfxConvMpsrtLoweringKind::MpsConv2D ||
            lowering == GfxConvMpsrtLoweringKind::MpsGroupConv2D) {
            auto source_plan = make_mpsrt_kernel_source_plan_from_module(source.module);
            if (source_plan.valid()) {
                gfx_attach_mpsrt_conv_const_tensors(source_plan.source, node);
                return source_plan;
            }
        }
    }

    if (stage_type == "MaxPool" || stage_type == "AvgPool") {
        GfxMpsrtPool2DAbiDesc pool_desc{};
        if (gfx_apple_make_mps_pool2d_desc(node, pool_desc)) {
            const auto plan = select_stage_optimization_plan(
                buffer_manager, GpuBackend::Metal, std::string(stage_type), node,
                node->get_output_element_type(0),
                /*has_bias=*/false,
                /*has_activation=*/false,
                /*has_batchnorm=*/false, GfxStageRuntimeTraits{});
            if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
                plan.placement.storage == GfxStageStorageKind::Image) {
                GfxAppleMpsVendorPrimitiveContract contract{};
                if (!gfx_apple_make_mps_pool2d_contract(node, pool_desc, contract)) {
                    return {};
                }
                auto source_module = source.module;
                const auto materialized = materialize_apple_mps_vendor_contract_program(
                    source_module, plan, std::string(stage_type), contract);
                if (materialized.valid) {
                    auto source_plan = make_mpsrt_kernel_source_plan_from_module(source_module);
                    if (source_plan.valid()) {
                        return source_plan;
                    }
                }
            }
        }
    }

    if (stage_type == "Interpolate") {
        GfxMpsrtResize2DAbiDesc resize_desc{};
        if (gfx_apple_make_mps_resize2d_desc(node, resize_desc)) {
            const auto plan =
                select_stage_optimization_plan(buffer_manager, GpuBackend::Metal, "Interpolate",
                                               node, node->get_output_element_type(0),
                                               /*has_bias=*/false,
                                               /*has_activation=*/false,
                                               /*has_batchnorm=*/false, GfxStageRuntimeTraits{});
            if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
                plan.placement.storage == GfxStageStorageKind::Image) {
                GfxAppleMpsVendorPrimitiveContract contract{};
                if (!gfx_apple_make_mps_resize2d_contract(node, resize_desc, contract)) {
                    return {};
                }
                auto source_module = source.module;
                const auto materialized = materialize_apple_mps_vendor_contract_program(
                    source_module, plan, "Interpolate", contract);
                if (materialized.valid) {
                    auto source_plan = make_mpsrt_kernel_source_plan_from_module(source_module);
                    if (source_plan.valid()) {
                        return source_plan;
                    }
                }
            }
        }
    }

    if (stage_type == "Softmax") {
        GfxMpsrtSoftmaxAbiDesc softmax_desc{};
        if (gfx_apple_make_mps_softmax_desc(node, softmax_desc)) {
            const auto plan =
                select_stage_optimization_plan(buffer_manager, GpuBackend::Metal, "Softmax", node,
                                               node->get_output_element_type(0),
                                               /*has_bias=*/false,
                                               /*has_activation=*/false,
                                               /*has_batchnorm=*/false, GfxStageRuntimeTraits{});
            if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
                plan.placement.storage == GfxStageStorageKind::Matrix) {
                GfxAppleMpsVendorPrimitiveContract contract{};
                if (!gfx_apple_make_mps_softmax_contract(node, softmax_desc, contract)) {
                    return {};
                }
                auto source_module = source.module;
                const auto materialized = materialize_apple_mps_vendor_contract_program(
                    source_module, plan, "Softmax", contract);
                if (materialized.valid) {
                    auto source_plan = make_mpsrt_kernel_source_plan_from_module(source_module);
                    if (source_plan.valid()) {
                        return source_plan;
                    }
                }
            }
        }
    }

    if (stage_type == "TopK") {
        GfxMpsrtTopKAbiDesc topk_desc{};
        if (gfx_apple_make_mps_topk_desc(node, topk_desc)) {
            const auto plan = select_stage_optimization_plan(
                buffer_manager, GpuBackend::Metal, "TopK", node, node->get_output_element_type(0),
                /*has_bias=*/false,
                /*has_activation=*/false,
                /*has_batchnorm=*/false, GfxStageRuntimeTraits{});
            if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
                plan.placement.storage == GfxStageStorageKind::Matrix) {
                GfxAppleMpsVendorPrimitiveContract contract{};
                if (!gfx_apple_make_mps_topk_contract(node, topk_desc, contract)) {
                    return {};
                }
                auto source_module = source.module;
                const auto materialized = materialize_apple_mps_vendor_contract_program(
                    source_module, plan, "TopK", contract);
                if (materialized.valid) {
                    auto source_plan = make_mpsrt_kernel_source_plan_from_module(source_module);
                    if (source_plan.valid()) {
                        return source_plan;
                    }
                }
            }
        }
    }

    return {};
}

GfxMpsrtKernelSourcePlan try_configure_clean_apple_mps_vendor_kernel_source_plan_for_node(
    const std::shared_ptr<const ov::Node>& node, const GpuBufferManager* buffer_manager,
    std::string_view stage_type, bool has_bias, bool has_activation, bool has_batchnorm,
    ActivationKind activation) {
    if (!node) {
        return {};
    }

    KernelSource clean_source;
    clean_source.module = build_mlir_for_node(node, gfx_mlir_context());
    if (!clean_source.module) {
        return {};
    }
    return try_configure_apple_mps_vendor_kernel_source_plan_for_node(
        clean_source, node, buffer_manager, stage_type, has_bias, has_activation, has_batchnorm,
        activation);
}

}  // namespace

GfxMpsrtKernelSourcePlan configure_apple_mps_vendor_kernel_source_plan_for_node(
    KernelSource source, const std::shared_ptr<const ov::Node>& node,
    const GpuBufferManager* buffer_manager, std::string_view stage_type, bool has_bias,
    bool has_activation, bool has_batchnorm, ActivationKind activation) {
    auto source_plan = try_configure_apple_mps_vendor_kernel_source_plan_for_node(
        source, node, buffer_manager, stage_type, has_bias, has_activation, has_batchnorm,
        activation);
    if (source_plan.valid() || !source.module ||
        !has_apple_msl_custom_kernel_manifest(source.module)) {
        return source_plan;
    }

    return try_configure_clean_apple_mps_vendor_kernel_source_plan_for_node(
        node, buffer_manager, stage_type, has_bias, has_activation, has_batchnorm, activation);
}

}  // namespace gfx_plugin
}  // namespace ov
