// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/compiler/metal_kernel_artifacts.hpp"

#include <memory>
#include <string>
#include <utility>

#include "compiler/pipeline_stage_fusion.hpp"
#include "kernel_ir/gfx_kernel_source.hpp"
#include "kernel_ir/metal_kernels/reduction_kernels.hpp"
#include "kernel_ir/metal_kernels/softmax_kernels.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_activation.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_eltwise.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_ops.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_reduction.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_shape.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_slice_static.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_softmax.hpp"
#include "backends/metal/compiler/msl_codegen_apple_msl_split.hpp"
#include "backends/metal/compiler/msl_codegen_attention.hpp"

namespace ov {
namespace gfx_plugin {
namespace compiler {
namespace {

constexpr const char *kMetalShapeOfMslKernelUnit = "metal/generated/shapeof";
constexpr const char *kMetalRangeMslKernelUnit = "metal/generated/range";
constexpr const char *kMetalTileMslKernelUnit = "metal/generated/tile";
constexpr const char *kMetalConcatMslKernelUnit = "metal/generated/concat";
constexpr const char *kMetalSplitMslKernelUnit = "metal/generated/split";
constexpr const char *kMetalSliceMslKernelUnit = "metal/generated/slice";
constexpr const char *kMetalTransposeF32MslKernelUnit =
    "metal/generated/transpose_f32";
constexpr const char *kMetalCausalSdpaMslKernelUnit =
    "metal/generated/sdpa_causal_mask";
constexpr const char *kMetalActivationMslKernelUnit =
    "metal/generated/activation";
constexpr const char *kMetalEltwiseMslKernelUnit = "metal/generated/eltwise";
constexpr const char *kMetalReductionF32MslKernelUnit =
    "metal/generated/reduction_f32";
constexpr const char *kMetalReductionLogicalBoolMslKernelUnit =
    "metal/generated/reduction_logical_bool";
constexpr const char *kMetalSoftmaxF32MslKernelUnit =
    "metal/generated/softmax_f32";
constexpr const char *kMetalSoftmaxF16MslKernelUnit =
    "metal/generated/softmax_f16";
constexpr const char *kMetalLogSoftmaxF32MslKernelUnit =
    "metal/generated/logsoftmax_f32";
constexpr const char *kMetalLogSoftmaxF16MslKernelUnit =
    "metal/generated/logsoftmax_f16";
constexpr const char *kMetalMpsSoftmaxVendorUnit = "metal/vendor/mps_softmax";
constexpr const char *kMetalMpsGemmVendorUnit = "metal/vendor/mps_gemm";
constexpr const char *kMetalMpsPool2DVendorUnit = "metal/vendor/mps_pool2d";
constexpr const char *kMetalMpsResize2DVendorUnit = "metal/vendor/mps_resize2d";
constexpr const char *kMetalMpsGraphSdpaVendorUnit =
    "metal/vendor/mpsgraph_sdpa";

uint32_t
vendor_abi_arg_count(const GfxAppleMpsVendorPrimitiveContract &contract) {
  if (contract.external_buffer_abi.valid &&
      contract.external_buffer_abi.has_buffer_count) {
    return contract.external_buffer_abi.buffer_count;
  }
  return static_cast<uint32_t>(contract.input_descs.size() +
                               contract.output_descs.size());
}

uint32_t vendor_abi_output_arg_count(
    const GfxAppleMpsVendorPrimitiveContract &contract) {
  if (contract.external_buffer_abi.valid &&
      contract.external_buffer_abi.has_output_buffer_count) {
    return contract.external_buffer_abi.output_buffer_count;
  }
  return static_cast<uint32_t>(contract.output_descs.size());
}

std::shared_ptr<const KernelArtifactPayload>
materialize_generated_msl_payload(KernelArtifactDescriptor &descriptor,
                                  GfxMslGeneratedKernelSourcePlan source_plan) {
  if (!source_plan.valid()) {
    return {};
  }
  descriptor.entry_point = source_plan.source.entry_point;
  descriptor.compile_options_key = "metal_msl_source";
  descriptor.abi_arg_count = source_plan.source.signature.arg_count;
  descriptor.abi_output_arg_count =
      source_plan.source.signature.output_arg_count;
  return std::make_shared<GfxKernelSourcePayload>(
      descriptor.kernel.kernel_id, descriptor.kernel.backend_domain,
      descriptor.entry_point, GfxKernelSourceLanguage::MetalShadingLanguage,
      std::move(source_plan.source.msl_source));
}

std::shared_ptr<const KernelArtifactPayload>
materialize_vendor_payload(KernelArtifactDescriptor &descriptor,
                           const char *entry_point,
                           GfxAppleMpsVendorPrimitiveContract contract) {
  if (!contract.valid ||
      contract.descriptor.kind == GfxAppleMpsVendorPrimitiveKind::None) {
    return {};
  }
  descriptor.entry_point = entry_point;
  descriptor.compile_options_key = "metal_vendor_descriptor";
  descriptor.abi_arg_count = vendor_abi_arg_count(contract);
  descriptor.abi_output_arg_count = vendor_abi_output_arg_count(contract);
  return std::make_shared<GfxMetalVendorPrimitiveArtifactPayload>(
      descriptor.kernel.kernel_id, descriptor.kernel.backend_domain,
      descriptor.entry_point, std::move(contract));
}

std::shared_ptr<const KernelArtifactPayload>
materialize_mps_softmax_payload(KernelArtifactDescriptor &descriptor,
                                const std::shared_ptr<const ov::Node> &node) {
  GfxMpsrtSoftmaxAbiDesc desc{};
  if (!gfx_apple_make_mps_softmax_desc(node, desc)) {
    return {};
  }
  GfxAppleMpsVendorPrimitiveContract contract{};
  if (!gfx_apple_make_mps_softmax_contract(node, desc, contract)) {
    return {};
  }
  return materialize_vendor_payload(descriptor, "mps_softmax",
                                    std::move(contract));
}

std::shared_ptr<const KernelArtifactPayload>
materialize_mps_gemm_payload(KernelArtifactDescriptor &descriptor,
                             const std::shared_ptr<const ov::Node> &node) {
  GfxAppleMpsVendorPrimitiveContract contract{};
  if (!gfx_apple_make_mps_gemm_contract(node, contract)) {
    return {};
  }
  return materialize_vendor_payload(descriptor, "mps_gemm",
                                    std::move(contract));
}

std::shared_ptr<const KernelArtifactPayload>
materialize_mps_pool2d_payload(KernelArtifactDescriptor &descriptor,
                               const std::shared_ptr<const ov::Node> &node) {
  GfxMpsrtPool2DAbiDesc desc{};
  if (!gfx_apple_make_mps_pool2d_desc(node, desc)) {
    return {};
  }
  GfxAppleMpsVendorPrimitiveContract contract{};
  if (!gfx_apple_make_mps_pool2d_contract(node, desc, contract)) {
    return {};
  }
  return materialize_vendor_payload(descriptor, "mps_pool2d",
                                    std::move(contract));
}

std::shared_ptr<const KernelArtifactPayload>
materialize_mps_resize2d_payload(KernelArtifactDescriptor &descriptor,
                                 const std::shared_ptr<const ov::Node> &node) {
  GfxMpsrtResize2DAbiDesc desc{};
  if (!gfx_apple_make_mps_resize2d_desc(node, desc)) {
    return {};
  }
  GfxAppleMpsVendorPrimitiveContract contract{};
  if (!gfx_apple_make_mps_resize2d_contract(node, desc, contract)) {
    return {};
  }
  return materialize_vendor_payload(descriptor, "mps_resize2d",
                                    std::move(contract));
}

std::shared_ptr<const KernelArtifactPayload>
materialize_mpsgraph_sdpa_payload(KernelArtifactDescriptor &descriptor,
                                  const std::shared_ptr<const ov::Node> &node) {
  GfxMpsrtSdpaAbiDesc desc{};
  if (!gfx_apple_make_mps_sdpa_desc(node, desc)) {
    return {};
  }
  GfxAppleMpsVendorPrimitiveContract contract{};
  if (!gfx_apple_make_mps_sdpa_contract(node, desc, contract)) {
    return {};
  }
  return materialize_vendor_payload(descriptor, "mps_sdpa",
                                    std::move(contract));
}

std::shared_ptr<const KernelArtifactPayload>
materialize_fused_mpsgraph_sdpa_payload(
    KernelArtifactDescriptor &descriptor,
    const PipelineVendorAttentionPlan &plan) {
  if (descriptor.kernel.backend_domain != "metal" ||
      descriptor.payload_kind != KernelArtifactPayloadKind::VendorDescriptor ||
      descriptor.kernel.kernel_id != kMetalMpsGraphSdpaVendorUnit) {
    return {};
  }
  GfxAppleMpsVendorPrimitiveContract contract{};
  if (!gfx_apple_make_mps_transposed_sdpa_contract(
          plan.name, plan.element_type, plan.query_shape, plan.key_shape,
          plan.value_shape, plan.output_shape, plan.scale, contract)) {
    return {};
  }
  return materialize_vendor_payload(descriptor, "mps_sdpa",
                                    std::move(contract));
}

KernelArtifactDescriptor make_mpsgraph_sdpa_vendor_attention_artifact_descriptor(
    uint64_t stage_record_key, const PipelineVendorAttentionPlan &plan) {
  KernelArtifactDescriptor descriptor;
  descriptor.stage_record_key = stage_record_key;
  descriptor.payload_kind = KernelArtifactPayloadKind::VendorDescriptor;
  descriptor.kernel.kernel_id = kMetalMpsGraphSdpaVendorUnit;
  descriptor.kernel.op_family = "VendorAttention";
  descriptor.kernel.backend_domain = "metal";
  descriptor.kernel.origin = KernelArtifactOrigin::VendorPrimitive;
  descriptor.kernel.tensor_roles = {"tensor_input", "tensor_input",
                                    "tensor_input", "tensor_output"};
  descriptor.kernel.scalar_roles = {"scalar:f32"};
  descriptor.kernel.layout_contract = "ndarray";
  descriptor.kernel.precision_contract = plan.element_type.get_type_name();
  descriptor.kernel.dispatch_contract = "mps_sdpa";
  descriptor.kernel.requires_runtime_shape_args = false;
  descriptor.entry_point = "mps_sdpa";
  descriptor.compile_options_key = "metal_vendor_descriptor";
  descriptor.abi_arg_count = 4;
  descriptor.abi_output_arg_count = 1;
  finalize_kernel_artifact_descriptor_identity(descriptor);
  return descriptor;
}

PipelineVendorAttentionArtifact materialize_fused_mpsgraph_sdpa_artifact(
    uint64_t stage_record_key, const PipelineVendorAttentionPlan &plan) {
  PipelineVendorAttentionArtifact artifact;
  artifact.descriptor =
      make_mpsgraph_sdpa_vendor_attention_artifact_descriptor(stage_record_key,
                                                             plan);
  artifact.payload =
      materialize_fused_mpsgraph_sdpa_payload(artifact.descriptor, plan);
  if (!artifact.valid()) {
    return {};
  }
  return artifact;
}

std::shared_ptr<const KernelArtifactPayload>
resolve_metal_payload(KernelArtifactDescriptor &descriptor,
                      const PlannedOperation &op) {
  if (descriptor.kernel.backend_domain != "metal" || !op.source_node) {
    return {};
  }
  if (descriptor.payload_kind == KernelArtifactPayloadKind::VendorDescriptor) {
    if (descriptor.kernel.kernel_id == kMetalMpsGemmVendorUnit) {
      return materialize_mps_gemm_payload(descriptor, op.source_node);
    }
    if (descriptor.kernel.kernel_id == kMetalMpsSoftmaxVendorUnit) {
      return materialize_mps_softmax_payload(descriptor, op.source_node);
    }
    if (descriptor.kernel.kernel_id == kMetalMpsPool2DVendorUnit) {
      return materialize_mps_pool2d_payload(descriptor, op.source_node);
    }
    if (descriptor.kernel.kernel_id == kMetalMpsResize2DVendorUnit) {
      return materialize_mps_resize2d_payload(descriptor, op.source_node);
    }
    if (descriptor.kernel.kernel_id == kMetalMpsGraphSdpaVendorUnit) {
      return materialize_mpsgraph_sdpa_payload(descriptor, op.source_node);
    }
    return {};
  }
  if (descriptor.payload_kind != KernelArtifactPayloadKind::MslSource) {
    return {};
  }

  if (descriptor.kernel.kernel_id == kMetalShapeOfMslKernelUnit) {
    return materialize_generated_msl_payload(
        descriptor, make_shapeof_msl_kernel_source_plan(op.source_node));
  }
  if (descriptor.kernel.kernel_id == kMetalRangeMslKernelUnit) {
    return materialize_generated_msl_payload(
        descriptor, make_range_msl_kernel_source_plan(op.source_node));
  }
  if (descriptor.kernel.kernel_id == kMetalTileMslKernelUnit) {
    return materialize_generated_msl_payload(
        descriptor, make_tile_msl_kernel_source_plan(op.source_node));
  }
  if (descriptor.kernel.kernel_id == kMetalConcatMslKernelUnit) {
    return materialize_generated_msl_payload(
        descriptor, make_concat_msl_kernel_source_plan(op.source_node));
  }
  if (descriptor.kernel.kernel_id == kMetalSplitMslKernelUnit) {
    return materialize_generated_msl_payload(
        descriptor, make_split_msl_kernel_source_plan(op.source_node));
  }
  if (descriptor.kernel.kernel_id == kMetalSliceMslKernelUnit) {
    return materialize_generated_msl_payload(
        descriptor,
        make_direct_static_slice_msl_kernel_source_plan(
            op.source_node, op.source_node->get_output_element_type(0)));
  }
  if (descriptor.kernel.kernel_id == kMetalTransposeF32MslKernelUnit) {
    return materialize_generated_msl_payload(
        descriptor, make_transpose_msl_kernel_source_plan(op.source_node));
  }
  if (descriptor.kernel.kernel_id == kMetalCausalSdpaMslKernelUnit) {
    return materialize_generated_msl_payload(
        descriptor, make_causal_sdpa_msl_kernel_source_plan(
                        op.source_node->get_output_element_type(0)));
  }
  if (descriptor.kernel.kernel_id == kMetalActivationMslKernelUnit) {
    return materialize_generated_msl_payload(
        descriptor, make_activation_msl_kernel_source_plan(op.source_node));
  }
  if (descriptor.kernel.kernel_id == kMetalEltwiseMslKernelUnit) {
    return materialize_generated_msl_payload(
        descriptor, make_eltwise_msl_kernel_source_plan(op.source_node));
  }
  if (descriptor.kernel.kernel_id == kMetalReductionF32MslKernelUnit ||
      descriptor.kernel.kernel_id == kMetalReductionLogicalBoolMslKernelUnit) {
    return materialize_generated_msl_payload(
        descriptor, make_reduction_msl_kernel_source_plan(op.source_node));
  }
  if (descriptor.kernel.kernel_id == kMetalSoftmaxF32MslKernelUnit ||
      descriptor.kernel.kernel_id == kMetalSoftmaxF16MslKernelUnit ||
      descriptor.kernel.kernel_id == kMetalLogSoftmaxF32MslKernelUnit ||
      descriptor.kernel.kernel_id == kMetalLogSoftmaxF16MslKernelUnit) {
    return materialize_generated_msl_payload(
        descriptor,
        make_softmax_runtime_params_msl_kernel_source_plan(op.source_node));
  }

  return {};
}

} // namespace

GfxMetalVendorPrimitiveArtifactPayload::GfxMetalVendorPrimitiveArtifactPayload(
    std::string kernel_id, std::string backend_domain, std::string entry_point,
    GfxAppleMpsVendorPrimitiveContract contract)
    : m_kernel_id(std::move(kernel_id)),
      m_backend_domain(std::move(backend_domain)),
      m_entry_point(std::move(entry_point)), m_contract(std::move(contract)) {}

KernelArtifactPayloadKind
GfxMetalVendorPrimitiveArtifactPayload::payload_kind() const noexcept {
  return KernelArtifactPayloadKind::VendorDescriptor;
}

std::string_view
GfxMetalVendorPrimitiveArtifactPayload::backend_domain() const noexcept {
  return m_backend_domain;
}

std::string_view
GfxMetalVendorPrimitiveArtifactPayload::source_id() const noexcept {
  return m_kernel_id;
}

std::string_view
GfxMetalVendorPrimitiveArtifactPayload::entry_point() const noexcept {
  return m_entry_point;
}

bool GfxMetalVendorPrimitiveArtifactPayload::valid() const noexcept {
  return !m_kernel_id.empty() && !m_backend_domain.empty() &&
         !m_entry_point.empty() && m_contract.valid &&
         m_contract.descriptor.kind != GfxAppleMpsVendorPrimitiveKind::None;
}

KernelArtifactPayloadResolver make_metal_kernel_artifact_payload_resolver() {
  return [](KernelArtifactDescriptor &descriptor, const PlannedOperation &op) {
    return resolve_metal_payload(descriptor, op);
  };
}

std::string_view metal_mpsgraph_sdpa_vendor_kernel_unit_id() noexcept {
  return kMetalMpsGraphSdpaVendorUnit;
}

PipelineVendorAttentionArtifactResolver
make_metal_vendor_attention_artifact_resolver() {
  return [](uint64_t stage_record_key, const PipelineVendorAttentionPlan &plan) {
    return materialize_fused_mpsgraph_sdpa_artifact(stage_record_key, plan);
  };
}

} // namespace compiler
} // namespace gfx_plugin
} // namespace ov
