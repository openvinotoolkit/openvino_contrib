// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/mpsrt_vendor_primitive_stage.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "backends/metal/compiler/metal_kernel_artifacts.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_context.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_request.hpp"
#include "common/constant_tensor_evaluator.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "openvino/core/except.hpp"
#include "backends/metal/runtime/mpsrt/gfx_mpsrt_model.hpp"
#include "backends/metal/common/mpsrt/gfx_mpsrt_program.hpp"
#include "runtime/gfx_profiler.hpp"
#include "runtime/gpu_buffer_manager.hpp"
#include "runtime/gpu_tensor.hpp"

namespace ov {
namespace gfx_plugin {
namespace {

namespace runtime_mpsrt = ::ov::gfx_plugin::mpsrt;

const compiler::GfxMetalVendorPrimitiveArtifactPayload*
vendor_payload_from_descriptor(
    const RuntimeStageExecutableDescriptor& descriptor) noexcept {
  if (descriptor.payload_kind !=
          compiler::KernelArtifactPayloadKind::VendorDescriptor ||
      descriptor.origin != compiler::KernelArtifactOrigin::VendorPrimitive ||
      descriptor.backend_domain != "metal" || !descriptor.payload) {
    return nullptr;
  }
  return dynamic_cast<const compiler::GfxMetalVendorPrimitiveArtifactPayload*>(
      descriptor.payload.get());
}

GfxMpsrtStageKind stage_kind_from_vendor_kind(
    GfxAppleMpsVendorPrimitiveKind kind,
    const std::shared_ptr<const ov::Node>& node) noexcept {
  switch (kind) {
    case GfxAppleMpsVendorPrimitiveKind::Gemm:
      return GfxMpsrtStageKind::MPSGemm;
    case GfxAppleMpsVendorPrimitiveKind::Conv2D:
      return node && node->get_type_name() == std::string("GroupConvolution")
                 ? GfxMpsrtStageKind::MPSGroupConv2D
                 : GfxMpsrtStageKind::MPSConv2D;
    case GfxAppleMpsVendorPrimitiveKind::Pool2D:
      return GfxMpsrtStageKind::MPSPool2D;
    case GfxAppleMpsVendorPrimitiveKind::Resize2D:
      return GfxMpsrtStageKind::MPSResize2D;
    case GfxAppleMpsVendorPrimitiveKind::Softmax:
      return GfxMpsrtStageKind::MPSSoftmax;
    case GfxAppleMpsVendorPrimitiveKind::TopK:
      return GfxMpsrtStageKind::MPSTopK;
    case GfxAppleMpsVendorPrimitiveKind::Sdpa:
      return GfxMpsrtStageKind::MPSSdpa;
    case GfxAppleMpsVendorPrimitiveKind::None:
    default:
      return GfxMpsrtStageKind::Unknown;
  }
}

GfxKernelStageFamily stage_family_from_vendor_kind(
    GfxAppleMpsVendorPrimitiveKind kind,
    const std::shared_ptr<const ov::Node>& node) noexcept {
  switch (kind) {
    case GfxAppleMpsVendorPrimitiveKind::Gemm:
      return GfxKernelStageFamily::Gemm;
    case GfxAppleMpsVendorPrimitiveKind::Conv2D:
      return node && node->get_type_name() == std::string("GroupConvolution")
                 ? GfxKernelStageFamily::GroupConvolution
                 : GfxKernelStageFamily::Convolution;
    case GfxAppleMpsVendorPrimitiveKind::Pool2D:
      return GfxKernelStageFamily::Pooling;
    case GfxAppleMpsVendorPrimitiveKind::Resize2D:
      return GfxKernelStageFamily::Resize;
    case GfxAppleMpsVendorPrimitiveKind::Softmax:
      return GfxKernelStageFamily::Softmax;
    case GfxAppleMpsVendorPrimitiveKind::TopK:
      return GfxKernelStageFamily::TopK;
    case GfxAppleMpsVendorPrimitiveKind::Sdpa:
      return GfxKernelStageFamily::AttentionSoftmax;
    case GfxAppleMpsVendorPrimitiveKind::None:
    default:
      return GfxKernelStageFamily::Unknown;
  }
}

GfxKernelStorageKind kernel_storage_from_mpsrt_storage(
    GfxMpsrtStorage storage) noexcept {
  switch (storage) {
    case GfxMpsrtStorage::Buffer:
      return GfxKernelStorageKind::Buffer;
    case GfxMpsrtStorage::Image:
      return GfxKernelStorageKind::Image;
    case GfxMpsrtStorage::Matrix:
      return GfxKernelStorageKind::Matrix;
    case GfxMpsrtStorage::NDArray:
      return GfxKernelStorageKind::NDArray;
    case GfxMpsrtStorage::Alias:
      return GfxKernelStorageKind::Alias;
    case GfxMpsrtStorage::Unknown:
    default:
      return GfxKernelStorageKind::Unknown;
  }
}

GfxMpsrtStorage first_storage_or_unknown(
    const std::vector<GfxMpsrtTensorDesc>& descs) noexcept {
  if (descs.empty()) {
    return GfxMpsrtStorage::Unknown;
  }
  return descs.front().storage;
}

void apply_vendor_descriptor_to_stage(
    const GfxAppleMpsVendorPrimitiveDescriptor& vendor,
    GfxMpsrtStageDesc& stage) {
  switch (vendor.kind) {
    case GfxAppleMpsVendorPrimitiveKind::Gemm:
      stage.gemm_desc = vendor.gemm;
      break;
    case GfxAppleMpsVendorPrimitiveKind::Conv2D:
      stage.conv2d_desc = vendor.conv2d;
      break;
    case GfxAppleMpsVendorPrimitiveKind::Pool2D:
      stage.pool2d_desc = vendor.pool2d;
      break;
    case GfxAppleMpsVendorPrimitiveKind::Resize2D:
      stage.resize2d_desc = vendor.resize2d;
      break;
    case GfxAppleMpsVendorPrimitiveKind::Softmax:
      stage.softmax_desc = vendor.softmax;
      break;
    case GfxAppleMpsVendorPrimitiveKind::TopK:
      stage.topk_desc = vendor.topk;
      break;
    case GfxAppleMpsVendorPrimitiveKind::Sdpa:
      stage.sdpa_desc = vendor.sdpa;
      break;
    case GfxAppleMpsVendorPrimitiveKind::None:
    default:
      break;
  }
}

GfxMpsrtStageDesc make_stage_desc_from_contract(
    const RuntimeStageExecutableDescriptor& descriptor,
    const std::shared_ptr<const ov::Node>& node,
    const GfxAppleMpsVendorPrimitiveContract& contract) {
  GfxMpsrtStageDesc stage{};
  stage.kind = stage_kind_from_vendor_kind(contract.descriptor.kind, node);
  stage.domain = GfxStageBackendDomain::AppleMps;
  stage.input_storage = first_storage_or_unknown(contract.input_descs);
  stage.output_storage = first_storage_or_unknown(contract.output_descs);
  if (stage.output_storage == GfxMpsrtStorage::Unknown) {
    stage.output_storage = stage.input_storage;
  }
  stage.layout = gfx_mpsrt_stage_layout_for_storage(stage.output_storage);
  stage.kernel_name = descriptor.entry_point.empty()
                          ? std::string(gfx_mpsrt_stage_kind_name(stage.kind))
                          : descriptor.entry_point;
  const auto family =
      stage_family_from_vendor_kind(contract.descriptor.kind, node);
  const auto storage = kernel_storage_from_mpsrt_storage(stage.output_storage);
  if (family != GfxKernelStageFamily::Unknown &&
      storage != GfxKernelStorageKind::Unknown) {
    const std::string specialization =
        descriptor.kernel_id.empty() ? stage.kernel_name : descriptor.kernel_id;
    stage.stage_manifest = make_gfx_vendor_stage_manifest(
        family, GfxKernelBackendDomain::AppleMps, storage, specialization);
  }
  apply_vendor_descriptor_to_stage(contract.descriptor, stage);
  return stage;
}

std::vector<GfxMpsrtValue> sequential_values(size_t count,
                                             GfxMpsrtValue first = 0) {
  std::vector<GfxMpsrtValue> values;
  values.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    values.push_back(first + static_cast<GfxMpsrtValue>(i));
  }
  return values;
}

runtime_mpsrt::MpsrtModel make_mpsrt_model_from_contract(
    const RuntimeStageExecutableDescriptor& descriptor,
    const std::shared_ptr<const ov::Node>& node,
    const GfxAppleMpsVendorPrimitiveContract& contract) {
  OPENVINO_ASSERT(contract.valid,
                  "GFX Metal MPSRT: vendor primitive contract is invalid");
  OPENVINO_ASSERT(contract.external_buffer_abi.valid &&
                      contract.external_buffer_abi.has_buffer_count &&
                      contract.external_buffer_abi.has_output_buffer_count,
                  "GFX Metal MPSRT: vendor primitive requires exact external "
                  "buffer ABI");

  GfxMpsrtProgram program{};
  program.record_key = descriptor.manifest_ref.empty()
                           ? descriptor.artifact_key
                           : descriptor.manifest_ref;
  program.inputs = contract.input_descs;
  program.external_buffer_abi = contract.external_buffer_abi;

  GfxMpsrtBuilderStageSpec stage_spec{};
  stage_spec.stage = make_stage_desc_from_contract(descriptor, node, contract);
  stage_spec.inputs = sequential_values(contract.input_descs.size());
  stage_spec.outputs =
      sequential_values(contract.output_descs.size(),
                        static_cast<GfxMpsrtValue>(contract.input_descs.size()));
  stage_spec.output_descs = contract.output_descs;
  program.stages.push_back(std::move(stage_spec));
  program.output_values = program.stages.front().outputs;

  GfxMpsrtBuilderPlan builder_plan{};
  OPENVINO_ASSERT(gfx_mpsrt_build_builder_plan_from_program(program,
                                                            builder_plan),
                  "GFX Metal MPSRT: failed to build vendor primitive builder "
                  "plan for ",
                  descriptor.kernel_id);

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  OPENVINO_ASSERT(runtime_mpsrt::build_mpsrt_model_from_builder_plan(
                      builder_plan, model, &error),
                  "GFX Metal MPSRT: failed to build runtime model for ",
                  descriptor.kernel_id, ": ", error);
  OPENVINO_ASSERT(runtime_mpsrt::adapt_mpsrt_model_to_external_buffer_abi(
                      model, contract.external_buffer_abi.buffer_count,
                      contract.external_buffer_abi.output_buffer_count, &error),
                  "GFX Metal MPSRT: failed to adapt vendor primitive ABI for ",
                  descriptor.kernel_id, ": ", error);
  return model;
}

ov::Shape shape_from_mpsrt_desc(const GfxMpsrtTensorDesc& desc) {
  ov::Shape shape;
  shape.reserve(desc.rank);
  for (uint32_t i = 0; i < desc.rank && i < desc.dims.size(); ++i) {
    shape.push_back(static_cast<size_t>(desc.dims[i]));
  }
  return shape;
}

ov::element::Type element_type_from_mpsrt_dtype(uint32_t dtype) {
  switch (static_cast<GfxMpsrtDType>(dtype)) {
    case GfxMpsrtDType::F16:
      return ov::element::f16;
    case GfxMpsrtDType::F32:
      return ov::element::f32;
    case GfxMpsrtDType::I8:
      return ov::element::i8;
    case GfxMpsrtDType::U8:
      return ov::element::u8;
    case GfxMpsrtDType::I16:
      return ov::element::i16;
    case GfxMpsrtDType::U16:
      return ov::element::u16;
    case GfxMpsrtDType::I32:
      return ov::element::i32;
    case GfxMpsrtDType::U32:
      return ov::element::u32;
    case GfxMpsrtDType::I64:
      return ov::element::i64;
    case GfxMpsrtDType::U64:
      return ov::element::u64;
    case GfxMpsrtDType::Bool:
      return ov::element::boolean;
    case GfxMpsrtDType::Unknown:
    default:
      return ov::element::dynamic;
  }
}

void materialize_output_contracts(
    const std::vector<GfxMpsrtTensorDesc>& output_descs,
    const std::vector<GpuTensor*>& outputs) {
  const size_t count = std::min(output_descs.size(), outputs.size());
  for (size_t i = 0; i < count; ++i) {
    auto* output = outputs[i];
    if (!output) {
      continue;
    }
    if (output->shape.empty()) {
      output->shape = shape_from_mpsrt_desc(output_descs[i]);
    }
    if (output->expected_type == ov::element::dynamic) {
      output->expected_type =
          element_type_from_mpsrt_dtype(
              static_cast<uint32_t>(output_descs[i].dtype));
    }
  }
}

std::vector<GpuTensor*> active_outputs(std::vector<GpuTensor*> outputs,
                                       GpuTensor* output) {
  if (outputs.empty() && output) {
    outputs.push_back(output);
  }
  return outputs;
}

metal::mpsrt::MpsrtBoundBuffer tensor_bound_buffer(const GpuTensor& tensor) {
  return {tensor.buf.buffer, tensor.buf.offset};
}

struct VendorConstInputBuffers {
  std::vector<GpuTensor> buffers;
  std::vector<bool> present;
};

std::string make_vendor_const_input_key(const std::string& kernel_id,
                                        const ov::Node& node,
                                        size_t input_index,
                                        const ov::Tensor& tensor) {
  std::ostringstream key;
  key << "metal/vendor/" << kernel_id << "/const/"
      << node.get_friendly_name() << "/" << input_index << "/"
      << tensor.get_element_type().get_type_name() << "/"
      << tensor.get_byte_size() << "/"
      << gfx_hash_bytes(tensor.data(), tensor.get_byte_size());
  return key.str();
}

void prepare_vendor_const_inputs(const std::shared_ptr<const ov::Node>& node,
                                 GpuBufferManager* buffer_manager,
                                 const std::string& kernel_id,
                                 VendorConstInputBuffers& const_inputs) {
  if (!node) {
    return;
  }
  const size_t input_count = node->get_input_size();
  if (const_inputs.buffers.size() < input_count) {
    const_inputs.buffers.resize(input_count);
    const_inputs.present.assign(input_count, false);
  }

  bool const_cache_checked = false;
  for (size_t input_index = 0; input_index < input_count; ++input_index) {
    if (const_inputs.present[input_index] &&
        const_inputs.buffers[input_index].buf.valid()) {
      continue;
    }
    auto tensor =
        gfx_evaluate_constant_source_tensor(node->input_value(input_index));
    if (!tensor.has_value() || tensor->get_byte_size() == 0) {
      continue;
    }
    if (!const_cache_checked) {
      OPENVINO_ASSERT(buffer_manager,
                      "GFX Metal MPSRT: const buffer manager is required for "
                      "vendor primitive ",
                      kernel_id);
      OPENVINO_ASSERT(buffer_manager->supports_const_cache(),
                      "GFX Metal MPSRT: const cache is required for vendor "
                      "primitive ",
                      kernel_id);
      const_cache_checked = true;
    }

    const auto key =
        make_vendor_const_input_key(kernel_id, *node, input_index, *tensor);
    GpuBuffer buffer = buffer_manager->wrap_const(
        key, tensor->data(), tensor->get_byte_size(),
        tensor->get_element_type());
    OPENVINO_ASSERT(buffer.valid(),
                    "GFX Metal MPSRT: failed to materialize const input ",
                    input_index, " for ", kernel_id);
    buffer.owned = false;

    auto& const_tensor = const_inputs.buffers[input_index];
    const_tensor.buf = buffer;
    const_tensor.shape = tensor->get_shape();
    const_tensor.expected_type = tensor->get_element_type();
    const_tensor.prefer_private = false;
    const_inputs.present[input_index] = true;
  }
}

GpuTensor* resolve_vendor_input_tensor(
    const std::vector<GpuTensor*>& inputs,
    const VendorConstInputBuffers& const_inputs,
    size_t input_index) {
  if (input_index < inputs.size() && inputs[input_index] &&
      inputs[input_index]->buf.valid()) {
    return inputs[input_index];
  }
  if (input_index < const_inputs.buffers.size() &&
      input_index < const_inputs.present.size() &&
      const_inputs.present[input_index] &&
      const_inputs.buffers[input_index].buf.valid()) {
    return const_cast<GpuTensor*>(&const_inputs.buffers[input_index]);
  }
  return nullptr;
}

std::vector<metal::mpsrt::MpsrtBoundBuffer>
make_external_buffers_for_roles(
    const GfxAppleMpsVendorPrimitiveContract& contract,
    const std::vector<GpuTensor*>& inputs,
    const VendorConstInputBuffers& const_inputs,
    const std::vector<GpuTensor*>& outputs,
    const std::string& kernel_id) {
  OPENVINO_ASSERT(contract.external_buffer_abi.valid &&
                      contract.external_buffer_abi.has_buffer_roles,
                  "GFX Metal MPSRT: vendor primitive is missing external "
                  "buffer roles for ",
                  kernel_id);
  std::vector<metal::mpsrt::MpsrtBoundBuffer> buffers;
  buffers.reserve(contract.external_buffer_abi.buffer_roles.size());
  size_t input_index = 0;
  size_t output_index = 0;
  for (const auto role : contract.external_buffer_abi.buffer_roles) {
    switch (role) {
      case GfxMpsrtExternalBufferRole::TensorInput:
      case GfxMpsrtExternalBufferRole::ConstBuffer:
        if (auto* input =
                resolve_vendor_input_tensor(inputs, const_inputs, input_index)) {
          buffers.push_back(tensor_bound_buffer(*input));
          ++input_index;
          break;
        }
        OPENVINO_ASSERT(false,
                        "GFX Metal MPSRT: missing input binding ",
                        input_index, " for ", kernel_id);
        break;
      case GfxMpsrtExternalBufferRole::TensorOutput:
        OPENVINO_ASSERT(output_index < outputs.size() && outputs[output_index],
                        "GFX Metal MPSRT: missing output binding ",
                        output_index, " for ", kernel_id);
        buffers.push_back(tensor_bound_buffer(*outputs[output_index]));
        ++output_index;
        break;
      case GfxMpsrtExternalBufferRole::RuntimeParams:
      case GfxMpsrtExternalBufferRole::Metadata:
        buffers.push_back({});
        break;
      case GfxMpsrtExternalBufferRole::Unknown:
      default:
        OPENVINO_THROW("GFX Metal MPSRT: unknown external buffer role for ",
                       kernel_id);
    }
  }
  return buffers;
}

class MpsrtVendorPrimitiveStage final : public GpuStage {
public:
  MpsrtVendorPrimitiveStage(
      const std::shared_ptr<const ov::Node>& node,
      MetalDeviceHandle device,
      MetalCommandQueueHandle queue,
      const RuntimeStageExecutableDescriptor& descriptor)
      : m_node(node),
        m_device(device),
        m_queue(queue),
        m_descriptor(descriptor),
        m_name(node ? node->get_friendly_name() : descriptor.kernel_id) {
    const auto* payload = vendor_payload_from_descriptor(descriptor);
    OPENVINO_ASSERT(payload && payload->valid(),
                    "GFX Metal MPSRT: invalid vendor primitive descriptor for ",
                    descriptor.kernel_id);
    m_contract = payload->contract();
    m_type = "MpsrtVendorPrimitive";
  }

  void init(GpuBufferManager* buffer_manager) override {
    m_buffer_manager = buffer_manager;
  }

  void compile(GpuBufferManager* buffer_manager) override {
    if (buffer_manager) {
      m_buffer_manager = buffer_manager;
    }
    OPENVINO_ASSERT(m_device,
                    "GFX Metal MPSRT: Metal device handle is null for ",
                    m_descriptor.kernel_id);
    prepare_vendor_const_inputs(m_node, m_buffer_manager,
                                m_descriptor.kernel_id, m_const_inputs);
    m_model = make_mpsrt_model_from_contract(m_descriptor, m_node, m_contract);
    auto context =
        std::make_shared<metal::mpsrt::MpsrtContext>((id<MTLDevice>)m_device);
    auto prepared = std::make_shared<metal::mpsrt::MpsrtPreparedModel>();
    std::string error;
    OPENVINO_ASSERT(context->prepare_model(m_model, "", *prepared, &error),
                    error);
    m_context = std::move(context);
    m_prepared = std::move(prepared);
  }

  void execute(GpuCommandBufferHandle command_buffer) override {
    OPENVINO_ASSERT(m_context && m_prepared,
                    "GFX Metal MPSRT: vendor primitive stage is not compiled");
    auto outputs = active_outputs(m_outputs, m_output);
    OPENVINO_ASSERT(outputs.size() == m_contract.output_descs.size(),
                    "GFX Metal MPSRT: output binding count mismatch for ",
                    m_descriptor.kernel_id);
    materialize_output_contracts(m_contract.output_descs, outputs);

    const auto external_buffers = make_external_buffers_for_roles(
        m_contract, m_inputs, m_const_inputs, outputs, m_descriptor.kernel_id);
    metal::mpsrt::MpsrtRequest request;
    metal::mpsrt::MpsrtRequestBindingSet binding_set;
    KernelExecutionHooks hooks;
    KernelExecutionHooks* hooks_ptr = prepare_hooks(hooks);
    std::string error;
    OPENVINO_ASSERT(request.build_binding_set_from_external_buffers(
                        m_model, external_buffers, m_prepared.get(),
                        binding_set, hooks_ptr, &error),
                    error);

    std::vector<KernelDispatch> dispatches(m_model.stages.size());
    metal::mpsrt::MpsrtModelEncodeResult result;
    OPENVINO_ASSERT(request.encode_prepared_model_with_binding_set(
                        command_buffer, *m_context, m_model, *m_prepared,
                        dispatches, binding_set, hooks_ptr, &result, &error),
                    error);
  }

  void set_inputs(const std::vector<GpuTensor*>& inputs) override {
    m_inputs = inputs;
  }

  void set_output(GpuTensor* output) override {
    m_output = output;
    m_outputs.clear();
    if (output) {
      m_outputs.push_back(output);
    }
  }

  void set_output_refs(const std::vector<GpuTensor*>& outputs) override {
    m_outputs = outputs;
    m_output = outputs.empty() ? nullptr : outputs.front();
  }

  void set_outputs(
      const std::vector<std::unique_ptr<GpuTensor>>& outputs) override {
    m_outputs.clear();
    m_outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
      m_outputs.push_back(output.get());
    }
    m_output = m_outputs.empty() ? nullptr : m_outputs.front();
  }

  void enable_profiling(bool enable) override {
    m_profiling_enabled = enable;
  }

  void set_profiler(void* profiler,
                    uint32_t node_id,
                    const std::string& node_name,
                    const std::string& node_type) override {
    m_profiler = profiler;
    m_profile_node_id = node_id;
    m_profile_node_name = node_name;
    m_profile_node_type = node_type;
  }

  const std::string& name() const override {
    return m_name;
  }

  const std::string& type() const override {
    return m_type;
  }

  std::unique_ptr<GpuStage> clone() const override {
    auto stage = std::make_unique<MpsrtVendorPrimitiveStage>(
        m_node, m_device, m_queue, m_descriptor);
    stage->m_contract = m_contract;
    stage->m_model = m_model;
    stage->m_context = m_context;
    stage->m_prepared = m_prepared;
    stage->m_const_inputs = m_const_inputs;
    stage->m_buffer_manager = m_buffer_manager;
    stage->m_profiling_enabled = m_profiling_enabled;
    stage->m_profiler = m_profiler;
    stage->m_profile_node_id = m_profile_node_id;
    stage->m_profile_node_name = m_profile_node_name;
    stage->m_profile_node_type = m_profile_node_type;
    return stage;
  }

private:
  KernelExecutionHooks* prepare_hooks(KernelExecutionHooks& hooks) {
    if (!m_profiling_enabled || !m_profiler) {
      return nullptr;
    }
    auto* profiler = static_cast<GfxProfiler*>(m_profiler);
    hooks.stage_name =
        m_profile_node_name.empty() ? m_name : m_profile_node_name;
    hooks.stage_type =
        m_profile_node_type.empty() ? m_type : m_profile_node_type;
    hooks.on_counter = [profiler](std::string_view name, uint64_t delta) {
      profiler->increment_counter(name, delta);
    };
    hooks.on_segment =
        [profiler](std::string_view phase, std::string_view name,
                   std::chrono::microseconds cpu_us, uint64_t gpu_us,
                   uint32_t dispatches, uint64_t bytes_in, uint64_t bytes_out,
                   uint64_t macs_est, uint64_t flops_est,
                   int64_t inflight_slot, uint64_t queue_id,
                   uint64_t cmd_buffer_id) {
          profiler->record_segment(phase, name, cpu_us, gpu_us, dispatches,
                                   bytes_in, bytes_out, macs_est, flops_est,
                                   inflight_slot, queue_id, cmd_buffer_id);
        };
    return &hooks;
  }

  std::shared_ptr<const ov::Node> m_node;
  MetalDeviceHandle m_device = nullptr;
  [[maybe_unused]] MetalCommandQueueHandle m_queue = nullptr;
  RuntimeStageExecutableDescriptor m_descriptor;
  GfxAppleMpsVendorPrimitiveContract m_contract;
  GpuBufferManager* m_buffer_manager = nullptr;
  std::string m_name;
  std::string m_type;
  std::vector<GpuTensor*> m_inputs;
  std::vector<GpuTensor*> m_outputs;
  GpuTensor* m_output = nullptr;
  VendorConstInputBuffers m_const_inputs;
  runtime_mpsrt::MpsrtModel m_model;
  std::shared_ptr<metal::mpsrt::MpsrtContext> m_context;
  std::shared_ptr<metal::mpsrt::MpsrtPreparedModel> m_prepared;
  bool m_profiling_enabled = false;
  void* m_profiler = nullptr;
  uint32_t m_profile_node_id = 0;
  std::string m_profile_node_name;
  std::string m_profile_node_type;
};

}  // namespace

bool is_metal_mpsrt_vendor_primitive_descriptor(
    const RuntimeStageExecutableDescriptor& descriptor) noexcept {
  const auto* payload = vendor_payload_from_descriptor(descriptor);
  return payload && payload->valid();
}

std::unique_ptr<GpuStage> create_metal_mpsrt_vendor_primitive_stage(
    const std::shared_ptr<const ov::Node>& node,
    MetalDeviceHandle device,
    MetalCommandQueueHandle queue,
    const RuntimeStageExecutableDescriptor& descriptor) {
  return std::make_unique<MpsrtVendorPrimitiveStage>(node, device, queue,
                                                    descriptor);
}

}  // namespace gfx_plugin
}  // namespace ov
