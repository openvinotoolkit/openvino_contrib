// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/codegen/metal_codegen_backend.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "backends/metal/codegen/metal_compiler.hpp"
#include "backends/metal/runtime/metal_command_encoder.hpp"
#include "backends/metal/runtime/mpsrt/mpsrt_request.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_cache.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/mlir_passes.hpp"
#include "openvino/core/except.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_mpsrt_storage_bridge.hpp"

#include "llvm/Support/raw_ostream.h"

namespace ov {
namespace gfx_plugin {

namespace runtime_mpsrt = ::ov::gfx_plugin::mpsrt;

class MetalBindingSchema final {
public:
  explicit MetalBindingSchema(uint32_t arg_count) : m_arg_count(arg_count) {}

  uint32_t arg_count() const { return m_arg_count; }

private:
  uint32_t m_arg_count = 0;
};

class MetalDeviceReuseContext final {
public:
  explicit MetalDeviceReuseContext(MetalDeviceHandle device)
      : m_device(device) {}

  std::shared_ptr<MetalBindingSchema>
  acquire_binding_schema(uint32_t arg_count) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (auto it = m_binding_schemas.find(arg_count);
        it != m_binding_schemas.end()) {
      if (auto schema = it->second.lock()) {
        return schema;
      }
    }
    auto schema = std::make_shared<MetalBindingSchema>(arg_count);
    m_binding_schemas[arg_count] = schema;
    return schema;
  }

private:
  MetalDeviceHandle m_device = nullptr;
  std::mutex m_mutex;
  std::unordered_map<uint32_t, std::weak_ptr<MetalBindingSchema>>
      m_binding_schemas;
};

class MetalDeviceReuseRegistry final {
public:
  static MetalDeviceReuseRegistry &instance() {
    static MetalDeviceReuseRegistry registry;
    return registry;
  }

  std::shared_ptr<MetalDeviceReuseContext> acquire(MetalDeviceHandle device) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto key = reinterpret_cast<uintptr_t>(device);
    if (auto it = m_contexts.find(key); it != m_contexts.end()) {
      if (auto context = it->second.lock()) {
        return context;
      }
    }
    auto context = std::make_shared<MetalDeviceReuseContext>(device);
    m_contexts[key] = context;
    return context;
  }

private:
  std::mutex m_mutex;
  std::unordered_map<uintptr_t, std::weak_ptr<MetalDeviceReuseContext>>
      m_contexts;
};

namespace {
inline id<MTLBuffer> to_mtl(const GpuBuffer &buf) {
  return (__bridge id<MTLBuffer>)buf.buffer;
}

bool source_has_exact_mpsrt_external_buffer_abi(
    const KernelSource &source, GfxMpsrtExternalBufferAbiPlan *out = nullptr) {
  GfxMpsrtExternalBufferAbiPlan abi{};
  if (source.module) {
    abi = read_module_mpsrt_external_buffer_abi(source.module);
  }
  const bool exact = abi.valid && abi.has_buffer_count &&
                     abi.has_output_buffer_count && abi.has_buffer_roles;
  if (out) {
    *out = std::move(abi);
  }
  return exact;
}

bool source_has_typed_mpsrt_program(const KernelSource &source) {
  return source.module && module_has_mpsrt_ops_program(source.module);
}

struct MetalRuntimeSignature {
  uint32_t arg_count = 0;
  uint32_t output_arg_count = 0;

  bool valid() const { return arg_count != 0 && output_arg_count != 0; }
};

std::optional<MetalRuntimeSignature>
exact_metal_kernel_runtime_signature_from_stage_manifest(
    const GfxKernelStageManifest &manifest) {
  if (!manifest.valid || !manifest.custom_kernel.valid ||
      !manifest.custom_kernel.external_buffer_abi.valid) {
    return std::nullopt;
  }

  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      manifest.custom_kernel.external_buffer_abi);
  if (roles.empty()) {
    return std::nullopt;
  }

  MetalRuntimeSignature signature{};
  for (const auto role : roles) {
    if (is_gfx_kernel_buffer_role(role) || is_gfx_kernel_scalar_role(role)) {
      ++signature.arg_count;
    }
    if (is_gfx_kernel_output_role(role)) {
      ++signature.output_arg_count;
    }
  }
  if (!signature.valid()) {
    return std::nullopt;
  }
  return signature;
}

std::optional<MetalRuntimeSignature>
exact_metal_kernel_runtime_signature_from_typed_program(
    mlir::ModuleOp module, const std::string &entry_point) {
  if (!module || !module_has_mpsrt_ops_program(module)) {
    return std::nullopt;
  }

  GfxMpsrtProgram program{};
  if (!read_module_mpsrt_program(module, program) || !program.valid) {
    OPENVINO_THROW("GFX Metal MPSRT: invalid typed gfx_mpsrt_ops program");
  }

  std::optional<MetalRuntimeSignature> fallback_signature;
  bool saw_msl_stage = false;
  for (const auto &stage_spec : program.stages) {
    const auto &stage = stage_spec.stage;
    if (stage.kind != GfxMpsrtStageKind::MSLDispatch &&
        stage.domain != GfxStageBackendDomain::AppleMsl) {
      continue;
    }
    saw_msl_stage = true;

    const auto signature =
        exact_metal_kernel_runtime_signature_from_stage_manifest(
            stage.stage_manifest);
    if (!signature) {
      OPENVINO_THROW(
          "GFX Metal MPSRT: typed MSL dispatch is missing exact custom-kernel "
          "manifest ABI");
    }

    const auto &manifest_entry =
        stage.stage_manifest.custom_kernel.entry_point;
    if (!entry_point.empty()) {
      if (manifest_entry == entry_point || stage.kernel_name == entry_point) {
        return signature;
      }
      if (!manifest_entry.empty() || !stage.kernel_name.empty()) {
        continue;
      }
    }

    if (!fallback_signature) {
      fallback_signature = signature;
    }
  }

  if (saw_msl_stage && !entry_point.empty()) {
    OPENVINO_THROW("GFX Metal MPSRT: typed MSL dispatch entry point not found: ",
                   entry_point);
  }
  return fallback_signature;
}

std::optional<MetalRuntimeSignature>
exact_metal_kernel_runtime_signature_from_module_stage_manifest(
    mlir::ModuleOp module) {
  GfxKernelStageManifest manifest{};
  if (!module ||
      !detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest)) {
    return std::nullopt;
  }
  return exact_metal_kernel_runtime_signature_from_stage_manifest(manifest);
}

bool source_has_custom_kernel_stage_manifest(const KernelSource &source) {
  GfxKernelStageManifest manifest{};
  return source.module &&
         detail::gfx_mpsrt_read_stage_manifest_attrs(source.module, manifest) &&
         manifest.valid &&
         manifest.execution_kind == GfxKernelExecutionKind::CustomKernel &&
         manifest.custom_kernel.valid;
}

bool source_allows_no_manifest_msl_signature_fallback(
    const KernelSource &source) {
  if (!source.module) {
    return true;
  }
  if (source_has_typed_mpsrt_program(source) ||
      source_has_custom_kernel_stage_manifest(source)) {
    return false;
  }
  return !source_has_exact_mpsrt_external_buffer_abi(source);
}

std::optional<MetalRuntimeSignature>
exact_metal_runtime_signature_from_source(const KernelSource &source) {
  if (auto typed_signature =
          exact_metal_kernel_runtime_signature_from_typed_program(
              source.module, source.entry_point)) {
    return typed_signature;
  }

  if (auto manifest_signature =
          exact_metal_kernel_runtime_signature_from_module_stage_manifest(
              source.module)) {
    return manifest_signature;
  }

  GfxMpsrtExternalBufferAbiPlan external_abi{};
  if (!source_has_exact_mpsrt_external_buffer_abi(source, &external_abi)) {
    return std::nullopt;
  }

  if (source.module && module_has_mpsrt_ops_program(source.module)) {
    return MetalRuntimeSignature{external_abi.buffer_count,
                                 external_abi.output_buffer_count};
  }

  return MetalRuntimeSignature{external_abi.buffer_count,
                               external_abi.output_buffer_count};
}

std::string make_resolved_msl_cache_key(const KernelSource &source) {
  if (!source.module) {
    return {};
  }
  std::string module_text;
  llvm::raw_string_ostream os(module_text);
  auto module = source.module;
  module.print(os);
  os.flush();

  uint32_t arg_count = source.signature.arg_count;
  uint32_t output_arg_count = source.signature.output_arg_count;
  if (auto exact_signature = exact_metal_runtime_signature_from_source(source)) {
    arg_count = exact_signature->arg_count;
    output_arg_count = exact_signature->output_arg_count;
  }

  std::ostringstream key;
  key << source.entry_point << '\n'
      << arg_count << ':' << output_arg_count << '\n'
      << module_text;
  return key.str();
}

uint32_t resolve_kernel_output_arg_count(const KernelSource &source) {
  if (auto exact_signature = exact_metal_runtime_signature_from_source(source)) {
    return exact_signature->output_arg_count;
  }
  if (source_has_custom_kernel_stage_manifest(source)) {
    OPENVINO_THROW(
        "GFX Metal: custom-kernel stage manifest is missing exact runtime "
        "output ABI for entry ",
        source.entry_point);
  }
  if (source.module) {
    const auto abi = read_module_mpsrt_external_buffer_abi(source.module);
    if (abi.valid && abi.has_output_buffer_count) {
      return abi.output_buffer_count;
    }
    if (module_has_mpsrt_ops_program(source.module)) {
      OPENVINO_THROW(
          "GFX Metal MPSRT: typed gfx_mpsrt_ops program is missing exact "
          "external output buffer ABI");
    }
  }
  if (source.signature.output_arg_count != 0) {
    return source.signature.output_arg_count;
  }
  return 0;
}

uint32_t resolve_kernel_arg_count(const KernelSource &source) {
  if (auto exact_signature = exact_metal_runtime_signature_from_source(source)) {
    return exact_signature->arg_count;
  }
  if (source_has_custom_kernel_stage_manifest(source)) {
    OPENVINO_THROW(
        "GFX Metal: custom-kernel stage manifest is missing exact runtime ABI "
        "for entry ",
        source.entry_point);
  }
  return source.signature.arg_count;
}

uint32_t resolve_no_manifest_msl_source_arg_count(
    const KernelSource &source, const std::string &msl,
    uint32_t fallback_arg_count) {
  if (!source_allows_no_manifest_msl_signature_fallback(source)) {
    return 0;
  }
  uint32_t resolved_arg_count = fallback_arg_count;
  if (!msl.empty()) {
    const auto source_msl_arg_count =
        infer_msl_buffer_arg_count_from_source(msl);
    if (source_msl_arg_count != 0) {
      resolved_arg_count = resolved_arg_count == 0
                               ? source_msl_arg_count
                               : std::max(resolved_arg_count,
                                          source_msl_arg_count);
    }
  }
  return resolved_arg_count;
}

uint32_t resolve_initial_compile_source_arg_count(const KernelSource &source) {
  const uint32_t source_arg_count = resolve_kernel_arg_count(source);
  if (!source_allows_no_manifest_msl_signature_fallback(source)) {
    return source_arg_count;
  }
  return resolve_no_manifest_msl_source_arg_count(source, source.msl_source,
                                                 source_arg_count);
}

uint32_t resolve_metal_runtime_arg_count(
    const KernelSource &source, const std::string &msl,
    uint32_t source_arg_count,
    const std::shared_ptr<const runtime_mpsrt::MpsrtModel> &mpsrt_model) {
  if (mpsrt_model) {
    const auto external_abi_count = static_cast<uint32_t>(
        runtime_mpsrt::mpsrt_model_external_buffer_abi_count(*mpsrt_model));
    if (external_abi_count != 0) {
      if (auto exact_signature =
              exact_metal_runtime_signature_from_source(source)) {
        return exact_signature->arg_count;
      }
      return external_abi_count;
    }
  }

  if (auto exact_signature = exact_metal_runtime_signature_from_source(source)) {
    return exact_signature->arg_count;
  }
  if (source_has_custom_kernel_stage_manifest(source)) {
    OPENVINO_THROW(
        "GFX Metal: custom-kernel stage manifest is missing exact runtime ABI "
        "for entry ",
        source.entry_point);
  }

  const uint32_t source_buffer_arg_count =
      resolve_no_manifest_msl_source_arg_count(source, msl, source_arg_count);
  if (source_buffer_arg_count != 0) {
    return source_buffer_arg_count;
  }
  if (source.module) {
    const auto abi = read_module_mpsrt_external_buffer_abi(source.module);
    if (abi.valid && abi.has_buffer_count && abi.buffer_count != 0) {
      return abi.buffer_count;
    }
  }
  return 0;
}

uint32_t resolve_expected_mpsrt_external_arg_count(
    const KernelSource &source, uint32_t runtime_binding_arg_count) {
  GfxMpsrtExternalBufferAbiPlan abi{};
  if (source_has_exact_mpsrt_external_buffer_abi(source, &abi) &&
      abi.has_buffer_count) {
    return abi.buffer_count;
  }
  return runtime_binding_arg_count;
}

uint32_t mpsrt_model_external_output_buffer_count(
    const runtime_mpsrt::MpsrtModel &model) {
  uint32_t output_count = 0;
  for (const auto role : model.external_buffer_roles) {
    if (gfx_mpsrt_is_external_output_buffer_role(role)) {
      ++output_count;
    }
  }
  return output_count;
}

bool set_error(std::string *error, const std::string &message) {
  if (error) {
    *error = message;
  }
  return false;
}

bool register_mpsrt_const_tensor_sources(
    MetalCompiledKernel &kernel, const runtime_mpsrt::MpsrtModel &model,
    const std::vector<MpsrtConstTensorSource> &const_tensors,
    std::string *log) {
  for (const auto &payload : const_tensors) {
    if (payload.bytes.empty()) {
      return set_error(log, "GFX MPSRT: const tensor source payload is empty");
    }
    const auto *tensor = runtime_mpsrt::find_mpsrt_tensor(model, payload.value);
    if (!tensor) {
      std::ostringstream stream;
      stream << "GFX MPSRT: const tensor source references unknown value "
             << payload.value;
      return set_error(log, stream.str());
    }
    if (!kernel.register_mpsrt_const_tensor_data(payload.value, tensor->desc,
                                                 payload.bytes.data(),
                                                 payload.bytes.size(), log)) {
      return false;
    }
  }
  return true;
}

struct ResolvedMpsrtProgramPlan {
  bool valid = false;
  GfxMpsrtProgram program{};
  GfxMpsrtBuilderPlan builder_plan{};
};

bool resolve_module_mpsrt_program_plan(mlir::ModuleOp module,
                                       ResolvedMpsrtProgramPlan &program_plan) {
  program_plan = {};
  if (!module) {
    return false;
  }

  GfxMpsrtProgram program{};
  if (!read_module_mpsrt_program(module, program)) {
    return false;
  }

  GfxMpsrtBuilderPlan builder_plan{};
  if (!gfx_mpsrt_build_builder_plan_from_program(program, builder_plan)) {
    return false;
  }
  if (current_compile_trace()) {
    increment_compile_counter(
        "mpsrt_typed_program_builder_plan_consumed_count");
  }

  program_plan.valid = true;
  program_plan.program = std::move(program);
  program_plan.builder_plan = std::move(builder_plan);
  return program_plan.valid;
}

void record_mpsrt_plan_counters(mlir::ModuleOp module) {
  if (!module || !current_compile_trace()) {
    return;
  }

  ResolvedMpsrtProgramPlan program_plan;
  if (!resolve_module_mpsrt_program_plan(module, program_plan)) {
    return;
  }
  const auto &builder_plan = program_plan.builder_plan;
  const auto &program = program_plan.program;
  auto record_stage_counters = [](const GfxMpsrtStageDesc &stage) {
    switch (stage.domain) {
    case GfxStageBackendDomain::AppleMps:
      increment_compile_counter("mpsrt_plan_apple_mps_count");
      break;
    case GfxStageBackendDomain::AppleMsl:
      increment_compile_counter("mpsrt_plan_apple_msl_count");
      break;
    case GfxStageBackendDomain::OpenCl:
      increment_compile_counter("mpsrt_plan_opencl_count");
      break;
    case GfxStageBackendDomain::Spirv:
      increment_compile_counter("mpsrt_plan_spirv_count");
      break;
    case GfxStageBackendDomain::Unknown:
    default:
      break;
    }
    increment_compile_counter(std::string("mpsrt_stage_kind_") +
                              gfx_mpsrt_stage_kind_name(stage.kind));
    const std::string builder_symbol = gfx_mpsrt_builder_symbol(stage.kind);
    increment_compile_counter(std::string("mpsrt_builder_symbol_") +
                              builder_symbol + "_count");
    const auto dispatch = gfx_mpsrt_custom_dispatch_spec_from_kernel_manifest(
        stage.stage_manifest.custom_kernel);
    if (!dispatch.kernel_family.empty()) {
      increment_compile_counter(std::string("mpsrt_dispatch_family_") +
                                dispatch.kernel_family + "_count");
    }
    if (dispatch.kernel_family_id != 0) {
      increment_compile_counter(std::string("mpsrt_dispatch_family_id_") +
                                std::to_string(dispatch.kernel_family_id) +
                                "_count");
    }
    if (!dispatch.entry_point.empty()) {
      increment_compile_counter(std::string("mpsrt_dispatch_entry_") +
                                dispatch.entry_point + "_count");
    }
    if (dispatch.threads_per_threadgroup != 0) {
      increment_compile_counter(
          std::string("mpsrt_dispatch_tg_") +
          std::to_string(dispatch.threads_per_threadgroup) + "_count");
    }
    if (dispatch.flags != GfxMpsrtMslDispatchFlagNone) {
      increment_compile_counter("mpsrt_dispatch_flags",
                                static_cast<uint64_t>(dispatch.flags));
    }
    if (dispatch.precompiled_binary_required) {
      increment_compile_counter(
          "mpsrt_dispatch_precompiled_kernel_required_count");
    }
    increment_compile_counter(std::string("mpsrt_storage_") +
                              gfx_mpsrt_storage_name(stage.output_storage) +
                              "_count");
    const auto &manifest = stage.stage_manifest;
    if (manifest.valid) {
      increment_compile_counter(std::string("mpsrt_stage_family_") +
                                gfx_kernel_stage_family_name(
                                    manifest.stage_family) +
                                "_count");
      increment_compile_counter(std::string("mpsrt_stage_backend_") +
                                gfx_kernel_backend_domain_name(
                                    manifest.backend_domain) +
                                "_count");
      increment_compile_counter(std::string("mpsrt_stage_execution_") +
                                gfx_kernel_execution_kind_name(
                                    manifest.execution_kind) +
                                "_count");
      increment_compile_counter(std::string("mpsrt_stage_precision_") +
                                gfx_kernel_compute_precision_name(
                                    manifest.compute_precision) +
                                "_count");
      increment_compile_counter(
          std::string("mpsrt_stage_backend_") +
          gfx_kernel_backend_domain_name(manifest.backend_domain) +
          "_precision_" +
          gfx_kernel_compute_precision_name(manifest.compute_precision) +
          "_count");
      increment_compile_counter(
          std::string("mpsrt_stage_backend_") +
          gfx_kernel_backend_domain_name(manifest.backend_domain) +
          "_family_" +
          gfx_kernel_stage_family_name(manifest.stage_family) + "_count");
    }
    if (stage.stage_manifest.execution_kind ==
        GfxKernelExecutionKind::VendorPrimitive) {
      increment_compile_counter("mpsrt_vendor_primitive_stage_count");
    }
    if (stage.stage_manifest.execution_kind ==
        GfxKernelExecutionKind::CustomKernel) {
      increment_compile_counter("mpsrt_custom_kernel_stage_count");
    }
  };

  increment_compile_counter("mpsrt_builder_record_count",
                            static_cast<uint64_t>(builder_plan.records.size()));
  increment_compile_counter(
      "mpsrt_builder_storage_bridge_count",
      static_cast<uint64_t>(builder_plan.storage_bridges.size()));
  uint64_t encode_record_count = 0;
  for (const auto &record : builder_plan.records) {
    if (record.kind == GfxMpsrtBuilderRecordKind::EncodeStage) {
      ++encode_record_count;
    }
    const auto msl_dispatch_desc = gfx_mpsrt_make_msl_dispatch_desc(
        record.stage_desc, static_cast<uint32_t>(record.inputs.size()),
        static_cast<uint32_t>(record.outputs.size()));
    if (record.stage_desc.kind == GfxMpsrtStageKind::MSLDispatch &&
        msl_dispatch_desc.kernel_family != 0) {
      increment_compile_counter("mpsrt_msl_dispatch_descriptor_count");
      increment_compile_counter(
          std::string("mpsrt_msl_dispatch_descriptor_family_id_") +
          std::to_string(msl_dispatch_desc.kernel_family) + "_count");
    }
  }
  increment_compile_counter("mpsrt_builder_encode_record_count",
                            encode_record_count);

  uint64_t input_bytes = 0;
  uint64_t output_bytes = 0;
  uint64_t output_descriptor_count = 0;
  if (program.multi_stage) {
    increment_compile_counter("mpsrt_multi_stage_module_plan_count");
    increment_compile_counter("mpsrt_multi_stage_module_stage_count",
                              static_cast<uint64_t>(program.stages.size()));
    for (const auto &desc : program.inputs) {
      input_bytes += desc.byte_length;
    }
    for (const auto &stage : program.stages) {
      record_stage_counters(stage.stage);
      output_descriptor_count += stage.output_descs.size();
      for (const auto &desc : stage.output_descs) {
        output_bytes += desc.byte_length;
      }
    }
  } else if (!program.stages.empty()) {
    const auto &stage = program.stages.front();
    record_stage_counters(stage.stage);
    for (const auto &desc : program.inputs) {
      input_bytes += desc.byte_length;
    }
    output_descriptor_count = stage.output_descs.size();
    for (const auto &desc : stage.output_descs) {
      output_bytes += desc.byte_length;
    }
  }
  increment_compile_counter(
      "mpsrt_input_descriptor_count",
      static_cast<uint64_t>(builder_plan.input_values.size()));
  increment_compile_counter("mpsrt_output_descriptor_count",
                            output_descriptor_count);
  increment_compile_counter("mpsrt_input_byte_length", input_bytes);
  increment_compile_counter("mpsrt_output_byte_length", output_bytes);
}

std::shared_ptr<const runtime_mpsrt::MpsrtModel>
build_metal_mpsrt_runtime_model(mlir::ModuleOp module) {
  if (!module) {
    return nullptr;
  }
  if (!module_has_mpsrt_ops_program(module)) {
    return nullptr;
  }

  ResolvedMpsrtProgramPlan program_plan;
  if (!resolve_module_mpsrt_program_plan(module, program_plan)) {
    OPENVINO_THROW("GFX Metal MPSRT: invalid typed gfx_mpsrt_ops program");
  }

  runtime_mpsrt::MpsrtModel model;
  std::string error;
  if (!runtime_mpsrt::build_mpsrt_model_from_builder_plan(
          program_plan.builder_plan, model, &error)) {
    OPENVINO_THROW("GFX Metal MPSRT: failed to build runtime model: ", error);
  }
  if (!program_plan.builder_plan.external_buffer_abi_valid) {
    OPENVINO_THROW(
        "GFX Metal MPSRT: typed gfx_mpsrt_ops program is missing exact "
        "external buffer ABI");
  }
  const uint32_t mpsrt_arg_count =
      program_plan.builder_plan.external_buffer_count;
  const uint32_t mpsrt_output_arg_count =
      program_plan.builder_plan.external_output_buffer_count;
  if (mpsrt_arg_count == 0 || mpsrt_output_arg_count == 0) {
    OPENVINO_THROW(
        "GFX Metal MPSRT: typed gfx_mpsrt_ops program external buffer ABI "
        "is incomplete");
  }
  if (!runtime_mpsrt::adapt_mpsrt_model_to_external_buffer_abi(
          model, mpsrt_arg_count, mpsrt_output_arg_count, &error)) {
    OPENVINO_THROW("GFX Metal MPSRT: failed to adapt runtime model ABI: ",
                   error);
  }

  if (current_compile_trace()) {
    increment_compile_counter("mpsrt_runtime_model_prepare_count");
    if (program_plan.builder_plan.external_buffer_abi_valid) {
      increment_compile_counter(
          "mpsrt_runtime_model_mlir_external_buffer_abi_count");
    }
    increment_compile_counter("mpsrt_runtime_model_stage_count",
                              static_cast<uint64_t>(model.stages.size()));
    increment_compile_counter("mpsrt_runtime_model_tensor_count",
                              static_cast<uint64_t>(model.tensors.size()));
    for (const auto &stage : model.stages) {
      increment_compile_counter(std::string("mpsrt_runtime_model_stage_kind_") +
                                gfx_mpsrt_stage_kind_name(stage.kind) +
                                "_count");
      if (stage.kind == GfxMpsrtStageKind::MSLDispatch) {
        increment_compile_counter(
            "mpsrt_runtime_model_msl_dispatch_stage_count");
      }
    }
  }

  return std::make_shared<runtime_mpsrt::MpsrtModel>(std::move(model));
}

class MetalResolvedMslCache final {
public:
  static MetalResolvedMslCache &instance() {
    static MetalResolvedMslCache cache;
    return cache;
  }

  bool lookup(const std::string &key, std::string &msl) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_cache.find(key);
    if (it == m_cache.end()) {
      return false;
    }
    msl = it->second;
    return true;
  }

  void store(std::string key, std::string msl) {
    if (key.empty() || msl.empty()) {
      return;
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_cache.emplace(std::move(key), std::move(msl));
  }

private:
  std::mutex m_mutex;
  std::unordered_map<std::string, std::string> m_cache;
};

class MetalPreparedState final {
public:
  explicit MetalPreparedState(const KernelBindingTable &table) {
    const auto &bindings = table.buffers;
    buffers.reserve(bindings.size());
    buffer_ptrs.reserve(bindings.size());
    offsets.reserve(bindings.size());
    for (const auto &binding : bindings) {
      auto *buffer = to_mtl(binding.buffer);
      buffers.push_back(buffer);
      buffer_ptrs.push_back(buffer);
      offsets.push_back(binding.offset);
    }
  }

  std::vector<id<MTLBuffer>> buffers;
  std::vector<void *> buffer_ptrs;
  std::vector<size_t> offsets;
};

bool mpsrt_model_has_msl_dispatch(
    const std::shared_ptr<const runtime_mpsrt::MpsrtModel> &model) {
  if (!model) {
    return false;
  }
  return metal::mpsrt::mpsrt_model_has_msl_dispatch(*model);
}

bool mpsrt_conv2d_stage_supported_by_image_bridge(
    const runtime_mpsrt::MpsrtModel &model,
    const runtime_mpsrt::MpsrtRuntimeStage &stage) {
  if ((stage.inputs.size() != 2 && stage.inputs.size() != 3) ||
      stage.outputs.size() != 1 ||
      stage.output_descs.size() != 1) {
    return false;
  }
  const auto *input = runtime_mpsrt::find_mpsrt_tensor(model, stage.inputs[0]);
  const auto *weights =
      runtime_mpsrt::find_mpsrt_tensor(model, stage.inputs[1]);
  if (!input || !weights) {
    return false;
  }
  const auto &input_desc = input->desc;
  const auto &weights_desc = weights->desc;
  const auto &output_desc = stage.output_descs.front();
  if (!gfx_mpsrt_tensor_is_image(input_desc) ||
      !gfx_mpsrt_tensor_is_image(output_desc) ||
      !gfx_mpsrt_image_bridge_supported(input_desc) ||
      !gfx_mpsrt_image_bridge_supported(output_desc)) {
    return false;
  }
  if (input_desc.dtype != output_desc.dtype ||
      weights_desc.dtype != output_desc.dtype) {
    return false;
  }
  if (stage.inputs.size() == 3) {
    const auto *bias = runtime_mpsrt::find_mpsrt_tensor(model, stage.inputs[2]);
    if (!bias || (bias->desc.flags & GfxMpsrtTensorFlagConst) == 0 ||
        bias->desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Buffer)) {
      return false;
    }
  }
  if (stage.conv2d_desc.groups == 0 ||
      input_desc.image_feature_channels % stage.conv2d_desc.groups != 0 ||
      output_desc.image_feature_channels % stage.conv2d_desc.groups != 0) {
    return false;
  }
  return true;
}

bool mpsrt_pool2d_stage_supported_by_image_bridge(
    const runtime_mpsrt::MpsrtModel &model,
    const runtime_mpsrt::MpsrtRuntimeStage &stage) {
  if (stage.inputs.size() != 1 || stage.outputs.size() != 1 ||
      stage.output_descs.size() != 1) {
    return false;
  }
  const auto *input = runtime_mpsrt::find_mpsrt_tensor(model, stage.inputs[0]);
  if (!input) {
    return false;
  }
  const auto &input_desc = input->desc;
  const auto &output_desc = stage.output_descs.front();
  if (!gfx_mpsrt_tensor_is_image(input_desc) ||
      !gfx_mpsrt_tensor_is_image(output_desc) ||
      !gfx_mpsrt_image_bridge_supported(input_desc) ||
      !gfx_mpsrt_image_bridge_supported(output_desc)) {
    return false;
  }
  if (input_desc.dtype != output_desc.dtype ||
      input_desc.image_batch != output_desc.image_batch ||
      input_desc.image_feature_channels != output_desc.image_feature_channels) {
    return false;
  }
  if ((input_desc.image_feature_channels % 4u) != 0) {
    return false;
  }
  if (stage.pool2d_desc.kernel[0] == 0 || stage.pool2d_desc.kernel[1] == 0 ||
      stage.pool2d_desc.strides[0] == 0 || stage.pool2d_desc.strides[1] == 0 ||
      stage.pool2d_desc.dilations[0] != 1 ||
      stage.pool2d_desc.dilations[1] != 1) {
    return false;
  }
  return true;
}

bool mpsrt_resize2d_stage_supported_by_image_bridge(
    const runtime_mpsrt::MpsrtModel &model,
    const runtime_mpsrt::MpsrtRuntimeStage &stage) {
  if (stage.inputs.size() != 1 || stage.outputs.size() != 1 ||
      stage.output_descs.size() != 1) {
    return false;
  }
  if (stage.resize2d_desc.nearest != 0) {
    return false;
  }
  const auto *input = runtime_mpsrt::find_mpsrt_tensor(model, stage.inputs[0]);
  if (!input) {
    return false;
  }
  const auto &input_desc = input->desc;
  const auto &output_desc = stage.output_descs.front();
  if (!gfx_mpsrt_tensor_is_image(input_desc) ||
      !gfx_mpsrt_tensor_is_image(output_desc) ||
      !gfx_mpsrt_image_bridge_supported(input_desc) ||
      !gfx_mpsrt_image_bridge_supported(output_desc)) {
    return false;
  }
  if (input_desc.dtype != output_desc.dtype ||
      input_desc.image_batch != output_desc.image_batch ||
      input_desc.image_feature_channels != output_desc.image_feature_channels) {
    return false;
  }
  return input_desc.image_width != 0 && input_desc.image_height != 0 &&
         output_desc.image_width != 0 && output_desc.image_height != 0;
}

bool mpsrt_softmax_stage_supported_by_matrix_bridge(
    const runtime_mpsrt::MpsrtModel &model,
    const runtime_mpsrt::MpsrtRuntimeStage &stage) {
  if (stage.inputs.size() != 1 || stage.outputs.size() != 1 ||
      stage.output_descs.size() != 1) {
    return false;
  }
  const auto *input = runtime_mpsrt::find_mpsrt_tensor(model, stage.inputs[0]);
  if (!input) {
    return false;
  }
  const auto &input_desc = input->desc;
  const auto &output_desc = stage.output_descs.front();
  if (input_desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix) ||
      output_desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
    return false;
  }
  if (stage.softmax_desc.log_softmax != 0) {
    return false;
  }
  if (input_desc.dtype != output_desc.dtype ||
      (input_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F16) &&
       input_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32))) {
    return false;
  }
  if (input_desc.matrix_rows == 0 || input_desc.matrix_columns == 0 ||
      input_desc.matrix_row_bytes == 0 ||
      output_desc.matrix_rows != input_desc.matrix_rows ||
      output_desc.matrix_columns != input_desc.matrix_columns ||
      output_desc.matrix_row_bytes == 0) {
    return false;
  }
  const uint32_t input_count =
      input_desc.matrix_count == 0 ? 1 : input_desc.matrix_count;
  const uint32_t output_count =
      output_desc.matrix_count == 0 ? 1 : output_desc.matrix_count;
  return input_count == output_count;
}

bool mpsrt_topk_stage_supported_by_matrix_bridge(
    const runtime_mpsrt::MpsrtModel &model,
    const runtime_mpsrt::MpsrtRuntimeStage &stage) {
  if (stage.inputs.size() != 1 || stage.outputs.size() != 2 ||
      stage.output_descs.size() != 2) {
    return false;
  }
  const auto *input = runtime_mpsrt::find_mpsrt_tensor(model, stage.inputs[0]);
  if (!input) {
    return false;
  }
  const auto &input_desc = input->desc;
  const auto &values_desc = stage.output_descs[0];
  const auto &indices_desc = stage.output_descs[1];
  if (input_desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix) ||
      values_desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix) ||
      indices_desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::Matrix)) {
    return false;
  }
  if (stage.topk_desc.mode_max == 0 || stage.topk_desc.k == 0 ||
      stage.topk_desc.k > input_desc.matrix_columns) {
    return false;
  }
  if (stage.topk_desc.k > 16 && stage.topk_desc.sort_type == 2u) {
    return false;
  }
  if (input_desc.dtype != values_desc.dtype ||
      (input_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F16) &&
       input_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32))) {
    return false;
  }
  if (indices_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::I32) &&
      indices_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::U32) &&
      indices_desc.dtype != static_cast<uint32_t>(GfxMpsrtDType::I64)) {
    return false;
  }
  if (input_desc.matrix_rows == 0 || input_desc.matrix_columns == 0 ||
      input_desc.matrix_row_bytes == 0 ||
      values_desc.matrix_rows != input_desc.matrix_rows ||
      values_desc.matrix_columns != stage.topk_desc.k ||
      values_desc.matrix_row_bytes == 0 ||
      indices_desc.matrix_rows != input_desc.matrix_rows ||
      indices_desc.matrix_columns != stage.topk_desc.k ||
      indices_desc.matrix_row_bytes == 0) {
    return false;
  }
  const uint32_t input_count =
      input_desc.matrix_count == 0 ? 1 : input_desc.matrix_count;
  const uint32_t values_count =
      values_desc.matrix_count == 0 ? 1 : values_desc.matrix_count;
  const uint32_t indices_count =
      indices_desc.matrix_count == 0 ? 1 : indices_desc.matrix_count;
  return input_count == values_count && input_count == indices_count;
}

bool mpsrt_dense_ndarray_graph_tensor_supported(
    const GfxMpsrtTensorAbiDesc &desc) {
  if (desc.storage != static_cast<uint32_t>(GfxMpsrtStorage::NDArray)) {
    return false;
  }
  const auto dtype = static_cast<GfxMpsrtDType>(desc.dtype);
  const uint32_t element_bytes = gfx_mpsrt_element_size_bytes(dtype);
  if (element_bytes == 0 || desc.byte_offset != 0 || desc.rank == 0 ||
      desc.rank > 8) {
    return false;
  }
  uint64_t dense_elements = 1;
  for (uint32_t i = 0; i < desc.rank; ++i) {
    if (desc.dims[i] == 0) {
      return false;
    }
    dense_elements *= desc.dims[i];
  }
  return desc.byte_length == dense_elements * element_bytes;
}

bool mpsrt_sdpa_stage_supported_by_graph_bridge(
    const runtime_mpsrt::MpsrtModel &model,
    const runtime_mpsrt::MpsrtRuntimeStage &stage) {
  if (stage.inputs.size() != 3 || stage.outputs.size() != 1 ||
      stage.output_descs.size() != 1 || stage.sdpa_desc.has_mask != 0 ||
      stage.sdpa_desc.causal != 0) {
    return false;
  }
  const bool transposed_layout =
      stage.sdpa_desc.layout == GfxMpsrtSdpaLayoutTransposedBHDN;
  if (!transposed_layout) {
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, *)) {
    } else {
      return false;
    }
  }
  const auto *query = runtime_mpsrt::find_mpsrt_tensor(model, stage.inputs[0]);
  const auto *key = runtime_mpsrt::find_mpsrt_tensor(model, stage.inputs[1]);
  const auto *value = runtime_mpsrt::find_mpsrt_tensor(model, stage.inputs[2]);
  if (!query || !key || !value) {
    return false;
  }
  const auto &q = query->desc;
  const auto &k = key->desc;
  const auto &v = value->desc;
  const auto &out = stage.output_descs.front();
  if (q.dtype != k.dtype || q.dtype != v.dtype || q.dtype != out.dtype ||
      (q.dtype != static_cast<uint32_t>(GfxMpsrtDType::F16) &&
       q.dtype != static_cast<uint32_t>(GfxMpsrtDType::F32))) {
    return false;
  }
  if (q.rank != 4 || k.rank != 4 || v.rank != 4 || out.rank != 4) {
    return false;
  }
  if (!mpsrt_dense_ndarray_graph_tensor_supported(q) ||
      !mpsrt_dense_ndarray_graph_tensor_supported(k) ||
      !mpsrt_dense_ndarray_graph_tensor_supported(v) ||
      !mpsrt_dense_ndarray_graph_tensor_supported(out)) {
    return false;
  }
  if (stage.sdpa_desc.layout == GfxMpsrtSdpaLayoutNativeBHND) {
    return q.dims[0] == k.dims[0] && q.dims[0] == v.dims[0] &&
           q.dims[1] == k.dims[1] && q.dims[1] == v.dims[1] &&
           q.dims[3] == k.dims[3] && q.dims[3] == v.dims[3] &&
           k.dims[2] == v.dims[2] &&
           out.dims[0] == q.dims[0] && out.dims[1] == q.dims[1] &&
           out.dims[2] == q.dims[2] && out.dims[3] == v.dims[3];
  }
  if (stage.sdpa_desc.layout == GfxMpsrtSdpaLayoutTransposedBHDN) {
    return q.dims[0] == k.dims[0] && q.dims[0] == v.dims[0] &&
           q.dims[1] == k.dims[1] && q.dims[1] == v.dims[1] &&
           q.dims[2] == k.dims[2] && k.dims[3] == v.dims[3] &&
           out.dims[0] == q.dims[0] && out.dims[1] == q.dims[1] &&
           out.dims[2] == v.dims[2] && out.dims[3] == q.dims[3];
  }
  return false;
}

bool mpsrt_stage_supported_by_current_runtime(
    const runtime_mpsrt::MpsrtModel &model,
    const runtime_mpsrt::MpsrtRuntimeStage &stage) {
  switch (stage.kind) {
  case GfxMpsrtStageKind::MPSGemm:
    return true;
  case GfxMpsrtStageKind::MPSConv2D:
  case GfxMpsrtStageKind::MPSGroupConv2D:
    return mpsrt_conv2d_stage_supported_by_image_bridge(model, stage);
  case GfxMpsrtStageKind::MPSPool2D:
    return mpsrt_pool2d_stage_supported_by_image_bridge(model, stage);
  case GfxMpsrtStageKind::MPSResize2D:
    return mpsrt_resize2d_stage_supported_by_image_bridge(model, stage);
  case GfxMpsrtStageKind::MPSSoftmax:
    return mpsrt_softmax_stage_supported_by_matrix_bridge(model, stage);
  case GfxMpsrtStageKind::MPSTopK:
    return mpsrt_topk_stage_supported_by_matrix_bridge(model, stage);
  case GfxMpsrtStageKind::MPSSdpa:
    return mpsrt_sdpa_stage_supported_by_graph_bridge(model, stage);
  default:
    return false;
  }
}

bool mpsrt_model_has_supported_vendor_stage(
    const std::shared_ptr<const runtime_mpsrt::MpsrtModel> &model) {
  if (!model) {
    return false;
  }
  return std::any_of(
      model->stages.begin(), model->stages.end(), [&](const auto &stage) {
        return mpsrt_stage_supported_by_current_runtime(*model, stage);
      });
}

bool mpsrt_model_is_executable_by_mpsrt(
    const std::shared_ptr<const runtime_mpsrt::MpsrtModel> &model,
    const std::string &msl_source) {
  if (!model || model->stages.empty()) {
    return false;
  }

  bool has_msl_dispatch = false;
  for (const auto &stage : model->stages) {
    switch (stage.kind) {
    case GfxMpsrtStageKind::MPSGemm:
    case GfxMpsrtStageKind::MPSConv2D:
    case GfxMpsrtStageKind::MPSGroupConv2D:
    case GfxMpsrtStageKind::MPSPool2D:
    case GfxMpsrtStageKind::MPSResize2D:
    case GfxMpsrtStageKind::MPSSoftmax:
    case GfxMpsrtStageKind::MPSTopK:
    case GfxMpsrtStageKind::MPSSdpa:
      if (!mpsrt_stage_supported_by_current_runtime(*model, stage)) {
        return false;
      }
      break;
    case GfxMpsrtStageKind::MSLDispatch:
      has_msl_dispatch = true;
      break;
    default:
      return false;
    }
  }
  return !has_msl_dispatch || !msl_source.empty();
}

bool mpsrt_model_should_use_context_execution(
    const std::shared_ptr<const runtime_mpsrt::MpsrtModel> &model,
    const std::string &msl_source) {
  if (!mpsrt_model_is_executable_by_mpsrt(model, msl_source)) {
    return false;
  }

  const bool has_vendor_stage = mpsrt_model_has_supported_vendor_stage(model);
  const bool has_msl_dispatch = mpsrt_model_has_msl_dispatch(model);
  if (!has_vendor_stage) {
    return false;
  }

  return !has_msl_dispatch || model->stages.size() > 1;
}

void record_mpsrt_prepared_model_counters(
    const KernelExecutionHooks *hooks,
    const metal::mpsrt::MpsrtPreparedModel &prepared_model) {
  if (!hooks || !hooks->on_counter) {
    return;
  }
  uint64_t transient_resource_bytes = 0;
  for (const auto &resource : prepared_model.resources) {
    if (resource.lifetime ==
        runtime_mpsrt::MpsrtRuntimeResourceLifetime::Transient) {
      transient_resource_bytes +=
          static_cast<uint64_t>(resource.heap_allocation_size);
    }
  }
  uint64_t image_bridge_resource_bytes = 0;
  for (const auto &resource : prepared_model.image_bridge_resources) {
    image_bridge_resource_bytes +=
        static_cast<uint64_t>(resource.heap_allocation_size);
  }

  hooks->on_counter("mpsrt_prepared_resource_heap_size_bytes",
                    static_cast<uint64_t>(prepared_model.resource_heap_size));
  hooks->on_counter(
      "mpsrt_prepared_resource_heap_unaliased_size_bytes",
      static_cast<uint64_t>(prepared_model.resource_heap_unaliased_size));
  hooks->on_counter(
      "mpsrt_prepared_resource_heap_aliasable_size_bytes",
      static_cast<uint64_t>(prepared_model.resource_heap_aliasable_size));
  hooks->on_counter(
      "mpsrt_prepared_resource_heap_alias_reuse_count",
      static_cast<uint64_t>(prepared_model.resource_heap_alias_reuse_count));
  hooks->on_counter(
      "mpsrt_prepared_image_bridge_resource_count",
      static_cast<uint64_t>(prepared_model.image_bridge_resource_count));
  hooks->on_counter(
      "mpsrt_prepared_transient_resource_size_bytes",
      transient_resource_bytes);
  hooks->on_counter(
      "mpsrt_prepared_image_bridge_resource_size_bytes",
      image_bridge_resource_bytes);
}

} // namespace

MetalCodegenBackend::MetalCodegenBackend(MetalDeviceHandle device)
    : m_device(device),
      m_reuse_context(MetalDeviceReuseRegistry::instance().acquire(device)) {}

std::shared_ptr<ICompiledKernel>
MetalCodegenBackend::compile(const KernelSource &source, std::string *log) {
  std::string local_log;
  std::string *log_ptr = log ? log : &local_log;
  if (gfx_log_debug_enabled()) {
    gfx_log_debug("MetalCodegen")
        << "compile entry=" << source.entry_point
        << " arg_count=" << source.signature.arg_count
        << " has_module=" << (source.module ? "yes" : "no")
        << " has_msl=" << (!source.msl_source.empty() ? "yes" : "no")
        << " has_generator=" << (source.msl_generator ? "yes" : "no");
  }
  std::string msl;
  std::string resolved_msl_cache_key;
  const bool can_cache_resolved_msl =
      source.module && source.msl_source.empty() && source.msl_generator;
  record_mpsrt_plan_counters(source.module);
  const uint32_t source_arg_count =
      resolve_initial_compile_source_arg_count(source);
  const uint32_t output_arg_count = resolve_kernel_output_arg_count(source);
  const uintptr_t device_key = reinterpret_cast<uintptr_t>(m_device);
  auto mpsrt_model = build_metal_mpsrt_runtime_model(source.module);
  const bool vendor_only_mpsrt_model =
      mpsrt_model && mpsrt_model_has_supported_vendor_stage(mpsrt_model) &&
      !mpsrt_model_has_msl_dispatch(mpsrt_model);
  if (vendor_only_mpsrt_model && source.msl_source.empty() &&
      !source.msl_generator) {
    if (current_compile_trace()) {
      increment_compile_counter("metal_mpsrt_vendor_only_kernel_count");
    }
    const uint32_t runtime_binding_arg_count = static_cast<uint32_t>(
        runtime_mpsrt::mpsrt_model_external_buffer_abi_count(*mpsrt_model));
    auto shared_prepared_cache = acquire_shared_prepared_binding_cache(
        GpuBackend::Metal, device_key, runtime_binding_arg_count);
    auto binding_schema =
        m_reuse_context->acquire_binding_schema(runtime_binding_arg_count);
    auto binding_plan = std::make_shared<KernelBindingPlan>(
        runtime_binding_arg_count, output_arg_count);
    auto kernel = std::make_shared<MetalCompiledKernel>(
        m_device, nullptr, std::move(binding_plan), shared_prepared_cache,
        binding_schema);
    kernel->set_mpsrt_model(mpsrt_model);
    if (!source.mpsrt_const_tensors.empty() &&
        !register_mpsrt_const_tensor_sources(
            *kernel, *mpsrt_model, source.mpsrt_const_tensors, log_ptr)) {
      return nullptr;
    }
    return kernel;
  }
  if (can_cache_resolved_msl) {
    const auto cache_key_start = current_compile_trace()
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
    resolved_msl_cache_key = make_resolved_msl_cache_key(source);
    if (current_compile_trace()) {
      add_compile_segment(
          "metal_resolved_msl_cache_key",
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - cache_key_start)
                  .count()));
    }
    if (MetalResolvedMslCache::instance().lookup(resolved_msl_cache_key, msl)) {
      if (current_compile_trace()) {
        increment_compile_counter("metal_resolved_msl_cache_hit_count");
        add_compile_segment("metal_resolved_msl_cache_hit", 0);
      }
    }
  }

  if (msl.empty() && source.module && source.msl_source.empty() &&
      source.msl_generator) {
    const auto mlir_preprocess_start =
        current_compile_trace() ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
    try {
      if (gfx_log_debug_enabled()) {
        gfx_log_debug("MetalCodegen")
            << "before run_mlir_pipeline entry=" << source.entry_point;
      }
      run_mlir_pipeline(source.module, /*use_alloca=*/true,
                        /*use_parallel_loops=*/false);
      if (gfx_log_debug_enabled()) {
        gfx_log_debug("MetalCodegen")
            << "after run_mlir_pipeline entry=" << source.entry_point;
      }
      if (current_compile_trace()) {
        increment_compile_counter("metal_mlir_preprocess_count");
        add_compile_segment(
            "metal_mlir_preprocess",
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - mlir_preprocess_start)
                    .count()));
      }
    } catch (const std::exception &e) {
      if (log_ptr) {
        *log_ptr = std::string("MLIR preprocessing failed: ") + e.what();
      }
      return nullptr;
    }
  }
  if (msl.empty()) {
    const auto resolve_msl_start =
        current_compile_trace() ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MetalCodegen")
          << "before resolve_msl_source entry=" << source.entry_point;
    }
    msl = resolve_msl_source(source, log_ptr);
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MetalCodegen")
          << "after resolve_msl_source entry=" << source.entry_point
          << " msl_size=" << msl.size();
    }
    if (current_compile_trace()) {
      increment_compile_counter("metal_resolve_msl_count");
      add_compile_segment(
          "metal_resolve_msl",
          static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - resolve_msl_start)
                  .count()));
    }
    if (can_cache_resolved_msl && !resolved_msl_cache_key.empty()) {
      MetalResolvedMslCache::instance().store(std::move(resolved_msl_cache_key),
                                              msl);
    }
  }
  OPENVINO_ASSERT(!msl.empty(), "MetalCodegenBackend: missing MSL source");
  OPENVINO_ASSERT(!source.entry_point.empty(),
                  "MetalCodegenBackend: missing entry point");
  const bool use_mpsrt_context_execution =
      mpsrt_model_should_use_context_execution(mpsrt_model, msl);
  const uint32_t mpsrt_external_arg_count =
      use_mpsrt_context_execution
          ? static_cast<uint32_t>(
                runtime_mpsrt::mpsrt_model_external_buffer_abi_count(
                    *mpsrt_model))
          : 0u;
  const uint32_t resolved_output_arg_count =
      use_mpsrt_context_execution
          ? mpsrt_model_external_output_buffer_count(*mpsrt_model)
          : resolve_kernel_output_arg_count(source);
  const uint32_t runtime_binding_arg_count =
      mpsrt_external_arg_count != 0
          ? mpsrt_external_arg_count
          : resolve_metal_runtime_arg_count(source, msl, source_arg_count,
                                            mpsrt_model);
  OPENVINO_ASSERT(
      runtime_binding_arg_count != 0,
      "MetalCodegenBackend: failed to resolve Metal buffer argument count");
  const auto current_mpsrt_binding_arg_count =
      mpsrt_model ? static_cast<uint32_t>(
                        runtime_mpsrt::mpsrt_model_external_buffer_abi_count(
                            *mpsrt_model))
                  : 0u;
  const uint32_t expected_mpsrt_external_arg_count =
      resolve_expected_mpsrt_external_arg_count(source,
                                                runtime_binding_arg_count);
  if (!mpsrt_model ||
      (current_mpsrt_binding_arg_count != 0 &&
       expected_mpsrt_external_arg_count != current_mpsrt_binding_arg_count) ||
      resolved_output_arg_count != output_arg_count) {
    mpsrt_model = build_metal_mpsrt_runtime_model(source.module);
  }
  auto shared_prepared_cache = acquire_shared_prepared_binding_cache(
      GpuBackend::Metal, device_key, runtime_binding_arg_count);
  auto binding_schema =
      m_reuse_context->acquire_binding_schema(runtime_binding_arg_count);

  auto kernel = lookup_or_compile_kernel(
      GpuBackend::Metal, device_key, msl.data(), msl.size(), source.entry_point,
      runtime_binding_arg_count, [&]() -> std::shared_ptr<ICompiledKernel> {
        MetalKernelCompiler compiler((id<MTLDevice>)m_device);
        std::string local_log;
        std::string &compile_log = log ? *log : local_log;
        const auto backend_compile_start =
            current_compile_trace() ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
        id<MTLComputePipelineState> pipeline = compiler.compile_msl_from_source(
            msl, source.entry_point.c_str(), compile_log);
        if (current_compile_trace()) {
          increment_compile_counter("metal_backend_compile_count");
          add_compile_segment(
              "metal_backend_compile",
              static_cast<uint64_t>(
                  std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - backend_compile_start)
                      .count()));
        }
        if (!pipeline) {
          return nullptr;
        }
        auto binding_plan = std::make_shared<KernelBindingPlan>(
            runtime_binding_arg_count, resolved_output_arg_count);
        return std::make_shared<MetalCompiledKernel>(
            m_device, (void *)pipeline, std::move(binding_plan),
            shared_prepared_cache, binding_schema);
      });
  const bool compiled_model_has_msl_dispatch =
      mpsrt_model_has_msl_dispatch(mpsrt_model);
  if (auto metal_kernel =
          std::dynamic_pointer_cast<MetalCompiledKernel>(kernel)) {
    metal_kernel->set_mpsrt_model(mpsrt_model);
    if (mpsrt_model && !source.mpsrt_const_tensors.empty() &&
        !register_mpsrt_const_tensor_sources(
            *metal_kernel, *mpsrt_model, source.mpsrt_const_tensors, log_ptr)) {
      return nullptr;
    }
    if (compiled_model_has_msl_dispatch) {
      metal_kernel->set_mpsrt_msl_source(msl);
    }
  }
  return kernel;
}

MetalCompiledKernel::MetalCompiledKernel(MetalDeviceHandle device,
                                         void *pipeline, uint32_t arg_count)
    : CompiledKernelBase(arg_count), m_device(device), m_pipeline(pipeline) {}

MetalCompiledKernel::MetalCompiledKernel(
    MetalDeviceHandle device, void *pipeline,
    std::shared_ptr<const KernelBindingPlan> binding_plan)
    : CompiledKernelBase(std::move(binding_plan)), m_device(device),
      m_pipeline(pipeline) {}

MetalCompiledKernel::MetalCompiledKernel(
    MetalDeviceHandle device, void *pipeline,
    std::shared_ptr<const KernelBindingPlan> binding_plan,
    std::shared_ptr<void> prepared_binding_cache,
    std::shared_ptr<MetalBindingSchema> binding_schema)
    : CompiledKernelBase(std::move(binding_plan),
                         std::move(prepared_binding_cache)),
      m_device(device), m_pipeline(pipeline),
      m_binding_schema(std::move(binding_schema)) {}

size_t MetalCompiledKernel::clamp_threadgroup_size(size_t desired) const {
  return metal_clamp_tg_size(m_pipeline, desired);
}

std::shared_ptr<ICompiledKernel> MetalCompiledKernel::fork() const {
  auto kernel = std::make_shared<MetalCompiledKernel>(
      m_device, m_pipeline, binding_plan(), prepared_binding_cache(),
      m_binding_schema);
  kernel->set_mpsrt_model(m_mpsrt_model);
  kernel->set_mpsrt_msl_source(m_mpsrt_msl_source);
  return kernel;
}

const void *MetalCompiledKernel::shared_binding_schema_identity() const {
  return m_binding_schema.get();
}

void MetalCompiledKernel::set_mpsrt_model(
    std::shared_ptr<const runtime_mpsrt::MpsrtModel> model) {
  m_mpsrt_model = std::move(model);
  reset_mpsrt_prepared_model_cache();
}

void MetalCompiledKernel::set_mpsrt_msl_source(std::string msl_source) {
  m_mpsrt_msl_source = std::move(msl_source);
  reset_mpsrt_prepared_model_cache();
}

const runtime_mpsrt::MpsrtModel *MetalCompiledKernel::mpsrt_model() const {
  return m_mpsrt_model.get();
}

bool MetalCompiledKernel::register_mpsrt_const_tensor_data(
    GfxMpsrtValue value, GfxMpsrtTensorAbiDesc desc, const void *data,
    size_t bytes, std::string *log) {
  if (!m_mpsrt_model) {
    return set_error(
        log, "GFX MPSRT: cannot register const tensor without an MPSRT model");
  }
  if (!m_mpsrt_context) {
    m_mpsrt_context = std::make_shared<metal::mpsrt::MpsrtContext>(
        static_cast<id<MTLDevice>>(m_device));
  }
  desc.flags |= GfxMpsrtTensorFlagConst;
  const bool registered = m_mpsrt_context->register_const_tensor_data(
      value, desc, data, bytes, log);
  if (registered) {
    reset_mpsrt_prepared_model_cache();
  }
  return registered;
}

void MetalCompiledKernel::reset_mpsrt_prepared_model_cache() {
  std::lock_guard<std::mutex> lock(m_mpsrt_prepared_model_cache_mutex);
  m_mpsrt_prepared_model_cache_slots.clear();
  m_mpsrt_prepared_model_cache_next_slot = 0;
}

const char *MetalCompiledKernel::mpsrt_prepared_model_cache_kind_name(
    MpsrtPreparedModelCacheKind kind) {
  switch (kind) {
  case MpsrtPreparedModelCacheKind::ContextExecution:
    return "context_execution";
  case MpsrtPreparedModelCacheKind::SingleMslDispatch:
    return "single_msl_dispatch";
  case MpsrtPreparedModelCacheKind::None:
    return "none";
  }
  return "unknown";
}

MetalCompiledKernel::MpsrtPreparedModelCacheKind
MetalCompiledKernel::resolve_mpsrt_prepared_model_cache_kind() const {
  const auto execution_binding_plan = binding_plan();
  const bool mpsrt_external_abi_matches_bindings =
      m_mpsrt_model && execution_binding_plan &&
      metal::mpsrt::mpsrt_external_abi_matches_exact_binding_plan(
          *m_mpsrt_model, *execution_binding_plan);
  const bool mpsrt_has_msl_dispatch =
      mpsrt_model_has_msl_dispatch(m_mpsrt_model);
  const bool mpsrt_context_external_abi_ready =
      !mpsrt_has_msl_dispatch || mpsrt_external_abi_matches_bindings;
  if (mpsrt_context_external_abi_ready &&
      mpsrt_model_should_use_context_execution(m_mpsrt_model,
                                               m_mpsrt_msl_source)) {
    return MpsrtPreparedModelCacheKind::ContextExecution;
  }

  if (mpsrt_external_abi_matches_bindings && m_mpsrt_model &&
      m_mpsrt_model->stages.size() == 1 &&
      m_mpsrt_model->stages.front().kind == GfxMpsrtStageKind::MSLDispatch) {
    return MpsrtPreparedModelCacheKind::SingleMslDispatch;
  }

  return MpsrtPreparedModelCacheKind::None;
}

std::shared_ptr<const metal::mpsrt::MpsrtPreparedModel>
MetalCompiledKernel::get_or_prepare_mpsrt_model(
    MpsrtPreparedModelCacheKind kind, std::string *error, bool *cache_hit,
    const KernelExecutionHooks *hooks) {
  if (cache_hit) {
    *cache_hit = false;
  }
  if (kind == MpsrtPreparedModelCacheKind::None) {
    (void)set_error(error, "GFX MPSRT: prepared model cache kind is not set");
    return nullptr;
  }
  if (!m_mpsrt_model) {
    (void)set_error(error,
                    "GFX MPSRT: cannot prepare model without an MPSRT model");
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(m_mpsrt_prepared_model_cache_mutex);
  if (m_mpsrt_prepared_model_cache_slots.empty()) {
    m_mpsrt_prepared_model_cache_slots.resize(
        kMpsrtPreparedModelCacheSlotCount);
  }

  for (const auto &cached_slot : m_mpsrt_prepared_model_cache_slots) {
    if (cached_slot.model && cached_slot.kind == kind) {
      if (cache_hit) {
        *cache_hit = true;
      }
      return cached_slot.model;
    }
  }

  const size_t slot_index = m_mpsrt_prepared_model_cache_next_slot;
  m_mpsrt_prepared_model_cache_next_slot =
      (m_mpsrt_prepared_model_cache_next_slot + 1) %
      m_mpsrt_prepared_model_cache_slots.size();
  auto &slot = m_mpsrt_prepared_model_cache_slots[slot_index];
  if (!m_mpsrt_context) {
    m_mpsrt_context = std::make_shared<metal::mpsrt::MpsrtContext>(
        static_cast<id<MTLDevice>>(m_device));
  }

  const bool trace_prepare = hooks && hooks->on_segment;
  const auto prepare_start =
      trace_prepare ? std::chrono::steady_clock::now()
                    : std::chrono::steady_clock::time_point{};
  auto prepared_model = std::make_shared<metal::mpsrt::MpsrtPreparedModel>();
  switch (kind) {
  case MpsrtPreparedModelCacheKind::ContextExecution:
    if (!m_mpsrt_context->prepare_model(*m_mpsrt_model, m_mpsrt_msl_source,
                                        *prepared_model, error)) {
      return nullptr;
    }
    break;
  case MpsrtPreparedModelCacheKind::SingleMslDispatch: {
    if (!m_pipeline) {
      (void)set_error(
          error,
          "GFX MPSRT: cannot prepare single MSL dispatch without a pipeline");
      return nullptr;
    }
    if (!m_mpsrt_context->prepare_model_resources(*m_mpsrt_model,
                                                  *prepared_model, error)) {
      return nullptr;
    }
    prepared_model->msl_dispatches.push_back(
        metal::mpsrt::make_prepared_msl_dispatch_from_pipeline(
            m_mpsrt_model->stages.front(), 0,
            static_cast<id<MTLComputePipelineState>>(m_pipeline)));
    break;
  }
  case MpsrtPreparedModelCacheKind::None:
    return nullptr;
  }

  if (trace_prepare) {
    std::string segment_name;
    if (!hooks->stage_type.empty()) {
      segment_name += hooks->stage_type;
      segment_name += ":";
    }
    if (!hooks->stage_name.empty()) {
      segment_name += hooks->stage_name;
      segment_name += ":";
    }
    segment_name += mpsrt_prepared_model_cache_kind_name(kind);
    hooks->on_segment(
        "mpsrt_prepare_model", segment_name,
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - prepare_start),
        0, 0, 0, 0, 0, 0, -1, 0, 0);
  }

  slot.model = std::move(prepared_model);
  slot.kind = kind;
  return slot.model;
}

void MetalCompiledKernel::prewarm_bindings(const std::vector<KernelArg> &args) {
  auto prepared_base =
      get_or_create_prepared_bindings(args, "MetalCompiledKernel prewarm");
  (void)prepared_base->get_or_create_backend_state<MetalPreparedState>(
      reinterpret_cast<uintptr_t>(
          m_binding_schema.get() ? m_binding_schema.get() : m_device),
      [&]() {
        return std::make_shared<MetalPreparedState>(
            prepared_base->binding_table());
      });

  const auto mpsrt_cache_kind = resolve_mpsrt_prepared_model_cache_kind();
  if (mpsrt_cache_kind != MpsrtPreparedModelCacheKind::None) {
    std::string mpsrt_error;
    const auto prepared_model =
        get_or_prepare_mpsrt_model(mpsrt_cache_kind, &mpsrt_error, nullptr);
    OPENVINO_ASSERT(prepared_model, mpsrt_error);
  }
}

void MetalCompiledKernel::execute(GpuCommandBufferHandle command_buffer,
                                  const KernelDispatch &dispatch,
                                  const std::vector<KernelArg> &args,
                                  const KernelExecutionHooks *hooks) {
  id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(command_buffer);
  OPENVINO_ASSERT(cb, "MetalCompiledKernel: command buffer is null");

  const auto execution_binding_plan = binding_plan();
  const auto prepared_base =
      get_or_create_prepared_bindings(args, "MetalCompiledKernel");

  const bool trace_bindings = hooks && (hooks->on_segment || hooks->on_counter);
  const auto binding_start = trace_bindings
                                 ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
  bool prepared_state_created = false;
  auto prepared =
      prepared_base->get_or_create_backend_state<MetalPreparedState>(
          reinterpret_cast<uintptr_t>(
              m_binding_schema.get() ? m_binding_schema.get() : m_device),
          [&]() {
            prepared_state_created = true;
            return std::make_shared<MetalPreparedState>(
                prepared_base->binding_table());
          });
  if (trace_bindings && prepared_state_created) {
    const auto binding_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - binding_start);
    if (hooks->on_counter) {
      hooks->on_counter("binding_prepare_count", 1);
    }
    if (hooks->on_segment) {
      hooks->on_segment("binding_prepare", "metal_prepared_state",
                        binding_cpu_us, 0, 0, 0, 0, 0, 0, -1, 0,
                        reinterpret_cast<uint64_t>(cb));
    }
  } else if (hooks && hooks->on_counter) {
    hooks->on_counter("prepared_binding_cache_hit_count", 1);
  }

  const auto mpsrt_cache_kind = resolve_mpsrt_prepared_model_cache_kind();
  if (mpsrt_cache_kind == MpsrtPreparedModelCacheKind::ContextExecution) {
    std::string mpsrt_error;
    bool mpsrt_prepared_cache_hit = false;
    const auto prepared_model = get_or_prepare_mpsrt_model(
        MpsrtPreparedModelCacheKind::ContextExecution, &mpsrt_error,
        &mpsrt_prepared_cache_hit, hooks);
    OPENVINO_ASSERT(prepared_model, mpsrt_error);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> completed_command_buffer) {
      (void)completed_command_buffer;
      (void)prepared_model;
    }];
    if (hooks && hooks->on_counter) {
      hooks->on_counter(mpsrt_prepared_cache_hit
                            ? "mpsrt_prepared_model_cache_hit_count"
                            : "mpsrt_prepared_model_cache_miss_count",
                        1);
    }
    record_mpsrt_prepared_model_counters(hooks, *prepared_model);
    metal::mpsrt::MpsrtRequest request;
    metal::mpsrt::MpsrtRequestBindingSet binding_set;
    const bool bindings_built =
        request.build_binding_set_from_runtime_buffers(
            *m_mpsrt_model, *execution_binding_plan, prepared->buffer_ptrs,
            prepared->offsets, prepared_model.get(), binding_set, hooks,
            &mpsrt_error);
    OPENVINO_ASSERT(bindings_built, mpsrt_error);

    std::vector<KernelDispatch> stage_dispatches(m_mpsrt_model->stages.size(),
                                                 dispatch);
    metal::mpsrt::MpsrtModelEncodeResult encode_result;
    const bool encoded = request.encode_prepared_model_with_binding_set(
        command_buffer, *m_mpsrt_context, *m_mpsrt_model, *prepared_model,
        stage_dispatches, binding_set, hooks, &encode_result, &mpsrt_error);
    OPENVINO_ASSERT(encoded, mpsrt_error);
    return;
  }

  if (mpsrt_cache_kind == MpsrtPreparedModelCacheKind::SingleMslDispatch) {
    OPENVINO_ASSERT(m_pipeline,
                    "MetalCompiledKernel: MSL MPSRT pipeline is null");
    std::string mpsrt_error;
    bool mpsrt_prepared_cache_hit = false;
    const auto prepared_model = get_or_prepare_mpsrt_model(
        MpsrtPreparedModelCacheKind::SingleMslDispatch, &mpsrt_error,
        &mpsrt_prepared_cache_hit, hooks);
    OPENVINO_ASSERT(prepared_model, mpsrt_error);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> completed_command_buffer) {
      (void)completed_command_buffer;
      (void)prepared_model;
    }];
    if (hooks && hooks->on_counter) {
      hooks->on_counter(mpsrt_prepared_cache_hit
                            ? "mpsrt_prepared_model_cache_hit_count"
                            : "mpsrt_prepared_model_cache_miss_count",
                        1);
    }
    record_mpsrt_prepared_model_counters(hooks, *prepared_model);
    metal::mpsrt::MpsrtRequest request;
    metal::mpsrt::MpsrtRequestBindingSet binding_set;
    const bool bindings_built =
        request.build_binding_set_from_runtime_buffers(
            *m_mpsrt_model, *execution_binding_plan, prepared->buffer_ptrs,
            prepared->offsets, prepared_model.get(), binding_set, hooks,
            &mpsrt_error);
    OPENVINO_ASSERT(bindings_built, mpsrt_error);

    std::vector<KernelDispatch> stage_dispatches = {dispatch};
    metal::mpsrt::MpsrtModelEncodeResult encode_result;
    const bool encoded = request.encode_prepared_model_with_binding_set(
        command_buffer, *m_mpsrt_context, *m_mpsrt_model, *prepared_model,
        stage_dispatches, binding_set, hooks, &encode_result, &mpsrt_error);
    OPENVINO_ASSERT(encoded, mpsrt_error);
    return;
  }

  OPENVINO_ASSERT(m_pipeline, "MetalCompiledKernel: pipeline is null");
  const bool trace_encoder_setup =
      hooks && (hooks->on_segment || hooks->on_counter);
  const auto encoder_setup_start =
      trace_encoder_setup ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
  bool encoder_created = false;
  id<MTLComputeCommandEncoder> enc = static_cast<id<MTLComputeCommandEncoder>>(
      metal_get_or_create_compute_encoder(command_buffer, &encoder_created));
  OPENVINO_ASSERT(enc, "MetalCompiledKernel: failed to create compute encoder");
  const auto pipeline_bind_start =
      trace_encoder_setup ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
  const bool pipeline_bound = metal_set_compute_pipeline_if_needed(
      command_buffer, reinterpret_cast<GpuCommandEncoderHandle>(enc),
      m_pipeline);
  const auto after_pipeline_bind =
      trace_encoder_setup ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};

  const auto buffer_bind_start = trace_encoder_setup
                                     ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
  const size_t bound_buffers = metal_bind_compute_buffers_if_needed(
      command_buffer, reinterpret_cast<GpuCommandEncoderHandle>(enc),
      prepared->buffer_ptrs, prepared->offsets);
  if (trace_encoder_setup) {
    const auto encoder_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - encoder_setup_start);
    const auto pipeline_bind_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            after_pipeline_bind - pipeline_bind_start);
    const auto buffer_bind_cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - buffer_bind_start);
    if (hooks->on_counter) {
      if (encoder_created) {
        hooks->on_counter("encoder_setup_count", 1);
      }
      if (pipeline_bound) {
        hooks->on_counter("pipeline_bind_count", 1);
      }
      hooks->on_counter("buffer_bind_count",
                        static_cast<uint64_t>(bound_buffers));
    }
    if (hooks->on_segment) {
      hooks->on_segment("descriptor_update", "metal_pipeline_bind",
                        pipeline_bind_cpu_us, 0, 0, 0, 0, 0, 0, -1, 0,
                        reinterpret_cast<uint64_t>(cb));
      hooks->on_segment("descriptor_update", "metal_buffer_bind",
                        buffer_bind_cpu_us, 0,
                        static_cast<uint32_t>(bound_buffers), 0, 0, 0, 0, -1, 0,
                        reinterpret_cast<uint64_t>(cb));
      if (encoder_created &&
          encoder_cpu_us > pipeline_bind_cpu_us + buffer_bind_cpu_us) {
        hooks->on_segment(
            "descriptor_update", "metal_encoder_setup_overhead",
            encoder_cpu_us - pipeline_bind_cpu_us - buffer_bind_cpu_us, 0, 0, 0,
            0, 0, 0, -1, 0, reinterpret_cast<uint64_t>(cb));
      }
    }
  }

  if (hooks && hooks->on_begin) {
    hooks->on_begin(enc);
  }

  const size_t grid_x = dispatch.grid[0];
  const size_t grid_y = dispatch.grid[1];
  const size_t grid_z = dispatch.grid[2];
  if (grid_x == 0 || grid_y == 0 || grid_z == 0) {
    if (hooks && hooks->on_end) {
      hooks->on_end(enc);
    }
    return;
  }

  MTLSize grid = MTLSizeMake(grid_x, grid_y, grid_z);
  MTLSize tg =
      MTLSizeMake(dispatch.threads_per_group[0], dispatch.threads_per_group[1],
                  dispatch.threads_per_group[2]);
  [enc dispatchThreads:grid threadsPerThreadgroup:tg];

  if (hooks && hooks->on_end) {
    hooks->on_end(enc);
  }
}

} // namespace gfx_plugin
} // namespace ov
