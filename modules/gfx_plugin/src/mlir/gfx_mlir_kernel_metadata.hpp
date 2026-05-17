// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "kernel_ir/gfx_kernel_dispatch.hpp"
#include "kernel_ir/gfx_kernel_inputs.hpp"
#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "kernel_ir/gfx_kernel_signature.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "openvino/core/except.hpp"
#include "llvm/Support/Casting.h"

namespace ov {
namespace gfx_plugin {

struct KernelOperandMetadata {
  std::vector<int32_t> operand_kinds;
  std::vector<int32_t> operand_arg_indices;
  std::vector<int32_t> scalar_args;
};

struct KernelRuntimeBindingState {
  std::vector<size_t> inputs;
  size_t input_arg_count = 0;
  std::vector<int32_t> operand_kinds;
  std::vector<int32_t> operand_arg_indices;
  std::vector<int32_t> scalar_args;
};

struct GfxKernelRuntimeBindingPlan {
  bool valid = false;
  KernelRuntimeBindingState runtime_binding;
  size_t scalar_arg_count = 0;
  GfxKernelStageManifest stage_manifest;
};

inline KernelRuntimeBindingState
make_kernel_runtime_binding_state(std::vector<size_t> inputs,
                                  size_t input_arg_count,
                                  std::vector<int32_t> operand_kinds,
                                  std::vector<int32_t> operand_arg_indices,
                                  std::vector<int32_t> scalar_args = {}) {
  return KernelRuntimeBindingState{
      std::move(inputs), input_arg_count, std::move(operand_kinds),
      std::move(operand_arg_indices), std::move(scalar_args)};
}

inline GfxKernelRuntimeBindingPlan
make_kernel_runtime_binding_plan_from_stage_manifest(
    const GfxKernelStageManifest &manifest) {
  GfxKernelRuntimeBindingPlan plan{};
  if (!manifest.valid || !manifest.custom_kernel.valid ||
      !manifest.custom_kernel.external_buffer_abi.valid) {
    return plan;
  }

  const auto roles = materialize_gfx_kernel_external_buffer_roles(
      manifest.custom_kernel.external_buffer_abi);
  if (roles.empty()) {
    return plan;
  }

  plan.valid = true;
  plan.stage_manifest = manifest;
  plan.runtime_binding.operand_kinds.reserve(roles.size());
  plan.runtime_binding.operand_arg_indices.reserve(roles.size());

  size_t logical_input_arg_count = 0;
  size_t tensor_input_count = 0;
  for (const auto role : roles) {
    if (is_gfx_kernel_buffer_role(role) && !is_gfx_kernel_output_role(role)) {
      ++logical_input_arg_count;
    }
    if (role == GfxKernelBufferRole::TensorInput) {
      ++tensor_input_count;
    }
  }

  size_t next_tensor_input = 0;
  size_t next_extra_buffer = 0;
  size_t output_count = 0;
  for (const auto role : roles) {
    if (is_gfx_kernel_scalar_role(role)) {
      plan.runtime_binding.operand_kinds.push_back(0);
      plan.runtime_binding.operand_arg_indices.push_back(-1);
      ++plan.scalar_arg_count;
      continue;
    }

    plan.runtime_binding.operand_kinds.push_back(1);
    switch (role) {
    case GfxKernelBufferRole::TensorInput:
      plan.runtime_binding.operand_arg_indices.push_back(
          static_cast<int32_t>(next_tensor_input));
      plan.runtime_binding.inputs.push_back(next_tensor_input++);
      break;
    case GfxKernelBufferRole::ConstTensor:
    case GfxKernelBufferRole::RuntimeParams:
      plan.runtime_binding.operand_arg_indices.push_back(
          static_cast<int32_t>(tensor_input_count + next_extra_buffer++));
      break;
    case GfxKernelBufferRole::TensorOutput:
      plan.runtime_binding.operand_arg_indices.push_back(
          static_cast<int32_t>(logical_input_arg_count + output_count));
      ++output_count;
      break;
    case GfxKernelBufferRole::ScalarParam:
      break;
    case GfxKernelBufferRole::Unknown:
    default:
      return {};
    }
  }
  if (next_tensor_input != tensor_input_count || output_count == 0) {
    return {};
  }
  plan.runtime_binding.input_arg_count = logical_input_arg_count;
  return plan;
}

inline mlir::ArrayAttr
make_kernel_i32_array_attr(mlir::MLIRContext *ctx,
                           const std::vector<int32_t> &vals) {
  mlir::OpBuilder builder(ctx);
  llvm::SmallVector<mlir::Attribute, 8> attrs;
  attrs.reserve(vals.size());
  for (auto v : vals) {
    attrs.push_back(builder.getI32IntegerAttr(v));
  }
  return builder.getArrayAttr(attrs);
}

inline void annotate_kernel_operand_abi_attrs_for_spirv_adapter(
    mlir::ModuleOp module, const KernelRuntimeBindingState &binding) {
  OPENVINO_ASSERT(
      module, "GFX MLIR: SPIR-V kernel operand adapter attrs require a module");
  OPENVINO_ASSERT(
      binding.operand_kinds.size() == binding.operand_arg_indices.size(),
      "GFX MLIR: SPIR-V kernel operand adapter attr sizes mismatch");
  module->setAttr(
      "gfx.kernel_operand_kinds",
      make_kernel_i32_array_attr(module.getContext(), binding.operand_kinds));
  module->setAttr("gfx.kernel_operand_arg_indices",
                  make_kernel_i32_array_attr(module.getContext(),
                                             binding.operand_arg_indices));
  if (!binding.scalar_args.empty()) {
    module->setAttr(
        "gfx.kernel_scalar_values",
        make_kernel_i32_array_attr(module.getContext(), binding.scalar_args));
  } else {
    module->removeAttr("gfx.kernel_scalar_values");
  }
}

struct KernelRuntimeMetadata {
  bool valid = false;
  ParallelDispatchConfig dispatch;
  bool force_single_dispatch = false;
  KernelOperandMetadata operands;
  std::vector<size_t> kernel_inputs;
  size_t kernel_input_arg_count = 0;
};

struct KernelSignatureInfo {
  KernelFunctionSignature signature;
  size_t scalar_inputs = 0;
};

struct KernelArgMappingInfo {
  KernelFunctionSignature signature;
  size_t scalar_inputs = 0;
  size_t func_inputs = 0;
  size_t func_results = 0;
  size_t output_args = 0;
  size_t buffer_inputs = 0;
  KernelInputMapping mapping;
};

inline size_t count_scalar_inputs(mlir::FunctionType type) {
  if (!type) {
    return 0;
  }
  size_t scalar_inputs = 0;
  for (auto input_type : type.getInputs()) {
    if (!mlir::isa<mlir::ShapedType>(input_type)) {
      ++scalar_inputs;
    }
  }
  return scalar_inputs;
}

inline size_t infer_extra_inputs_for_mapping(size_t buffer_inputs,
                                             size_t node_inputs,
                                             size_t extra_inputs) {
  if (buffer_inputs <= node_inputs) {
    return 0;
  }
  const size_t inferred = buffer_inputs - node_inputs;
  return std::min(inferred, extra_inputs);
}

inline ParallelDispatchConfig
extract_kernel_dispatch_metadata(mlir::ModuleOp module) {
  ParallelDispatchConfig meta;
  if (!module) {
    return meta;
  }
  if (auto attr =
          module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch")) {
    meta.enabled = attr.getValue();
  }
  if (auto attr =
          module->getAttrOfType<mlir::IntegerAttr>("gfx.parallel_loop_dims")) {
    meta.loop_dims = static_cast<size_t>(attr.getInt());
  }
  if (auto attr =
          module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_tile_h")) {
    meta.tile_h = static_cast<uint32_t>(attr.getInt());
  }
  if (auto attr =
          module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_tile_w")) {
    meta.tile_w = static_cast<uint32_t>(attr.getInt());
  }
  if (auto attr =
          module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_h")) {
    meta.threads_h = static_cast<uint32_t>(attr.getInt());
  }
  if (auto attr =
          module->getAttrOfType<mlir::IntegerAttr>("gfx.dispatch_threads_w")) {
    meta.threads_w = static_cast<uint32_t>(attr.getInt());
  }
  if (meta.threads_h == 1) {
    meta.threads_h = meta.tile_h;
  }
  if (meta.threads_w == 1) {
    meta.threads_w = meta.tile_w;
  }
  return meta;
}

inline bool extract_kernel_force_single_dispatch(mlir::ModuleOp module) {
  if (!module) {
    return false;
  }
  if (auto attr =
          module->getAttrOfType<mlir::BoolAttr>("gfx.force_single_dispatch")) {
    return attr.getValue();
  }
  return false;
}

inline std::vector<int32_t> extract_kernel_scalar_args(mlir::ModuleOp module) {
  std::vector<int32_t> scalars;
  if (!module) {
    return scalars;
  }
  if (auto attr = module->getAttr("gfx.kernel_scalar_args")) {
    if (auto attrs = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      scalars.reserve(attrs.size());
      for (auto attr_val : attrs) {
        if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(attr_val)) {
          scalars.push_back(static_cast<int32_t>(iattr.getInt()));
        }
      }
      return scalars;
    }
    if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr)) {
      auto vals = dense.asArrayRef();
      scalars.assign(vals.begin(), vals.end());
      return scalars;
    }
    if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
      scalars.reserve(dense.getNumElements());
      for (auto v : dense.getValues<int32_t>()) {
        scalars.push_back(v);
      }
      return scalars;
    }
  }
  return scalars;
}

inline std::vector<int32_t>
extract_kernel_scalar_values(mlir::ModuleOp module) {
  if (!module) {
    return {};
  }
  if (auto attr = module->getAttr("gfx.kernel_scalar_values")) {
    if (auto attrs = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      std::vector<int32_t> values;
      values.reserve(attrs.size());
      for (auto attr_val : attrs) {
        if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(attr_val)) {
          values.push_back(static_cast<int32_t>(iattr.getInt()));
        }
      }
      return values;
    }
    if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr)) {
      auto vals = dense.asArrayRef();
      return std::vector<int32_t>(vals.begin(), vals.end());
    }
    if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
      std::vector<int32_t> values;
      values.reserve(dense.getNumElements());
      for (auto v : dense.getValues<int32_t>()) {
        values.push_back(v);
      }
      return values;
    }
  }
  return extract_kernel_scalar_args(module);
}

inline std::vector<int32_t>
extract_kernel_operand_kinds(mlir::ModuleOp module) {
  std::vector<int32_t> kinds;
  if (!module) {
    return kinds;
  }
  if (auto attr = module->getAttr("gfx.kernel_operand_kinds")) {
    if (auto attrs = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      kinds.reserve(attrs.size());
      for (auto attr_val : attrs) {
        if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(attr_val)) {
          kinds.push_back(static_cast<int32_t>(iattr.getInt()));
        }
      }
      return kinds;
    }
    if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr)) {
      auto vals = dense.asArrayRef();
      kinds.assign(vals.begin(), vals.end());
      return kinds;
    }
    if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
      kinds.reserve(dense.getNumElements());
      for (auto v : dense.getValues<int32_t>()) {
        kinds.push_back(v);
      }
      return kinds;
    }
  }
  return kinds;
}

inline std::vector<int32_t>
extract_kernel_operand_arg_indices(mlir::ModuleOp module) {
  std::vector<int32_t> indices;
  if (!module) {
    return indices;
  }
  if (auto attr = module->getAttr("gfx.kernel_operand_arg_indices")) {
    if (auto attrs = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
      indices.reserve(attrs.size());
      for (auto attr_val : attrs) {
        if (auto iattr = mlir::dyn_cast<mlir::IntegerAttr>(attr_val)) {
          indices.push_back(static_cast<int32_t>(iattr.getInt()));
        }
      }
      return indices;
    }
    if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(attr)) {
      auto vals = dense.asArrayRef();
      indices.assign(vals.begin(), vals.end());
      return indices;
    }
    if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
      indices.reserve(dense.getNumElements());
      for (auto v : dense.getValues<int32_t>()) {
        indices.push_back(v);
      }
      return indices;
    }
  }
  return indices;
}

inline KernelSignatureInfo
extract_kernel_signature_info(mlir::ModuleOp module, const std::string &entry) {
  KernelSignatureInfo info;
  info.signature = infer_kernel_signature(module, entry);
  if (!module) {
    return info;
  }
  if (!entry.empty()) {
    if (auto func = module.lookupSymbol<mlir::func::FuncOp>(entry)) {
      info.scalar_inputs = count_scalar_inputs(func.getFunctionType());
      return info;
    }
    if (auto gpu_func = module.lookupSymbol<mlir::gpu::GPUFuncOp>(entry)) {
      info.scalar_inputs = count_scalar_inputs(gpu_func.getFunctionType());
      return info;
    }
  }
  mlir::gpu::GPUFuncOp gpu_func;
  module.walk([&](mlir::gpu::GPUFuncOp f) {
    if (!gpu_func) {
      gpu_func = f;
    }
  });
  if (gpu_func) {
    info.scalar_inputs = count_scalar_inputs(gpu_func.getFunctionType());
    return info;
  }
  mlir::func::FuncOp func;
  module.walk([&](mlir::func::FuncOp f) {
    if (!func) {
      func = f;
    }
  });
  if (func) {
    info.scalar_inputs = count_scalar_inputs(func.getFunctionType());
  }
  return info;
}

inline KernelArgMappingInfo
build_kernel_arg_mapping(mlir::ModuleOp module, const std::string &entry,
                         const std::shared_ptr<const ov::Node> &node,
                         size_t output_args_override, size_t extra_inputs,
                         const char *stage_name) {
  KernelArgMappingInfo info;
  const auto sig_info = extract_kernel_signature_info(module, entry);
  info.signature = sig_info.signature;
  info.scalar_inputs = sig_info.scalar_inputs;
  info.func_inputs = info.signature.inputs;
  info.func_results = info.signature.results;
  info.output_args = output_args_override;
  if (info.func_results == 0 && info.output_args == 0 && node) {
    info.output_args = node->get_output_size();
  }
  size_t buffer_inputs = info.func_inputs;
  if (info.scalar_inputs <= buffer_inputs) {
    buffer_inputs -= info.scalar_inputs;
  } else {
    buffer_inputs = 0;
  }
  // Only memref-style kernels pass outputs as trailing function arguments.
  // Tensor-returning kernels keep outputs in the result list and must not
  // lose real input operands here, otherwise constant OV inputs (for example
  // Split axis/lengths) get misclassified as runtime buffers.
  if (info.func_results == 0) {
    if (info.output_args <= buffer_inputs) {
      buffer_inputs -= info.output_args;
    } else {
      buffer_inputs = 0;
    }
  }
  info.buffer_inputs = buffer_inputs;
  const size_t node_inputs = node ? node->get_input_size() : 0;
  const size_t extra_inputs_for_mapping =
      infer_extra_inputs_for_mapping(buffer_inputs, node_inputs, extra_inputs);
  info.mapping = build_kernel_inputs(node, buffer_inputs, stage_name,
                                     extra_inputs_for_mapping);
  if (info.mapping.func_inputs != 0) {
    info.func_inputs = info.mapping.func_inputs;
  }
  return info;
}

inline KernelOperandMetadata
extract_kernel_operand_metadata(mlir::ModuleOp module) {
  KernelOperandMetadata meta;
  meta.operand_kinds = extract_kernel_operand_kinds(module);
  meta.operand_arg_indices = extract_kernel_operand_arg_indices(module);
  meta.scalar_args = extract_kernel_scalar_values(module);
  return meta;
}

inline std::vector<GfxKernelBufferRole>
materialize_kernel_external_buffer_roles(
    const GfxKernelExternalBufferAbiSpec &abi) {
  return materialize_gfx_kernel_external_buffer_roles(abi);
}

inline bool extract_kernel_operand_metadata_from_custom_kernel_manifest(
    const GfxKernelStageManifest &manifest, KernelOperandMetadata &meta,
    size_t &kernel_input_arg_count, std::string_view entry_point = {},
    std::optional<GfxKernelBackendDomain> expected_backend_domain =
        std::nullopt) {
  if (!manifest.valid ||
      (expected_backend_domain &&
       manifest.backend_domain != *expected_backend_domain) ||
      manifest.execution_kind != GfxKernelExecutionKind::CustomKernel ||
      !manifest.custom_kernel.valid ||
      !manifest.custom_kernel.external_buffer_abi.valid) {
    return false;
  }
  if (!entry_point.empty() &&
      manifest.backend_domain != GfxKernelBackendDomain::Spirv &&
      manifest.custom_kernel.entry_point != entry_point) {
    return false;
  }

  const auto roles = materialize_kernel_external_buffer_roles(
      manifest.custom_kernel.external_buffer_abi);
  if (roles.empty()) {
    return false;
  }

  size_t tensor_input_count = 0;
  size_t logical_input_arg_count = 0;
  for (const auto role : roles) {
    if (role == GfxKernelBufferRole::TensorInput) {
      ++tensor_input_count;
    }
    if (is_gfx_kernel_buffer_role(role) && !is_gfx_kernel_output_role(role)) {
      ++logical_input_arg_count;
    }
  }

  KernelOperandMetadata manifest_meta;
  manifest_meta.operand_kinds.reserve(roles.size());
  manifest_meta.operand_arg_indices.reserve(roles.size());
  manifest_meta.scalar_args = manifest.custom_kernel.scalar_args;

  size_t next_tensor_input = 0;
  size_t next_extra_buffer = 0;
  size_t output_count = 0;
  for (const auto role : roles) {
    if (is_gfx_kernel_scalar_role(role)) {
      manifest_meta.operand_kinds.push_back(0);
      manifest_meta.operand_arg_indices.push_back(-1);
      continue;
    }

    if (!is_gfx_kernel_buffer_role(role)) {
      return false;
    }
    manifest_meta.operand_kinds.push_back(1);
    switch (role) {
    case GfxKernelBufferRole::TensorInput:
      manifest_meta.operand_arg_indices.push_back(
          static_cast<int32_t>(next_tensor_input++));
      break;
    case GfxKernelBufferRole::ConstTensor:
    case GfxKernelBufferRole::RuntimeParams:
      manifest_meta.operand_arg_indices.push_back(
          static_cast<int32_t>(tensor_input_count + next_extra_buffer++));
      break;
    case GfxKernelBufferRole::TensorOutput:
      manifest_meta.operand_arg_indices.push_back(
          static_cast<int32_t>(logical_input_arg_count + output_count));
      ++output_count;
      break;
    case GfxKernelBufferRole::ScalarParam:
    case GfxKernelBufferRole::Unknown:
    default:
      return false;
    }
  }
  if (next_tensor_input != tensor_input_count || output_count == 0) {
    return false;
  }

  meta = std::move(manifest_meta);
  kernel_input_arg_count = logical_input_arg_count;
  return true;
}

inline bool extract_kernel_operand_metadata_from_stage_manifest(
    mlir::ModuleOp module, KernelOperandMetadata &meta,
    size_t &kernel_input_arg_count, std::string_view entry_point = {},
    std::optional<GfxKernelBackendDomain> expected_backend_domain =
        std::nullopt) {
  GfxKernelStageManifest manifest{};
  if (!detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest)) {
    return false;
  }
  return extract_kernel_operand_metadata_from_custom_kernel_manifest(
      manifest, meta, kernel_input_arg_count, entry_point,
      expected_backend_domain);
}

inline bool extract_kernel_operand_metadata_from_mpsrt_typed_program_manifest(
    mlir::ModuleOp module, KernelOperandMetadata &meta,
    size_t &kernel_input_arg_count, std::string_view entry_point = {},
    std::optional<GfxKernelBackendDomain> expected_backend_domain =
        std::nullopt) {
  if (expected_backend_domain &&
      *expected_backend_domain != GfxKernelBackendDomain::AppleMsl) {
    return false;
  }
  if (!module_has_mpsrt_ops_program(module)) {
    return false;
  }
  GfxMpsrtProgram program{};
  if (!read_module_mpsrt_program(module, program) || !program.valid) {
    return false;
  }

  bool saw_msl_stage = false;
  for (const auto &stage_spec : program.stages) {
    const auto &stage = stage_spec.stage;
    if (stage.kind != GfxMpsrtStageKind::MSLDispatch &&
        stage.domain != GfxStageBackendDomain::AppleMsl) {
      continue;
    }
    saw_msl_stage = true;
    if (extract_kernel_operand_metadata_from_custom_kernel_manifest(
            stage.stage_manifest, meta, kernel_input_arg_count, entry_point,
            GfxKernelBackendDomain::AppleMsl)) {
      return true;
    }
  }
  if (!saw_msl_stage || !entry_point.empty()) {
    return false;
  }

  for (const auto &stage_spec : program.stages) {
    const auto &stage = stage_spec.stage;
    if (stage.kind != GfxMpsrtStageKind::MSLDispatch &&
        stage.domain != GfxStageBackendDomain::AppleMsl) {
      continue;
    }
    if (extract_kernel_operand_metadata_from_custom_kernel_manifest(
            stage.stage_manifest, meta, kernel_input_arg_count,
            /*entry_point=*/{}, GfxKernelBackendDomain::AppleMsl)) {
      return true;
    }
  }
  return false;
}

inline bool extract_kernel_operand_metadata_from_mpsrt_external_buffer_abi(
    mlir::ModuleOp module, KernelOperandMetadata &meta,
    size_t &kernel_input_arg_count) {
  const auto external_buffer_abi =
      read_module_mpsrt_external_buffer_abi(module);
  if (!external_buffer_abi.valid || !external_buffer_abi.has_buffer_roles ||
      external_buffer_abi.buffer_roles.empty()) {
    return false;
  }

  size_t tensor_input_count = 0;
  size_t logical_input_arg_count = 0;
  for (const auto role : external_buffer_abi.buffer_roles) {
    if (role == GfxMpsrtExternalBufferRole::TensorInput) {
      ++tensor_input_count;
    }
    if (role != GfxMpsrtExternalBufferRole::TensorOutput) {
      ++logical_input_arg_count;
    }
  }

  KernelOperandMetadata abi_meta;
  abi_meta.operand_kinds.reserve(external_buffer_abi.buffer_roles.size());
  abi_meta.operand_arg_indices.reserve(external_buffer_abi.buffer_roles.size());
  abi_meta.scalar_args = extract_kernel_scalar_values(module);

  size_t next_tensor_input = 0;
  size_t next_extra_buffer = 0;
  size_t output_count = 0;
  for (const auto role : external_buffer_abi.buffer_roles) {
    abi_meta.operand_kinds.push_back(1);
    switch (role) {
    case GfxMpsrtExternalBufferRole::TensorInput:
      abi_meta.operand_arg_indices.push_back(
          static_cast<int32_t>(next_tensor_input++));
      break;
    case GfxMpsrtExternalBufferRole::ConstBuffer:
    case GfxMpsrtExternalBufferRole::RuntimeParams:
    case GfxMpsrtExternalBufferRole::Metadata:
      abi_meta.operand_arg_indices.push_back(
          static_cast<int32_t>(tensor_input_count + next_extra_buffer++));
      break;
    case GfxMpsrtExternalBufferRole::TensorOutput:
      abi_meta.operand_arg_indices.push_back(
          static_cast<int32_t>(logical_input_arg_count + output_count));
      ++output_count;
      break;
    case GfxMpsrtExternalBufferRole::Unknown:
    default:
      return false;
    }
  }

  if (next_tensor_input != tensor_input_count || output_count == 0 ||
      output_count != external_buffer_abi.output_buffer_count) {
    return false;
  }

  meta = std::move(abi_meta);
  kernel_input_arg_count = logical_input_arg_count;
  return true;
}

inline bool extract_spirv_kernel_operand_metadata_from_adapter_attrs(
    mlir::ModuleOp module, KernelOperandMetadata &meta,
    size_t &kernel_input_arg_count, size_t output_arg_count,
    std::vector<size_t> *kernel_inputs = nullptr,
    std::optional<GfxKernelBackendDomain> expected_backend_domain =
        std::nullopt) {
  if (!module ||
      (expected_backend_domain &&
       *expected_backend_domain != GfxKernelBackendDomain::Spirv)) {
    return false;
  }

  GfxKernelStageManifest manifest{};
  if (!detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest) ||
      !manifest.valid || manifest.backend_domain != GfxKernelBackendDomain::Spirv ||
      manifest.execution_kind != GfxKernelExecutionKind::CustomKernel ||
      !manifest.custom_kernel.valid) {
    return false;
  }

  auto adapter_meta = extract_kernel_operand_metadata(module);
  if (adapter_meta.operand_kinds.empty() ||
      adapter_meta.operand_arg_indices.size() !=
          adapter_meta.operand_kinds.size()) {
    return false;
  }

  if (kernel_inputs) {
    kernel_inputs->clear();
    const auto roles = materialize_kernel_external_buffer_roles(
        manifest.custom_kernel.external_buffer_abi);
    size_t next_tensor_input = 0;
    for (const auto role : roles) {
      if (role == GfxKernelBufferRole::TensorInput) {
        kernel_inputs->push_back(next_tensor_input++);
      }
    }
  }

  meta = std::move(adapter_meta);
  int32_t max_idx = -1;
  for (auto idx : meta.operand_arg_indices) {
    if (idx > max_idx) {
      max_idx = idx;
    }
  }
  if (max_idx < 0) {
    return false;
  }
  const size_t total_buffer_args = static_cast<size_t>(max_idx) + 1;
  if (output_arg_count > total_buffer_args) {
    return false;
  }
  kernel_input_arg_count = total_buffer_args - output_arg_count;
  return true;
}

inline bool
infer_kernel_arg_count_from_manifest(const GfxKernelStageManifest &manifest,
                                     size_t &arg_count,
                                     std::string_view entry_point = {},
                                     std::optional<GfxKernelBackendDomain>
                                         expected_backend_domain =
                                             std::nullopt) {
  if (!manifest.valid ||
      (expected_backend_domain &&
       manifest.backend_domain != *expected_backend_domain) ||
      manifest.execution_kind != GfxKernelExecutionKind::CustomKernel ||
      !manifest.custom_kernel.valid ||
      !manifest.custom_kernel.external_buffer_abi.valid) {
    return false;
  }
  if (!entry_point.empty() &&
      manifest.backend_domain != GfxKernelBackendDomain::Spirv &&
      manifest.custom_kernel.entry_point != entry_point) {
    return false;
  }
  const auto roles = materialize_kernel_external_buffer_roles(
      manifest.custom_kernel.external_buffer_abi);
  if (roles.empty()) {
    return false;
  }
  size_t runtime_arg_count = 0;
  for (const auto role : roles) {
    if (is_gfx_kernel_buffer_role(role) || is_gfx_kernel_scalar_role(role)) {
      ++runtime_arg_count;
    }
  }
  if (runtime_arg_count == 0) {
    return false;
  }
  arg_count = runtime_arg_count;
  return true;
}

inline bool
infer_kernel_arg_count_from_stage_manifest(mlir::ModuleOp module,
                                           size_t &arg_count,
                                           std::string_view entry_point = {},
                                           std::optional<GfxKernelBackendDomain>
                                               expected_backend_domain =
                                                   std::nullopt) {
  GfxKernelStageManifest manifest{};
  if (!detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest)) {
    return false;
  }
  return infer_kernel_arg_count_from_manifest(manifest, arg_count, entry_point,
                                              expected_backend_domain);
}

inline bool infer_kernel_arg_count_from_mpsrt_typed_program_manifest(
    mlir::ModuleOp module, size_t &arg_count,
    std::string_view entry_point = {},
    std::optional<GfxKernelBackendDomain> expected_backend_domain =
        std::nullopt) {
  if (expected_backend_domain &&
      *expected_backend_domain != GfxKernelBackendDomain::AppleMsl) {
    return false;
  }
  if (!module_has_mpsrt_ops_program(module)) {
    return false;
  }
  GfxMpsrtProgram program{};
  if (!read_module_mpsrt_program(module, program) || !program.valid) {
    return false;
  }

  bool saw_msl_stage = false;
  for (const auto &stage_spec : program.stages) {
    const auto &stage = stage_spec.stage;
    if (stage.kind != GfxMpsrtStageKind::MSLDispatch &&
        stage.domain != GfxStageBackendDomain::AppleMsl) {
      continue;
    }
    saw_msl_stage = true;
    if (infer_kernel_arg_count_from_manifest(stage.stage_manifest, arg_count,
                                             entry_point,
                                             GfxKernelBackendDomain::AppleMsl)) {
      return true;
    }
  }
  if (!saw_msl_stage || !entry_point.empty()) {
    return false;
  }

  for (const auto &stage_spec : program.stages) {
    const auto &stage = stage_spec.stage;
    if (stage.kind != GfxMpsrtStageKind::MSLDispatch &&
        stage.domain != GfxStageBackendDomain::AppleMsl) {
      continue;
    }
    if (infer_kernel_arg_count_from_manifest(stage.stage_manifest, arg_count,
                                             /*entry_point=*/{},
                                             GfxKernelBackendDomain::AppleMsl)) {
      return true;
    }
  }
  return false;
}

inline size_t
resolve_kernel_runtime_output_args(const KernelArgMappingInfo &mapping,
                                   const std::shared_ptr<const ov::Node> &node,
                                   size_t outputs_hint = 0) {
  if (mapping.output_args != 0) {
    return mapping.output_args;
  }
  if (outputs_hint != 0) {
    return outputs_hint;
  }
  if (node) {
    return node->get_output_size();
  }
  return 0;
}

inline size_t infer_kernel_input_arg_count_from_operand_indices(
    const std::vector<int32_t> &indices, size_t output_arg_count,
    size_t fallback) {
  if (indices.empty()) {
    return fallback;
  }
  int32_t max_idx = -1;
  for (auto idx : indices) {
    if (idx > max_idx) {
      max_idx = idx;
    }
  }
  if (max_idx < 0) {
    return fallback;
  }
  const size_t total_buffer_args = static_cast<size_t>(max_idx) + 1;
  if (output_arg_count > total_buffer_args) {
    return fallback;
  }
  return total_buffer_args - output_arg_count;
}

inline bool has_legacy_kernel_operand_metadata_attrs(mlir::ModuleOp module) {
  return module &&
         (module->hasAttr("gfx.fixed_arg_count") ||
          module->hasAttr("gfx.kernel_operand_kinds") ||
          module->hasAttr("gfx.kernel_operand_arg_indices") ||
          module->hasAttr("gfx.kernel_scalar_values") ||
          module->hasAttr("gfx.kernel_scalar_args"));
}

inline bool module_has_typed_custom_dispatch_mpsrt_program(mlir::ModuleOp module) {
  if (!module_has_mpsrt_ops_program(module)) {
    return false;
  }
  GfxMpsrtProgram program{};
  if (!read_module_mpsrt_program(module, program) || !program.valid) {
    return true;
  }
  return gfx_mpsrt_program_has_custom_dispatch_stage(program);
}

inline KernelRuntimeMetadata
extract_kernel_runtime_metadata(mlir::ModuleOp module, size_t output_arg_count,
                                size_t fallback_input_arg_count,
                                std::string_view entry_point = {},
                                std::optional<GfxKernelBackendDomain>
                                    expected_backend_domain = std::nullopt) {
  KernelRuntimeMetadata meta;
  if (!module) {
    return meta;
  }
  meta.valid = true;
  meta.dispatch = extract_kernel_dispatch_metadata(module);
  meta.force_single_dispatch = extract_kernel_force_single_dispatch(module);
  if (extract_spirv_kernel_operand_metadata_from_adapter_attrs(
          module, meta.operands, meta.kernel_input_arg_count,
          output_arg_count, &meta.kernel_inputs, expected_backend_domain)) {
    return meta;
  }
  if (extract_kernel_operand_metadata_from_mpsrt_typed_program_manifest(
          module, meta.operands, meta.kernel_input_arg_count, entry_point,
          expected_backend_domain)) {
    return meta;
  }
  if ((!expected_backend_domain ||
       *expected_backend_domain == GfxKernelBackendDomain::AppleMsl) &&
      module_has_typed_custom_dispatch_mpsrt_program(module)) {
    meta.valid = false;
    return meta;
  }
  if (extract_kernel_operand_metadata_from_stage_manifest(
          module, meta.operands, meta.kernel_input_arg_count, entry_point,
          expected_backend_domain)) {
    return meta;
  }
  if ((!expected_backend_domain ||
       *expected_backend_domain == GfxKernelBackendDomain::AppleMsl) &&
      extract_kernel_operand_metadata_from_mpsrt_external_buffer_abi(
          module, meta.operands, meta.kernel_input_arg_count)) {
    return meta;
  }
  if (has_legacy_kernel_operand_metadata_attrs(module)) {
    meta.valid = false;
    return meta;
  }
  meta.operands = {};
  meta.kernel_input_arg_count = fallback_input_arg_count;
  return meta;
}

inline KernelRuntimeMetadata extract_kernel_runtime_metadata(
    mlir::ModuleOp module, const KernelArgMappingInfo &mapping,
    const std::shared_ptr<const ov::Node> &node, size_t outputs_hint = 0,
    std::string_view entry_point = {},
    std::optional<GfxKernelBackendDomain> expected_backend_domain =
        std::nullopt) {
  const size_t output_arg_count =
      resolve_kernel_runtime_output_args(mapping, node, outputs_hint);
  return extract_kernel_runtime_metadata(module, output_arg_count,
                                         mapping.buffer_inputs, entry_point,
                                         expected_backend_domain);
}

inline size_t
infer_kernel_arg_count_from_module(mlir::ModuleOp module, size_t fallback,
                                   std::string_view entry_point = {},
                                   std::optional<GfxKernelBackendDomain>
                                       expected_backend_domain =
                                           std::nullopt) {
  if (!module) {
    return fallback;
  }
  KernelOperandMetadata adapter_meta;
  size_t adapter_arg_count = 0;
  if (extract_spirv_kernel_operand_metadata_from_adapter_attrs(
          module, adapter_meta, adapter_arg_count,
          /*output_arg_count=*/0, /*kernel_inputs=*/nullptr,
          expected_backend_domain)) {
    if (!adapter_meta.operand_kinds.empty()) {
      return adapter_meta.operand_kinds.size();
    }
    return adapter_arg_count;
  }
  size_t typed_program_arg_count = 0;
  if (infer_kernel_arg_count_from_mpsrt_typed_program_manifest(
          module, typed_program_arg_count, entry_point,
          expected_backend_domain)) {
    return typed_program_arg_count;
  }
  if ((!expected_backend_domain ||
       *expected_backend_domain == GfxKernelBackendDomain::AppleMsl) &&
      module_has_typed_custom_dispatch_mpsrt_program(module)) {
    return fallback;
  }
  size_t manifest_arg_count = 0;
  if (infer_kernel_arg_count_from_stage_manifest(module, manifest_arg_count,
                                                 entry_point,
                                                 expected_backend_domain)) {
    return manifest_arg_count;
  }
  size_t launch_operand_count = 0;
  module.walk([&](mlir::gpu::LaunchFuncOp launch) {
    if (launch_operand_count == 0) {
      launch_operand_count = launch.getKernelOperands().size();
    }
  });
  if (launch_operand_count != 0) {
    return launch_operand_count;
  }
  return fallback;
}

} // namespace gfx_plugin
} // namespace ov
