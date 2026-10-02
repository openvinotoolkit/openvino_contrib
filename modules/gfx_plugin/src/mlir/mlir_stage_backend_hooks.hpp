// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "common/gpu_backend.hpp"
#include "compiler/stage_policy.hpp"
#include "kernel_ir/gfx_codegen_desc.hpp"
#include "kernel_ir/gfx_codegen_backend.hpp"
#include "kernel_ir/gfx_kernel_manifest.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "common/gfx_activation.hpp"
#include "common/gfx_bias.hpp"
#include "common/gpu_parallelism_plan.hpp"

namespace ov {
class Tensor;
namespace gfx_plugin {

class GpuBufferManager;

struct MlirStageBackendSourcePlan {
  KernelSource source;
  KernelRuntimeBindingState runtime_binding;
  bool has_runtime_binding = false;
  bool requires_backend_model = false;

  bool valid() const {
    return source.module || source.msl_generator || !source.msl_source.empty();
  }
};

struct MlirStageBackendCompressedMatMulPlan {
  bool valid = false;
  MlirStageBackendSourcePlan source_plan;
  uint32_t reduction_threads = 1;
  uint32_t output_block = 1;
  uint32_t output_columns = 0;
  std::string packed_weight_suffix;
  ov::element::Type packed_scale_type = ov::element::dynamic;
  ov::Shape packed_scale_shape;
  std::vector<uint8_t> packed_weights;
  std::vector<uint8_t> packed_scales;
};

struct MlirStageBackendRuntimeParamsPlan {
  bool valid = false;
  std::vector<int32_t> params;
  KernelRuntimeBindingState runtime_binding;
};

struct MlirStageBackendHooks {
  virtual ~MlirStageBackendHooks() = default;

  virtual GfxKernelBackendDomain custom_kernel_backend_domain() const {
    return GfxKernelBackendDomain::Unknown;
  }

  virtual bool should_pack_matmul_const_input_as_f16(
      const std::shared_ptr<const ov::Node> & /*node*/,
      size_t /*input_idx*/, const ov::Tensor & /*tensor*/) const {
    return false;
  }

  virtual bool should_pack_conv2d_const_weights_oc4(
      const std::shared_ptr<const ov::Node> & /*node*/,
      size_t /*input_idx*/, const ov::Tensor & /*tensor*/) const {
    return false;
  }

  virtual void
  attach_const_tensor_sources(KernelSource & /*source*/,
                              const std::shared_ptr<const ov::Node>
                                  & /*node*/) const {}

  virtual bool apply_stage_optimization(
      mlir::ModuleOp /*module*/, const GfxStageOptimizationPlan & /*plan*/,
      const std::shared_ptr<const ov::Node> & /*node*/,
      std::string_view /*stage_type*/, bool /*has_bias*/,
      const BiasParams * /*bias_params*/, bool /*has_activation*/,
      ActivationKind /*activation*/) const {
    return false;
  }

  virtual uint32_t static_matmul_reduction_threads(
      const std::shared_ptr<const ov::Node> & /*node*/, bool /*has_activation*/,
      ActivationKind /*activation*/, float /*activation_alpha*/) const {
    return 0;
  }

  virtual MlirStageBackendCompressedMatMulPlan make_compressed_matmul_plan(
      const std::shared_ptr<const ov::Node> & /*node*/,
      const GfxParallelismCaps & /*caps*/) const {
    return {};
  }

  virtual MlirStageBackendSourcePlan make_shapeof_source_plan(
      const std::shared_ptr<const ov::Node> & /*node*/,
      mlir::ModuleOp /*module*/) const {
    return {};
  }

  virtual MlirStageBackendSourcePlan make_concat_source_plan(
      const std::shared_ptr<const ov::Node> & /*node*/,
      mlir::ModuleOp /*module*/) const {
    return {};
  }

  virtual MlirStageBackendSourcePlan make_causal_sdpa_source_plan(
      ov::element::Type /*element_type*/) const {
    return {};
  }

  virtual MlirStageBackendSourcePlan make_sdpa_source_plan(
      mlir::MLIRContext & /*ctx*/,
      const std::shared_ptr<const ov::Node> & /*node*/,
      const GpuBufferManager * /*buffer_manager*/,
      const GfxStageRuntimeTraits & /*runtime_traits*/) const {
    return {};
  }

  virtual MlirStageBackendSourcePlan make_range_source_plan(
      const std::shared_ptr<const ov::Node> & /*node*/,
      mlir::ModuleOp /*module*/) const {
    return {};
  }

  virtual MlirStageBackendSourcePlan make_tile_source_plan(
      const std::shared_ptr<const ov::Node> & /*node*/,
      mlir::ModuleOp /*module*/) const {
    return {};
  }

  virtual MlirStageBackendSourcePlan make_activation_source_plan(
      const std::shared_ptr<const ov::Node> & /*node*/,
      mlir::ModuleOp /*module*/) const {
    return {};
  }

  virtual MlirStageBackendSourcePlan make_runtime_matmul_source_plan(
      mlir::MLIRContext & /*ctx*/, const GpuBufferManager * /*buffer_manager*/,
      const std::shared_ptr<const ov::Node> & /*node*/,
      const MatMulCodegenDesc & /*desc*/, const ov::Shape & /*shape_a*/,
      const ov::Shape & /*shape_b*/, std::string_view /*stage_name*/) const {
    return {};
  }

  virtual MlirStageBackendSourcePlan make_static_slice_source_plan(
      const std::shared_ptr<const ov::Node> & /*node*/,
      const ov::element::Type & /*storage_type*/,
      mlir::ModuleOp /*module*/) const {
    return {};
  }

  virtual MlirStageBackendSourcePlan make_direct_split_source_plan(
      std::string_view /*stage_type*/,
      const ov::element::Type & /*element_type*/,
      const ov::Shape & /*input_shape*/,
      const std::vector<size_t> & /*split_sizes*/, uint32_t /*axis_len*/,
      uint32_t /*inner_stride*/, mlir::ModuleOp /*module*/) const {
    return {};
  }

  virtual MlirStageBackendRuntimeParamsPlan make_causal_sdpa_runtime_params_plan(
      const ov::Shape & /*q_shape*/, const ov::Shape & /*k_shape*/,
      const ov::Shape & /*v_shape*/, const ov::Shape & /*mask_shape*/,
      float /*scale*/, bool /*k_gqa*/, size_t /*k_heads*/, bool /*v_gqa*/,
      size_t /*v_heads*/) const {
    return {};
  }

  virtual MlirStageBackendRuntimeParamsPlan make_sdpa_runtime_params_plan(
      const ov::Shape & /*q_shape*/, const ov::Shape & /*k_shape*/,
      const ov::Shape & /*v_shape*/, const ov::Shape & /*mask_shape*/,
      bool /*has_mask*/, float /*scale*/, bool /*k_gqa*/, size_t /*k_heads*/,
      bool /*v_gqa*/, size_t /*v_heads*/) const {
    return {};
  }
};

bool register_mlir_stage_backend_hooks(GpuBackend backend,
                                       const MlirStageBackendHooks &hooks);
const MlirStageBackendHooks *mlir_stage_backend_hooks_for(GpuBackend backend);

} // namespace gfx_plugin
} // namespace ov
