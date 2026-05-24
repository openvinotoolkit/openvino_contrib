// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_stage.hpp"
#include "mlir/mlir_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <optional>
#include <sstream>

#include "kernel_ir/gfx_kernel_args.hpp"
#include "kernel_ir/gfx_kernel_plan.hpp"
#include "mlir/codegen_common.hpp"
#include "mlir/gfx_apple_stage_pipeline.hpp"
#include "mlir/gfx_backend_custom_kernel_adapter.hpp"
#include "mlir/gfx_kernel_runtime_params.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mpsrt_const_tensor_sources.hpp"
#include "mlir/gfx_mpsrt_conv_metadata.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/gfx_stage_kernel_binding.hpp"
#include "mlir/gfx_stage_runtime_values.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_kernel_plan_utils.hpp"
#include "mlir/msl_codegen.hpp"
#include "mlir/msl_codegen_apple_mps.hpp"
#include "mlir/msl_codegen_apple_msl_split.hpp"
#include "mlir/msl_codegen_attention.hpp"
#include "mlir/msl_codegen_compressed_matmul.hpp"
#include "mlir/msl_codegen_matmul_metal.hpp"
#include "mlir/spirv_kernel_binding_adapter.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_parallelism.hpp"
#include "runtime/gfx_profiler.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gfx_stage_policy.hpp"
#include "runtime/memory_manager.hpp"
#include "transforms/gfx_llm_ops.hpp"
#include "transforms/mlir_fused_ops.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/topk_base.hpp"
#include "openvino/op/variadic_split.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

// MLIR 18 switches AttrSizedOperandSegments properties to DenseArrayAttr in
// assembly; generated converters still expect DenseI32ArrayAttr. Provide a
// bridge so verification succeeds with either representation.
namespace mlir {
template <size_t N>
inline LogicalResult
convertFromAttribute(std::array<int32_t, N> &storage, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError) {
  if (auto a = dyn_cast<DenseI32ArrayAttr>(attr)) {
    if (a.size() != static_cast<int64_t>(N)) {
      if (emitError)
        emitError() << "DenseI32ArrayAttr has wrong size";
      return failure();
    }
    llvm::copy(a.asArrayRef(), storage.begin());
    return success();
  }
  if (auto arr = dyn_cast<DenseArrayAttr>(attr)) {
    if (auto ity = dyn_cast<IntegerType>(arr.getElementType())) {
      SmallVector<int32_t> vals;
      vals.reserve(static_cast<size_t>(arr.getSize()));
      if (ity.isInteger(32)) {
        auto raw = arr.getRawData();
        auto ptr = reinterpret_cast<const int32_t *>(raw.data());
        vals.append(ptr, ptr + arr.getSize());
      } else if (ity.isIndex() || ity.getWidth() == 64) {
        auto raw = arr.getRawData();
        auto ptr = reinterpret_cast<const int64_t *>(raw.data());
        for (int64_t i = 0; i < arr.getSize(); ++i) {
          vals.push_back(static_cast<int32_t>(ptr[i]));
        }
      }
      if (vals.size() == N) {
        llvm::copy(vals, storage.begin());
        return success();
      }
    }
  }
  return convertFromAttribute(
      MutableArrayRef<int32_t>(storage.data(), storage.size()), attr,
      emitError);
}

inline LogicalResult
convertFromAttribute(DenseI32ArrayAttr &storage, Attribute attr,
                     function_ref<InFlightDiagnostic()> emitError) {
  if (auto a = dyn_cast<DenseI32ArrayAttr>(attr)) {
    storage = a;
    return success();
  }
  if (auto arr = dyn_cast<DenseArrayAttr>(attr)) {
    if (auto ity = dyn_cast<IntegerType>(arr.getElementType())) {
      SmallVector<int32_t> vals;
      vals.reserve(static_cast<size_t>(arr.getSize()));
      if (ity.isInteger(32)) {
        auto raw = arr.getRawData();
        auto ptr = reinterpret_cast<const int32_t *>(raw.data());
        vals.append(ptr, ptr + arr.getSize());
      } else if (ity.isIndex() || ity.getWidth() == 64) {
        auto raw = arr.getRawData();
        auto ptr = reinterpret_cast<const int64_t *>(raw.data());
        for (int64_t i = 0; i < arr.getSize(); ++i) {
          vals.push_back(static_cast<int32_t>(ptr[i]));
        }
      }
      storage = DenseI32ArrayAttr::get(attr.getContext(), vals);
      return success();
    }
  }
  return emitError ? emitError() : failure();
}
} // namespace mlir

namespace ov {
namespace gfx_plugin {

namespace {

uint32_t kernel_activation_code(ActivationKind kind) {
  switch (kind) {
  case ActivationKind::Relu:
    return 1u;
  case ActivationKind::Sigmoid:
    return 2u;
  case ActivationKind::Tanh:
    return 3u;
  case ActivationKind::Elu:
    return 4u;
  case ActivationKind::Prelu:
    return 5u;
  case ActivationKind::Gelu:
    return 6u;
  case ActivationKind::Swish:
    return 7u;
  case ActivationKind::HSwish:
    return 8u;
  case ActivationKind::HSigmoid:
    return 9u;
  case ActivationKind::Abs:
    return 10u;
  case ActivationKind::Sign:
    return 11u;
  case ActivationKind::Clamp:
    return 12u;
  case ActivationKind::Identity:
  default:
    return 0u;
  }
}

void record_conv_compile_profile(const GfxStageOptimizationPlan &plan) {
  if (plan.archetype != GfxStageArchetype::Convolution &&
      plan.archetype != GfxStageArchetype::GroupConvolution) {
    return;
  }
  const auto &algorithm = plan.conv.algorithm;
  set_compile_counter("conv_requires_multi_kernel_manifest",
                      algorithm.requires_multi_kernel_manifest ? 1u : 0u);
  set_compile_counter("conv_reduction_work", algorithm.reduction_work);
  set_compile_counter("conv_output_elements", algorithm.output_elements);
  set_compile_counter("conv_intermediate_elements",
                      algorithm.intermediate_elements);
  set_compile_counter("conv_reduction_chunk_count",
                      algorithm.reduction_chunk_count);
  set_compile_counter("conv_reduction_chunk_size",
                      algorithm.reduction_chunk_size);
  set_compile_counter("conv_workgroup_reduction_lanes",
                      algorithm.workgroup_reduction_lanes);
  set_compile_counter("conv_workgroup_output_lanes",
                      algorithm.workgroup_output_lanes);
  set_compile_counter("conv_output_channel_reuse_lanes",
                      algorithm.output_channel_reuse_lanes);
  set_compile_counter("conv_spatial_output_reuse_lanes",
                      algorithm.spatial_output_reuse_lanes);
  set_compile_counter("conv_output_reuse_lanes", algorithm.output_reuse_lanes);
  set_compile_counter("conv_spatial_input_reuse_lanes",
                      algorithm.spatial_input_reuse_lanes);
  set_compile_counter("conv_spatial_input_reuse_unique_width_loads",
                      algorithm.spatial_input_reuse_unique_width_loads);
  set_compile_counter("conv_spatial_input_reuse_saved_width_loads",
                      algorithm.spatial_input_reuse_saved_width_loads);
  const auto &manifest = algorithm.multi_kernel_manifest;
  set_compile_counter("conv_multi_kernel_stage_count", manifest.stages.size());
  set_compile_counter("conv_multi_kernel_requires_owned_intermediates",
                      manifest.requires_owned_intermediates ? 1u : 0u);
  set_compile_counter("conv_multi_kernel_requires_owned_launch_sequence",
                      manifest.requires_owned_launch_sequence ? 1u : 0u);
  set_compile_counter("conv_multi_kernel_requires_output_reuse",
                      manifest.requires_output_reuse ? 1u : 0u);
  set_compile_counter("conv_multi_kernel_requires_spatial_input_reuse",
                      manifest.requires_spatial_input_reuse ? 1u : 0u);
  set_compile_counter("conv_multi_kernel_requires_coarse_output_tile_preservation",
                      manifest.requires_coarse_output_tile_preservation ? 1u : 0u);
  set_compile_counter("conv_multi_kernel_has_workgroup_local_reduction_plan",
                      manifest.has_workgroup_local_reduction_plan ? 1u : 0u);
  set_compile_counter("conv_multi_kernel_coarse_spatial_tile_elements",
                      manifest.coarse_spatial_tile_elements);
  set_compile_counter("conv_multi_kernel_coarse_output_channel_block",
                      manifest.coarse_output_channel_block);
  set_compile_counter("conv_multi_kernel_coarse_output_tile_elements",
                      manifest.coarse_output_tile_elements);
  set_compile_counter("conv_multi_kernel_workgroup_output_tile_deficit",
                      manifest.workgroup_output_tile_deficit);
  set_compile_counter("conv_multi_kernel_partial_sum_elements",
                      manifest.partial_sum_elements);
  set_compile_counter("conv_multi_kernel_reduced_accumulator_elements",
                      manifest.reduced_accumulator_elements);
  set_compile_counter("conv_multi_kernel_owned_intermediate_elements",
                      manifest.owned_intermediate_elements);
  set_compile_counter("conv_multi_kernel_owned_intermediate_bytes",
                      manifest.owned_intermediate_bytes);
  set_compile_counter("conv_multi_kernel_owned_intermediate_buffer_count",
                      manifest.owned_intermediate_buffer_count);
  set_compile_counter("conv_multi_kernel_workgroup_local_accumulator_elements",
                      manifest.workgroup_local_accumulator_elements);
  set_compile_counter("conv_multi_kernel_workgroup_local_accumulator_bytes",
                      manifest.workgroup_local_accumulator_bytes);
  set_compile_counter("conv_multi_kernel_launch_dispatch_count",
                      manifest.launch_dispatch_count);
}

void record_conv_dispatch_compile_profile(const ConvParallelismPlan &plan) {
  set_compile_counter("conv_dispatch_prefer_parallel",
                      plan.prefer_parallel ? 1u : 0u);
  set_compile_counter("conv_dispatch_tile_h", plan.dispatch.tile_h);
  set_compile_counter("conv_dispatch_tile_w", plan.dispatch.tile_w);
  set_compile_counter("conv_dispatch_threads_h", plan.dispatch.threads_h);
  set_compile_counter("conv_dispatch_threads_w", plan.dispatch.threads_w);
  set_compile_counter("conv_dispatch_channel_block", plan.output_channel_block);
  set_compile_counter("conv_dispatch_channel_block_accumulation_serial",
                      plan.channel_block_accumulation ==
                              ConvChannelBlockAccumulation::Serial
                          ? 1u
                          : 0u);
}

void record_runtime_dispatch_profile(GfxProfiler *profiler,
                                     const ParallelDispatchConfig &cfg,
                                     const KernelDispatch &dispatch) {
  if (!profiler || !cfg.enabled) {
    return;
  }
  profiler->set_counter("runtime_parallel_loop_dims", cfg.loop_dims);
  profiler->set_counter("runtime_dispatch_tile_h", cfg.tile_h);
  profiler->set_counter("runtime_dispatch_tile_w", cfg.tile_w);
  profiler->set_counter("runtime_dispatch_threads_h", cfg.threads_h);
  profiler->set_counter("runtime_dispatch_threads_w", cfg.threads_w);
  profiler->set_counter("runtime_dispatch_channel_block", cfg.channel_block);
  profiler->set_counter("runtime_dispatch_grid_x", dispatch.grid[0]);
  profiler->set_counter("runtime_dispatch_grid_y", dispatch.grid[1]);
  profiler->set_counter("runtime_dispatch_grid_z", dispatch.grid[2]);
  profiler->set_counter("runtime_dispatch_tpg_x",
                        dispatch.threads_per_group[0]);
  profiler->set_counter("runtime_dispatch_tpg_y",
                        dispatch.threads_per_group[1]);
  profiler->set_counter("runtime_dispatch_tpg_z",
                        dispatch.threads_per_group[2]);
}

const char *stage_archetype_attr(GfxStageArchetype archetype) {
  switch (archetype) {
  case GfxStageArchetype::Convolution:
    return "convolution";
  case GfxStageArchetype::GroupConvolution:
    return "group_convolution";
  case GfxStageArchetype::MatMul:
    return "matmul";
  case GfxStageArchetype::UnaryElementwise:
    return "unary_elementwise";
  case GfxStageArchetype::BinaryElementwise:
    return "binary_elementwise";
  case GfxStageArchetype::Reduction:
    return "reduction";
  case GfxStageArchetype::Layout:
    return "layout";
  case GfxStageArchetype::Convert:
    return "convert";
  case GfxStageArchetype::SplitConcat:
    return "split_concat";
  default:
    return "unknown";
  }
}

const char *tensor_layout_kind_attr(GfxTensorLayoutKind kind) {
  switch (kind) {
  case GfxTensorLayoutKind::Materialized:
    return "materialized";
  case GfxTensorLayoutKind::ViewOnly:
    return "view_only";
  default:
    return "unknown";
  }
}

const char *conv_route_kind_attr(GfxConvRouteKind kind) {
  switch (kind) {
  case GfxConvRouteKind::Direct1x1:
    return "direct_1x1";
  case GfxConvRouteKind::Direct3x3:
    return "direct_3x3";
  case GfxConvRouteKind::Chunked:
    return "chunked";
  default:
    return "none";
  }
}

const char *conv_family_attr(GfxConvFamily family) {
  switch (family) {
  case GfxConvFamily::Pointwise1x1:
    return "pointwise_1x1";
  case GfxConvFamily::Spatial3x3:
    return "spatial_3x3";
  case GfxConvFamily::Depthwise:
    return "depthwise";
  case GfxConvFamily::Grouped:
    return "grouped";
  case GfxConvFamily::General:
    return "general";
  default:
    return "unknown";
  }
}

const char *conv_algorithm_kind_attr(GfxConvAlgorithmKind kind) {
  switch (kind) {
  case GfxConvAlgorithmKind::Direct1x1:
    return "direct_1x1";
  case GfxConvAlgorithmKind::Direct3x3Stride1:
    return "direct_3x3_stride1";
  case GfxConvAlgorithmKind::Direct3x3Stride2:
    return "direct_3x3_stride2";
  case GfxConvAlgorithmKind::DepthwiseDirect:
    return "depthwise_direct";
  case GfxConvAlgorithmKind::ChunkedDirect:
    return "chunked_direct";
  case GfxConvAlgorithmKind::Indirect:
    return "indirect";
  default:
    return "none";
  }
}

bool is_vulkan_pipeline_creation_failure(const std::exception &ex) {
  return std::string(ex.what()).find("vkCreateComputePipelines failed") !=
         std::string::npos;
}

std::optional<MatMulParallelismPlan>
select_safe_matmul_fallback_plan(const GfxParallelismCaps &caps,
                                 const ov::Shape &output_shape) {
  const auto plans =
      enumerate_matmul_parallelism_candidates(caps, output_shape);
  if (plans.size() <= 1) {
    return std::nullopt;
  }
  auto score = [](const MatMulParallelismPlan &plan) {
    const uint32_t threads = plan.dispatch.threads_h * plan.dispatch.threads_w;
    const uint32_t aspect = plan.dispatch.tile_h > plan.dispatch.tile_w
                                ? (plan.dispatch.tile_h - plan.dispatch.tile_w)
                                : (plan.dispatch.tile_w - plan.dispatch.tile_h);
    const uint32_t distance_to_safe_threads =
        threads > 16 ? (threads - 16) : (16 - threads);
    return std::tuple<uint32_t, uint32_t, uint32_t>{distance_to_safe_threads,
                                                    aspect, threads};
  };
  auto best = std::min_element(plans.begin() + 1, plans.end(),
                               [&](const auto &lhs, const auto &rhs) {
                                 return score(lhs) < score(rhs);
                               });
  if (best == plans.end()) {
    return std::nullopt;
  }
  return *best;
}

MatMulParallelismPlan make_serial_matmul_fallback_plan() {
  MatMulParallelismPlan plan;
  plan.prefer_parallel = false;
  plan.variant = "serial";
  return plan;
}

uint64_t shape_batch_product_prefix(const ov::Shape &shape) {
  if (shape.size() <= 2) {
    return 1;
  }
  uint64_t batch = 1;
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    batch *= static_cast<uint64_t>(shape[i]);
  }
  return batch;
}

ov::Shape topk_row_dispatch_shape(const std::shared_ptr<const ov::Node> &node) {
  auto topk = ov::as_type_ptr<const ov::op::util::TopKBase>(node);
  if (!topk || !topk->get_input_partial_shape(0).is_static()) {
    return {};
  }
  const ov::Shape input_shape = topk->get_input_shape(0);
  if (input_shape.empty()) {
    return {};
  }
  const int64_t axis_norm =
      normalize_axis(topk->get_axis(), input_shape.size(), "GFX MLIR: TopK");
  uint64_t rows = 1;
  for (size_t dim = 0; dim < input_shape.size(); ++dim) {
    if (dim == static_cast<size_t>(axis_norm)) {
      continue;
    }
    rows *= static_cast<uint64_t>(input_shape[dim]);
  }
  return ov::Shape{static_cast<size_t>(std::max<uint64_t>(rows, 1))};
}

} // namespace

namespace {

bool should_pack_matmul_const_input_as_f16(
    const std::shared_ptr<const ov::Node> &node, size_t input_idx,
    const ov::Tensor &tensor) {
  auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
  return matmul && input_idx == 1 &&
         (!matmul->get_input_partial_shape(0).is_static() ||
          !matmul->get_output_partial_shape(0).is_static()) &&
         tensor.get_element_type() == ov::element::f32 &&
         matmul->get_output_element_type(0) == ov::element::f32;
}

std::vector<ov::float16> pack_f32_tensor_as_f16(const ov::Tensor &tensor) {
  const auto elements = tensor.get_size();
  const auto *src = tensor.data<const float>();
  std::vector<ov::float16> packed(elements);
  for (size_t i = 0; i < elements; ++i) {
    packed[i] = ov::float16(src[i]);
  }
  return packed;
}

bool requires_scalar_fp32_convolution_path(
    const std::shared_ptr<const ov::Node> &node) {
  return node && node->get_output_size() > 0 &&
         node->get_output_element_type(0) == ov::element::f32 &&
         ov::fp16_compression_is_disabled(node);
}

bool should_pack_conv2d_const_weights_oc4(
    const std::shared_ptr<const ov::Node> &node, size_t input_idx,
    const ov::Tensor &tensor) {
  auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node);
  if (!conv || input_idx != 1 || tensor.get_shape().size() != 4) {
    return false;
  }
  if (requires_scalar_fp32_convolution_path(node)) {
    return false;
  }
  const auto et = tensor.get_element_type();
  if (et != ov::element::f16 && et != ov::element::f32) {
    return false;
  }
  if (!conv->get_input_partial_shape(0).is_static() ||
      !conv->get_input_partial_shape(1).is_static() ||
      !conv->get_output_partial_shape(0).is_static()) {
    return false;
  }
  const auto in_shape = conv->get_input_shape(0);
  const auto w_shape = conv->get_input_shape(1);
  const auto out_shape = conv->get_output_shape(0);
  if (in_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4) {
    return false;
  }
  Conv2DCodegenDesc desc{};
  desc.input_type = conv->get_input_element_type(0);
  desc.weight_type = conv->get_input_element_type(1);
  desc.output_type = conv->get_output_element_type(0);
  desc.C_out = static_cast<uint32_t>(w_shape[0]);
  desc.kH = static_cast<uint32_t>(w_shape[2]);
  desc.kW = static_cast<uint32_t>(w_shape[3]);
  const uint32_t cin_pg = static_cast<uint32_t>(w_shape[1]);
  const uint32_t c_in = static_cast<uint32_t>(in_shape[1]);
  desc.groups = (cin_pg != 0 && c_in % cin_pg == 0) ? (c_in / cin_pg) : 1;
  return gfx_conv2d_output_channel_block(desc) >= 4;
}

template <typename T>
std::vector<uint8_t> pack_conv2d_weights_oc4_typed(const ov::Tensor &tensor) {
  const auto shape = tensor.get_shape();
  OPENVINO_ASSERT(shape.size() == 4,
                  "GFX Conv2D weight pack expects OIHW rank-4 weights");
  const size_t c_out = shape[0];
  const size_t c_in = shape[1];
  const size_t k_h = shape[2];
  const size_t k_w = shape[3];
  const size_t channel_blocks = (c_out + 3u) / 4u;
  const auto *src = tensor.data<const T>();
  std::vector<T> packed(channel_blocks * c_in * k_h * k_w * 4u, T{});
  for (size_t block = 0; block < channel_blocks; ++block) {
    for (size_t ci = 0; ci < c_in; ++ci) {
      for (size_t kh = 0; kh < k_h; ++kh) {
        for (size_t kw = 0; kw < k_w; ++kw) {
          const size_t dst_base =
              (((block * c_in + ci) * k_h + kh) * k_w + kw) * 4u;
          for (size_t lane = 0; lane < 4; ++lane) {
            const size_t co = block * 4u + lane;
            if (co >= c_out) {
              continue;
            }
            const size_t src_idx = ((co * c_in + ci) * k_h + kh) * k_w + kw;
            packed[dst_base + lane] = src[src_idx];
          }
        }
      }
    }
  }
  std::vector<uint8_t> bytes(packed.size() * sizeof(T));
  std::memcpy(bytes.data(), packed.data(), bytes.size());
  return bytes;
}

std::vector<uint8_t> pack_conv2d_weights_oc4(const ov::Tensor &tensor) {
  const auto et = tensor.get_element_type();
  if (et == ov::element::f16) {
    return pack_conv2d_weights_oc4_typed<ov::float16>(tensor);
  }
  if (et == ov::element::f32) {
    return pack_conv2d_weights_oc4_typed<float>(tensor);
  }
  OPENVINO_THROW("GFX Conv2D weight pack: unsupported element type ", et);
}

std::optional<RuntimeReduceInfo>
get_runtime_reduce_info(const std::shared_ptr<const ov::Node> &node) {
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceSum>(node)) {
    OPENVINO_ASSERT(reduce->reduction_axes_constant(),
                    "GFX MLIR: ReduceSum axes must be constant");
    return RuntimeReduceInfo{reduce->get_reduction_axes(),
                             reduce->get_keep_dims()};
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceMean>(node)) {
    OPENVINO_ASSERT(reduce->reduction_axes_constant(),
                    "GFX MLIR: ReduceMean axes must be constant");
    return RuntimeReduceInfo{reduce->get_reduction_axes(),
                             reduce->get_keep_dims()};
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceMax>(node)) {
    OPENVINO_ASSERT(reduce->reduction_axes_constant(),
                    "GFX MLIR: ReduceMax axes must be constant");
    return RuntimeReduceInfo{reduce->get_reduction_axes(),
                             reduce->get_keep_dims()};
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceMin>(node)) {
    OPENVINO_ASSERT(reduce->reduction_axes_constant(),
                    "GFX MLIR: ReduceMin axes must be constant");
    return RuntimeReduceInfo{reduce->get_reduction_axes(),
                             reduce->get_keep_dims()};
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceProd>(node)) {
    OPENVINO_ASSERT(reduce->reduction_axes_constant(),
                    "GFX MLIR: ReduceProd axes must be constant");
    return RuntimeReduceInfo{reduce->get_reduction_axes(),
                             reduce->get_keep_dims()};
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v4::ReduceL1>(node)) {
    OPENVINO_ASSERT(reduce->reduction_axes_constant(),
                    "GFX MLIR: ReduceL1 axes must be constant");
    return RuntimeReduceInfo{reduce->get_reduction_axes(),
                             reduce->get_keep_dims()};
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v4::ReduceL2>(node)) {
    OPENVINO_ASSERT(reduce->reduction_axes_constant(),
                    "GFX MLIR: ReduceL2 axes must be constant");
    return RuntimeReduceInfo{reduce->get_reduction_axes(),
                             reduce->get_keep_dims()};
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceLogicalAnd>(node)) {
    OPENVINO_ASSERT(reduce->reduction_axes_constant(),
                    "GFX MLIR: ReduceLogicalAnd axes must be constant");
    return RuntimeReduceInfo{reduce->get_reduction_axes(),
                             reduce->get_keep_dims()};
  }
  if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceLogicalOr>(node)) {
    OPENVINO_ASSERT(reduce->reduction_axes_constant(),
                    "GFX MLIR: ReduceLogicalOr axes must be constant");
    return RuntimeReduceInfo{reduce->get_reduction_axes(),
                             reduce->get_keep_dims()};
  }
  return std::nullopt;
}

inline void normalize_operand_segment_sizes(mlir::ModuleOp module) {
  (void)module;
}

bool try_alias_contiguous_split_outputs(
    const std::shared_ptr<const ov::Node> &node, GpuTensor *input,
    const std::vector<GpuTensor *> &outputs, const char *stage_name) {
  if (!node || !input || !input->buf.valid() || input->shape.empty() ||
      outputs.empty()) {
    return false;
  }
  for (const auto *out : outputs) {
    if (!out || !out->prefer_private) {
      return false;
    }
  }

  const auto plan = plan_split_runtime_values(node.get(), input->shape,
                                              outputs.size(), stage_name);
  size_t outer = 1;
  for (size_t d = 0; d < static_cast<size_t>(plan.axis_norm); ++d) {
    outer *= input->shape[d];
  }
  if (outer != 1) {
    return false;
  }

  ov::element::Type element_type = input->expected_type == ov::element::dynamic
                                       ? input->buf.type
                                       : input->expected_type;
  if (element_type == ov::element::dynamic) {
    element_type = node->get_input_element_type(0);
  }
  OPENVINO_ASSERT(element_type != ov::element::dynamic,
                  "GFX MLIR: Split input element type is unknown for stage ",
                  stage_name);
  const size_t elem_size = element_type.size();
  OPENVINO_ASSERT(elem_size != 0,
                  "GFX MLIR: Split element size is zero for stage ",
                  stage_name);

  size_t axis_offset = 0;
  for (size_t oi = 0; oi < outputs.size(); ++oi) {
    auto *out = outputs[oi];
    if (!out) {
      axis_offset += plan.split_sizes[oi];
      continue;
    }

    ov::Shape out_shape = input->shape;
    out_shape[static_cast<size_t>(plan.axis_norm)] = plan.split_sizes[oi];
    const size_t byte_offset =
        axis_offset * static_cast<size_t>(plan.inner_stride) * elem_size;
    const size_t bytes = ov::shape_size(out_shape) * elem_size;
    OPENVINO_ASSERT(byte_offset + bytes <= input->buf.size,
                    "GFX MLIR: Split view exceeds input buffer for stage ",
                    stage_name, " (offset=", byte_offset, ", bytes=", bytes,
                    ", input=", input->buf.size, ")");

    GpuBuffer alias = input->buf;
    alias.offset += byte_offset;
    alias.size = bytes;
    alias.external = true;
    alias.owned = false;
    out->buf = alias;
    out->shape = std::move(out_shape);
    out->expected_type = element_type;
    axis_offset += plan.split_sizes[oi];
  }
  return true;
}

void propagate_view_metadata(GpuTensor &dst, const GpuTensor &src) {
  dst.gqa_broadcast_view = src.gqa_broadcast_view;
  dst.gqa_storage_shape = src.gqa_storage_shape;
  dst.gqa_kv_heads = src.gqa_kv_heads;
}

bool alias_tensor_view(GpuTensor &dst, const GpuTensor &src,
                       const ov::Shape &shape, ov::element::Type element_type) {
  if (!src.buf.valid()) {
    return false;
  }
  dst.buf = src.buf;
  dst.buf.external = true;
  dst.buf.owned = false;
  dst.shape = shape;
  dst.expected_type =
      element_type == ov::element::dynamic
          ? (src.expected_type == ov::element::dynamic ? src.buf.type
                                                       : src.expected_type)
          : element_type;
  propagate_view_metadata(dst, src);
  if (!src.i64_values.empty() &&
      src.i64_values.size() == ov::shape_size(shape)) {
    dst.i64_values = src.i64_values;
  }
  return true;
}

bool try_alias_same_shape_unary_view(GpuTensor *input,
                                     const std::vector<GpuTensor *> &outputs,
                                     const ov::Shape &out_shape,
                                     ov::element::Type output_type) {
  if (!input || outputs.size() != 1 || !outputs[0] ||
      input->shape != out_shape) {
    return false;
  }
  return alias_tensor_view(*outputs[0], *input, out_shape, output_type);
}

bool output_consumers_are_reshape_or_sdpa(
    const std::shared_ptr<const ov::Node> &node) {
  if (!node || node->get_output_size() != 1) {
    return false;
  }
  bool has_consumer = false;
  for (const auto &target : node->output(0).get_target_inputs()) {
    has_consumer = true;
    const auto *consumer = target.get_node();
    if (dynamic_cast<const ov::op::v1::Reshape *>(consumer) ||
        dynamic_cast<const ov::op::v13::ScaledDotProductAttention *>(
            consumer)) {
      continue;
    }
    return false;
  }
  return has_consumer;
}

bool try_alias_gqa_broadcast_view(const std::shared_ptr<const ov::Node> &node,
                                  GpuTensor *input,
                                  const std::vector<GpuTensor *> &outputs) {
  if (!node || !input || !input->buf.valid() || input->shape.size() != 5 ||
      outputs.size() != 1 || !outputs[0] || outputs[0]->shape.size() != 5 ||
      !output_consumers_are_reshape_or_sdpa(node)) {
    return false;
  }
  if (!ov::as_type_ptr<const ov::op::v3::Broadcast>(node) &&
      !ov::as_type_ptr<const ov::op::v1::Broadcast>(node)) {
    return false;
  }

  const auto &in_shape = input->shape;
  const auto &out_shape = outputs[0]->shape;
  if (in_shape[2] != 1 || out_shape[2] <= 1 || in_shape[0] != out_shape[0] ||
      in_shape[1] != out_shape[1] || in_shape[3] != out_shape[3] ||
      in_shape[4] != out_shape[4]) {
    return false;
  }

  auto *out = outputs[0];
  out->buf = input->buf;
  out->buf.external = true;
  out->buf.owned = false;
  out->expected_type = input->expected_type == ov::element::dynamic
                           ? input->buf.type
                           : input->expected_type;
  out->gqa_broadcast_view = true;
  out->gqa_storage_shape =
      ov::Shape{in_shape[0], in_shape[1], in_shape[3], in_shape[4]};
  out->gqa_kv_heads = in_shape[1];
  return true;
}

bool concat_has_runtime_shape(const std::shared_ptr<const ov::Node> &node) {
  auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(node);
  if (!concat) {
    return false;
  }
  if (!concat->get_output_partial_shape(0).is_static()) {
    return true;
  }
  for (size_t input_idx = 0; input_idx < concat->get_input_size();
       ++input_idx) {
    if (!concat->get_input_partial_shape(input_idx).is_static()) {
      return true;
    }
  }
  return false;
}

} // namespace

MlirStage::MlirStage(const std::shared_ptr<const ov::Node> &node)
    : m_node(node),
      m_name(node ? node->get_friendly_name() : std::string("mlir_stage")),
      m_type(node ? node->get_type_name() : std::string("Unknown")) {
  if (node && node->get_output_partial_shape(0).is_static()) {
    m_output_shape = node->get_output_shape(0);
  }
  m_is_view_op = select_tensor_layout_plan(m_type, m_node).view_only;
}

void MlirStage::apply_kernel_metadata(const KernelRuntimeMetadata &meta,
                                      size_t scalar_inputs) {
  if (!meta.valid) {
    return;
  }
  m_parallel_cfg = meta.dispatch;
  if (meta.force_single_dispatch) {
    m_force_single_dispatch = true;
  }
  apply_kernel_runtime_binding_state(make_stage_direct_kernel_runtime_binding(
      meta.kernel_inputs, meta.kernel_input_arg_count,
      std::move(meta.operands.operand_kinds),
      std::move(meta.operands.operand_arg_indices),
      std::move(meta.operands.scalar_args)));
  if (m_kernel_binding.operand_kinds.empty() && scalar_inputs != 0) {
    OPENVINO_ASSERT(m_kernel_binding.scalar_args.size() == scalar_inputs,
                    "GFX MLIR: kernel scalar args mismatch for ", m_name,
                    " (expected ", scalar_inputs, ", got ",
                    m_kernel_binding.scalar_args.size(), ")");
  }
}

void MlirStage::apply_kernel_runtime_binding_state(
    KernelRuntimeBindingState binding) {
  m_kernel_binding = std::move(binding);
  m_kernel_binding_owned_by_source_plan = false;
}

void MlirStage::apply_source_plan_kernel_runtime_binding_state(
    KernelRuntimeBindingState binding) {
  m_kernel_binding = std::move(binding);
  m_kernel_binding_owned_by_source_plan = true;
}

KernelRuntimeBindingState
MlirStage::resolved_kernel_runtime_binding_state() const {
  KernelRuntimeBindingState binding = m_kernel_binding;
  if (binding.inputs.empty() && m_node) {
    const size_t in_count = m_node->get_input_size();
    binding.inputs.reserve(in_count);
    for (size_t i = 0; i < in_count; ++i) {
      binding.inputs.push_back(i);
    }
  }
  return binding;
}

void MlirStage::append_const_kernel_extra_input(
    std::vector<GpuTensor> &extra_inputs, size_t input_idx,
    std::string_view role_name) const {
  OPENVINO_ASSERT(
      m_const_buffers && input_idx < m_const_buffers->buffers.size() &&
          input_idx < m_const_buffers->present.size() &&
          m_const_buffers->present[input_idx] &&
          m_const_buffers->buffers[input_idx].buf.valid(),
      "GFX MLIR: ", role_name,
      " must be available as a ConstTensor extra buffer for stage ", m_name);
  extra_inputs.push_back(m_const_buffers->buffers[input_idx]);
}

bool MlirStage::refresh_conv2d_kernel_extra_inputs(
    const ov::Shape &input_shape, const ov::Shape &output_shape,
    ov::element::Type output_type) {
  if (!is_conv_like() || !m_node || input_shape.size() != 4) {
    return false;
  }
  OPENVINO_ASSERT(output_shape.size() == 4,
                  "GFX MLIR: Conv2D expects NCHW output for stage ", m_name);

  ov::element::Type et = output_type == ov::element::dynamic
                             ? m_node->get_output_element_type(0)
                             : output_type;
  if (et == ov::element::dynamic) {
    et = ov::element::f32;
  }

  uint32_t groups = 1;
  uint32_t C_out = 0, C_in_pg = 0, C_out_pg = 0, kH = 0, kW = 0;
  uint32_t strideH = 1, strideW = 1, dilationH = 1, dilationW = 1;
  uint32_t padTop = 0, padLeft = 0, padBottom = 0, padRight = 0;
  float epsilon = 0.f;

  if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node)) {
    OPENVINO_ASSERT(conv->get_input_partial_shape(1).is_static(),
                    "GFX MLIR: Conv2D weights shape must be static for stage ",
                    m_name);
    const auto &w = conv->get_input_shape(1);
    OPENVINO_ASSERT(w.size() == 4,
                    "GFX MLIR: Conv2D expects rank-4 weights for stage ",
                    m_name);
    C_out = static_cast<uint32_t>(w.at(0));
    C_in_pg = static_cast<uint32_t>(w.at(1));
    C_out_pg = C_out;
    const uint32_t C_in = static_cast<uint32_t>(input_shape.at(1));
    if (C_in_pg != 0 && C_in % C_in_pg == 0) {
      groups = C_in / C_in_pg;
      if (groups == 0) {
        groups = 1;
      }
    }
    kH = static_cast<uint32_t>(w.at(2));
    kW = static_cast<uint32_t>(w.at(3));
    strideH = static_cast<uint32_t>(conv->get_strides().at(0));
    strideW = static_cast<uint32_t>(conv->get_strides().at(1));
    dilationH = static_cast<uint32_t>(conv->get_dilations().at(0));
    dilationW = static_cast<uint32_t>(conv->get_dilations().at(1));
    padTop = static_cast<uint32_t>(conv->get_pads_begin().at(0));
    padLeft = static_cast<uint32_t>(conv->get_pads_begin().at(1));
    padBottom = static_cast<uint32_t>(conv->get_pads_end().at(0));
    padRight = static_cast<uint32_t>(conv->get_pads_end().at(1));
  } else if (auto gconv =
                 ov::as_type_ptr<const ov::op::v1::GroupConvolution>(m_node)) {
    OPENVINO_ASSERT(
        gconv->get_input_partial_shape(1).is_static(),
        "GFX MLIR: GroupConv2D weights shape must be static for stage ",
        m_name);
    const auto &w = gconv->get_input_shape(1);
    OPENVINO_ASSERT(w.size() == 5,
                    "GFX MLIR: GroupConv2D expects rank-5 weights for stage ",
                    m_name);
    groups = static_cast<uint32_t>(w.at(0));
    C_out_pg = static_cast<uint32_t>(w.at(1));
    C_in_pg = static_cast<uint32_t>(w.at(2));
    C_out = groups * C_out_pg;
    kH = static_cast<uint32_t>(w.at(3));
    kW = static_cast<uint32_t>(w.at(4));
    strideH = static_cast<uint32_t>(gconv->get_strides().at(0));
    strideW = static_cast<uint32_t>(gconv->get_strides().at(1));
    dilationH = static_cast<uint32_t>(gconv->get_dilations().at(0));
    dilationW = static_cast<uint32_t>(gconv->get_dilations().at(1));
    padTop = static_cast<uint32_t>(gconv->get_pads_begin().at(0));
    padLeft = static_cast<uint32_t>(gconv->get_pads_begin().at(1));
    padBottom = static_cast<uint32_t>(gconv->get_pads_end().at(0));
    padRight = static_cast<uint32_t>(gconv->get_pads_end().at(1));
  } else {
    return false;
  }

  if (!is_vulkan_backend() && m_type == "Convolution") {
    Conv2DCodegenDesc desc{};
    desc.input_type = m_node->get_input_element_type(0);
    desc.weight_type = m_node->get_input_element_type(1);
    desc.output_type = m_node->get_output_element_type(0);
    desc.C_out = C_out;
    desc.groups = groups;
    desc.kH = kH;
    desc.kW = kW;
    desc.outW = static_cast<uint32_t>(output_shape.at(3));
    if (requires_scalar_fp32_convolution_path(m_node)) {
      m_conv_output_channels_per_thread = 1;
      m_conv_output_width_per_thread = 1;
    } else {
      m_conv_output_channels_per_thread = gfx_conv2d_output_channel_block(desc);
      m_conv_output_width_per_thread = gfx_conv2d_output_width_block(desc);
    }
  }

  const size_t channels = static_cast<size_t>(C_out);
  std::vector<float> bias(channels, 0.0f);
  std::vector<float> gamma(channels, 1.0f);
  std::vector<float> beta(channels, 0.0f);
  std::vector<float> mean(channels, 0.0f);
  std::vector<float> var(channels, 1.0f);
  if (m_has_bias && !m_bias_params.values.empty()) {
    const size_t bias_count = std::min(channels, m_bias_params.values.size());
    std::copy_n(m_bias_params.values.begin(), bias_count, bias.begin());
  }
  if (m_has_bn && !m_bn_params.gamma.empty()) {
    gamma = m_bn_params.gamma;
    beta = m_bn_params.beta;
    mean = m_bn_params.mean;
    var = m_bn_params.var;
    epsilon = m_bn_params.epsilon;
  }

  std::vector<GpuTensor> extras;
  append_const_kernel_extra_input(extras, 1, "Conv2D weights");
  auto conv_payload = make_conv2d_runtime_param_payload(
      *m_buffer_manager, m_name, input_shape, output_shape, et, C_out, groups,
      C_in_pg, C_out_pg, kH, kW, strideH, strideW, dilationH, dilationW, padTop,
      padLeft, padBottom, padRight, m_has_bias, m_has_bn,
      m_has_activation ? kernel_activation_code(m_activation) : 0u,
      m_activation_alpha, epsilon, bias, gamma, beta, mean, var);
  OPENVINO_ASSERT(conv_payload.extra_inputs.size() == 6,
                  "GFX MLIR: Conv2D params payload must contain bias, BN and "
                  "metadata buffers for ",
                  m_name);
  extras.insert(extras.end(),
                std::make_move_iterator(conv_payload.extra_inputs.begin()),
                std::make_move_iterator(conv_payload.extra_inputs.end()));
  m_kernel_extra_inputs = std::move(extras);
  return true;
}

bool MlirStage::refresh_conv3d_kernel_extra_inputs(
    const ov::Shape &input_shape, const ov::Shape &output_shape) {
  auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
  if (!conv || input_shape.size() != 5) {
    return false;
  }
  OPENVINO_ASSERT(output_shape.size() == 5,
                  "GFX MLIR: Conv3D expects NCDHW output for stage ", m_name);
  OPENVINO_ASSERT(conv->get_input_partial_shape(1).is_static(),
                  "GFX MLIR: Conv3D weights shape must be static for stage ",
                  m_name);

  std::vector<GpuTensor> extras;
  append_const_kernel_extra_input(extras, 1, "Conv3D weights");
  auto conv3d_payload = make_conv3d_runtime_param_payload(
      *m_buffer_manager, m_name, input_shape, output_shape,
      conv->get_input_shape(1), conv->get_strides(), conv->get_dilations(),
      conv->get_pads_begin(), conv->get_pads_end());
  extras.insert(extras.end(),
                std::make_move_iterator(conv3d_payload.extra_inputs.begin()),
                std::make_move_iterator(conv3d_payload.extra_inputs.end()));
  m_kernel_extra_inputs = std::move(extras);
  return true;
}

void MlirStage::compile_prebuilt_kernel_source(
    const KernelSource &source, KernelRuntimeBindingState binding,
    std::string_view stage_kind) {
  std::string log;
  m_last_compiled_kernel_entry_point = source.entry_point;
  try {
    m_kernel = compile_kernel(source, &log);
  } catch (const std::exception &e) {
    OPENVINO_THROW("GFX MLIR: failed to compile ", stage_kind, " stage ",
                   m_name, " (", m_type, "): ", e.what());
  }
  OPENVINO_ASSERT(m_kernel, "GFX MLIR: failed to compile ", stage_kind,
                  " stage ", m_name, " (", m_type, "): ", log);
  m_kernel->prepare_runtime_artifacts();
  apply_source_plan_kernel_runtime_binding_state(std::move(binding));
}

void MlirStage::compile_generated_msl_source_plan(
    const GfxMslGeneratedKernelSourcePlan &source_plan,
    std::string_view stage_kind) {
  OPENVINO_ASSERT(source_plan.valid(),
                  "GFX MLIR: generated MSL source plan is "
                  "invalid for stage ",
                  m_name, " (", m_type, ")");
  compile_prebuilt_kernel_source(
      source_plan.source, source_plan.binding.runtime_binding, stage_kind);
}

void MlirStage::compile_from_plan(MlirKernelPlanContext &plan_ctx,
                                  mlir::ModuleOp module,
                                  const char *stage_kind) {
  auto &build_info = plan_ctx.build_info;
  if (module) {
    normalize_operand_segment_sizes(module);
    if (m_force_single_dispatch) {
      module->setAttr("gfx.force_single_dispatch",
                      mlir::BoolAttr::get(module.getContext(), true));
    }
  }
  const size_t scalar_inputs = plan_ctx.scalar_inputs;
  const size_t buffer_inputs = plan_ctx.buffer_inputs;
  const size_t output_arg_count =
      plan_ctx.output_args != 0 ? plan_ctx.output_args
                                : (m_node ? m_node->get_output_size() : 0);
  if (!m_kernel_binding_owned_by_source_plan) {
    apply_kernel_runtime_binding_state(make_stage_direct_kernel_runtime_binding(
        std::move(build_info.mapping.mapping.kernel_inputs), buffer_inputs, {},
        {}));
  }
  KernelSource src = build_info.plan.to_source();
  const bool has_custom_kernel_signature =
      configure_backend_custom_kernel_source_signature_from_module(src);
  if (!has_custom_kernel_signature) {
    src.signature.output_arg_count = static_cast<uint32_t>(output_arg_count);
  }
  if (is_vulkan_backend() && !has_custom_kernel_signature) {
    src.signature.arg_count =
        static_cast<uint32_t>(infer_kernel_arg_count_from_module(
            src.module, buffer_inputs + output_arg_count, src.entry_point,
            GfxKernelBackendDomain::Spirv));
  }
  if (backend_kind() == GpuBackend::Metal) {
    gfx_attach_mpsrt_const_tensors(src, m_node);
  }
  if (src.module) {
    normalize_operand_segment_sizes(src.module);
    if (m_force_single_dispatch) {
      src.module->setAttr("gfx.force_single_dispatch",
                          mlir::BoolAttr::get(src.module.getContext(), true));
    }
    if (gfx_log_debug_enabled()) {
      llvm::errs() << "[GFX][MLIRExec] module before backend compile for "
                   << m_name << " (" << m_type << "):\n";
      src.module.dump();
    }
  }
  std::string log;
  m_last_compiled_kernel_entry_point = src.entry_point;
  try {
    m_kernel = compile_kernel(src, &log);
  } catch (const std::exception &e) {
    OPENVINO_THROW("GFX MLIR: failed to compile ", stage_kind, " stage ",
                   m_name, " (", m_type, "): ", e.what());
  }
  OPENVINO_ASSERT(m_kernel, "GFX MLIR: failed to compile ", stage_kind,
                  " stage ", m_name, " (", m_type, "): ", log);
  m_kernel->prepare_runtime_artifacts();
  const std::string runtime_metadata_entry_point =
      m_last_compiled_kernel_entry_point.empty()
          ? src.entry_point
          : m_last_compiled_kernel_entry_point;
  if (m_kernel_binding_owned_by_source_plan) {
    return;
  }
  if (src.module) {
    const std::string metadata_entry_point =
        is_vulkan_backend() ? runtime_metadata_entry_point : std::string{};
    auto runtime_meta = extract_kernel_runtime_metadata(
        src.module, output_arg_count, buffer_inputs, metadata_entry_point,
        is_vulkan_backend() ? std::optional<GfxKernelBackendDomain>(
                                  GfxKernelBackendDomain::Spirv)
                            : std::optional<GfxKernelBackendDomain>(
                                  GfxKernelBackendDomain::AppleMsl));
    apply_kernel_metadata(runtime_meta, scalar_inputs);
  } else if (module) {
    auto runtime_meta =
        build_info.runtime_metadata(m_node, plan_ctx.output_args);
    apply_kernel_metadata(runtime_meta, scalar_inputs);
  }
}

void MlirStage::init(GpuBufferManager *buffer_manager) {
  m_buffer_manager = buffer_manager;
}

void MlirStage::prepare_constant_input_buffers(bool skip_matmul_weight_const) {
  if (!m_node) {
    return;
  }

  if (gfx_log_debug_enabled() && m_type == "MatMul") {
    if (auto mm = std::dynamic_pointer_cast<const ov::op::v0::MatMul>(m_node)) {
      std::ostringstream meta;
      meta << "MatMul ta=" << mm->get_transpose_a()
           << " tb=" << mm->get_transpose_b()
           << " A=" << mm->get_input_partial_shape(0)
           << " B=" << mm->get_input_partial_shape(1);
      gfx_log_debug("MLIRConst") << meta.str();
    }
  }

  const size_t in_count = m_node->get_input_size();
  if (!m_const_buffers) {
    m_const_buffers = std::make_shared<ConstBufferSet>();
  }
  if (m_const_buffers->buffers.size() < in_count) {
    m_const_buffers->buffers.resize(in_count);
    m_const_buffers->present.assign(in_count, false);
  }

  const bool use_const_cache = true;
  bool const_cache_checked = false;
  for (size_t i = 0; i < in_count; ++i) {
    if (skip_matmul_weight_const && i == 1) {
      continue;
    }
    auto const_tensor =
        gfx_evaluate_constant_source_tensor(m_node->input_value(i));
    if (!const_tensor.has_value()) {
      continue;
    }
    if (!const_cache_checked) {
      OPENVINO_ASSERT(
          m_buffer_manager,
          "GFX MLIR: const buffer manager is required for constants (stage ",
          m_name, ")");
      OPENVINO_ASSERT(m_buffer_manager->supports_const_cache(),
                      "GFX MLIR: const cache must be supported for stage ",
                      m_name);
      const_cache_checked = true;
    }
    if (m_const_buffers->present[i] &&
        m_const_buffers->buffers[i].buf.valid()) {
      continue;
    }
    std::vector<ov::float16> packed_f16;
    std::vector<uint8_t> packed_conv_weights;
    const void *const_data = const_tensor->data();
    size_t bytes = const_tensor->get_byte_size();
    auto et = const_tensor->get_element_type();
    if (backend_kind() == GpuBackend::Metal &&
        should_pack_matmul_const_input_as_f16(m_node, i, *const_tensor)) {
      const size_t original_bytes = bytes;
      packed_f16 = pack_f32_tensor_as_f16(*const_tensor);
      const_data = packed_f16.data();
      bytes = packed_f16.size() * sizeof(ov::float16);
      et = ov::element::f16;
      increment_compile_counter("matmul_const_f32_to_f16_pack_count");
      increment_compile_counter("matmul_const_f32_to_f16_original_bytes",
                                original_bytes);
      increment_compile_counter("matmul_const_f32_to_f16_packed_bytes", bytes);
    }
    if (backend_kind() == GpuBackend::Metal &&
        should_pack_conv2d_const_weights_oc4(m_node, i, *const_tensor)) {
      const size_t original_bytes = bytes;
      packed_conv_weights = pack_conv2d_weights_oc4(*const_tensor);
      et = const_tensor->get_element_type();
      m_conv_weight_storage_type = et;
      const_data = packed_conv_weights.data();
      bytes = packed_conv_weights.size();
      m_conv_weights_packed_oc4 = true;
      increment_compile_counter("conv2d_const_oc4_pack_count");
      increment_compile_counter("conv2d_const_oc4_original_bytes",
                                original_bytes);
      increment_compile_counter("conv2d_const_oc4_packed_bytes", bytes);
    }
    if (gfx_log_debug_enabled() && et == ov::element::f32 &&
        bytes >= sizeof(float)) {
      const float *vals = const_tensor->data<const float>();
      const size_t count = bytes / sizeof(float);
      std::ostringstream oss;
      oss << "const[" << i << "] ";
      const size_t dump_n = std::min<size_t>(count, 6);
      for (size_t vi = 0; vi < dump_n; ++vi) {
        if (vi) {
          oss << ", ";
        }
        oss << vals[vi];
      }
      gfx_log_debug("MLIRConst") << oss.str();
    }
    if (use_const_cache && bytes) {
      const uint64_t hash = gfx_hash_bytes(const_data, bytes);
      std::ostringstream key;
      key << m_name << "/const/" << i << "/" << et.get_type_name() << "/"
          << bytes << "/" << hash;
      GpuBuffer buf =
          m_buffer_manager->wrap_const(key.str(), const_data, bytes, et);
      OPENVINO_ASSERT(buf.valid(),
                      "GFX MLIR: failed to wrap const buffer for stage ",
                      m_name);
      buf.owned = false;
      m_const_buffers->buffers[i].buf = buf;
    }
    m_const_buffers->buffers[i].shape = const_tensor->get_shape();
    m_const_buffers->buffers[i].expected_type = et;
    m_const_buffers->present[i] = true;
  }
}

void MlirStage::compile(GpuBufferManager *buffer_manager) {
  auto &ctx = gfx_mlir_context();
  if (m_is_view_op) {
    return;
  }
  if (m_kernel) {
    return;
  }
  if (!m_buffer_manager) {
    m_buffer_manager = buffer_manager;
  }
  m_kernel_extra_inputs.clear();
  m_kernel_binding = {};
  m_kernel_binding_owned_by_source_plan = false;
  m_uses_mpsrt_sdpa_plan = false;
  m_force_single_dispatch = false;
  m_matmul_reduction_threads = 1;
  m_compressed_matmul_output_block = 1;
  m_compressed_matmul_n = 0;
  m_conv_output_channels_per_thread = 1;
  m_conv_output_width_per_thread = 1;
  m_conv_weight_storage_type = ov::element::dynamic;
  m_conv_weights_packed_oc4 = false;
  m_rms_reduction_threads = 1;
  m_rms_hidden = 0;
  if (auto desc = make_static_matmul_codegen_desc_for_node(m_node)) {
    desc->has_activation = m_has_activation;
    desc->activation = m_activation;
    desc->alpha = m_activation_alpha;
    m_matmul_reduction_threads = gfx_matmul_parallel_reduction_threads(*desc);
  }
  if (m_type == "RMS" && m_node &&
      m_node->get_input_partial_shape(0).rank().is_static()) {
    const auto pshape = m_node->get_input_partial_shape(0);
    const auto rank = pshape.rank().get_length();
    if (rank > 0 && pshape[rank - 1].is_static()) {
      m_rms_hidden = static_cast<uint32_t>(pshape[rank - 1].get_length());
      m_rms_reduction_threads =
          gfx_rms_parallel_reduction_threads(m_rms_hidden);
    }
  }
  std::optional<CompressedMatMulInfo> compressed_matmul_info;
  if (!is_vulkan_backend()) {
    compressed_matmul_info = detect_compressed_matmul_weights(m_node);
  }
  prepare_constant_input_buffers(compressed_matmul_info.has_value());
  if (!is_vulkan_backend() && m_type == "Concat" &&
      concat_has_runtime_shape(m_node) && !has_absorbed_input_transpose()) {
    return;
  }
  if (compressed_matmul_info) {
    OPENVINO_ASSERT(m_buffer_manager,
                    "GFX MLIR: const buffer manager is required for compressed "
                    "MatMul stage ",
                    m_name);
    const auto caps = query_parallelism_caps(m_buffer_manager);
    m_matmul_reduction_threads = compressed_matmul_parallel_reduction_threads(
        *compressed_matmul_info, caps);
    m_compressed_matmul_output_block = compressed_matmul_output_block(
        *compressed_matmul_info, caps, m_matmul_reduction_threads);
    m_compressed_matmul_n = static_cast<uint32_t>(compressed_matmul_info->n);
    const std::vector<uint8_t> packed_weights =
        pack_compressed_matmul_weights_for_output_block(
            *compressed_matmul_info, m_compressed_matmul_output_block);
    const std::vector<uint8_t> packed_scales =
        pack_compressed_matmul_scales(*compressed_matmul_info);
    if (gfx_log_debug_enabled()) {
      std::ostringstream oss;
      oss << "compressed MatMul reduction threads="
          << m_matmul_reduction_threads
          << " output_block=" << m_compressed_matmul_output_block
          << " K=" << compressed_matmul_info->k
          << " N=" << compressed_matmul_info->n
          << " parts=" << compressed_matmul_info->parts.size()
          << " packed_weight_bytes=" << packed_weights.size()
          << " packed_scale_bytes=" << packed_scales.size()
          << " max_threads=" << caps.max_total_threads_per_group << " simd="
          << std::max(caps.subgroup_size, caps.preferred_simd_width);
      gfx_log_debug("MLIRConst") << oss.str();
    }

    auto compressed_source_plan = make_compressed_matmul_msl_kernel_source_plan(
        *compressed_matmul_info, m_matmul_reduction_threads,
        m_compressed_matmul_output_block);
    compile_generated_msl_source_plan(compressed_source_plan,
                                      "compressed MatMul");

    m_kernel_extra_inputs.clear();
    std::ostringstream weight_suffix;
    weight_suffix
        << "compressed_matmul/packed_weights/"
        << compressed_matmul_info->weights->get_element_type().get_type_name()
        << "/block" << m_compressed_matmul_output_block;
    m_kernel_extra_inputs.push_back(make_hashed_kernel_bytes_param_tensor(
        *m_buffer_manager, m_name, weight_suffix.str(), packed_weights.data(),
        packed_weights.size(), ov::element::u8,
        ov::Shape{packed_weights.size()}));
    m_kernel_extra_inputs.push_back(make_hashed_kernel_bytes_param_tensor(
        *m_buffer_manager, m_name, "compressed_matmul/packed_scale",
        packed_scales.data(), packed_scales.size(),
        compressed_matmul_info->scale->get_element_type(),
        ov::Shape{compressed_matmul_info->n, compressed_matmul_info->groups,
                  1}));
    m_parallel_cfg = ParallelDispatchConfig{};
    m_force_single_dispatch = false;
    m_is_compressed_matmul = true;
    increment_compile_counter("compressed_matmul_i4_stage_count");
    return;
  }
  // Backend-side broadcast helpers for elementwise ops: prepare constant
  // buffers with output dims and input strides to avoid runtime CPU copies.
  auto prepare_eltwise_broadcast_meta = [&]() {
    if (!m_node) {
      return;
    }
    // Binary elementwise only; other kinds handled elsewhere.
    const size_t inputs = m_node->get_input_size();
    if (inputs < 2) {
      return;
    }
    // Gather shapes (must be static for now).
    if (!m_node->get_output_partial_shape(0).is_static() ||
        !m_node->get_input_partial_shape(0).is_static() ||
        !m_node->get_input_partial_shape(1).is_static()) {
      return;
    }
    const ov::Shape out_shape = m_node->get_output_shape(0);
    if (out_shape.empty()) {
      return;
    }
    const ov::Shape in0_shape = compile_time_input_shape(0);
    const ov::Shape in1_shape = compile_time_input_shape(1);
    auto stride0 = compile_time_broadcast_strides(0, out_shape);
    auto stride1 = compile_time_broadcast_strides(1, out_shape);
    if (stride0.empty() && m_node) {
      stride0 = compute_broadcast_element_strides(m_node->get_input_shape(0),
                                                  out_shape);
    }
    if (stride1.empty() && m_node) {
      stride1 = compute_broadcast_element_strides(m_node->get_input_shape(1),
                                                  out_shape);
    }

    auto broadcast_payload = make_binary_broadcast_runtime_param_payload(
        *m_buffer_manager, m_name, out_shape, std::move(stride0),
        std::move(stride1));
    m_kernel_binding.scalar_args = broadcast_payload.scalar_args;
    m_kernel_extra_inputs.insert(
        m_kernel_extra_inputs.end(),
        std::make_move_iterator(broadcast_payload.extra_inputs.begin()),
        std::make_move_iterator(broadcast_payload.extra_inputs.end()));
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MLIRExec")
          << "Prepared eltwise broadcast meta: dims=" << out_shape.size()
          << " extras=" << m_kernel_extra_inputs.size();
    }
  };
  // Only for classic binary eltwise ops; keep list tight to avoid surprises.
  if ((m_type == "Add" || m_type == "Subtract" || m_type == "Multiply" ||
       m_type == "Divide" || m_type == "Power" || m_type == "Mod" ||
       m_type == "FloorMod" || m_type == "Minimum" || m_type == "Maximum" ||
       m_type == "Equal" || m_type == "NotEqual" || m_type == "Less" ||
       m_type == "Greater" || m_type == "LessEqual" ||
       m_type == "GreaterEqual" || m_type == "LogicalAnd" ||
       m_type == "LogicalOr" || m_type == "LogicalXor" ||
       m_type == "SquaredDifference" || m_type == "PRelu")) {
    prepare_eltwise_broadcast_meta();
  }
  ov::element::Type out_et = ov::element::dynamic;
  if (m_node) {
    out_et = m_node->get_output_element_type(0);
  }
  if (m_has_bias) {
    const size_t count = m_bias_params.values.size();
    if (count) {
      const bool conv_like =
          (m_type == "Convolution" || m_type == "GroupConvolution");
      size_t out_rank = m_bias_params.shape.size();
      if (m_node) {
        const auto &pshape = m_node->get_output_partial_shape(0);
        if (pshape.rank().is_static()) {
          out_rank = static_cast<size_t>(pshape.rank().get_length());
        }
      }
      GpuTensor tensor = make_bias_runtime_param_tensor(
          *m_buffer_manager, m_name, m_bias_params.values, m_bias_params.shape,
          m_bias_params.element_type, out_et, out_rank, conv_like);
      m_kernel_extra_inputs.push_back(std::move(tensor));
    }
  }
  if (m_has_bn) {
    const size_t channels = m_bn_params.gamma.size();
    if (channels) {
      auto bn_payload = make_batchnorm_scale_bias_runtime_param_payload(
          *m_buffer_manager, m_name, m_bn_params.gamma, m_bn_params.beta,
          m_bn_params.mean, m_bn_params.var, m_bn_params.epsilon, out_et);
      m_kernel_extra_inputs.insert(
          m_kernel_extra_inputs.end(),
          std::make_move_iterator(bn_payload.extra_inputs.begin()),
          std::make_move_iterator(bn_payload.extra_inputs.end()));
    }
  }
  // Convolution: build fixed extra inputs (bias + BN + params) to align with
  // MSL/Spir-V kernel signatures and avoid runtime CPU-side packing.
  if (m_node && is_conv_like()) {
    m_kernel_extra_inputs.clear();
    const auto &out_shape = m_node->get_output_shape(0);
    const auto &in_shape = m_node->get_input_shape(0);
    const size_t in_rank = in_shape.size();
    if (in_rank == 5) {
      OPENVINO_ASSERT(refresh_conv3d_kernel_extra_inputs(in_shape, out_shape),
                      "GFX MLIR: Conv3D node cast failed for stage ", m_name);
    } else {
      OPENVINO_ASSERT(
          refresh_conv2d_kernel_extra_inputs(in_shape, out_shape, out_et),
          "GFX MLIR: Conv2D node cast failed for stage ", m_name);
    }
  }
  if (m_type == "MaxPool" || m_type == "AvgPool") {
    m_kernel_extra_inputs.clear();
    auto maxpool =
        std::dynamic_pointer_cast<const ov::op::util::MaxPoolBase>(m_node);
    auto avgpool = std::dynamic_pointer_cast<const ov::op::v1::AvgPool>(m_node);
    OPENVINO_ASSERT(maxpool || avgpool, "GFX MLIR: pool node cast failed");
    const auto in = m_node->get_input_shape(0);
    const auto out = m_node->get_output_shape(0);
    OPENVINO_ASSERT(in.size() == 4 && out.size() == 4,
                    "GFX MLIR: pool expects NCHW");
    const auto &kernel =
        maxpool ? maxpool->get_kernel() : avgpool->get_kernel();
    const auto &strides =
        maxpool ? maxpool->get_strides() : avgpool->get_strides();
    const auto &pads_begin =
        maxpool ? maxpool->get_pads_begin() : avgpool->get_pads_begin();
    const auto &pads_end =
        maxpool ? maxpool->get_pads_end() : avgpool->get_pads_end();
    ov::Strides dilations(kernel.size(), 1);
    if (maxpool) {
      if (auto p =
              std::dynamic_pointer_cast<const ov::op::v8::MaxPool>(m_node)) {
        dilations = p->get_dilations();
      } else if (auto p = std::dynamic_pointer_cast<const ov::op::v14::MaxPool>(
                     m_node)) {
        dilations = p->get_dilations();
      }
    }
    const bool is_avg = avgpool != nullptr;
    const bool exclude_pad = is_avg ? avgpool->get_exclude_pad() : true;
    auto pool_payload = make_pool_runtime_param_payload(
        *m_buffer_manager, m_name, in, out, kernel, strides, pads_begin,
        pads_end, dilations, is_avg, exclude_pad);
    m_kernel_extra_inputs = std::move(pool_payload.extra_inputs);
  }
  if (m_type == "ShapeOf") {
    OPENVINO_ASSERT(m_node, "GFX MLIR: ShapeOf stage requires node");
    const auto input_pshape = m_node->get_input_partial_shape(0);
    OPENVINO_ASSERT(input_pshape.rank().is_static(),
                    "GFX MLIR: ShapeOf input rank must be static");
    const size_t rank = static_cast<size_t>(input_pshape.rank().get_length());
    const auto output_et = m_node->get_output_element_type(0);
    OPENVINO_ASSERT(output_et == ov::element::i32 ||
                        output_et == ov::element::i64,
                    "GFX MLIR: ShapeOf output must be i32/i64");

    ov::Shape compile_shape(rank, 0);
    if (!m_inputs.empty() && m_inputs[0] && !m_inputs[0]->shape.empty()) {
      for (size_t i = 0; i < rank && i < m_inputs[0]->shape.size(); ++i) {
        compile_shape[i] = m_inputs[0]->shape[i];
      }
    } else if (input_pshape.is_static()) {
      compile_shape = m_node->get_input_shape(0);
    } else {
      for (size_t i = 0; i < rank; ++i) {
        if (input_pshape[i].is_static()) {
          compile_shape[i] = static_cast<size_t>(input_pshape[i].get_length());
        }
      }
    }

    auto shapeof_payload = make_shapeof_runtime_param_payload(
        *m_buffer_manager, m_name, compile_shape, output_et);
    const auto shapeof_scalars = shapeof_payload.scalar_args;
    m_kernel_extra_inputs = std::move(shapeof_payload.extra_inputs);
    auto shapeof_plan = require_backend_custom_kernel_binding_plan(
        is_vulkan_backend(), "ShapeOf", "shapeof_kernel", shapeof_scalars,
        m_name);
    apply_kernel_runtime_binding_state(shapeof_plan.runtime_binding);
  }
  if (m_type == "GfxSDPAWithCausalMask") {
    if (is_vulkan_backend()) {
      OPENVINO_THROW("GFX Vulkan SDPA causal mask fusion is not enabled yet");
    }
    const auto et =
        m_node ? m_node->get_output_element_type(0) : ov::element::dynamic;
    auto sdpa_source_plan = make_causal_sdpa_msl_kernel_source_plan(et);
    compile_generated_msl_source_plan(sdpa_source_plan, "SDPA causal mask");
    m_force_single_dispatch = false;
    return;
  }
  if (m_type == "ScaledDotProductAttention") {
    if (is_vulkan_backend()) {
      OPENVINO_THROW("GFX Vulkan SDPA: native ScaledDotProductAttention is not "
                     "enabled yet");
    }
    KernelSource sdpa_mps_source;
    sdpa_mps_source.module =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    auto sdpa_mps_source_plan =
        configure_apple_mps_vendor_kernel_source_plan_for_node(
            sdpa_mps_source, m_node, m_buffer_manager,
            "ScaledDotProductAttention",
            /*has_bias=*/false,
            /*has_activation=*/false,
            /*has_batchnorm=*/false, ActivationKind::Identity, nullptr,
            m_runtime_traits);
    if (sdpa_mps_source_plan.valid()) {
      OPENVINO_ASSERT(
          sdpa_mps_source_plan.has_runtime_binding,
          "GFX MLIR: MPS SDPA source plan must provide runtime binding for ",
          m_name);
      compile_prebuilt_kernel_source(sdpa_mps_source_plan.source,
                                     sdpa_mps_source_plan.runtime_binding,
                                     "MPS SDPA");
      m_kernel_extra_inputs.clear();
      m_force_single_dispatch = false;
      m_uses_mpsrt_sdpa_plan = true;
      return;
    }
    const auto et =
        m_node ? m_node->get_output_element_type(0) : ov::element::dynamic;
    const bool has_mask = m_node && m_node->get_input_size() >= 4;
    auto sdpa_source_plan = make_sdpa_msl_kernel_source_plan(et, has_mask);
    compile_generated_msl_source_plan(sdpa_source_plan, "SDPA");
    m_force_single_dispatch = false;
    return;
  }
  if (m_type == "Softmax" || m_type == "LogSoftmax" || m_type == "Split" ||
      m_type == "VariadicSplit") {
    return;
  }
  const auto optimization_plan = stage_optimization_plan();
  if (should_skip_generic_kernel_compile(optimization_plan)) {
    return;
  }
  KernelPlan plan = [&]() {
    if (m_type == "Add" && has_absorbed_input_transpose()) {
      auto module = build_mlir_add_from_node(m_node, ctx, m_input_transforms);
      return KernelPlan(module, resolve_entry_point(module, {}, "gfx_kernel"),
                        0);
    }
    if (m_type == "Convolution" && has_absorbed_input_transpose()) {
      auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
      OPENVINO_ASSERT(
          conv, "GFX MLIR: expected Convolution node for absorbed transpose");
      auto module = build_mlir_conv2d_from_node(conv, ctx, input_transform(0));
      return KernelPlan(module, resolve_entry_point(module, {}, "conv2d_main"),
                        0);
    }
    if (m_type == "GroupConvolution") {
      auto gconv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(m_node);
      OPENVINO_ASSERT(gconv, "GFX MLIR: expected GroupConvolution node");
      auto module =
          build_mlir_group_conv2d_from_node(gconv, ctx, input_transform(0));
      return KernelPlan(
          module, resolve_entry_point(module, {}, "group_conv2d_main"), 0);
    }
    MlirKernelPlanBuilder plan_builder;
    return plan_builder.build_plan(m_node, ctx, 0);
  }();
  auto module = plan.module();
  auto should_skip_vulkan_conv_parallel = [&]() {
    if (!is_vulkan_backend() || m_type != "Convolution" ||
        has_absorbed_input_transpose() || !m_node) {
      return false;
    }
    if (optimization_plan.conv.kind != GfxConvRouteKind::None) {
      return false;
    }
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    if (!conv || conv->get_input_size() != 2 || conv->get_output_size() != 1) {
      return false;
    }
    const auto &in_shape = conv->get_input_shape(0);
    const auto &w_shape = conv->get_input_shape(1);
    const auto &out_shape = conv->get_output_shape(0);
    if (in_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4) {
      return false;
    }
    return out_shape[1] == 1 && w_shape[2] == 1 && w_shape[3] == 1 &&
           conv->get_strides().at(0) == 1 && conv->get_strides().at(1) == 1 &&
           conv->get_dilations().at(0) == 1 &&
           conv->get_dilations().at(1) == 1 &&
           conv->get_pads_begin().at(0) == 0 &&
           conv->get_pads_begin().at(1) == 0 &&
           conv->get_pads_end().at(0) == 0 && conv->get_pads_end().at(1) == 0 &&
           in_shape[2] == out_shape[2] && in_shape[3] == out_shape[3];
  };
  const bool force_vulkan_conv_serial_fallback =
      is_vulkan_backend() && m_vulkan_conv_serial_retry_attempted &&
      is_conv_like() && !has_absorbed_input_transpose() &&
      optimization_plan.conv.kind == GfxConvRouteKind::None;
  apply_stage_optimization_attrs(module, optimization_plan);
  apply_input_transform_attrs(module);
  set_parallel_preference(module);
  if (m_conv_weights_packed_oc4 && module) {
    module->setAttr("gfx.conv2d_weights_packed_oc4",
                    mlir::BoolAttr::get(module.getContext(), true));
    if (m_conv_weight_storage_type == ov::element::f16) {
      module->setAttr("gfx.conv2d_weight_storage_f16",
                      mlir::BoolAttr::get(module.getContext(), true));
    }
  }
  if (should_skip_vulkan_conv_parallel() || force_vulkan_conv_serial_fallback) {
    module->setAttr("gfx.skip_conv_parallel",
                    mlir::BoolAttr::get(module.getContext(), true));
    module->setAttr("gfx.prefer_parallel",
                    mlir::BoolAttr::get(module.getContext(), false));
    // This path keeps the shared serial loop nest instead of rewriting the
    // convolution to scf.parallel. Launch it as a single kernel instance;
    // default element-wise dispatch would otherwise execute the full serial
    // loop body once per output element and corrupt the accumulation.
    m_force_single_dispatch = true;
  }
  if (is_vulkan_backend() && m_type == "GroupConvolution" &&
      optimization_plan.conv.algorithm.kind !=
          GfxConvAlgorithmKind::DepthwiseDirect) {
    // Unsupported grouped convolutions still use the shared serial MLIR loop
    // nest. Depthwise group convs are lowered later to scf.parallel.
    m_force_single_dispatch = true;
  }
  apply_fused_operations(module);
  if (module && is_conv_like() && m_node &&
      m_node->get_input_partial_shape(0).is_static()) {
    const auto input_rank = m_node->get_input_shape(0).size();
    if (m_type == "Convolution" && input_rank == 5) {
      auto conv3d_plan = annotate_required_backend_custom_kernel_abi_binding(
          module, is_vulkan_backend(), "Conv3D", "conv3d_kernel", m_name);
      if (is_vulkan_backend()) {
        apply_kernel_runtime_binding_state(conv3d_plan.runtime_binding);
      }
    } else if (input_rank == 4) {
      if (force_vulkan_conv_serial_fallback) {
        auto conv2d_plan =
            annotate_required_backend_custom_kernel_direct_io_binding(
                module, /*is_vulkan_backend=*/true, m_type, "conv2d_main",
                /*tensor_input_count=*/2, /*output_count=*/1, m_name);
        auto direct_binding = make_stage_direct_kernel_runtime_binding(
            {0, 1}, /*input_arg_count=*/2, {1, 1, 1}, {0, 1, 2});
        annotate_spirv_kernel_binding_attrs(module, direct_binding);
        apply_kernel_runtime_binding_state(std::move(direct_binding));
        m_kernel_extra_inputs.clear();
        (void)conv2d_plan;
      } else {
        auto conv2d_plan = annotate_required_backend_custom_kernel_abi_binding(
            module, is_vulkan_backend(), m_type, "conv2d_kernel", m_name);
        if (is_vulkan_backend()) {
          apply_kernel_runtime_binding_state(conv2d_plan.runtime_binding);
        }
      }
    }
  }
  if (m_type == "ShapeOf" && module) {
    auto shapeof_plan = annotate_required_backend_custom_kernel_binding(
        module, is_vulkan_backend(), "ShapeOf", "shapeof_kernel",
        m_kernel_binding.scalar_args, m_name);
    if (is_vulkan_backend()) {
      apply_kernel_runtime_binding_state(shapeof_plan.runtime_binding);
    }
  }
  if (m_type == "Tile" && m_node && module) {
    OPENVINO_ASSERT(
        m_node->get_input_partial_shape(0).is_static() &&
            m_node->get_output_partial_shape(0).is_static(),
        "GFX MLIR: Tile requires static input/output shapes for stage ",
        m_name);
    const ov::Shape in_shape = m_node->get_input_shape(0);
    const ov::Shape out_shape = m_node->get_output_shape(0);
    auto tile_payload = make_tile_runtime_param_payload(
        *m_buffer_manager, m_name, in_shape, out_shape);
    const auto tile_scalars = tile_payload.scalar_args;
    m_kernel_extra_inputs = std::move(tile_payload.extra_inputs);
    auto tile_plan = annotate_required_backend_custom_kernel_binding(
        module, is_vulkan_backend(), "Tile", "tile_kernel", tile_scalars,
        m_name);
    if (is_vulkan_backend()) {
      apply_kernel_runtime_binding_state(tile_plan.runtime_binding);
    }
  }
  if (auto gather_elements =
          ov::as_type_ptr<const ov::op::v6::GatherElements>(m_node)) {
    OPENVINO_ASSERT(
        gather_elements->get_input_partial_shape(0).is_static() &&
            gather_elements->get_input_partial_shape(1).is_static() &&
            gather_elements->get_output_partial_shape(0).is_static(),
        "GFX MLIR: GatherElements requires static shapes for stage ", m_name);
    const ov::Shape data_shape = gather_elements->get_input_shape(0);
    const ov::Shape out_shape = gather_elements->get_output_shape(0);
    const int64_t axis_norm =
        normalize_axis(gather_elements->get_axis(), out_shape.size(),
                       "GFX MLIR: GatherElements");
    auto gather_elements_payload = make_gather_elements_runtime_param_payload(
        *m_buffer_manager, m_name, data_shape, out_shape,
        static_cast<uint32_t>(axis_norm));
    m_kernel_extra_inputs = std::move(gather_elements_payload.extra_inputs);
    const std::vector<int32_t> gather_elements_scalars;
    auto gather_elements_plan = annotate_required_backend_custom_kernel_binding(
        module, is_vulkan_backend(), "GatherElements", "gather_elements_kernel",
        gather_elements_scalars, m_name);
    if (is_vulkan_backend()) {
      apply_kernel_runtime_binding_state(gather_elements_plan.runtime_binding);
    }
  }
  if (!is_vulkan_backend()) {
    if (auto gather = ov::as_type_ptr<const ov::op::util::GatherBase>(m_node)) {
      int64_t batch_dims = 0;
      if (auto gather_v7 = ov::as_type_ptr<const ov::op::v7::Gather>(m_node)) {
        batch_dims = gather_v7->get_batch_dims();
      } else if (auto gather_v8 =
                     ov::as_type_ptr<const ov::op::v8::Gather>(m_node)) {
        batch_dims = gather_v8->get_batch_dims();
      }
      OPENVINO_ASSERT(batch_dims == 0,
                      "GFX MLIR: Gather batch_dims not supported for stage ",
                      m_name);
      OPENVINO_ASSERT(
          m_node->get_input_partial_shape(0).is_static() &&
              m_node->get_input_partial_shape(1).is_static(),
          "GFX MLIR: Gather requires static input/index shapes for stage ",
          m_name);
      const ov::Shape data_shape = m_node->get_input_shape(0);
      const ov::Shape indices_shape = m_node->get_input_shape(1);
      const int64_t axis_norm = normalize_axis(
          gather->get_axis(), data_shape.size(), "GFX MLIR: Gather");
      auto gather_payload = make_gather_runtime_param_payload(
          *m_buffer_manager, m_name, data_shape, indices_shape,
          static_cast<uint32_t>(axis_norm));
      m_kernel_extra_inputs = std::move(gather_payload.extra_inputs);
      auto gather_plan = annotate_required_backend_custom_kernel_binding(
          module, is_vulkan_backend(), "Gather", "gather_kernel", {}, m_name);
      apply_kernel_runtime_binding_state(gather_plan.runtime_binding);
    }
  }
  auto is_unary_eltwise_compile_stage = [&]() {
    return m_type == "Relu" || m_type == "Sigmoid" || m_type == "Tanh" ||
           m_type == "Elu" || m_type == "Gelu" || m_type == "Swish" ||
           m_type == "HSwish" || m_type == "HSigmoid" || m_type == "SoftPlus" ||
           m_type == "Mish" || m_type == "SoftSign" || m_type == "Abs" ||
           m_type == "Sign" || m_type == "Clamp" || m_type == "LogicalNot" ||
           m_type == "Exp" ||
           m_type == "Log" || m_type == "Sqrt" || m_type == "Floor" ||
           m_type == "Ceiling" || m_type == "Negative" || m_type == "Sin" ||
           m_type == "Cos" || m_type == "Tan" || m_type == "Erf" ||
           m_type == "Asin" || m_type == "Acos" || m_type == "Atan" ||
           m_type == "Asinh" || m_type == "Acosh" || m_type == "Atanh" ||
           m_type == "Sinh" || m_type == "Cosh" || m_type == "Round";
  };
  auto is_binary_eltwise_compile_stage = [&]() {
    return m_type == "Add" || m_type == "Subtract" || m_type == "Multiply" ||
           m_type == "Divide" || m_type == "Power" || m_type == "Mod" ||
           m_type == "FloorMod" || m_type == "Minimum" || m_type == "Maximum" ||
           m_type == "Equal" || m_type == "NotEqual" || m_type == "Less" ||
           m_type == "Greater" || m_type == "LessEqual" ||
           m_type == "GreaterEqual" || m_type == "LogicalAnd" ||
           m_type == "LogicalOr" || m_type == "LogicalXor" ||
           m_type == "SquaredDifference" || m_type == "PRelu";
  };
  if (is_vulkan_backend() && module && is_unary_eltwise_compile_stage()) {
    auto unary_plan = annotate_required_backend_custom_kernel_abi_binding(
        module, /*is_vulkan_backend=*/true, m_type, "unary_kernel", m_name);
    apply_kernel_runtime_binding_state(unary_plan.runtime_binding);
  }
  if (is_vulkan_backend() && module && is_binary_eltwise_compile_stage()) {
    auto binary_plan = annotate_required_backend_custom_kernel_abi_binding(
        module, /*is_vulkan_backend=*/true, m_type, "eltwise_kernel", m_name);
    apply_kernel_runtime_binding_state(binary_plan.runtime_binding);
  }
  if (!is_vulkan_backend() && (m_type == "MaxPool" || m_type == "AvgPool") &&
      module) {
    const std::vector<int32_t> pool_scalars;
    (void)annotate_required_backend_custom_kernel_binding(
        module, /*is_vulkan_backend=*/false, m_type, "pool2d_kernel",
        pool_scalars, m_name);
  }
  if (!is_vulkan_backend() && m_type == "Concat" && module) {
    auto concat_binding_plan =
        annotate_required_backend_custom_kernel_direct_io_binding(
            module, /*is_vulkan_backend=*/false, "Concat", "concat_kernel",
            m_node ? m_node->get_input_size() : 0,
            m_node ? m_node->get_output_size() : 0, m_name);
    apply_source_plan_kernel_runtime_binding_state(
        concat_binding_plan.runtime_binding);
  }
  const bool use_msl_stage_manifest_arg_count =
      !is_vulkan_backend() &&
      (m_type == "ShapeOf" || m_type == "Tile" || m_type == "GatherElements" ||
       m_type == "MaxPool" || m_type == "AvgPool" || m_type == "Concat");
  auto plan_ctx = build_mlir_kernel_plan(
      module, plan.entry_point(), m_node,
      /*output_args_override=*/0, m_kernel_extra_inputs.size(), m_name.c_str(),
      "gfx_kernel", [&](const KernelArgMappingInfo &info) -> size_t {
        const size_t fallback = fallback_arg_count_from_func_signature(
            info, m_node ? m_node->get_output_size() : 0);
        return use_msl_stage_manifest_arg_count
                   ? resolve_apple_msl_manifest_arg_count_or_fallback(
                         module, /*is_vulkan_backend=*/false, fallback)
                   : fallback;
      });
  if (m_type == "MaxPool" || m_type == "AvgPool") {
    plan_ctx = build_mlir_kernel_plan(
        module, plan.entry_point(), m_node,
        /*output_args_override=*/1, m_kernel_extra_inputs.size(),
        m_name.c_str(), "gfx_kernel",
        [&](const KernelArgMappingInfo &info) -> size_t {
          const size_t fallback = fallback_arg_count_from_kernel_mapping(
              info, info.output_args, m_kernel_extra_inputs.size());
          return use_msl_stage_manifest_arg_count
                     ? resolve_apple_msl_manifest_arg_count_or_fallback(
                           module, /*is_vulkan_backend=*/false, fallback)
                     : fallback;
        });
  }
  if (is_vulkan_backend() && is_matmul_like() && module) {
    auto matmul_binding_plan =
        annotate_required_backend_custom_kernel_direct_io_binding(
            module, /*is_vulkan_backend=*/true, "MatMul", plan.entry_point(),
            /*tensor_input_count=*/2,
            /*output_count=*/1, m_name);
    apply_kernel_runtime_binding_state(matmul_binding_plan.runtime_binding);
    plan_ctx = build_mlir_kernel_plan(
        module, plan.entry_point(), m_node,
        /*output_args_override=*/0, m_kernel_extra_inputs.size(),
        m_name.c_str(), "gfx_kernel",
        [&](const KernelArgMappingInfo &) -> size_t { return 3; });
    plan_ctx.build_info.plan =
        KernelPlan(module, plan_ctx.build_info.plan.entry_point(),
                   /*arg_count=*/0);
  }
  auto &build_info = plan_ctx.build_info;
  const auto signature = build_info.mapping.signature;
  const size_t scalar_inputs = plan_ctx.scalar_inputs;
  size_t output_args = plan_ctx.output_args;
  const size_t buffer_inputs = plan_ctx.buffer_inputs;
  const size_t kernel_inputs_size = plan_ctx.kernel_inputs_size;
  const size_t node_inputs = plan_ctx.node_inputs;
  const size_t extra_inputs_for_mapping = plan_ctx.extra_inputs_for_mapping;
  if (gfx_log_debug_enabled()) {
    gfx_log_debug("MLIRExec")
        << "Kernel signature: entry=" << build_info.plan.entry_point()
        << " func_inputs=" << signature.inputs
        << " func_results=" << signature.results
        << " scalar_inputs=" << scalar_inputs << " output_args=" << output_args
        << " buffer_inputs=" << buffer_inputs
        << " extra_inputs=" << m_kernel_extra_inputs.size()
        << " extra_inputs_map=" << extra_inputs_for_mapping
        << " kernel_inputs=" << kernel_inputs_size
        << " node_inputs=" << node_inputs;
  }
  if (m_type == "Concat") {
    if (is_vulkan_backend()) {
      module->setAttr("gfx.prefer_parallel",
                      mlir::BoolAttr::get(module.getContext(), false));
      m_force_single_dispatch = true;
    }
  }
  if (is_vulkan_backend() &&
      (m_type == "Interpolate" || m_type == "Transpose")) {
    m_force_single_dispatch = true;
  }
  if (gfx_log_debug_enabled() && is_vulkan_backend() &&
      (m_type == "Convolution" || m_type == "GroupConvolution")) {
    gfx_log_debug("MLIRExec")
        << "Vulkan conv lowering: stage=" << m_name << " type=" << m_type
        << " route=" << conv_route_kind_attr(optimization_plan.conv.kind)
        << " algorithm="
        << conv_algorithm_kind_attr(optimization_plan.conv.algorithm.kind)
        << " lowering=shared_mlir";
  }
  if (m_has_residual_add && module) {
    module->setAttr("gfx.fused_residual_add",
                    mlir::BoolAttr::get(module.getContext(), true));
  }
  if (!is_vulkan_backend() && m_has_residual_add && m_type == "RMS" && module) {
    (void)annotate_required_backend_custom_kernel_binding(
        module, /*is_vulkan_backend=*/false, "RMSResidual", "rms_kernel", {},
        m_name);
  }
  compile_from_plan(plan_ctx, module, "stage");
  if (m_type == "MatMul" && m_node &&
      (!m_node->get_input_partial_shape(0).is_static() ||
       !m_node->get_input_partial_shape(1).is_static() ||
       !m_node->get_output_partial_shape(0).is_static())) {
    // Compile-time MLIR may contain placeholder dimensions for dynamic
    // sequence length. Keep constants/metadata, but force the first
    // execution to specialize the MatMul kernel from concrete tensor shapes.
    m_kernel.reset();
    m_last_input_shape.clear();
  }
  if (m_type == "ShapeOf") {
    if (is_vulkan_backend()) {
      apply_kernel_runtime_binding_state(
          make_stage_direct_kernel_runtime_binding({0}, 1, {1, 1, 1},
                                                   {0, 1, 2}));
    }
  }
  if (m_has_residual_add && m_type == "RMS") {
    if (is_vulkan_backend()) {
      apply_kernel_runtime_binding_state(
          make_stage_direct_kernel_runtime_binding({0, 1, 2}, 3, {1, 1, 1, 1},
                                                   {0, 1, 2, 3}));
    }
  }
  if (module) {
    if (gfx_log_debug_enabled() && !m_kernel_binding.scalar_args.empty()) {
      std::ostringstream oss;
      oss << "Kernel scalar args: ";
      const size_t dump_n =
          std::min<size_t>(m_kernel_binding.scalar_args.size(), 8);
      for (size_t i = 0; i < dump_n; ++i) {
        if (i) {
          oss << ", ";
        }
        oss << m_kernel_binding.scalar_args[i];
      }
      if (m_kernel_binding.scalar_args.size() > dump_n) {
        oss << ", ...";
      }
      gfx_log_debug("MLIRExec") << oss.str();
    }
    if (gfx_log_debug_enabled()) {
      const bool has_kinds = module->hasAttr("gfx.kernel_operand_kinds");
      const bool has_scalars = module->hasAttr("gfx.kernel_scalar_values");
      gfx_log_debug("MLIRExec")
          << "Kernel attrs: operand_kinds=" << (has_kinds ? "yes" : "no")
          << " scalar_values=" << (has_scalars ? "yes" : "no");
      gfx_log_debug("MLIRExec") << "Kernel operand kinds size="
                                << m_kernel_binding.operand_kinds.size();
      if (!m_kernel_binding.operand_arg_indices.empty()) {
        std::ostringstream idxs;
        idxs << "Kernel operand arg indices: ";
        const size_t dump_n =
            std::min<size_t>(m_kernel_binding.operand_arg_indices.size(), 8);
        for (size_t i = 0; i < dump_n; ++i) {
          if (i) {
            idxs << ", ";
          }
          idxs << m_kernel_binding.operand_arg_indices[i];
        }
        if (m_kernel_binding.operand_arg_indices.size() > dump_n) {
          idxs << ", ...";
        }
        gfx_log_debug("MLIRExec") << idxs.str();
      }
      if (auto attr = module->getAttr("gfx.kernel_operand_kinds")) {
        std::string text;
        llvm::raw_string_ostream os(text);
        attr.print(os);
        gfx_log_debug("MLIRExec") << "Kernel operand_kinds attr=" << os.str();
        gfx_log_debug("MLIRExec")
            << "operand_kinds isa ArrayAttr="
            << (llvm::isa<mlir::ArrayAttr>(attr) ? "yes" : "no")
            << " DenseI32ArrayAttr="
            << (llvm::isa<mlir::DenseI32ArrayAttr>(attr) ? "yes" : "no")
            << " DenseIntElementsAttr="
            << (llvm::isa<mlir::DenseIntElementsAttr>(attr) ? "yes" : "no");
      }
    }
  }
}

std::vector<KernelArg> MlirStage::materialize_bound_kernel_args(
    const std::vector<GpuTensor *> &outputs) const {
  OPENVINO_ASSERT(m_kernel, "GFX MLIR: kernel is not compiled for stage ",
                  m_name);
  OPENVINO_ASSERT(m_buffer_manager,
                  "GFX MLIR: buffer manager is not initialized for stage ",
                  m_name);
  const KernelRuntimeBindingState binding =
      resolved_kernel_runtime_binding_state();

  const auto resolve_input_tensor = [&](size_t input_idx) -> GpuTensor * {
    GpuTensor *tensor =
        input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
    if (tensor && tensor->buf.valid()) {
      return tensor;
    }
    if (m_const_buffers && input_idx < m_const_buffers->buffers.size() &&
        input_idx < m_const_buffers->present.size() &&
        m_const_buffers->present[input_idx] &&
        m_const_buffers->buffers[input_idx].buf.valid()) {
      return const_cast<GpuTensor *>(&m_const_buffers->buffers[input_idx]);
    }
    return nullptr;
  };

  const size_t expected_inputs =
      binding.input_arg_count ? binding.input_arg_count : binding.inputs.size();
  const std::vector<GpuTensor> empty_extras;
  const std::vector<GpuTensor> *extras = &m_kernel_extra_inputs;
  if (binding.operand_kinds.empty() &&
      expected_inputs <= binding.inputs.size()) {
    extras = &empty_extras;
  }

  auto bundle = build_kernel_args_from_metadata(
      binding.operand_kinds, binding.operand_arg_indices, binding.scalar_args,
      binding.inputs, binding.input_arg_count, *extras, outputs,
      [&](size_t input_idx) { return resolve_input_tensor(input_idx); },
      m_name.c_str(), nullptr);
  return materialize_kernel_bytes_args(bundle.args, *m_buffer_manager,
                                       m_name.c_str());
}

void MlirStage::prepare_prewarm_kernel_runtime_state(
    const std::vector<GpuTensor *> &outputs) {
  if (!m_node || outputs.empty()) {
    return;
  }

  RuntimeInputResolver runtime_inputs{
      &m_inputs, m_const_buffers ? &m_const_buffers->buffers : nullptr,
      m_const_buffers ? &m_const_buffers->present : nullptr, m_node};

  if (!is_vulkan_backend() && m_type == "Interpolate") {
    const auto interpolate_plan = plan_interpolate_runtime_values(
        runtime_inputs, outputs, *m_node, is_vulkan_backend(), m_name);
    assign_runtime_value_outputs(interpolate_plan.values, outputs);
    m_output_shape = interpolate_plan.values.output_shape;
    m_kernel_extra_inputs.clear();
    auto interpolate_payload = make_interpolate_runtime_param_payload(
        *m_buffer_manager, m_name, interpolate_plan.input_shape,
        interpolate_plan.values.output_shape, interpolate_plan.align_corners,
        interpolate_plan.use_half_pixel, interpolate_plan.nearest_mode);
    m_kernel_extra_inputs = std::move(interpolate_payload.extra_inputs);

    if (!m_kernel || m_last_input_shape.empty() ||
        m_last_input_shape != interpolate_plan.input_shape) {
      auto &ctx = gfx_mlir_context();
      auto module = build_mlir_for_node(m_node, ctx);
      if (module) {
        const auto binding = make_backend_custom_kernel_roles_binding_plan(
            "Interpolate", "interpolate_kernel",
            {GfxKernelBufferRole::TensorInput,
             GfxKernelBufferRole::RuntimeParams,
             GfxKernelBufferRole::TensorOutput});
        OPENVINO_ASSERT(
            binding.valid &&
                annotate_backend_custom_kernel_module_with_binding_plan(
                    module, binding),
            "GFX MLIR: failed to annotate Interpolate runtime-param binding "
            "for stage ",
            m_name);
        apply_source_plan_kernel_runtime_binding_state(binding.runtime_binding);
      }
      auto plan_ctx = build_mlir_kernel_plan(
          module, {}, m_node, outputs.size(), m_kernel_extra_inputs.size(),
          m_name.c_str(), "interpolate_main",
          [&](const KernelArgMappingInfo &info) -> size_t {
            const size_t fallback = fallback_arg_count_from_kernel_mapping(
                info, outputs.size(), m_kernel_extra_inputs.size());
            return resolve_apple_msl_manifest_arg_count_or_fallback(
                module, is_vulkan_backend(), fallback);
          });
      compile_from_plan(plan_ctx, module, "interpolate");
      m_last_input_shape = interpolate_plan.input_shape;
    }
  }
}

void MlirStage::prewarm_runtime_state() {
  if (m_is_view_op) {
    const bool profiling_enabled = m_profiling_enabled;
    m_profiling_enabled = false;
    try {
      execute(nullptr);
    } catch (const std::exception &ex) {
      if (gfx_log_debug_enabled()) {
        gfx_log_debug("MLIR")
            << "View prewarm skipped for " << m_name << ": " << ex.what();
      }
    } catch (...) {
      if (gfx_log_debug_enabled()) {
        gfx_log_debug("MLIR")
            << "View prewarm skipped for " << m_name << ": unknown exception";
      }
    }
    m_profiling_enabled = profiling_enabled;
    return;
  }

  if (!m_kernel) {
    return;
  }

  std::vector<GpuTensor *> outputs;
  if (!m_outputs.empty()) {
    outputs = m_outputs;
  } else if (m_output) {
    outputs.push_back(m_output);
  }
  outputs.erase(std::remove(outputs.begin(), outputs.end(), nullptr),
                outputs.end());
  if (outputs.empty()) {
    return;
  }

  try {
    prepare_prewarm_kernel_runtime_state(outputs);
    const auto bound_args = materialize_bound_kernel_args(outputs);
    if (bound_args.empty()) {
      return;
    }
    m_kernel->prewarm_bindings(bound_args);
  } catch (const std::exception &ex) {
    // Keep runtime-state prewarm best-effort. Stages with custom launch
    // paths continue to use their validated lazy execution path.
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MLIR")
          << "Runtime prewarm skipped for " << m_name << ": " << ex.what();
    }
  } catch (...) {
    // Keep runtime-state prewarm best-effort. Stages with custom launch
    // paths continue to use their validated lazy execution path.
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MLIR")
          << "Runtime prewarm skipped for " << m_name << ": unknown exception";
    }
  }
}

void MlirStage::execute(GpuCommandBufferHandle command_buffer) {
  if (gfx_log_debug_enabled()) {
    gfx_log_debug("MLIRExec")
        << "Execute stage " << m_name << " (" << m_type << ")";
  }
  std::vector<GpuTensor *> outputs = m_outputs;
  if (outputs.empty() && m_output) {
    outputs.push_back(m_output);
  }
  if (outputs.empty()) {
    OPENVINO_THROW("GFX MLIR: output tensor is not bound for stage ", m_name);
  }
  RuntimeInputResolver runtime_inputs{
      &m_inputs, m_const_buffers ? &m_const_buffers->buffers : nullptr,
      m_const_buffers ? &m_const_buffers->present : nullptr, m_node};

  std::optional<KernelRuntimeBindingState> kernel_binding_override;
  std::optional<std::vector<int32_t>> kernel_scalar_args_override;

  auto set_kernel_binding_override = [&](KernelRuntimeBindingState binding) {
    kernel_binding_override = std::move(binding);
  };
  auto set_direct_kernel_binding_override =
      [&](const std::vector<size_t> &kernel_inputs, size_t input_arg_count,
          const std::vector<int32_t> &operand_kinds,
          const std::vector<int32_t> &operand_arg_indices) {
        set_kernel_binding_override(make_stage_direct_kernel_runtime_binding(
            kernel_inputs, input_arg_count, operand_kinds,
            operand_arg_indices));
      };
  auto set_backend_custom_kernel_binding_override =
      [&](std::string_view stage_type, std::string_view entry_point,
          const std::vector<int32_t> &scalar_args = {},
          bool install_scalar_args_override = false) {
        if (install_scalar_args_override) {
          kernel_scalar_args_override = scalar_args;
        }
        set_kernel_binding_override(
            require_stage_backend_custom_kernel_runtime_binding(
                is_vulkan_backend(), stage_type, entry_point, scalar_args,
                m_name));
      };
  auto set_spirv_kernel_binding_override = [&](mlir::ModuleOp module,
                                               KernelRuntimeBindingState
                                                   binding) {
    OPENVINO_ASSERT(
        is_vulkan_backend(),
        "GFX MLIR: SPIR-V execute binding override used on non-Vulkan backend");
    annotate_spirv_kernel_binding_attrs(module, binding);
    set_kernel_binding_override(std::move(binding));
  };

  auto bind_small_i64_const_outputs = [&](std::string_view suffix) -> bool {
    return bind_small_i64_const_stage_outputs(
        m_buffer_manager, outputs, m_small_i64_const_output_cache, m_node,
        static_cast<GfxProfiler *>(m_profiler), m_profiling_enabled, m_name,
        suffix);
  };

  if (m_type == "GfxSDPAWithCausalMask" && m_node) {
    OPENVINO_ASSERT(
        ov::as_type_ptr<const ov::gfx_plugin::op::GfxSDPAWithCausalMask>(
            m_node),
        "GFX MLIR: expected GfxSDPAWithCausalMask node for stage ", m_name);
    ov::Shape q_shape = runtime_inputs.shape(0);
    ov::Shape k_shape = runtime_inputs.shape(1);
    ov::Shape v_shape = runtime_inputs.shape(2);
    ov::Shape mask_shape = runtime_inputs.shape(3);
    ov::Shape pos_shape = runtime_inputs.shape(4);
    OPENVINO_ASSERT(
        q_shape.size() == 4 && k_shape.size() == 4 && v_shape.size() == 4,
        "GFX MLIR: fused causal-mask SDPA expects rank-4 Q/K/V for stage ",
        m_name);
    OPENVINO_ASSERT(mask_shape.size() == 2 && pos_shape.size() == 1,
                    "GFX MLIR: fused causal-mask SDPA expects "
                    "attention_mask[B,K] and cache_positions[Q] for stage ",
                    m_name);
    const bool k_heads_match =
        q_shape[1] == k_shape[1] ||
        (k_shape[1] > 0 && (q_shape[1] % k_shape[1]) == 0);
    const bool v_heads_match =
        q_shape[1] == v_shape[1] ||
        (v_shape[1] > 0 && (q_shape[1] % v_shape[1]) == 0);
    OPENVINO_ASSERT(
        q_shape[0] == k_shape[0] && q_shape[0] == v_shape[0] && k_heads_match &&
            v_heads_match && k_shape[2] == v_shape[2] &&
            q_shape[3] == k_shape[3] && q_shape[0] == mask_shape[0] &&
            q_shape[2] == pos_shape[0],
        "GFX MLIR: incompatible fused causal-mask SDPA shapes for stage ",
        m_name, " q=", q_shape, " k=", k_shape, " v=", v_shape,
        " mask=", mask_shape, " positions=", pos_shape);
    ov::Shape out_shape{q_shape[0], q_shape[1], q_shape[2], v_shape[3]};
    for (auto *out : outputs) {
      if (!out) {
        continue;
      }
      out->shape = out_shape;
      if (out->expected_type == ov::element::dynamic) {
        out->expected_type = m_node->get_output_element_type(0);
      }
    }
    m_output_shape = out_shape;

    float scale = 1.0f / std::sqrt(static_cast<float>(q_shape[3]));
    if (auto scale_const =
            ov::util::get_constant_from_source(m_node->input_value(5))) {
      const auto vals = scale_const->cast_vector<float>();
      if (!vals.empty()) {
        scale = vals[0];
      }
    }
    const auto *k_tensor = runtime_inputs.tensor(1);
    const auto *v_tensor = runtime_inputs.tensor(2);
    const bool k_view_gqa =
        k_tensor && k_tensor->gqa_broadcast_view && k_tensor->gqa_kv_heads > 0;
    const bool v_view_gqa =
        v_tensor && v_tensor->gqa_broadcast_view && v_tensor->gqa_kv_heads > 0;
    const bool k_gqa = k_view_gqa || k_shape[1] != q_shape[1];
    const bool v_gqa = v_view_gqa || v_shape[1] != q_shape[1];
    auto sdpa_plan = make_causal_sdpa_msl_runtime_params_plan(
        q_shape, k_shape, v_shape, mask_shape, scale, k_gqa,
        k_view_gqa ? k_tensor->gqa_kv_heads : k_shape[1], v_gqa,
        v_view_gqa ? v_tensor->gqa_kv_heads : v_shape[1]);
    OPENVINO_ASSERT(
        sdpa_plan.valid(),
        "GFX MLIR: invalid causal SDPA MSL runtime params for stage ", m_name);
    m_kernel_extra_inputs.clear();
    m_kernel_extra_inputs.push_back(make_kernel_i32_param_tensor(
        *m_buffer_manager, m_name, "sdpa_causal_mask_params",
        sdpa_plan.params));
    set_kernel_binding_override(sdpa_plan.binding.runtime_binding);
  }

  if (m_type == "ScaledDotProductAttention" && m_node) {
    auto sdpa =
        ov::as_type_ptr<const ov::op::v13::ScaledDotProductAttention>(m_node);
    OPENVINO_ASSERT(
        sdpa, "GFX MLIR: expected ScaledDotProductAttention node for stage ",
        m_name);
    ov::Shape q_shape = runtime_inputs.shape(0);
    ov::Shape k_shape = runtime_inputs.shape(1);
    ov::Shape v_shape = runtime_inputs.shape(2);
    OPENVINO_ASSERT(q_shape.size() == 4 && k_shape.size() == 4 &&
                        v_shape.size() == 4,
                    "GFX MLIR: SDPA expects rank-4 Q/K/V for stage ", m_name);
    OPENVINO_ASSERT(q_shape[0] == k_shape[0] && q_shape[0] == v_shape[0] &&
                        q_shape[1] == k_shape[1] && q_shape[1] == v_shape[1] &&
                        k_shape[2] == v_shape[2] && q_shape[3] == k_shape[3],
                    "GFX MLIR: incompatible SDPA Q/K/V shapes for stage ",
                    m_name, " q=", q_shape, " k=", k_shape, " v=", v_shape);
    ov::Shape out_shape{q_shape[0], q_shape[1], q_shape[2], v_shape[3]};
    for (auto *out : outputs) {
      if (!out) {
        continue;
      }
      out->shape = out_shape;
      if (out->expected_type == ov::element::dynamic) {
        out->expected_type = m_node->get_output_element_type(0);
      }
    }
    m_output_shape = out_shape;

    if (!m_uses_mpsrt_sdpa_plan) {
      float scale = 1.0f / std::sqrt(static_cast<float>(q_shape[3]));
      if (m_node->get_input_size() >= 5) {
        if (auto scale_const =
                ov::util::get_constant_from_source(m_node->input_value(4))) {
          const auto vals = scale_const->cast_vector<float>();
          if (!vals.empty()) {
            scale = vals[0];
          }
        }
      }
      const auto *k_tensor = runtime_inputs.tensor(1);
      const auto *v_tensor = runtime_inputs.tensor(2);
      const bool k_gqa = k_tensor && k_tensor->gqa_broadcast_view &&
                         k_tensor->gqa_kv_heads > 0;
      const bool v_gqa = v_tensor && v_tensor->gqa_broadcast_view &&
                         v_tensor->gqa_kv_heads > 0;

      ov::Shape mask_shape{1, 1, 1, 1};
      const bool has_mask = m_node->get_input_size() >= 4;
      if (has_mask) {
        mask_shape = runtime_inputs.shape(3);
        OPENVINO_ASSERT(mask_shape.size() == 4,
                        "GFX MLIR: SDPA mask expects rank-4 shape for stage ",
                        m_name);
      }
      auto sdpa_plan = make_sdpa_msl_runtime_params_plan(
          q_shape, k_shape, v_shape, mask_shape, has_mask, scale, k_gqa,
          k_gqa ? k_tensor->gqa_kv_heads : k_shape[1], v_gqa,
          v_gqa ? v_tensor->gqa_kv_heads : v_shape[1]);
      OPENVINO_ASSERT(sdpa_plan.valid(),
                      "GFX MLIR: invalid SDPA MSL runtime params for stage ",
                      m_name);
      m_kernel_extra_inputs.clear();
      m_kernel_extra_inputs.push_back(make_kernel_i32_param_tensor(
          *m_buffer_manager, m_name, "sdpa_params", sdpa_plan.params));
      set_kernel_binding_override(sdpa_plan.binding.runtime_binding);
    }
  }

  if (auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(m_node)) {
    OPENVINO_ASSERT(!outputs.empty(),
                    "GFX MLIR: missing concat outputs for stage ", m_name);
    const auto concat_plan =
        plan_concat_runtime_values(runtime_inputs, *concat, m_name);
    assign_runtime_value_outputs(concat_plan.values, outputs);
    const ov::Shape &out_shape = concat_plan.values.output_shape;
    if (concat_plan.values.has_i64_values &&
        bind_small_i64_const_outputs("concat_i64_const")) {
      return;
    }
    m_output_shape = out_shape;
    GpuTensor *single_live_input = nullptr;
    size_t live_input_count = 0;
    for (size_t input_idx = 0; input_idx < concat_plan.input_shapes.size();
         ++input_idx) {
      const auto &input_shape = concat_plan.input_shapes[input_idx];
      if (ov::shape_size(input_shape) == 0) {
        continue;
      }
      ++live_input_count;
      single_live_input = runtime_inputs.tensor(input_idx);
    }
    if (live_input_count == 1 && single_live_input &&
        single_live_input->shape == out_shape) {
      const auto output_type = m_node->get_output_element_type(0);
      bool all_aliased = true;
      for (auto *out : outputs) {
        if (!out || !alias_tensor_view(*out, *single_live_input, out_shape,
                                       output_type)) {
          all_aliased = false;
          break;
        }
      }
      if (all_aliased) {
        return;
      }
    }
  }

  auto gather_v1 = ov::as_type_ptr<const ov::op::v1::Gather>(m_node);
  auto gather_v7 = ov::as_type_ptr<const ov::op::v7::Gather>(m_node);
  auto gather_v8 = ov::as_type_ptr<const ov::op::v8::Gather>(m_node);
  if (gather_v1 || gather_v7 || gather_v8) {
    const auto gather_plan =
        plan_gather_runtime_values(runtime_inputs, *m_node, m_name);
    assign_runtime_value_outputs(gather_plan.values, outputs);
    const ov::Shape &out_shape = gather_plan.values.output_shape;
    m_output_shape = out_shape;
    if (gather_plan.values.has_i64_values &&
        bind_small_i64_const_outputs("gather_i64_const")) {
      return;
    }

    if (gather_plan.identity_view) {
      GpuTensor *input = runtime_inputs.tensor(0);
      OPENVINO_ASSERT(
          input && input->buf.valid(),
          "GFX MLIR: missing input buffer for Gather identity view ", m_name);
      for (auto *out : outputs) {
        if (!out) {
          continue;
        }
        out->buf = input->buf;
        out->buf.external = true;
        out->buf.owned = false;
        propagate_view_metadata(*out, *input);
      }
      return;
    }

    if (!is_vulkan_backend()) {
      const auto gather_plan =
          plan_gather_runtime_values(runtime_inputs, *m_node, m_name);
      auto gather_payload = make_gather_runtime_param_payload(
          *m_buffer_manager, m_name, gather_plan.data_shape,
          gather_plan.indices_shape,
          static_cast<uint32_t>(gather_plan.axis_norm));
      m_kernel_extra_inputs = std::move(gather_payload.extra_inputs);
      set_backend_custom_kernel_binding_override("Gather", "gather_kernel");
    }
  }

  if (!is_vulkan_backend()) {
    if (ov::as_type_ptr<const ov::op::v1::Transpose>(m_node)) {
      const auto transpose_plan =
          plan_transpose_runtime_values(runtime_inputs, *m_node, m_name);
      assign_runtime_value_outputs(transpose_plan.values, outputs);
      m_output_shape = transpose_plan.values.output_shape;

      if (transpose_plan.linear_view) {
        GpuTensor *input = runtime_inputs.tensor(0);
        OPENVINO_ASSERT(
            input && input->buf.valid(),
            "GFX MLIR: missing input buffer for runtime Transpose view ",
            m_name);
        for (auto *out : outputs) {
          if (!out) {
            continue;
          }
          out->buf = input->buf;
          out->buf.external = true;
          out->buf.owned = false;
        }
        return;
      }

      auto transpose_payload = make_transpose_runtime_param_payload(
          *m_buffer_manager, m_name, transpose_plan.input_shape,
          transpose_plan.values.output_shape, transpose_plan.permutation);
      m_kernel_extra_inputs = std::move(transpose_payload.extra_inputs);
      if (is_vulkan_backend()) {
        set_direct_kernel_binding_override({0}, 1, {1, 1, 1, 1, 1, 1, 1},
                                           {0, 1, 2, 3, 4, 5, 6});
      }
    }
  }

  if (m_type == "ShapeOf") {
    const auto shapeof_values =
        plan_shapeof_runtime_values(runtime_inputs, m_node.get(), m_name);
    assign_runtime_value_outputs(shapeof_values, outputs);
    if (bind_small_i64_const_outputs("shapeof_i64_const")) {
      return;
    }

    auto shapeof_payload = make_shapeof_runtime_param_payload(
        *m_buffer_manager, m_name, runtime_inputs.shape(0),
        shapeof_values.output_type);
    const auto shapeof_scalars = shapeof_payload.scalar_args;
    m_kernel_extra_inputs = std::move(shapeof_payload.extra_inputs);
    set_backend_custom_kernel_binding_override("ShapeOf", "shapeof_kernel",
                                               shapeof_scalars);
  } else if (auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(m_node)) {
    ov::Shape a_shape;
    ov::Shape b_shape;
    const bool a_known = runtime_inputs.shape_known(0, a_shape);
    bool b_known = runtime_inputs.shape_known(1, b_shape);
    if (a_known && b_known) {
      OPENVINO_ASSERT(a_shape.size() >= 2 && b_shape.size() >= 2,
                      "GFX MLIR: MatMul ranks must be at least 2 for stage ",
                      m_name);
      const bool ta = matmul->get_transpose_a();
      const bool tb = matmul->get_transpose_b();
      const size_t a_rank = a_shape.size();
      const size_t b_rank = b_shape.size();
      const size_t M = ta ? a_shape[a_rank - 1] : a_shape[a_rank - 2];
      const size_t K = ta ? a_shape[a_rank - 2] : a_shape[a_rank - 1];
      const size_t Kb = tb ? b_shape[b_rank - 1] : b_shape[b_rank - 2];
      const size_t N = tb ? b_shape[b_rank - 2] : b_shape[b_rank - 1];
      OPENVINO_ASSERT(K == Kb, "GFX MLIR: MatMul K mismatch for stage ", m_name,
                      " (", K, " vs ", Kb, ")");
      const ov::Shape batch_prefix =
          broadcast_batch_prefix(a_shape, b_shape, "GFX MLIR: MatMul");
      const size_t batch = static_cast<size_t>(
          shape_product(batch_prefix, 0, batch_prefix.size()));
      const size_t batch_a =
          a_rank > 2
              ? static_cast<size_t>(shape_product(a_shape, 0, a_rank - 2))
              : 1;
      const size_t batch_b =
          b_rank > 2
              ? static_cast<size_t>(shape_product(b_shape, 0, b_rank - 2))
              : 1;
      OPENVINO_ASSERT(batch_a == batch_b || batch_a == 1 || batch_b == 1,
                      "GFX MLIR: MatMul mixed batch-prefix broadcast is not "
                      "supported for stage ",
                      m_name);

      ov::Shape out_shape = batch_prefix;
      out_shape.push_back(M);
      out_shape.push_back(N);
      for (auto *out : outputs) {
        if (!out) {
          continue;
        }
        out->shape = out_shape;
        if (out->expected_type == ov::element::dynamic) {
          out->expected_type = m_node->get_output_element_type(0);
        }
      }
      m_output_shape = out_shape;

      ov::Shape matmul_key = a_shape;
      matmul_key.push_back(0);
      matmul_key.insert(matmul_key.end(), b_shape.begin(), b_shape.end());
      if (!m_is_compressed_matmul &&
          (m_last_input_shape != matmul_key || !m_kernel)) {
        auto runtime_input_type = [&](size_t input_idx,
                                      const ov::element::Type &fallback) {
          if (auto *tensor = runtime_inputs.tensor(input_idx)) {
            if (tensor->expected_type != ov::element::dynamic) {
              return tensor->expected_type;
            }
            if (tensor->buf.type != ov::element::dynamic) {
              return tensor->buf.type;
            }
          }
          return fallback;
        };
        MatMulCodegenDesc desc{};
        desc.element_type = matmul->get_output_element_type(0);
        desc.input_a_type =
            runtime_input_type(0, matmul->get_input_element_type(0));
        desc.input_b_type =
            runtime_input_type(1, matmul->get_input_element_type(1));
        desc.output_type = matmul->get_output_element_type(0);
        desc.a_transpose = ta;
        desc.b_transpose = tb;
        desc.b_is_nk_layout = tb;
        desc.M = static_cast<int64_t>(M);
        desc.N = static_cast<int64_t>(N);
        desc.K = static_cast<int64_t>(K);
        desc.batch = static_cast<int64_t>(batch);
        desc.batch_a = static_cast<int64_t>(batch_a);
        desc.batch_b = static_cast<int64_t>(batch_b);
        desc.has_activation = m_has_activation;
        desc.activation = m_activation;
        desc.alpha = m_activation_alpha;
        m_matmul_reduction_threads =
            gfx_matmul_parallel_reduction_threads(desc);

        std::string log;
        if (!is_vulkan_backend()) {
          auto source_plan = make_apple_metal_runtime_matmul_kernel_source_plan(
              gfx_mlir_context(), m_buffer_manager, m_node, desc, a_shape,
              b_shape, m_name);
          const KernelSource &src = source_plan.source;
          OPENVINO_ASSERT(
              src.msl_generator || !src.msl_source.empty() || src.module,
              "GFX MLIR: failed to build runtime MatMul source for stage ",
              m_name);
          try {
            m_kernel = compile_kernel(src, &log);
          } catch (const std::exception &e) {
            OPENVINO_THROW("GFX MLIR: failed to compile MatMul stage ", m_name,
                           ": ", e.what());
          }
          OPENVINO_ASSERT(m_kernel, "GFX MLIR: failed to compile MatMul stage ",
                          m_name, ": ", log);
          if (source_plan.has_runtime_binding) {
            apply_source_plan_kernel_runtime_binding_state(
                source_plan.runtime_binding);
          } else {
            auto runtime_meta = extract_kernel_runtime_metadata(
                src.module, src.signature.output_arg_count,
                m_node->get_input_size(), src.entry_point,
                GfxKernelBackendDomain::AppleMsl);
            apply_kernel_metadata(runtime_meta, /*scalar_inputs=*/0);
          }
        }
        if (is_vulkan_backend()) {
          set_direct_kernel_binding_override({0, 1}, 2, {1, 1, 1}, {0, 1, 2});
        }
        m_kernel_extra_inputs.clear();
        m_parallel_cfg = ParallelDispatchConfig{};
        m_force_single_dispatch = false;
        m_last_input_shape = std::move(matmul_key);
      }
    }
  } else if (ov::as_type_ptr<const ov::op::v1::Select>(m_node)) {
    const auto select_plan =
        plan_select_runtime_values(runtime_inputs, *m_node, m_name);
    if (select_plan.valid()) {
      assign_runtime_value_outputs(select_plan.values, outputs);
      m_output_shape = select_plan.values.output_shape;

      auto select_payload = make_select_runtime_param_payload(
          *m_buffer_manager, m_name, select_plan.condition_shape,
          select_plan.true_shape, select_plan.false_shape,
          select_plan.values.output_shape);
      const auto select_scalars = select_payload.scalar_args;
      m_kernel_extra_inputs = std::move(select_payload.extra_inputs);
      set_backend_custom_kernel_binding_override("Select", "select_kernel",
                                                 select_scalars);
    }
  } else if (auto scatter =
                 ov::as_type_ptr<const ov::op::v3::ScatterUpdate>(m_node)) {
    const auto scatter_plan =
        plan_scatter_update_runtime_values(runtime_inputs, *scatter, m_name);
    if (scatter_plan.valid()) {
      assign_runtime_value_outputs(scatter_plan.values, outputs);
      m_output_shape = scatter_plan.values.output_shape;

      auto scatter_update_payload = make_scatter_update_runtime_param_payload(
          *m_buffer_manager, m_name, scatter_plan.values.output_shape,
          scatter_plan.indices_shape, scatter_plan.updates_shape,
          static_cast<uint32_t>(scatter_plan.axis_norm));
      m_kernel_extra_inputs = std::move(scatter_update_payload.extra_inputs);
      if (is_vulkan_backend()) {
        set_direct_kernel_binding_override({0, 1, 2}, 4, {1, 1, 1, 1, 1},
                                           {0, 1, 2, 4, 3});
      }
    }
  } else if (m_type == "Slice" || m_type == "StridedSlice") {
    const auto slice_plan = plan_slice_runtime_values(
        runtime_inputs, outputs, is_vulkan_backend(), m_name);
    assign_runtime_value_outputs(slice_plan.values, outputs);
    m_output_shape = slice_plan.values.output_shape;

    if (slice_plan.linear_view) {
      GpuTensor *input = runtime_inputs.tensor(0);
      OPENVINO_ASSERT(input && input->buf.valid(),
                      "GFX MLIR: missing input buffer for runtime Slice view ",
                      m_name);
      for (auto *out : outputs) {
        if (!out) {
          continue;
        }
        out->buf = input->buf;
        out->buf.external = true;
        out->buf.owned = false;
      }
      m_output_shape = slice_plan.values.output_shape;
      return;
    }

    m_kernel_extra_inputs.clear();
    if (slice_plan.use_runtime_args) {
      auto slice_payload = make_slice_runtime_param_payload(
          *m_buffer_manager, m_name, slice_plan.input_shape,
          slice_plan.values.output_shape, slice_plan.starts_full,
          slice_plan.steps_full);
      m_kernel_extra_inputs = std::move(slice_payload.extra_inputs);
    }

    const bool needs_slice_kernel_compile =
        !m_kernel || m_last_input_shape.empty() ||
        (!slice_plan.use_runtime_args &&
         m_last_input_shape != slice_plan.input_shape);
    if (m_node && needs_slice_kernel_compile) {
      if (!slice_plan.use_runtime_args) {
        if (gfx_log_debug_enabled() && !m_inputs.empty() && m_inputs.front() &&
            !outputs.empty() && outputs.front()) {
          gfx_log_debug("MLIRExec")
              << "Slice types in_expected=" << m_inputs.front()->expected_type
              << " in_buf=" << m_inputs.front()->buf.type
              << " out_expected=" << outputs.front()->expected_type
              << " out_buf=" << outputs.front()->buf.type;
          std::ostringstream buf_info;
          buf_info << "Slice buffers in_handle=" << m_inputs.front()->buf.buffer
                   << " out_handle=" << outputs.front()->buf.buffer
                   << " in_size=" << m_inputs.front()->buf.size
                   << " out_size=" << outputs.front()->buf.size;
          gfx_log_debug("MLIRExec") << buf_info.str();
        }
      }
      auto &ctx = gfx_mlir_context();
      auto module = build_mlir_for_node(m_node, ctx);
      if (module) {
        if (!is_vulkan_backend() && slice_plan.use_runtime_args) {
          (void)annotate_required_backend_custom_kernel_binding(
              module, /*is_vulkan_backend=*/false, "Slice", "slice_kernel", {},
              m_name);
        } else if (is_vulkan_backend()) {
          set_spirv_kernel_binding_override(
              module, make_stage_direct_kernel_runtime_binding(
                          {0}, 1,
                          slice_plan.use_runtime_args
                              ? std::vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1}
                              : std::vector<int32_t>{1, 1},
                          slice_plan.use_runtime_args
                              ? std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}
                              : std::vector<int32_t>{0, 1}));
        }
      }
      auto plan_ctx = build_mlir_kernel_plan(
          module, {}, m_node, outputs.size(), m_kernel_extra_inputs.size(),
          m_name.c_str(), "slice_main",
          [&](const KernelArgMappingInfo &info) -> size_t {
            const size_t fallback = fallback_arg_count_from_kernel_mapping(
                info, outputs.size(), m_kernel_extra_inputs.size());
            return resolve_apple_msl_manifest_arg_count_or_fallback(
                module, is_vulkan_backend(), fallback);
          });
      compile_from_plan(plan_ctx, module, "slice");
      if (is_vulkan_backend()) {
        set_direct_kernel_binding_override(
            {0}, 1,
            slice_plan.use_runtime_args
                ? std::vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1}
                : std::vector<int32_t>{1, 1},
            slice_plan.use_runtime_args
                ? std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}
                : std::vector<int32_t>{0, 1});
      }
      m_last_input_shape = slice_plan.input_shape;
    }
  } else if (m_type == "Interpolate") {
    const auto interpolate_plan = plan_interpolate_runtime_values(
        runtime_inputs, outputs, *m_node, is_vulkan_backend(), m_name);
    assign_runtime_value_outputs(interpolate_plan.values, outputs);
    m_output_shape = interpolate_plan.values.output_shape;
    m_kernel_extra_inputs.clear();
    const bool use_interpolate_runtime_params =
        !is_vulkan_backend() || interpolate_plan.use_runtime_params;
    if (use_interpolate_runtime_params) {
      auto interpolate_payload = make_interpolate_runtime_param_payload(
          *m_buffer_manager, m_name, interpolate_plan.input_shape,
          interpolate_plan.values.output_shape, interpolate_plan.align_corners,
          interpolate_plan.use_half_pixel, interpolate_plan.nearest_mode);
      m_kernel_extra_inputs = std::move(interpolate_payload.extra_inputs);
    }
    if (m_node && (!m_kernel || m_last_input_shape.empty() ||
                   m_last_input_shape != interpolate_plan.input_shape)) {
      auto &ctx = gfx_mlir_context();
      auto module = build_mlir_for_node(m_node, ctx);
      if (module) {
        if (!is_vulkan_backend() && use_interpolate_runtime_params) {
          const auto binding = make_backend_custom_kernel_roles_binding_plan(
              "Interpolate", "interpolate_kernel",
              {GfxKernelBufferRole::TensorInput,
               GfxKernelBufferRole::RuntimeParams,
               GfxKernelBufferRole::TensorOutput});
          OPENVINO_ASSERT(
              binding.valid &&
                  annotate_backend_custom_kernel_module_with_binding_plan(
                      module, binding),
              "GFX MLIR: failed to annotate Interpolate runtime-param "
              "binding for stage ",
              m_name);
          apply_source_plan_kernel_runtime_binding_state(
              binding.runtime_binding);
        } else if (is_vulkan_backend()) {
          set_spirv_kernel_binding_override(
              module, make_stage_direct_kernel_runtime_binding(
                          {0}, 1,
                          interpolate_plan.use_runtime_params
                              ? std::vector<int32_t>{1, 1, 1}
                              : std::vector<int32_t>{1, 1},
                          interpolate_plan.use_runtime_params
                              ? std::vector<int32_t>{0, 1, 2}
                              : std::vector<int32_t>{0, 1}));
        }
      }
      auto plan_ctx = build_mlir_kernel_plan(
          module, {}, m_node, outputs.size(), m_kernel_extra_inputs.size(),
          m_name.c_str(), "interpolate_main",
          [&](const KernelArgMappingInfo &info) -> size_t {
            const size_t fallback = fallback_arg_count_from_kernel_mapping(
                info, outputs.size(), m_kernel_extra_inputs.size());
            return resolve_apple_msl_manifest_arg_count_or_fallback(
                module, is_vulkan_backend(), fallback);
          });
      compile_from_plan(plan_ctx, module, "interpolate");
      if (is_vulkan_backend()) {
        set_direct_kernel_binding_override(
            {0}, 1,
            interpolate_plan.use_runtime_params ? std::vector<int32_t>{1, 1, 1}
                                                : std::vector<int32_t>{1, 1},
            interpolate_plan.use_runtime_params ? std::vector<int32_t>{0, 1, 2}
                                                : std::vector<int32_t>{0, 1});
      }
      m_last_input_shape = interpolate_plan.input_shape;
    }
  } else if (m_type == "Softmax" || m_type == "LogSoftmax") {
    const auto softmax_plan =
        plan_softmax_runtime_values(runtime_inputs, *m_node, m_name);
    assign_runtime_value_outputs(softmax_plan.values, outputs);
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MLIRSoftmax")
          << "shape_rank=" << softmax_plan.values.output_shape.size()
          << " axis=" << softmax_plan.axis
          << " rows*inner=" << softmax_plan.rows << " tiled=0";
    }
    auto softmax_payload = make_softmax_runtime_param_payload(
        *m_buffer_manager, m_name, softmax_plan.rows, softmax_plan.axis_len,
        softmax_plan.inner);
    m_kernel_extra_inputs = std::move(softmax_payload.extra_inputs);
    if (m_node && (!m_kernel || m_last_input_shape.empty())) {
      auto &ctx = gfx_mlir_context();
      auto module = softmax_plan.log_softmax
                        ? build_mlir_logsoftmax_from_node(
                              m_node, ctx, softmax_plan.values.output_shape)
                        : build_mlir_softmax_from_node(
                              m_node, ctx, softmax_plan.values.output_shape);
      if (module) {
        module->setAttr("gfx.prefer_parallel",
                        mlir::BoolAttr::get(module.getContext(), false));
        if (!is_vulkan_backend()) {
          (void)annotate_required_backend_custom_kernel_binding(
              module, /*is_vulkan_backend=*/false, m_type, "softmax_kernel", {},
              m_name);
        } else {
          // Vulkan keeps a straightforward 0,1,2 mapping to avoid
          // backend-specific descriptor ordering issues.
          set_spirv_kernel_binding_override(
              module, make_stage_direct_kernel_runtime_binding(
                          {0}, 1, {1, 1, 1}, {0, 1, 2}));
        }
      }
      auto plan_ctx = build_mlir_kernel_plan(
          module, {}, m_node, outputs.size(), m_kernel_extra_inputs.size(),
          m_name.c_str(), "softmax_main",
          [&](const KernelArgMappingInfo &info) -> size_t {
            const size_t fallback = fallback_arg_count_from_kernel_mapping(
                info, outputs.size(), /*extra_inputs=*/0);
            return resolve_apple_msl_manifest_arg_count_or_fallback(
                module, is_vulkan_backend(), fallback);
          });
      compile_from_plan(plan_ctx, module, "softmax");
      if (is_vulkan_backend()) {
        set_direct_kernel_binding_override({0}, 1, {1, 1, 1}, {0, 1, 2});
      }
      m_last_input_shape = softmax_plan.values.output_shape;
    }
  } else if (m_type == "Split" || m_type == "VariadicSplit") {
    ov::Shape stage_input_shape = runtime_inputs.shape(0);
    if (stage_input_shape.empty()) {
      OPENVINO_THROW("GFX MLIR: Split input shape is unknown for stage ",
                     m_name);
    }
    ov::Shape logical_input_shape = stage_input_shape;
    if (m_node && m_node->get_input_partial_shape(0).is_static()) {
      logical_input_shape = m_node->get_input_shape(0);
    }
    const auto plan = plan_split_runtime_values(
        m_node.get(), logical_input_shape, outputs.size(), m_name);
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (!outputs[i]) {
        continue;
      }
      ov::Shape out_shape = logical_input_shape;
      out_shape[static_cast<size_t>(plan.axis_norm)] = plan.split_sizes[i];
      outputs[i]->shape = out_shape;
    }
    m_output_shape = logical_input_shape;
    const bool has_input_transform = has_absorbed_input_transpose();
    const bool needs_split_compile =
        (m_last_input_shape != stage_input_shape || !m_kernel);
    if (needs_split_compile) {
      auto &ctx = gfx_mlir_context();
      auto module = build_mlir_split_from_node(m_node, ctx, stage_input_shape,
                                               input_transform(0));
      if (!has_input_transform && m_node) {
        const ov::element::Type out_et =
            m_node ? m_node->get_output_element_type(0)
                   : (outputs.empty() ? ov::element::f32
                                      : outputs.front()->expected_type);
        auto split_source_plan = make_direct_split_msl_kernel_source_plan(
            m_type, out_et, logical_input_shape, plan.split_sizes,
            plan.axis_len, plan.inner_stride, module);
        compile_generated_msl_source_plan(split_source_plan, "direct Split");
      } else {
        auto plan_ctx = build_mlir_kernel_plan(
            module, "split_main", m_node, outputs.size(),
            /*extra_inputs=*/0, m_name.c_str(), "split_main",
            [&](const KernelArgMappingInfo &info) -> size_t {
              const auto sig = info.signature;
              return sig.total()
                         ? sig.total()
                         : (info.mapping.kernel_inputs.size() + outputs.size());
            });
        compile_from_plan(plan_ctx, module, "split");
      }
      m_last_input_shape = stage_input_shape;
    }
  } else {
    for (size_t i = 0; i < outputs.size(); ++i) {
      runtime_inputs.ensure_output_shape(i, outputs[i]);
    }
  }

  if (m_node && is_conv_like() && !outputs.empty() &&
      !m_kernel_binding_owned_by_source_plan) {
    ov::Shape input_shape;
    if (runtime_inputs.shape_known(0, input_shape) && input_shape.size() == 4) {
      const ov::Shape output_shape = !outputs.front()->shape.empty()
                                         ? outputs.front()->shape
                                         : m_node->get_output_shape(0);
      if (refresh_conv2d_kernel_extra_inputs(
              input_shape, output_shape, m_node->get_output_element_type(0))) {
        if (!is_vulkan_backend()) {
          set_backend_custom_kernel_binding_override(m_type, "conv2d_kernel");
        }
      }
    } else if (m_type == "Convolution" && input_shape.size() == 5) {
      const ov::Shape output_shape = !outputs.front()->shape.empty()
                                         ? outputs.front()->shape
                                         : m_node->get_output_shape(0);
      if (refresh_conv3d_kernel_extra_inputs(input_shape, output_shape)) {
        set_backend_custom_kernel_binding_override("Convolution",
                                                   "conv3d_kernel");
      }
    }
  }

  auto is_unary_eltwise_stage = [&]() {
    return m_type == "Relu" || m_type == "Sigmoid" || m_type == "Tanh" ||
           m_type == "Elu" || m_type == "Gelu" || m_type == "Swish" ||
           m_type == "HSwish" || m_type == "HSigmoid" || m_type == "SoftPlus" ||
           m_type == "Mish" || m_type == "SoftSign" || m_type == "Abs" ||
           m_type == "Sign" || m_type == "Clamp" || m_type == "LogicalNot" ||
           m_type == "Exp" ||
           m_type == "Log" || m_type == "Sqrt" || m_type == "Floor" ||
           m_type == "Ceiling" || m_type == "Negative" || m_type == "Sin" ||
           m_type == "Cos" || m_type == "Tan" || m_type == "Erf" ||
           m_type == "Asin" || m_type == "Acos" || m_type == "Atan" ||
           m_type == "Asinh" || m_type == "Acosh" || m_type == "Atanh" ||
           m_type == "Sinh" || m_type == "Cosh" || m_type == "Round";
  };

  if (m_node && is_unary_eltwise_stage() && m_node->get_input_size() >= 1 &&
      !outputs.empty()) {
    ov::Shape input_shape;
    if (runtime_inputs.shape_known(0, input_shape)) {
      for (auto *out : outputs) {
        if (!out) {
          continue;
        }
        out->shape = input_shape;
        if (out->expected_type == ov::element::dynamic) {
          out->expected_type = m_node->get_output_element_type(0);
        }
      }
      m_output_shape = input_shape;
      const std::vector<int32_t> unary_scalars = {
          static_cast<int32_t>(ov::shape_size(input_shape))};
      set_backend_custom_kernel_binding_override(m_type, "unary_kernel",
                                                 unary_scalars, true);
    }
  }

  auto is_binary_eltwise_stage = [&]() {
    return m_type == "Add" || m_type == "Subtract" || m_type == "Multiply" ||
           m_type == "Divide" || m_type == "Power" || m_type == "Mod" ||
           m_type == "FloorMod" || m_type == "Minimum" || m_type == "Maximum" ||
           m_type == "Equal" || m_type == "NotEqual" || m_type == "Less" ||
           m_type == "Greater" || m_type == "LessEqual" ||
           m_type == "GreaterEqual" || m_type == "LogicalAnd" ||
           m_type == "LogicalOr" || m_type == "LogicalXor" ||
           m_type == "SquaredDifference" || m_type == "PRelu";
  };
  if (m_node && is_binary_eltwise_stage() && m_node->get_input_size() >= 2 &&
      !outputs.empty()) {
    ov::Shape lhs_shape;
    ov::Shape rhs_shape;
    const bool lhs_known = runtime_inputs.shape_known(0, lhs_shape);
    const bool rhs_known = runtime_inputs.shape_known(1, rhs_shape);
    if (lhs_known && rhs_known) {
      const ov::Shape out_shape =
          compute_binary_broadcast_shape(lhs_shape, rhs_shape, m_name);
      for (auto *out : outputs) {
        if (!out) {
          continue;
        }
        out->shape = out_shape;
        if (out->expected_type == ov::element::dynamic) {
          out->expected_type = m_node->get_output_element_type(0);
        }
      }
      m_output_shape = out_shape;
      if (m_type == "Add") {
        auto lhs_values = runtime_inputs.i64_values(0);
        auto rhs_values = runtime_inputs.i64_values(1);
        if (lhs_values && rhs_values) {
          const size_t out_count = ov::shape_size(out_shape);
          const size_t lhs_count = lhs_values->size();
          const size_t rhs_count = rhs_values->size();
          if ((lhs_count == out_count || lhs_count == 1) &&
              (rhs_count == out_count || rhs_count == 1)) {
            std::vector<int64_t> values(out_count, 0);
            for (size_t vi = 0; vi < out_count; ++vi) {
              values[vi] = (*lhs_values)[lhs_count == 1 ? 0 : vi] +
                           (*rhs_values)[rhs_count == 1 ? 0 : vi];
            }
            for (auto *out : outputs) {
              assign_i64_values(out, values, out_shape);
            }
            if (bind_small_i64_const_outputs("add_i64_const")) {
              return;
            }
          }
        }
      }

      ov::Shape meta_shape = out_shape.empty() ? ov::Shape{1} : out_shape;
      auto stride0 = compute_broadcast_element_strides(lhs_shape, meta_shape);
      auto stride1 = compute_broadcast_element_strides(rhs_shape, meta_shape);
      auto broadcast_payload = make_binary_broadcast_runtime_param_payload(
          *m_buffer_manager, m_name, out_shape, std::move(stride0),
          std::move(stride1));
      m_kernel_extra_inputs = std::move(broadcast_payload.extra_inputs);
      kernel_scalar_args_override = std::move(broadcast_payload.scalar_args);
    }
  }

  if (m_is_view_op && m_type == "ReadValue" && !outputs.empty() &&
      !m_inputs.empty() && m_inputs[0]) {
    outputs.front()->shape = m_inputs[0]->shape;
    if (outputs.front()->expected_type == ov::element::dynamic) {
      outputs.front()->expected_type =
          m_inputs[0]->expected_type == ov::element::dynamic
              ? m_inputs[0]->buf.type
              : m_inputs[0]->expected_type;
    }
  }

  if (m_is_view_op && m_type == "Reshape" && !outputs.empty() &&
      !m_inputs.empty() && m_inputs[0]) {
    const auto reshape_values =
        plan_reshape_runtime_values(runtime_inputs, *m_node, m_name);
    assign_runtime_value_outputs(reshape_values, outputs);
    if (bind_small_i64_const_outputs("reshape_i64_const")) {
      return;
    }
  }

  if (m_is_view_op && (m_type == "Squeeze" || m_type == "Unsqueeze") &&
      !outputs.empty() && !m_inputs.empty() && m_inputs[0]) {
    const auto shape_view_values =
        plan_squeeze_unsqueeze_runtime_values(runtime_inputs, *m_node, m_name);
    assign_runtime_value_outputs(shape_view_values, outputs);
    if (bind_small_i64_const_outputs("shape_view_i64_const")) {
      return;
    }
  }

  if (auto reduce_info = get_runtime_reduce_info(m_node)) {
    const auto reduce_plan = plan_reduce_runtime_values(
        runtime_inputs, m_node.get(), m_type, *reduce_info, m_name);
    assign_runtime_value_outputs(reduce_plan.values, outputs);
    if (bind_small_i64_const_outputs("reduce_i64_const")) {
      return;
    }
    m_output_shape = reduce_plan.values.output_shape;
    auto reduce_payload = make_reduce_runtime_param_payload(
        *m_buffer_manager, m_name, reduce_plan.input_shape, reduce_info->axes,
        reduce_info->keep_dims, reduce_plan.values.output_shape);
    const auto reduce_scalars = reduce_payload.scalar_args;
    m_kernel_extra_inputs = std::move(reduce_payload.extra_inputs);
    set_backend_custom_kernel_binding_override(m_type, "reduce_kernel",
                                               reduce_scalars);
  }

  if (ov::as_type_ptr<const ov::op::v1::Broadcast>(m_node) ||
      ov::as_type_ptr<const ov::op::v3::Broadcast>(m_node)) {
    ov::Shape in_shape = runtime_inputs.shape(0);
    const auto broadcast_values = plan_broadcast_runtime_values(
        runtime_inputs, *m_node, in_shape, m_name);
    assign_runtime_value_outputs(broadcast_values, outputs);
    const ov::Shape &out_shape = broadcast_values.output_shape;
    m_output_shape = out_shape;
    if (bind_small_i64_const_outputs("broadcast_i64_const")) {
      return;
    }
    if (try_alias_same_shape_unary_view(runtime_inputs.tensor(0), outputs,
                                        out_shape,
                                        m_node->get_output_element_type(0))) {
      return;
    }
    if (try_alias_gqa_broadcast_view(m_node, runtime_inputs.tensor(0),
                                     outputs)) {
      return;
    }
    auto broadcast_payload = make_broadcast_runtime_param_payload(
        *m_buffer_manager, m_name, in_shape, out_shape);
    const auto broadcast_scalars = broadcast_payload.scalar_args;
    m_kernel_extra_inputs = std::move(broadcast_payload.extra_inputs);
    const bool has_target_shape_input =
        m_node && m_node->get_input_size() > 1 &&
        !ov::as_type_ptr<const ov::op::v0::Constant>(
            m_node->get_input_node_shared_ptr(1));
    if (!is_vulkan_backend() && has_target_shape_input) {
      auto binding_plan = make_backend_custom_kernel_roles_binding_plan(
          "Broadcast", "broadcast_kernel",
          {GfxKernelBufferRole::TensorInput, GfxKernelBufferRole::TensorInput,
           GfxKernelBufferRole::TensorOutput,
           GfxKernelBufferRole::ScalarParam,
           GfxKernelBufferRole::ScalarParam,
           GfxKernelBufferRole::ScalarParam,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams,
           GfxKernelBufferRole::RuntimeParams});
      OPENVINO_ASSERT(binding_plan.valid &&
                          binding_plan.scalar_arg_count ==
                              broadcast_scalars.size(),
                      "GFX MLIR: Broadcast dynamic target-shape binding is "
                      "invalid for stage ",
                      m_name);
      binding_plan.runtime_binding.scalar_args = broadcast_scalars;
      set_kernel_binding_override(std::move(binding_plan.runtime_binding));
    } else {
      set_backend_custom_kernel_binding_override("Broadcast", "broadcast_kernel",
                                                 broadcast_scalars);
    }
  }

  if (m_type == "Convert") {
    const auto convert_values =
        plan_convert_runtime_values(runtime_inputs, m_node.get(), m_name);
    assign_runtime_value_outputs(convert_values, outputs);
    m_output_shape = convert_values.output_shape;
    if (bind_small_i64_const_outputs("convert_i64_const")) {
      return;
    }
    const std::vector<int32_t> convert_scalars = {
        static_cast<int32_t>(ov::shape_size(convert_values.output_shape))};
    set_backend_custom_kernel_binding_override("Convert", "convert_kernel",
                                               convert_scalars);
  }

  if (m_type == "Range") {
    const auto range_values =
        plan_range_runtime_values(runtime_inputs, m_node.get(), m_name);
    assign_runtime_value_outputs(range_values, outputs);
    m_output_shape = range_values.output_shape;
    if (bind_small_i64_const_outputs("range_i64_const")) {
      return;
    }
    const std::vector<int32_t> range_scalars = {
        static_cast<int32_t>(ov::shape_size(range_values.output_shape))};
    set_backend_custom_kernel_binding_override("Range", "range_kernel",
                                               range_scalars);
  }

  if (m_type == "RMS" && m_node && !outputs.empty()) {
    const auto rms_values =
        plan_shape_preserving_runtime_values(runtime_inputs, *m_node, m_name);
    assign_runtime_value_outputs(rms_values, outputs);
    m_output_shape = rms_values.output_shape;
  }

  if (m_type == "RoPE" && m_node && !outputs.empty()) {
    const auto rope_values =
        plan_shape_preserving_runtime_values(runtime_inputs, *m_node, m_name);
    assign_runtime_value_outputs(rope_values, outputs);
    m_output_shape = rope_values.output_shape;
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    GpuTensor *out = outputs[i];
    if (!out) {
      continue;
    }
    bool output_shape_known = !out->shape.empty();
    if (!output_shape_known && m_node && i < m_node->get_output_size() &&
        m_node->get_output_partial_shape(i).is_static()) {
      out->shape = m_node->get_output_shape(i);
      output_shape_known = true;
    }
    if (!output_shape_known) {
      OPENVINO_THROW("GFX MLIR: output shape is not known for stage ", m_name);
    }
    ov::element::Type out_type = out->expected_type;
    if (out_type == ov::element::dynamic && m_node) {
      out_type = m_node->get_output_element_type(i);
    }
    const auto elem_size = out_type == ov::element::dynamic
                               ? out->buf.type.size()
                               : out_type.size();
    const size_t out_bytes = ov::shape_size(out->shape) * elem_size;
    if (m_is_view_op) {
      out->expected_type = out_type;
      continue;
    }
    auto allocate_output_buffer = [&](size_t bytes) {
      OPENVINO_ASSERT(m_buffer_manager,
                      "GFX MLIR: buffer manager is not set for stage ", m_name);
      GpuBufferDesc desc{};
      desc.bytes = bytes;
      desc.type = out_type;
      desc.usage =
          out->prefer_private ? BufferUsage::Intermediate : BufferUsage::IO;
      desc.cpu_read = !out->prefer_private;
      desc.cpu_write = !out->prefer_private;
      desc.prefer_device_local = out->prefer_private;
      desc.label = m_name.c_str();
      return m_buffer_manager->allocate_temp(desc);
    };
    auto output_aliases_input = [&]() {
      if (!out->buf.valid()) {
        return false;
      }
      for (size_t input_idx = 0; input_idx < m_inputs.size(); ++input_idx) {
        GpuTensor *src = runtime_inputs.tensor(input_idx);
        if (src && same_gpu_allocation(out->buf, src->buf)) {
          return true;
        }
      }
      return false;
    };
    if (!out->buf.valid()) {
      out->buf = allocate_output_buffer(out_bytes);
      OPENVINO_ASSERT(out->buf.valid(),
                      "GFX MLIR: output buffer is not allocated for stage ",
                      m_name);
    }
    const bool aliases_input = output_aliases_input();
    if (out->buf.size < out_bytes || aliases_input) {
      if (out->buf.owned && !out->buf.external) {
        if (!out->buf.from_handle && !aliases_input) {
          m_buffer_manager->release_temp(std::move(out->buf));
        }
        out->buf = allocate_output_buffer(out_bytes);
        OPENVINO_ASSERT(out->buf.valid(),
                        "GFX MLIR: output buffer is not allocated for stage ",
                        m_name);
      } else if (aliases_input) {
        out->buf = allocate_output_buffer(out_bytes);
        OPENVINO_ASSERT(out->buf.valid(),
                        "GFX MLIR: output buffer is not allocated for stage ",
                        m_name);
      }
    }
    if (out->buf.size < out_bytes) {
      OPENVINO_THROW("GFX MLIR: output buffer too small for stage ", m_name,
                     " (need ", out_bytes, ", have ", out->buf.size,
                     ", owned=", out->buf.owned,
                     ", external=", out->buf.external,
                     ", from_handle=", out->buf.from_handle,
                     ", prefer_private=", out->prefer_private, ")");
    }
    out->expected_type = out_type;
  }

  if ((m_type == "Split" || m_type == "VariadicSplit") &&
      try_alias_contiguous_split_outputs(m_node, runtime_inputs.tensor(0),
                                         outputs, m_name.c_str())) {
    return;
  }

  if (m_type == "Broadcast" &&
      try_alias_gqa_broadcast_view(m_node, runtime_inputs.tensor(0), outputs)) {
    return;
  }

  if (m_is_view_op) {
    if (m_inputs.empty() || !m_inputs[0] || !m_inputs[0]->buf.valid()) {
      OPENVINO_THROW("GFX MLIR: missing input buffer for view op ", m_name);
    }
    auto *in = m_inputs[0];
    auto *out = outputs.front();
    const auto in_type = in->expected_type == ov::element::dynamic
                             ? in->buf.type
                             : in->expected_type;
    const auto out_type = out->expected_type == ov::element::dynamic
                              ? in_type
                              : out->expected_type;
    const size_t in_bytes = ov::shape_size(in->shape) * in_type.size();
    const size_t out_bytes = ov::shape_size(out->shape) * out_type.size();
    OPENVINO_ASSERT(in_bytes == out_bytes,
                    "GFX MLIR: view op byte size mismatch for ", m_name, " (",
                    in_bytes, " vs ", out_bytes, ")");
    out->buf = in->buf;
    out->buf.external = true;
    out->buf.owned = false;
    out->expected_type = out_type;
    propagate_view_metadata(*out, *in);
    return;
  }

  const bool use_concat_buffer_copy =
      m_type == "Concat" && !has_absorbed_input_transpose() &&
      (!prefer_specialized_concat_execution() || concat_has_runtime_shape(m_node));
  if (use_concat_buffer_copy) {
    auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(m_node);
    OPENVINO_ASSERT(concat, "GFX MLIR: expected v0::Concat for stage ", m_name);
    auto *output = outputs.front();
    OPENVINO_ASSERT(output && output->buf.valid(),
                    "GFX MLIR: missing concat output buffer for stage ",
                    m_name);

    const ov::Shape &out_shape = output->shape;
    OPENVINO_ASSERT(!out_shape.empty(),
                    "GFX MLIR: concat output shape unknown for stage ", m_name);
    const size_t rank = out_shape.size();
    const int64_t axis_norm =
        normalize_axis(concat->get_axis(), rank, "GFX MLIR: Concat");
    size_t outer = 1;
    for (size_t dim = 0; dim < static_cast<size_t>(axis_norm); ++dim) {
      outer *= out_shape[dim];
    }
    size_t inner = 1;
    for (size_t dim = static_cast<size_t>(axis_norm) + 1; dim < rank; ++dim) {
      inner *= out_shape[dim];
    }
    const size_t axis_total = out_shape[static_cast<size_t>(axis_norm)];
    const auto out_type = output->expected_type == ov::element::dynamic
                              ? output->buf.type
                              : output->expected_type;
    const size_t elem_bytes = out_type.size();

    struct CopyBatch {
      const GpuBuffer *src = nullptr;
      std::vector<GpuBufferCopyRegion> regions;
      uint64_t total_bytes = 0;
    };

    std::vector<CopyBatch> batches;
    batches.reserve(concat->get_input_size());
    size_t axis_offset = 0;
    uint64_t copied_bytes = 0;
    uint64_t copied_regions = 0;
    uint64_t skipped_self_copy_bytes = 0;
    uint64_t skipped_self_copy_regions = 0;
    for (size_t input_idx = 0; input_idx < concat->get_input_size();
         ++input_idx) {
      GpuTensor *src = runtime_inputs.tensor(input_idx);
      OPENVINO_ASSERT(src && src->buf.valid(),
                      "GFX MLIR: missing concat input buffer for stage ",
                      m_name);
      ov::Shape src_shape = !src->shape.empty() ? src->shape : ov::Shape{};
      if (src_shape.empty() &&
          m_node->get_input_partial_shape(input_idx).is_static()) {
        src_shape = m_node->get_input_shape(input_idx);
      }
      OPENVINO_ASSERT(src_shape.size() == rank,
                      "GFX MLIR: concat rank mismatch for stage ", m_name);
      const size_t axis_len = src_shape[static_cast<size_t>(axis_norm)];
      const size_t region_bytes = axis_len * inner * elem_bytes;
      if (outer == 0 || region_bytes == 0) {
        axis_offset += axis_len;
        continue;
      }
      CopyBatch batch{};
      batch.src = &src->buf;
      batch.regions.reserve(outer);
      for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
        GpuBufferCopyRegion region{};
        region.src_offset = outer_idx * region_bytes;
        region.dst_offset =
            ((outer_idx * axis_total + axis_offset) * inner) * elem_bytes;
        region.bytes = region_bytes;
        if (src->buf.buffer == output->buf.buffer &&
            src->buf.offset + region.src_offset ==
                output->buf.offset + region.dst_offset) {
          skipped_self_copy_bytes += static_cast<uint64_t>(region.bytes);
          ++skipped_self_copy_regions;
          continue;
        }
        batch.regions.push_back(region);
      }
      if (batch.regions.empty()) {
        axis_offset += axis_len;
        continue;
      }
      batch.total_bytes = static_cast<uint64_t>(batch.regions.size()) *
                          static_cast<uint64_t>(region_bytes);
      copied_bytes += batch.total_bytes;
      copied_regions += static_cast<uint64_t>(batch.regions.size());
      axis_offset += axis_len;
      batches.push_back(std::move(batch));
    }

    auto *profiler = static_cast<GfxProfiler *>(m_profiler);
    const bool profiling = m_profiling_enabled && profiler;
    if (batches.empty()) {
      if (profiling) {
        profiler->increment_counter("concat_self_copy_skip_bytes",
                                    skipped_self_copy_bytes);
        profiler->increment_counter("concat_self_copy_skip_region_count",
                                    skipped_self_copy_regions);
      }
      return;
    }

    if (!batches.empty()) {
      const auto copy_start = profiling
                                  ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
      for (const auto &batch : batches) {
        gpu_copy_buffer_regions(
            reinterpret_cast<GpuCommandQueueHandle>(command_buffer), *batch.src,
            output->buf, batch.regions.data(), batch.regions.size());
      }
      if (profiling) {
        const auto cpu_us =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - copy_start);
        profiler->record_segment("transfer", "concat_buffer_copy", cpu_us, 0, 0,
                                 copied_bytes, copied_bytes, 0, 0, -1, 0,
                                 reinterpret_cast<uint64_t>(command_buffer));
        profiler->increment_counter("concat_copy_input_count",
                                    static_cast<uint64_t>(batches.size()));
        profiler->increment_counter("concat_copy_region_count", copied_regions);
        profiler->increment_counter("concat_self_copy_skip_bytes",
                                    skipped_self_copy_bytes);
        profiler->increment_counter("concat_self_copy_skip_region_count",
                                    skipped_self_copy_regions);
      }
      return;
    }
  }

  if (!m_kernel) {
    OPENVINO_THROW("GFX MLIR: kernel was not compiled for stage ", m_name);
  }

  ProfileState profile_state{};
  KernelExecutionHooks hooks;
  KernelExecutionHooks *hooks_ptr = nullptr;
  if (m_profiling_enabled && m_profiler) {
    profile_state.enabled = true;
    hooks_ptr = prepare_profiling(profile_state, hooks);
    if (hooks_ptr) {
      hooks_ptr->stage_name = m_name;
      hooks_ptr->stage_type = m_type;
    }
  }

  if (m_type == "Tile" && !outputs.empty() && outputs.front() &&
      !outputs.front()->shape.empty()) {
    const auto tile_plan = plan_tile_runtime_values(runtime_inputs, outputs);
    if (tile_plan.valid()) {
      std::vector<int32_t> tile_scalars = tile_plan.scalar_args;
      if (!tile_plan.input_shape.empty()) {
        auto tile_payload = make_tile_runtime_param_payload(
            *m_buffer_manager, m_name, tile_plan.input_shape,
            tile_plan.output_shape);
        tile_scalars = tile_payload.scalar_args;
        m_kernel_extra_inputs = std::move(tile_payload.extra_inputs);
      }
      if (!is_vulkan_backend()) {
        set_backend_custom_kernel_binding_override("Tile", "tile_kernel",
                                                   tile_scalars);
      }
    }
  }

  KernelRuntimeBindingState execution_binding =
      kernel_binding_override ? *kernel_binding_override
                              : resolved_kernel_runtime_binding_state();
  if (kernel_scalar_args_override) {
    const bool compact_buffer_only_kernel =
        execution_binding.operand_kinds.empty() && m_kernel &&
        m_kernel->args_count() != 0 &&
        m_kernel->args_count() ==
            execution_binding.inputs.size() + outputs.size();
    if (compact_buffer_only_kernel) {
      execution_binding.scalar_args.clear();
    } else if (!execution_binding.operand_kinds.empty()) {
      const size_t scalar_slots = static_cast<size_t>(
          std::count(execution_binding.operand_kinds.begin(),
                     execution_binding.operand_kinds.end(), 0));
      OPENVINO_ASSERT(scalar_slots == kernel_scalar_args_override->size(),
                      "GFX MLIR: scalar runtime arg count mismatch for stage ",
                      m_name, " (", m_type, "): manifest expects ",
                      scalar_slots, ", runtime supplied ",
                      kernel_scalar_args_override->size());
      execution_binding.scalar_args = *kernel_scalar_args_override;
    } else {
      execution_binding.scalar_args = *kernel_scalar_args_override;
    }
  }
  if (gfx_log_debug_enabled()) {
    gfx_log_debug("MLIRExec")
        << "Kernel args prep: scalars=" << execution_binding.scalar_args.size()
        << " inputs=" << execution_binding.inputs.size()
        << " outputs=" << outputs.size()
        << " kinds=" << execution_binding.operand_kinds.size();
    if (m_type == "MatMul") {
      for (size_t input_idx = 0; input_idx < m_inputs.size(); ++input_idx) {
        auto *tensor = runtime_inputs.tensor(input_idx);
        if (!tensor || !tensor->buf.valid()) {
          gfx_log_debug("MLIRExec")
              << "MatMul input[" << input_idx << "] <missing>";
          continue;
        }
        std::ostringstream oss;
        oss << "MatMul input[" << input_idx << "]"
            << " buf=" << tensor->buf.buffer
            << " uid=" << tensor->buf.allocation_uid
            << " off=" << tensor->buf.offset << " bytes=" << tensor->buf.size
            << " shape=" << tensor->shape << " type=" << tensor->expected_type;
        gfx_log_debug("MLIRExec") << oss.str();
      }
      for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
        auto *tensor = outputs[output_idx];
        if (!tensor || !tensor->buf.valid()) {
          gfx_log_debug("MLIRExec")
              << "MatMul output[" << output_idx << "] <missing>";
          continue;
        }
        std::ostringstream oss;
        oss << "MatMul output[" << output_idx << "]"
            << " buf=" << tensor->buf.buffer
            << " uid=" << tensor->buf.allocation_uid
            << " off=" << tensor->buf.offset << " bytes=" << tensor->buf.size
            << " shape=" << tensor->shape << " type=" << tensor->expected_type;
        gfx_log_debug("MLIRExec") << oss.str();
      }
    }
  }

  std::string arg_map;
  std::vector<GpuTensor> empty_extras;
  const std::vector<GpuTensor> *extras = &m_kernel_extra_inputs;
  if (execution_binding.operand_kinds.empty()) {
    const size_t expected_inputs = execution_binding.input_arg_count
                                       ? execution_binding.input_arg_count
                                       : execution_binding.inputs.size();
    if (expected_inputs <= execution_binding.inputs.size()) {
      extras =
          &empty_extras; // Kernel does not expect extra buffers; drop them.
    }
  }

  KernelArgsBundle bundle = build_kernel_args_from_metadata(
      execution_binding.operand_kinds, execution_binding.operand_arg_indices,
      execution_binding.scalar_args, execution_binding.inputs,
      execution_binding.input_arg_count, *extras, outputs,
      [&](size_t input_idx) { return runtime_inputs.tensor(input_idx); },
      m_name.c_str(), gfx_log_debug_enabled() ? &arg_map : nullptr);
  if (gfx_log_debug_enabled() && !arg_map.empty()) {
    gfx_log_debug("MLIRExec") << arg_map;
  }
  auto bound_args = materialize_kernel_bytes_args(
      bundle.args, *m_buffer_manager, m_name.c_str());

  KernelDispatch dispatch{};
  ov::Shape dispatch_shape =
      (outputs.front() && !outputs.front()->shape.empty())
          ? outputs.front()->shape
          : m_output_shape;
  if (m_type == "MatMul" && dispatch_shape.size() > 3) {
    ov::Shape collapsed;
    collapsed.reserve(3);
    collapsed.push_back(shape_batch_product_prefix(dispatch_shape));
    collapsed.push_back(dispatch_shape[dispatch_shape.size() - 2]);
    collapsed.push_back(dispatch_shape[dispatch_shape.size() - 1]);
    dispatch_shape = std::move(collapsed);
  }
  if ((m_type == "Split" || m_type == "VariadicSplit") &&
      !m_output_shape.empty()) {
    dispatch_shape = m_output_shape;
  }
  if (m_type == "TopK") {
    ov::Shape rows_shape = topk_row_dispatch_shape(m_node);
    if (!rows_shape.empty()) {
      dispatch_shape = std::move(rows_shape);
    }
  }
  if (m_parallel_cfg.enabled && m_parallel_cfg.loop_dims == 1) {
    dispatch_shape = {static_cast<size_t>(std::max<uint64_t>(
        static_cast<uint64_t>(ov::shape_size(dispatch_shape)), 1))};
  }
  if (m_parallel_cfg.enabled) {
    dispatch =
        make_parallel_dispatch(dispatch_shape, m_parallel_cfg, m_kernel.get());
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MLIRExec")
          << "Dispatch grid=(" << dispatch.grid[0] << ", " << dispatch.grid[1]
          << ", " << dispatch.grid[2] << ")"
          << " tpg=(" << dispatch.threads_per_group[0] << ", "
          << dispatch.threads_per_group[1] << ", "
          << dispatch.threads_per_group[2] << ")"
          << " loops=" << m_parallel_cfg.loop_dims;
    }
  } else if ((m_type == "ScaledDotProductAttention" ||
              m_type == "GfxSDPAWithCausalMask") &&
             dispatch_shape.size() == 4 && dispatch_shape[3] <= 256 &&
             !m_inputs.empty() && m_inputs[0] && !m_inputs[0]->shape.empty() &&
             m_inputs[0]->shape.size() == 4 && m_inputs[0]->shape[3] <= 256) {
    const uint64_t vectors = static_cast<uint64_t>(dispatch_shape[0]) *
                             static_cast<uint64_t>(dispatch_shape[1]) *
                             static_cast<uint64_t>(dispatch_shape[2]);
    const bool simdgroup_path =
        dispatch_shape[3] <= 64 && m_inputs[0]->shape[3] <= 64;
    const size_t threads =
        m_kernel->clamp_threadgroup_size(simdgroup_path ? 32 : 256);
    dispatch = make_1d_dispatch(
        static_cast<size_t>(vectors * static_cast<uint64_t>(threads)), threads);
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MLIRExec")
          << "SDPA streaming dispatch grid=(" << dispatch.grid[0] << ", "
          << dispatch.grid[1] << ", " << dispatch.grid[2] << ")"
          << " tpg=(" << dispatch.threads_per_group[0] << ", "
          << dispatch.threads_per_group[1] << ", "
          << dispatch.threads_per_group[2] << ")";
    }
  } else if (m_force_single_dispatch) {
    dispatch = make_1d_dispatch(1, 1);
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MLIRExec")
          << "Single dispatch grid=(" << dispatch.grid[0] << ", "
          << dispatch.grid[1] << ", " << dispatch.grid[2] << ")"
          << " tpg=(" << dispatch.threads_per_group[0] << ", "
          << dispatch.threads_per_group[1] << ", "
          << dispatch.threads_per_group[2] << ")";
    }
  } else if (m_type == "MatMul" && m_matmul_reduction_threads > 1) {
    const uint64_t outputs =
        static_cast<uint64_t>(ov::shape_size(dispatch_shape));
    uint64_t work_groups = outputs;
    if (m_is_compressed_matmul && m_compressed_matmul_output_block > 1 &&
        m_compressed_matmul_n > 0) {
      const uint64_t n = static_cast<uint64_t>(m_compressed_matmul_n);
      const uint64_t cols_per_group =
          static_cast<uint64_t>(m_compressed_matmul_output_block);
      const uint64_t rows = (outputs + n - 1) / n;
      work_groups = rows * ((n + cols_per_group - 1) / cols_per_group);
    }
    dispatch = make_1d_dispatch(
        static_cast<size_t>(work_groups * m_matmul_reduction_threads),
        m_kernel->clamp_threadgroup_size(m_matmul_reduction_threads));
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MLIRExec")
          << "MatMul reduction dispatch grid=(" << dispatch.grid[0] << ", "
          << dispatch.grid[1] << ", " << dispatch.grid[2] << ")"
          << " tpg=(" << dispatch.threads_per_group[0] << ", "
          << dispatch.threads_per_group[1] << ", "
          << dispatch.threads_per_group[2] << ")";
    }
  } else if (m_type == "Convolution" && m_conv_output_channels_per_thread > 1 &&
             dispatch_shape.size() == 4) {
    const uint64_t work_items = gfx_conv2d_dispatch_items(
        static_cast<uint64_t>(dispatch_shape[0]),
        static_cast<uint64_t>(dispatch_shape[1]),
        static_cast<uint64_t>(dispatch_shape[2]),
        static_cast<uint64_t>(dispatch_shape[3]),
        m_conv_output_channels_per_thread, m_conv_output_width_per_thread);
    dispatch = make_1d_dispatch(static_cast<size_t>(work_items),
                                m_kernel->clamp_threadgroup_size(256));
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MLIRExec")
          << "Conv channel-block dispatch grid=(" << dispatch.grid[0] << ", "
          << dispatch.grid[1] << ", " << dispatch.grid[2] << ")"
          << " tpg=(" << dispatch.threads_per_group[0] << ", "
          << dispatch.threads_per_group[1] << ", "
          << dispatch.threads_per_group[2] << ")"
          << " channels_per_thread=" << m_conv_output_channels_per_thread
          << " width_per_thread=" << m_conv_output_width_per_thread;
    }
  } else if (m_type == "RMS" && m_rms_reduction_threads > 1 &&
             m_rms_hidden > 0) {
    const uint64_t total =
        static_cast<uint64_t>(ov::shape_size(dispatch_shape));
    const uint64_t rows = (total + m_rms_hidden - 1) / m_rms_hidden;
    dispatch = make_1d_dispatch(
        static_cast<size_t>(rows * m_rms_reduction_threads),
        m_kernel->clamp_threadgroup_size(m_rms_reduction_threads));
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MLIRExec")
          << "RMS reduction dispatch grid=(" << dispatch.grid[0] << ", "
          << dispatch.grid[1] << ", " << dispatch.grid[2] << ")"
          << " tpg=(" << dispatch.threads_per_group[0] << ", "
          << dispatch.threads_per_group[1] << ", "
          << dispatch.threads_per_group[2] << ")";
    }
  } else {
    // Fallback: linear dispatch over total elements.
    dispatch = KernelPlan::make_default_dispatch(dispatch_shape, *m_kernel);
    if (gfx_log_debug_enabled()) {
      gfx_log_debug("MLIRExec")
          << "Default dispatch grid=(" << dispatch.grid[0] << ", "
          << dispatch.grid[1] << ", " << dispatch.grid[2] << ")"
          << " tpg=(" << dispatch.threads_per_group[0] << ", "
          << dispatch.threads_per_group[1] << ", "
          << dispatch.threads_per_group[2] << ")";
    }
  }

  record_runtime_dispatch_profile(static_cast<GfxProfiler *>(m_profiler),
                                  m_parallel_cfg, dispatch);

  try {
    m_kernel->execute(command_buffer, dispatch, bound_args, hooks_ptr);
  } catch (const std::exception &ex) {
    const auto opt_plan = stage_optimization_plan();
    const bool is_shared_vulkan_conv =
        is_vulkan_backend() && is_conv_like() &&
        opt_plan.conv.kind == GfxConvRouteKind::None &&
        !m_vulkan_conv_serial_retry_attempted;
    if (!is_vulkan_backend() || !is_vulkan_pipeline_creation_failure(ex) ||
        (!is_matmul_like() && !is_shared_vulkan_conv) || !m_buffer_manager ||
        (is_matmul_like() && m_matmul_serial_retry_attempted)) {
      throw;
    }

    ov::Shape tuning_shape = m_output_shape;
    const auto caps = query_parallelism_caps(m_buffer_manager);
    if (!m_matmul_safe_retry_attempted) {
      auto safe_plan = select_safe_matmul_fallback_plan(caps, tuning_shape);
      if (safe_plan.has_value()) {
        gfx_log_info("MLIRExec")
            << "Retrying " << m_name << " with safe matmul variant "
            << safe_plan->variant
            << " after pipeline creation failure: " << ex.what();
        remember_matmul_parallelism(caps, tuning_shape, *safe_plan);
        m_matmul_safe_retry_attempted = true;
        m_kernel.reset();
        m_last_input_shape = {};
        compile(m_buffer_manager);
        execute(command_buffer);
        return;
      }
    }

    if (!m_matmul_serial_retry_attempted) {
      gfx_log_info("MLIRExec")
          << "Retrying " << m_name
          << " with serial matmul fallback after pipeline creation failure: "
          << ex.what();
      remember_matmul_parallelism(caps, tuning_shape,
                                  make_serial_matmul_fallback_plan());
      m_matmul_serial_retry_attempted = true;
      m_kernel.reset();
      m_last_input_shape = {};
      compile(m_buffer_manager);
      execute(command_buffer);
      return;
    }

    if (is_shared_vulkan_conv) {
      gfx_log_info("MLIRExec") << "Retrying " << m_name
                               << " with serial Vulkan convolution fallback "
                                  "after pipeline creation failure: "
                               << ex.what();
      m_vulkan_conv_serial_retry_attempted = true;
      m_kernel.reset();
      m_last_input_shape = {};
      compile(m_buffer_manager);
      execute(command_buffer);
      return;
    }

    throw;
  }

  if (profile_state.enabled) {
    finalize_profiling(profile_state);
  }
}

void MlirStage::set_inputs(const std::vector<GpuTensor *> &inputs) {
  m_inputs = inputs;
  if (!m_const_buffers) {
    m_const_buffers = std::make_shared<ConstBufferSet>();
    m_const_buffers->buffers.resize(inputs.size());
    m_const_buffers->present.assign(inputs.size(), false);
  }
}

void MlirStage::set_output(GpuTensor *output) {
  m_output = output;
  m_outputs.clear();
  if (output) {
    m_outputs.push_back(output);
  }
}

void MlirStage::set_output_refs(const std::vector<GpuTensor *> &outputs) {
  m_outputs = outputs;
  m_output = m_outputs.empty() ? nullptr : m_outputs.front();
}

void MlirStage::set_outputs(
    const std::vector<std::unique_ptr<GpuTensor>> &outputs) {
  std::vector<GpuTensor *> refs;
  refs.reserve(outputs.size());
  for (const auto &output : outputs) {
    refs.push_back(output.get());
  }
  set_output_refs(refs);
}

void MlirStage::set_input_transform(size_t input_idx,
                                    const GfxInputTransform &transform) {
  if (m_input_transforms.size() <= input_idx) {
    m_input_transforms.resize(input_idx + 1);
  }
  m_input_transforms[input_idx] = transform;
}

void MlirStage::set_runtime_options(const GpuStageRuntimeOptions &options) {
  m_runtime_traits.diagnostic_f32_vendor_image =
      options.diagnostic_f32_vendor_image;
}

void MlirStage::enable_profiling(bool enable) { m_profiling_enabled = enable; }

void MlirStage::set_profiler(void *profiler, uint32_t node_id,
                             const std::string &node_name,
                             const std::string &node_type) {
  m_profiler = profiler;
  m_profile_node_id = node_id;
  m_profile_node_name = node_name;
  m_profile_node_type = node_type;
}

void MlirStage::on_command_buffer_complete() {
  if (m_kernel) {
    m_kernel->on_submission_complete();
  }
}

bool MlirStage::fuse_activation(ActivationKind kind, float alpha) {
  if (!allow_stage_activation_fusion(backend_kind(), m_type, kind) ||
      !stage_optimization_plan().execution.fusion.allow_activation) {
    return false;
  }
  OPENVINO_ASSERT(!m_kernel,
                  "MlirStage: cannot fuse activation after compilation");
  m_has_activation = true;
  m_activation = kind;
  m_activation_alpha = alpha;
  return true;
}

bool MlirStage::fuse_input_activation(size_t input_idx, ActivationKind kind,
                                      float alpha) {
  if (m_has_input_activation || m_has_activation || m_type != "Multiply" ||
      input_idx >= 2) {
    return false;
  }
  if (kind != ActivationKind::Relu && kind != ActivationKind::Sigmoid &&
      kind != ActivationKind::Tanh && kind != ActivationKind::Gelu &&
      kind != ActivationKind::Swish && kind != ActivationKind::HSwish &&
      kind != ActivationKind::HSigmoid) {
    return false;
  }
  if (!m_node || m_node->get_output_element_type(0).is_integral_number() ||
      m_node->get_output_element_type(0) == ov::element::boolean) {
    return false;
  }
  OPENVINO_ASSERT(!m_kernel,
                  "MlirStage: cannot fuse input activation after compilation");
  m_has_input_activation = true;
  m_input_activation_index = input_idx;
  m_input_activation = kind;
  m_input_activation_alpha = alpha;
  return true;
}

bool MlirStage::fuse_residual_add() {
  if (m_type != "RMS" || is_vulkan_backend()) {
    return false;
  }
  OPENVINO_ASSERT(!m_kernel,
                  "MlirStage: cannot fuse residual add after compilation");
  m_has_residual_add = true;
  return true;
}

bool MlirStage::fuse_batchnorm(const BatchNormParams &params) {
  OPENVINO_ASSERT(!m_kernel,
                  "MlirStage: cannot fuse batchnorm after compilation");
  if (!stage_optimization_plan().execution.fusion.allow_batchnorm) {
    return false;
  }
  if (!m_node) {
    return false;
  }
  if (m_type != "Convolution" && m_type != "GroupConvolution") {
    return false;
  }
  if (params.empty()) {
    return false;
  }
  const auto et = m_node->get_output_element_type(0);
  if (!et.is_real()) {
    return false;
  }
  const auto &pshape = m_node->get_output_partial_shape(0);
  if (pshape.rank().is_dynamic() || pshape.rank().get_length() < 2) {
    return false;
  }
  if (pshape[1].is_static() &&
      static_cast<size_t>(pshape[1].get_length()) != params.gamma.size()) {
    return false;
  }
  m_has_bn = true;
  m_bn_params = params;
  return true;
}

bool MlirStage::fuse_bias(const BiasParams &params) {
  if (!stage_optimization_plan().execution.fusion.allow_bias) {
    return false;
  }
  OPENVINO_ASSERT(!m_kernel, "MlirStage: cannot fuse bias after compilation");
  if (params.empty()) {
    return false;
  }
  m_has_bias = true;
  m_bias_params = params;
  return true;
}

const GfxInputTransform *MlirStage::input_transform(size_t input_idx) const {
  if (input_idx >= m_input_transforms.size() ||
      !m_input_transforms[input_idx].has_transpose()) {
    return nullptr;
  }
  return &m_input_transforms[input_idx];
}

ov::Shape MlirStage::compile_time_input_shape(size_t input_idx) const {
  if (const auto *transform = input_transform(input_idx)) {
    return transform->source_shape;
  }
  if (m_node) {
    try {
      return m_node->get_input_shape(input_idx);
    } catch (const std::exception &) {
    }
  }
  return {};
}

std::vector<int32_t>
MlirStage::compile_time_broadcast_strides(size_t input_idx,
                                          const ov::Shape &out_shape) const {
  const auto in_shape = compile_time_input_shape(input_idx);
  if (in_shape.empty()) {
    return {};
  }
  if (const auto *transform = input_transform(input_idx)) {
    OPENVINO_ASSERT(m_node &&
                        m_node->get_input_partial_shape(input_idx).is_static(),
                    "GFX MLIR: absorbed transpose requires static consumer "
                    "input shape for stage ",
                    m_name);
    return compute_permuted_broadcast_element_strides(
        transform->source_shape, m_node->get_input_shape(input_idx),
        transform->transpose_permutation, out_shape, "GFX MLIR");
  }
  return compute_broadcast_element_strides(in_shape, out_shape);
}

bool MlirStage::has_absorbed_input_transpose() const {
  return std::any_of(m_input_transforms.begin(), m_input_transforms.end(),
                     [](const GfxInputTransform &transform) {
                       return transform.has_transpose();
                     });
}

KernelExecutionHooks *MlirStage::prepare_profiling(ProfileState &,
                                                   KernelExecutionHooks &) {
  return nullptr;
}

void MlirStage::finalize_profiling(const ProfileState &) {}

GfxStageOptimizationPlan MlirStage::stage_optimization_plan() const {
  return select_stage_optimization_plan(
      m_buffer_manager, backend_kind(), m_type, m_node,
      m_node ? m_node->get_output_element_type(0) : ov::element::dynamic,
      m_has_bias, m_has_activation, m_has_bn, m_runtime_traits);
}

void MlirStage::clone_into(MlirStage &dst) const {
  dst.m_kernel = m_kernel;
  dst.m_is_compressed_matmul = m_is_compressed_matmul;
  dst.m_output_shape = m_output_shape;
  dst.m_last_input_shape = m_last_input_shape;
  dst.m_input_transforms = m_input_transforms;
  dst.m_kernel_binding = m_kernel_binding;
  dst.m_kernel_binding_owned_by_source_plan =
      m_kernel_binding_owned_by_source_plan;
  dst.m_uses_mpsrt_sdpa_plan = m_uses_mpsrt_sdpa_plan;
  dst.m_const_buffers = m_const_buffers;
  dst.m_parallel_cfg = m_parallel_cfg;
  dst.m_force_single_dispatch = m_force_single_dispatch;
  dst.m_matmul_reduction_threads = m_matmul_reduction_threads;
  dst.m_compressed_matmul_output_block = m_compressed_matmul_output_block;
  dst.m_compressed_matmul_n = m_compressed_matmul_n;
  dst.m_conv_output_channels_per_thread = m_conv_output_channels_per_thread;
  dst.m_conv_output_width_per_thread = m_conv_output_width_per_thread;
  dst.m_conv_weight_storage_type = m_conv_weight_storage_type;
  dst.m_conv_weights_packed_oc4 = m_conv_weights_packed_oc4;
  dst.m_rms_reduction_threads = m_rms_reduction_threads;
  dst.m_rms_hidden = m_rms_hidden;
  dst.m_vulkan_conv_serial_retry_attempted =
      m_vulkan_conv_serial_retry_attempted;
  dst.m_has_activation = m_has_activation;
  dst.m_activation = m_activation;
  dst.m_activation_alpha = m_activation_alpha;
  dst.m_has_input_activation = m_has_input_activation;
  dst.m_input_activation_index = m_input_activation_index;
  dst.m_input_activation = m_input_activation;
  dst.m_input_activation_alpha = m_input_activation_alpha;
  dst.m_has_residual_add = m_has_residual_add;
  dst.m_has_bn = m_has_bn;
  dst.m_bn_params = m_bn_params;
  dst.m_has_bias = m_has_bias;
  dst.m_bias_params = m_bias_params;
  dst.m_runtime_traits = m_runtime_traits;
  dst.m_kernel_extra_inputs = m_kernel_extra_inputs;
}

bool MlirStage::is_conv_like() const {
  return m_type == "Convolution" || m_type == "GroupConvolution" ||
         m_type == "GroupConv2D";
}

bool MlirStage::is_matmul_like() const { return m_type == "MatMul"; }

void MlirStage::apply_stage_optimization_attrs(
    mlir::ModuleOp module, const GfxStageOptimizationPlan &plan) {
  if (!module) {
    return;
  }
  auto *ctx = module.getContext();
  module->setAttr(
      "gfx.stage_archetype",
      mlir::StringAttr::get(ctx, stage_archetype_attr(plan.archetype)));
  const bool conv_mpsrt_annotated =
      is_conv_like() &&
      annotate_module_with_conv_mpsrt_plan(
          module, plan, m_node, m_type, m_has_bias,
          m_has_bias ? &m_bias_params : nullptr, m_has_activation,
          m_activation) != GfxConvMpsrtLoweringKind::None;
  const bool deferred_apple_mps_conv_materialization =
      is_conv_like() && !conv_mpsrt_annotated &&
      plan.placement.domain == GfxStageBackendDomain::AppleMps;
  if (!conv_mpsrt_annotated && !deferred_apple_mps_conv_materialization) {
    const bool materialize_typed_program =
        plan.placement.domain == GfxStageBackendDomain::AppleMps ||
        plan.placement.domain == GfxStageBackendDomain::AppleMsl;
    (void)run_gfx_apple_stage_pipeline(module, plan, m_type, {},
                                       materialize_typed_program);
  }
  module->setAttr(
      "gfx.tensor_layout_kind",
      mlir::StringAttr::get(ctx, tensor_layout_kind_attr(plan.layout.kind)));
  module->setAttr("gfx.tensor_view_only",
                  mlir::BoolAttr::get(ctx, plan.layout.view_only));
  module->setAttr("gfx.post_bias_allowed",
                  mlir::BoolAttr::get(ctx, plan.post_ops.bias));
  module->setAttr("gfx.post_activation_allowed",
                  mlir::BoolAttr::get(ctx, plan.post_ops.activation));
  module->setAttr("gfx.post_batchnorm_allowed",
                  mlir::BoolAttr::get(ctx, plan.post_ops.batchnorm));
  module->setAttr("gfx.submit_weight",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(ctx, 32),
                      static_cast<int32_t>(plan.execution.submit.weight)));
  module->setAttr(
      "gfx.conv_route_kind",
      mlir::StringAttr::get(ctx, conv_route_kind_attr(plan.conv.kind)));
  module->setAttr(
      "gfx.conv_family",
      mlir::StringAttr::get(ctx, conv_family_attr(plan.conv.family)));
  module->setAttr("gfx.conv_algorithm_kind",
                  mlir::StringAttr::get(
                      ctx, conv_algorithm_kind_attr(plan.conv.algorithm.kind)));
  module->setAttr("gfx.conv_variant",
                  mlir::StringAttr::get(ctx, plan.conv.algorithm.variant));
  module->setAttr("gfx.conv_requires_multi_kernel_manifest",
                  mlir::BoolAttr::get(
                      ctx, plan.conv.algorithm.requires_multi_kernel_manifest));
  module->setAttr(
      "gfx.conv_multi_kernel_family",
      mlir::StringAttr::get(ctx, plan.conv.algorithm.multi_kernel_family));
  module->setAttr(
      "gfx.conv_reduction_work",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(plan.conv.algorithm.reduction_work)));
  module->setAttr(
      "gfx.conv_output_elements",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(plan.conv.algorithm.output_elements)));
  module->setAttr(
      "gfx.conv_intermediate_elements",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(plan.conv.algorithm.intermediate_elements)));
  module->setAttr(
      "gfx.conv_reduction_chunk_count",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(plan.conv.algorithm.reduction_chunk_count)));
  module->setAttr(
      "gfx.conv_reduction_chunk_size",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(plan.conv.algorithm.reduction_chunk_size)));
  module->setAttr(
      "gfx.conv_workgroup_reduction_lanes",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(plan.conv.algorithm.workgroup_reduction_lanes)));
  module->setAttr(
      "gfx.conv_workgroup_output_lanes",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(plan.conv.algorithm.workgroup_output_lanes)));
  module->setAttr("gfx.conv_output_channel_reuse_lanes",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(ctx, 64),
                      static_cast<int64_t>(
                          plan.conv.algorithm.output_channel_reuse_lanes)));
  module->setAttr("gfx.conv_spatial_output_reuse_lanes",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(ctx, 64),
                      static_cast<int64_t>(
                          plan.conv.algorithm.spatial_output_reuse_lanes)));
  module->setAttr(
      "gfx.conv_output_reuse_lanes",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(plan.conv.algorithm.output_reuse_lanes)));
  if (plan.conv.algorithm.spatial_input_reuse_saved_width_loads > 0) {
    module->setAttr("gfx.conv_spatial_input_reuse",
                    mlir::StringAttr::get(ctx, "width"));
  }
  module->setAttr(
      "gfx.conv_spatial_input_reuse_lanes",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(plan.conv.algorithm.spatial_input_reuse_lanes)));
  module->setAttr(
      "gfx.conv_spatial_input_reuse_unique_width_loads",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(
              plan.conv.algorithm.spatial_input_reuse_unique_width_loads)));
  module->setAttr(
      "gfx.conv_spatial_input_reuse_saved_width_loads",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(
              plan.conv.algorithm.spatial_input_reuse_saved_width_loads)));
  const auto &manifest = plan.conv.algorithm.multi_kernel_manifest;
  module->setAttr(
      "gfx.conv_multi_kernel_stage_count",
      mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                             static_cast<int64_t>(manifest.stages.size())));
  module->setAttr(
      "gfx.conv_multi_kernel_requires_owned_intermediates",
      mlir::BoolAttr::get(ctx, manifest.requires_owned_intermediates));
  module->setAttr(
      "gfx.conv_multi_kernel_requires_owned_launch_sequence",
      mlir::BoolAttr::get(ctx, manifest.requires_owned_launch_sequence));
  module->setAttr("gfx.conv_multi_kernel_requires_output_reuse",
                  mlir::BoolAttr::get(ctx, manifest.requires_output_reuse));
  module->setAttr(
      "gfx.conv_multi_kernel_requires_spatial_input_reuse",
      mlir::BoolAttr::get(ctx, manifest.requires_spatial_input_reuse));
  module->setAttr(
      "gfx.conv_multi_kernel_requires_coarse_output_tile_preservation",
      mlir::BoolAttr::get(
          ctx, manifest.requires_coarse_output_tile_preservation));
  module->setAttr(
      "gfx.conv_multi_kernel_has_workgroup_local_reduction_plan",
      mlir::BoolAttr::get(ctx, manifest.has_workgroup_local_reduction_plan));
  module->setAttr(
      "gfx.conv_multi_kernel_coarse_spatial_tile_elements",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(manifest.coarse_spatial_tile_elements)));
  module->setAttr(
      "gfx.conv_multi_kernel_coarse_output_channel_block",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(manifest.coarse_output_channel_block)));
  module->setAttr(
      "gfx.conv_multi_kernel_coarse_output_tile_elements",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(manifest.coarse_output_tile_elements)));
  module->setAttr(
      "gfx.conv_multi_kernel_workgroup_output_tile_deficit",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(manifest.workgroup_output_tile_deficit)));
  module->setAttr("gfx.conv_multi_kernel_partial_sum_elements",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(ctx, 64),
                      static_cast<int64_t>(manifest.partial_sum_elements)));
  module->setAttr(
      "gfx.conv_multi_kernel_reduced_accumulator_elements",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(manifest.reduced_accumulator_elements)));
  module->setAttr(
      "gfx.conv_multi_kernel_owned_intermediate_elements",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(manifest.owned_intermediate_elements)));
  module->setAttr("gfx.conv_multi_kernel_owned_intermediate_bytes",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(ctx, 64),
                      static_cast<int64_t>(manifest.owned_intermediate_bytes)));
  module->setAttr(
      "gfx.conv_multi_kernel_owned_intermediate_buffer_count",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(manifest.owned_intermediate_buffer_count)));
  module->setAttr(
      "gfx.conv_multi_kernel_workgroup_local_accumulator_elements",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(
              manifest.workgroup_local_accumulator_elements)));
  module->setAttr(
      "gfx.conv_multi_kernel_workgroup_local_accumulator_bytes",
      mlir::IntegerAttr::get(
          mlir::IntegerType::get(ctx, 64),
          static_cast<int64_t>(manifest.workgroup_local_accumulator_bytes)));
  module->setAttr("gfx.conv_multi_kernel_launch_dispatch_count",
                  mlir::IntegerAttr::get(
                      mlir::IntegerType::get(ctx, 64),
                      static_cast<int64_t>(manifest.launch_dispatch_count)));
  for (size_t i = 0; i < manifest.stages.size(); ++i) {
    const auto &stage = manifest.stages[i];
    const auto suffix = std::to_string(i);
    module->setAttr(
        "gfx.conv_multi_kernel_stage" + suffix + "_kind",
        mlir::StringAttr::get(
            ctx, gfx_conv_multi_kernel_stage_kind_name(stage.kind)));
    module->setAttr("gfx.conv_multi_kernel_stage" + suffix + "_name",
                    mlir::StringAttr::get(ctx, stage.name));
    module->setAttr(
        "gfx.conv_multi_kernel_stage" + suffix + "_output_elements",
        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                               static_cast<int64_t>(stage.output_elements)));
    module->setAttr("gfx.conv_multi_kernel_stage" + suffix +
                        "_writes_intermediate",
                    mlir::BoolAttr::get(ctx, stage.writes_intermediate));
    module->setAttr("gfx.conv_multi_kernel_stage" + suffix +
                        "_writes_final_output",
                    mlir::BoolAttr::get(ctx, stage.writes_final_output));
  }
  record_conv_compile_profile(plan);
}

void MlirStage::apply_input_transform_attrs(mlir::ModuleOp module) const {
  if (!module) {
    return;
  }
  auto *ctx = module.getContext();
  mlir::OpBuilder b(ctx);
  for (size_t input_idx = 0; input_idx < m_input_transforms.size();
       ++input_idx) {
    const auto *transform = input_transform(input_idx);
    if (!transform) {
      continue;
    }
    llvm::SmallVector<mlir::Attribute> attrs;
    attrs.reserve(transform->transpose_permutation.size());
    for (int64_t axis : transform->transpose_permutation) {
      attrs.push_back(b.getI64IntegerAttr(axis));
    }
    const std::string attr_name =
        "gfx.absorbed_input" + std::to_string(input_idx) + "_perm";
    module->setAttr(attr_name, b.getArrayAttr(attrs));
  }
}

void MlirStage::set_parallel_preference(mlir::ModuleOp module) {
  if (!module) {
    return;
  }
  auto *ctx = module.getContext();
  bool conv2d = false;
  if (m_node && is_conv_like()) {
    auto in_shape = m_node->get_input_partial_shape(0);
    if (in_shape.rank().is_static() && (in_shape.rank().get_length() == 4 ||
                                        in_shape.rank().get_length() == 5)) {
      conv2d = true;
    }
  }
  bool prefer_parallel = conv2d;
  const auto optimization_plan = stage_optimization_plan();
  if (conv2d && m_type == "Convolution") {
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    if (conv && conv->get_input_size() == 2 && conv->get_output_size() == 1) {
      const auto &in_shape = conv->get_input_shape(0);
      const auto &w_shape = conv->get_input_shape(1);
      if (in_shape.size() == 4 && w_shape.size() == 4) {
        const auto caps = query_parallelism_caps(m_buffer_manager);
        const uint64_t input_channels =
            static_cast<uint64_t>(std::max<size_t>(1, in_shape[1]));
        const uint64_t output_channels =
            static_cast<uint64_t>(std::max<size_t>(1, w_shape[0]));
        const uint64_t kernel_work =
            input_channels *
            static_cast<uint64_t>(std::max<size_t>(1, w_shape[2])) *
            static_cast<uint64_t>(std::max<size_t>(1, w_shape[3]));
        const bool stride2 =
            conv->get_strides().at(0) > 1 || conv->get_strides().at(1) > 1;
        const bool depthwise = optimization_plan.conv.algorithm.kind ==
                               GfxConvAlgorithmKind::DepthwiseDirect;
        const auto plan = select_conv_parallelism(
            caps, m_output_shape, input_channels, output_channels, kernel_work,
            stride2, depthwise);
        record_conv_dispatch_compile_profile(plan);
        prefer_parallel = prefer_parallel || plan.prefer_parallel;
        if (plan.prefer_parallel) {
          module->setAttr("gfx.dispatch_tile_h",
                          mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                                 plan.dispatch.tile_h));
          module->setAttr("gfx.dispatch_tile_w",
                          mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                                 plan.dispatch.tile_w));
          module->setAttr("gfx.dispatch_threads_h",
                          mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                                 plan.dispatch.threads_h));
          module->setAttr("gfx.dispatch_threads_w",
                          mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                                 plan.dispatch.threads_w));
          module->setAttr("gfx.dispatch_channel_block",
                          mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                                 plan.output_channel_block));
          module->setAttr(
              "gfx.dispatch_channel_block_accumulation",
              mlir::StringAttr::get(ctx, conv_channel_block_accumulation_name(
                                             plan.channel_block_accumulation)));
        }
      }
    }
  } else if (conv2d && m_type == "GroupConvolution") {
    auto gconv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(m_node);
    if (gconv && gconv->get_input_size() == 2 &&
        gconv->get_output_size() == 1 &&
        optimization_plan.conv.algorithm.kind ==
            GfxConvAlgorithmKind::DepthwiseDirect) {
      const auto &in_shape = gconv->get_input_shape(0);
      const auto &w_shape = gconv->get_input_shape(1);
      if (in_shape.size() == 4 && w_shape.size() == 5) {
        const auto caps = query_parallelism_caps(m_buffer_manager);
        const uint64_t input_channels =
            static_cast<uint64_t>(std::max<size_t>(1, in_shape[1]));
        const uint64_t output_channels =
            static_cast<uint64_t>(std::max<size_t>(1, w_shape[0]));
        const uint64_t kernel_work =
            static_cast<uint64_t>(std::max<size_t>(1, w_shape[3])) *
            static_cast<uint64_t>(std::max<size_t>(1, w_shape[4]));
        const bool stride2 =
            gconv->get_strides().at(0) > 1 || gconv->get_strides().at(1) > 1;
        const auto plan = select_conv_parallelism(
            caps, m_output_shape, input_channels, output_channels, kernel_work,
            stride2, /*depthwise=*/true);
        record_conv_dispatch_compile_profile(plan);
        prefer_parallel = prefer_parallel || plan.prefer_parallel;
        if (plan.prefer_parallel) {
          module->setAttr("gfx.dispatch_tile_h",
                          mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                                 plan.dispatch.tile_h));
          module->setAttr("gfx.dispatch_tile_w",
                          mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                                 plan.dispatch.tile_w));
          module->setAttr("gfx.dispatch_threads_h",
                          mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                                 plan.dispatch.threads_h));
          module->setAttr("gfx.dispatch_threads_w",
                          mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                                 plan.dispatch.threads_w));
          module->setAttr("gfx.dispatch_channel_block",
                          mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                                 plan.output_channel_block));
          module->setAttr(
              "gfx.dispatch_channel_block_accumulation",
              mlir::StringAttr::get(ctx, conv_channel_block_accumulation_name(
                                             plan.channel_block_accumulation)));
        }
      }
    } else {
      prefer_parallel = false;
    }
  }
  if (is_matmul_like()) {
    const auto caps = query_parallelism_caps(m_buffer_manager);
    const auto plan = select_matmul_parallelism(caps, m_output_shape);
    prefer_parallel = prefer_parallel || plan.prefer_parallel;
    if (plan.prefer_parallel) {
      module->setAttr("gfx.dispatch_tile_h",
                      mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                             plan.dispatch.tile_h));
      module->setAttr("gfx.dispatch_tile_w",
                      mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                             plan.dispatch.tile_w));
      module->setAttr("gfx.dispatch_threads_h",
                      mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                             plan.dispatch.threads_h));
      module->setAttr("gfx.dispatch_threads_w",
                      mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                             plan.dispatch.threads_w));
    }
  }
  const bool explicit_linear_parallel =
      module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch") &&
      module->getAttrOfType<mlir::BoolAttr>("gfx.parallel_dispatch")
          .getValue() &&
      module->getAttrOfType<mlir::IntegerAttr>("gfx.parallel_loop_dims") &&
      module->getAttrOfType<mlir::IntegerAttr>("gfx.parallel_loop_dims")
              .getInt() == 1;
  if (explicit_linear_parallel && !module->hasAttr("gfx.dispatch_threads_w") &&
      !m_output_shape.empty()) {
    const auto caps = query_parallelism_caps(m_buffer_manager);
    const uint64_t total_elems = std::max<uint64_t>(
        static_cast<uint64_t>(ov::shape_size(m_output_shape)), 1);
    const uint64_t work_per_elem =
        std::max<uint64_t>(static_cast<uint64_t>(m_output_shape.size()), 1);
    const auto plan =
        select_chunk_dispatch_plan(caps, m_type, total_elems, work_per_elem);
    module->setAttr("gfx.dispatch_threads_h",
                    mlir::IntegerAttr::get(mlir::IndexType::get(ctx), 1));
    module->setAttr("gfx.dispatch_threads_w",
                    mlir::IntegerAttr::get(mlir::IndexType::get(ctx),
                                           plan.threads_per_group));
    prefer_parallel = true;
  }
  module->setAttr("gfx.prefer_parallel",
                  mlir::BoolAttr::get(ctx, prefer_parallel));
}

void MlirStage::apply_fused_operations(mlir::ModuleOp module) {
  if (!module) {
    return;
  }
  if (m_has_bn) {
    const bool applied = apply_fused_batchnorm(module, m_bn_params);
    OPENVINO_ASSERT(applied,
                    "GFX MLIR: failed to apply fused batchnorm for stage ",
                    m_name);
  }
  if (m_has_bias) {
    const bool applied = apply_fused_bias(module, m_bias_params);
    OPENVINO_ASSERT(applied, "GFX MLIR: failed to apply fused bias for stage ",
                    m_name);
  }
  if (m_has_activation) {
    if (!is_conv_like()) {
      module->setAttr("gfx.post_activation_only",
                      mlir::BoolAttr::get(module.getContext(), true));
    }
    const bool applied =
        apply_fused_activation(module, m_activation, m_activation_alpha);
    OPENVINO_ASSERT(applied,
                    "GFX MLIR: failed to apply fused activation for stage ",
                    m_name);
  }
  if (m_has_input_activation) {
    const bool applied = apply_fused_input_activation(
        module, m_input_activation_index, m_input_activation,
        m_input_activation_alpha);
    OPENVINO_ASSERT(
        applied, "GFX MLIR: failed to apply fused input activation for stage ",
        m_name);
  }
}

} // namespace gfx_plugin
} // namespace ov
