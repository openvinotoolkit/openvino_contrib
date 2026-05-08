// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_stage.hpp"
#include "mlir/mlir_support.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <iterator>
#include <numeric>
#include <optional>
#include <sstream>

#include "kernel_ir/gfx_kernel_args.hpp"
#include "kernel_ir/gfx_kernel_plan.hpp"
#include "kernel_ir/gfx_kernel_spec.hpp"
#include "mlir/gfx_apple_stage_pipeline.hpp"
#include "mlir/codegen_common.hpp"
#include "mlir/gfx_kernel_runtime_params.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mpsrt_const_tensor_sources.hpp"
#include "mlir/gfx_mpsrt_conv_metadata.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_kernel_plan_utils.hpp"
#include "mlir/msl_codegen.hpp"
#include "runtime/gfx_compile_profiling.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_parallelism.hpp"
#include "runtime/gfx_profiler.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gfx_stage_policy.hpp"
#include "runtime/memory_manager.hpp"
#include "transforms/gfx_llm_ops.hpp"
#include "transforms/mlir_fused_ops.hpp"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"

// MLIR 18 switches AttrSizedOperandSegments properties to DenseArrayAttr in
// assembly; generated converters still expect DenseI32ArrayAttr. Provide a
// bridge so verification succeeds with either representation.
namespace mlir {
template <size_t N>
inline LogicalResult convertFromAttribute(std::array<int32_t, N>& storage,
                                          Attribute attr,
                                          function_ref<InFlightDiagnostic()> emitError) {
    if (auto a = dyn_cast<DenseI32ArrayAttr>(attr)) {
        if (a.size() != static_cast<int64_t>(N)) {
            if (emitError) emitError() << "DenseI32ArrayAttr has wrong size";
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
                auto ptr = reinterpret_cast<const int32_t*>(raw.data());
                vals.append(ptr, ptr + arr.getSize());
            } else if (ity.isIndex() || ity.getWidth() == 64) {
                auto raw = arr.getRawData();
                auto ptr = reinterpret_cast<const int64_t*>(raw.data());
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
    return convertFromAttribute(MutableArrayRef<int32_t>(storage.data(), storage.size()),
                                attr,
                                emitError);
}

inline LogicalResult convertFromAttribute(DenseI32ArrayAttr& storage,
                                          Attribute attr,
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
                auto ptr = reinterpret_cast<const int32_t*>(raw.data());
                vals.append(ptr, ptr + arr.getSize());
            } else if (ity.isIndex() || ity.getWidth() == 64) {
                auto raw = arr.getRawData();
                auto ptr = reinterpret_cast<const int64_t*>(raw.data());
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
}  // namespace mlir

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

const char* stage_archetype_attr(GfxStageArchetype archetype) {
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

const char* tensor_layout_kind_attr(GfxTensorLayoutKind kind) {
    switch (kind) {
        case GfxTensorLayoutKind::Materialized:
            return "materialized";
        case GfxTensorLayoutKind::ViewOnly:
            return "view_only";
        default:
            return "unknown";
    }
}

const char* conv_route_kind_attr(GfxConvRouteKind kind) {
    switch (kind) {
        case GfxConvRouteKind::Direct1x1:
            return "direct_1x1";
        case GfxConvRouteKind::Direct3x3:
            return "direct_3x3";
        case GfxConvRouteKind::Chunked:
            return "chunked";
        case GfxConvRouteKind::GroupChunked:
            return "group_chunked";
        default:
            return "none";
    }
}

const char* conv_family_attr(GfxConvFamily family) {
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

const char* conv_algorithm_kind_attr(GfxConvAlgorithmKind kind) {
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
        case GfxConvAlgorithmKind::Im2ColMatMul:
            return "im2col_matmul";
        case GfxConvAlgorithmKind::Indirect:
            return "indirect";
        default:
            return "none";
    }
}

bool is_vulkan_pipeline_creation_failure(const std::exception& ex) {
    return std::string(ex.what()).find("vkCreateComputePipelines failed") != std::string::npos;
}

std::optional<MatMulParallelismPlan> select_safe_matmul_fallback_plan(const GfxParallelismCaps& caps,
                                                                      const ov::Shape& output_shape) {
    const auto plans = enumerate_matmul_parallelism_candidates(caps, output_shape);
    if (plans.size() <= 1) {
        return std::nullopt;
    }
    auto score = [](const MatMulParallelismPlan& plan) {
        const uint32_t threads = plan.dispatch.threads_h * plan.dispatch.threads_w;
        const uint32_t aspect = plan.dispatch.tile_h > plan.dispatch.tile_w
                                    ? (plan.dispatch.tile_h - plan.dispatch.tile_w)
                                    : (plan.dispatch.tile_w - plan.dispatch.tile_h);
        const uint32_t distance_to_safe_threads = threads > 16 ? (threads - 16) : (16 - threads);
        return std::tuple<uint32_t, uint32_t, uint32_t>{distance_to_safe_threads, aspect, threads};
    };
    auto best = std::min_element(plans.begin() + 1, plans.end(), [&](const auto& lhs, const auto& rhs) {
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

uint64_t shape_batch_product_prefix(const ov::Shape& shape) {
    if (shape.size() <= 2) {
        return 1;
    }
    uint64_t batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= static_cast<uint64_t>(shape[i]);
    }
    return batch;
}

std::optional<MatMulCodegenDesc> static_matmul_desc_for_node(const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return std::nullopt;
    }
    auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
    if (!matmul ||
        !matmul->get_input_partial_shape(0).is_static() ||
        !matmul->get_input_partial_shape(1).is_static() ||
        !matmul->get_output_partial_shape(0).is_static()) {
        return std::nullopt;
    }

    const ov::Shape a_shape = matmul->get_input_shape(0);
    const ov::Shape b_shape = matmul->get_input_shape(1);
    const ov::Shape out_shape = matmul->get_output_shape(0);
    if (a_shape.size() < 2 || b_shape.size() < 2 || out_shape.size() < 2) {
        return std::nullopt;
    }

    const bool ta = matmul->get_transpose_a();
    const bool tb = matmul->get_transpose_b();
    const size_t a_rank = a_shape.size();
    const size_t b_rank = b_shape.size();
    const size_t out_rank = out_shape.size();
    const int64_t M = static_cast<int64_t>(out_shape[out_rank - 2]);
    const int64_t N = static_cast<int64_t>(out_shape[out_rank - 1]);
    const int64_t K = static_cast<int64_t>(ta ? a_shape[a_rank - 2] : a_shape[a_rank - 1]);
    if (M <= 0 || N <= 0 || K <= 0) {
        return std::nullopt;
    }

    MatMulCodegenDesc desc{};
    desc.element_type = matmul->get_output_element_type(0);
    desc.input_a_type = matmul->get_input_element_type(0);
    desc.input_b_type = matmul->get_input_element_type(1);
    desc.output_type = matmul->get_output_element_type(0);
    desc.a_transpose = ta;
    desc.b_transpose = tb;
    desc.b_is_nk_layout = tb;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.batch = static_cast<int64_t>(ov::shape_size(out_shape) / static_cast<uint64_t>(M * N));
    desc.batch_a = static_cast<int64_t>(ov::shape_size(a_shape) / static_cast<uint64_t>(M * K));
    desc.batch_b = static_cast<int64_t>(ov::shape_size(b_shape) / static_cast<uint64_t>(K * N));
    return desc;
}

}  // namespace

namespace {

bool should_pack_matmul_const_input_as_f16(const std::shared_ptr<const ov::Node>& node,
                                           size_t input_idx,
                                           const ov::Tensor& tensor) {
    auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
    return matmul &&
           input_idx == 1 &&
           (!matmul->get_input_partial_shape(0).is_static() ||
            !matmul->get_output_partial_shape(0).is_static()) &&
           tensor.get_element_type() == ov::element::f32 &&
           matmul->get_output_element_type(0) == ov::element::f32;
}

std::vector<ov::float16> pack_f32_tensor_as_f16(const ov::Tensor& tensor) {
    const auto elements = tensor.get_size();
    const auto* src = tensor.data<const float>();
    std::vector<ov::float16> packed(elements);
    for (size_t i = 0; i < elements; ++i) {
        packed[i] = ov::float16(src[i]);
    }
    return packed;
}

bool should_pack_conv2d_const_weights_oc4(const std::shared_ptr<const ov::Node>& node,
                                          size_t input_idx,
                                          const ov::Tensor& tensor) {
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(node);
    if (!conv || input_idx != 1 || tensor.get_shape().size() != 4) {
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
std::vector<uint8_t> pack_conv2d_weights_oc4_typed(const ov::Tensor& tensor) {
    const auto shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 4, "GFX Conv2D weight pack expects OIHW rank-4 weights");
    const size_t c_out = shape[0];
    const size_t c_in = shape[1];
    const size_t k_h = shape[2];
    const size_t k_w = shape[3];
    const size_t channel_blocks = (c_out + 3u) / 4u;
    const auto* src = tensor.data<const T>();
    std::vector<T> packed(channel_blocks * c_in * k_h * k_w * 4u, T{});
    for (size_t block = 0; block < channel_blocks; ++block) {
        for (size_t ci = 0; ci < c_in; ++ci) {
            for (size_t kh = 0; kh < k_h; ++kh) {
                for (size_t kw = 0; kw < k_w; ++kw) {
                    const size_t dst_base = (((block * c_in + ci) * k_h + kh) * k_w + kw) * 4u;
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

std::vector<uint8_t> pack_conv2d_weights_oc4(const ov::Tensor& tensor) {
    const auto et = tensor.get_element_type();
    if (et == ov::element::f16) {
        return pack_conv2d_weights_oc4_typed<ov::float16>(tensor);
    }
    if (et == ov::element::f32) {
        return pack_conv2d_weights_oc4_typed<float>(tensor);
    }
    OPENVINO_THROW("GFX Conv2D weight pack: unsupported element type ", et);
}

std::vector<int64_t> evaluate_constant_source_i64(const ov::Output<ov::Node>& source, const char* what) {
    auto constant = ov::util::get_constant_from_source(source);
    OPENVINO_ASSERT(constant, "GFX MLIR: ", what, " must be Constant");
    return constant->cast_vector<int64_t>();
}

std::optional<std::vector<int64_t>> evaluate_optional_constant_source_i64(const ov::Output<ov::Node>& source) {
    auto constant = ov::util::get_constant_from_source(source);
    if (!constant) {
        return std::nullopt;
    }
    return constant->cast_vector<int64_t>();
}

struct RuntimeReduceInfo {
    ov::AxisSet axes;
    bool keep_dims = false;
};

std::optional<RuntimeReduceInfo> get_runtime_reduce_info(const std::shared_ptr<const ov::Node>& node) {
    if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceSum>(node)) {
        OPENVINO_ASSERT(reduce->reduction_axes_constant(), "GFX MLIR: ReduceSum axes must be constant");
        return RuntimeReduceInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
    }
    if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceMean>(node)) {
        OPENVINO_ASSERT(reduce->reduction_axes_constant(), "GFX MLIR: ReduceMean axes must be constant");
        return RuntimeReduceInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
    }
    if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceMax>(node)) {
        OPENVINO_ASSERT(reduce->reduction_axes_constant(), "GFX MLIR: ReduceMax axes must be constant");
        return RuntimeReduceInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
    }
    if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceMin>(node)) {
        OPENVINO_ASSERT(reduce->reduction_axes_constant(), "GFX MLIR: ReduceMin axes must be constant");
        return RuntimeReduceInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
    }
    if (auto reduce = ov::as_type_ptr<const ov::op::v1::ReduceProd>(node)) {
        OPENVINO_ASSERT(reduce->reduction_axes_constant(), "GFX MLIR: ReduceProd axes must be constant");
        return RuntimeReduceInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
    }
    if (auto reduce = ov::as_type_ptr<const ov::op::v4::ReduceL1>(node)) {
        OPENVINO_ASSERT(reduce->reduction_axes_constant(), "GFX MLIR: ReduceL1 axes must be constant");
        return RuntimeReduceInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
    }
    if (auto reduce = ov::as_type_ptr<const ov::op::v4::ReduceL2>(node)) {
        OPENVINO_ASSERT(reduce->reduction_axes_constant(), "GFX MLIR: ReduceL2 axes must be constant");
        return RuntimeReduceInfo{reduce->get_reduction_axes(), reduce->get_keep_dims()};
    }
    return std::nullopt;
}

int64_t normalize_slice_index(int64_t index, int64_t dim, bool is_begin) {
    if (index < 0) {
        index += dim;
    }
    if (is_begin) {
        return std::clamp<int64_t>(index, 0, dim);
    }
    return std::clamp<int64_t>(index, -1, dim);
}

void build_slice_runtime_spec(const std::shared_ptr<const ov::Node>& node,
                              const ov::Shape& in_shape,
                              const ov::Shape& out_shape,
                              std::vector<int32_t>& starts_full,
                              std::vector<int32_t>& steps_full) {
    const size_t rank = in_shape.size();
    OPENVINO_ASSERT(rank == out_shape.size(),
                    "GFX MLIR: rank-changing Slice/StridedSlice is not supported");
    starts_full.assign(rank, 0);
    steps_full.assign(rank, 1);

    if (auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node)) {
        auto starts = evaluate_constant_source_i64(slice->input_value(1), "Slice starts");
        auto ends = evaluate_optional_constant_source_i64(slice->input_value(2));
        auto steps = evaluate_constant_source_i64(slice->input_value(3), "Slice steps");
        std::vector<int64_t> axes;
        if (slice->get_input_size() > 4) {
            axes = evaluate_constant_source_i64(slice->input_value(4), "Slice axes");
        } else {
            axes.resize(starts.size());
            std::iota(axes.begin(), axes.end(), 0);
        }
        OPENVINO_ASSERT(starts.size() == steps.size() && starts.size() == axes.size(),
                        "GFX MLIR: Slice starts/ends/steps/axes size mismatch for stage ",
                        node->get_friendly_name());
        if (ends.has_value()) {
            OPENVINO_ASSERT(starts.size() == ends->size(),
                            "GFX MLIR: Slice starts/ends/steps/axes size mismatch for stage ",
                            node->get_friendly_name());
        }
        for (size_t i = 0; i < axes.size(); ++i) {
            int64_t axis = axes[i];
            if (axis < 0) {
                axis += static_cast<int64_t>(rank);
            }
            OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank,
                            "GFX MLIR: Slice axis out of range for stage ",
                            node->get_friendly_name());
            OPENVINO_ASSERT(steps[i] != 0,
                            "GFX MLIR: Slice zero step is not supported for stage ",
                            node->get_friendly_name());
            const auto dim = static_cast<int64_t>(in_shape[static_cast<size_t>(axis)]);
            starts_full[static_cast<size_t>(axis)] =
                static_cast<int32_t>(normalize_slice_index(starts[i], dim, true));
            steps_full[static_cast<size_t>(axis)] = static_cast<int32_t>(steps[i]);
        }
        return;
    }

    auto slice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node);
    OPENVINO_ASSERT(slice, "GFX MLIR: expected Slice/StridedSlice node");
    OPENVINO_ASSERT(std::all_of(slice->get_new_axis_mask().begin(),
                                slice->get_new_axis_mask().end(),
                                [](int64_t v) { return v == 0; }),
                    "GFX MLIR: StridedSlice new_axis_mask is not supported for stage ",
                    node->get_friendly_name());
    OPENVINO_ASSERT(std::all_of(slice->get_shrink_axis_mask().begin(),
                                slice->get_shrink_axis_mask().end(),
                                [](int64_t v) { return v == 0; }),
                    "GFX MLIR: StridedSlice shrink_axis_mask is not supported for stage ",
                    node->get_friendly_name());
    OPENVINO_ASSERT(std::all_of(slice->get_ellipsis_mask().begin(),
                                slice->get_ellipsis_mask().end(),
                                [](int64_t v) { return v == 0; }),
                    "GFX MLIR: StridedSlice ellipsis_mask is not supported for stage ",
                    node->get_friendly_name());

    auto begin = evaluate_constant_source_i64(slice->input_value(1), "StridedSlice begin");
    auto end = evaluate_optional_constant_source_i64(slice->input_value(2));
    std::vector<int64_t> strides(rank, 1);
    if (slice->get_input_size() > 3) {
        auto values = evaluate_constant_source_i64(slice->input_value(3), "StridedSlice strides");
        OPENVINO_ASSERT(values.size() <= rank,
                        "GFX MLIR: StridedSlice strides rank mismatch for stage ",
                        node->get_friendly_name());
        std::copy(values.begin(), values.end(), strides.begin());
    }
    OPENVINO_ASSERT(begin.size() <= rank && (!end.has_value() || end->size() <= rank),
                    "GFX MLIR: StridedSlice begin/end rank mismatch for stage ",
                    node->get_friendly_name());
    const auto& begin_mask = slice->get_begin_mask();
    const auto& end_mask = slice->get_end_mask();
    for (size_t axis = 0; axis < rank; ++axis) {
        const auto dim = static_cast<int64_t>(in_shape[axis]);
        const bool masked_begin = axis < begin_mask.size() && begin_mask[axis] != 0;
        const bool masked_end = axis < end_mask.size() && end_mask[axis] != 0;
        const int64_t step = strides[axis];
        OPENVINO_ASSERT(step != 0,
                        "GFX MLIR: StridedSlice zero step is not supported for stage ",
                        node->get_friendly_name());
        int64_t start = axis < begin.size() ? begin[axis] : 0;
        int64_t finish = end.has_value() && axis < end->size() ? (*end)[axis] : dim;
        start = masked_begin ? (step < 0 ? dim - 1 : 0) : normalize_slice_index(start, dim, true);
        finish = masked_end ? (step < 0 ? -1 : dim) : normalize_slice_index(finish, dim, false);
        (void)finish;
        starts_full[axis] = static_cast<int32_t>(start);
        steps_full[axis] = static_cast<int32_t>(step);
    }
}

bool slice_requires_runtime_indexing(const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }
    if (auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node)) {
        auto steps = evaluate_optional_constant_source_i64(slice->input_value(3));
        return steps && std::any_of(steps->begin(), steps->end(), [](int64_t step) { return step < 0; });
    }
    if (auto slice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node)) {
        if (slice->get_input_size() <= 3) {
            return false;
        }
        auto steps = evaluate_optional_constant_source_i64(slice->input_value(3));
        return steps && std::any_of(steps->begin(), steps->end(), [](int64_t step) { return step < 0; });
    }
    return false;
}

inline void normalize_operand_segment_sizes(mlir::ModuleOp module) {
    (void)module;
}

struct SplitPlan {
    ov::Shape input_shape;
    std::vector<size_t> split_sizes;
    int64_t axis_norm = 0;
    uint32_t axis_len = 0;
    uint32_t inner_stride = 1;
};

inline SplitPlan make_split_plan(const std::shared_ptr<const ov::Node>& node,
                                 const ov::Shape& input_shape,
                                 const std::vector<GpuTensor*>& outputs) {
    SplitPlan plan;
    plan.input_shape = input_shape;

    int64_t axis = 0;
    size_t parts = 0;
    bool is_split = false;
    if (auto s = ov::as_type_ptr<const ov::op::v1::Split>(node)) {
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(s->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "Split axis must be constant");
        axis = axis_const->cast_vector<int64_t>().at(0);
        parts = s->get_num_splits();
        is_split = true;
    } else if (auto vs = ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "VariadicSplit axis must be constant");
        axis = axis_const->cast_vector<int64_t>().at(0);
        auto lengths_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(lengths_const, "VariadicSplit lengths must be constant");
        auto lengths = lengths_const->cast_vector<int64_t>();
        plan.split_sizes.reserve(lengths.size());
        int64_t infer_index = -1;
        size_t known_sum = 0;
        for (size_t i = 0; i < lengths.size(); ++i) {
            const int64_t v = lengths[i];
            if (v < 0) {
                OPENVINO_ASSERT(v == -1, "VariadicSplit only -1 negative length is supported");
                OPENVINO_ASSERT(infer_index < 0, "VariadicSplit supports only one inferred -1 length");
                infer_index = static_cast<int64_t>(i);
                plan.split_sizes.push_back(0);
            } else {
                known_sum += static_cast<size_t>(v);
                plan.split_sizes.push_back(static_cast<size_t>(v));
            }
        }
        if (infer_index >= 0) {
            const auto axis_norm = normalize_axis(axis, input_shape.size(), "GFX MLIR: VariadicSplit");
            const size_t axis_len = input_shape[static_cast<size_t>(axis_norm)];
            OPENVINO_ASSERT(known_sum <= axis_len, "VariadicSplit known lengths exceed axis dimension");
            plan.split_sizes[static_cast<size_t>(infer_index)] = axis_len - known_sum;
        }
    }

    plan.axis_norm = normalize_axis(axis, input_shape.size(), "GFX MLIR: Split");
    plan.axis_len = static_cast<uint32_t>(input_shape[static_cast<size_t>(plan.axis_norm)]);

    if (is_split) {
        OPENVINO_ASSERT(parts > 0, "Split number of splits is zero");
        OPENVINO_ASSERT(plan.axis_len % parts == 0, "Split dimension not divisible by parts");
        plan.split_sizes.assign(parts, plan.axis_len / parts);
    }

    size_t sum = 0;
    for (auto s : plan.split_sizes) {
        sum += s;
    }
    OPENVINO_ASSERT(sum == plan.axis_len,
                    "Split sizes do not sum to axis length (",
                    sum,
                    " vs ",
                    plan.axis_len,
                    ")");
    OPENVINO_ASSERT(!plan.split_sizes.empty(), "Split sizes are empty");
    OPENVINO_ASSERT(outputs.size() == plan.split_sizes.size(),
                    "Split output count mismatch (expected ",
                    plan.split_sizes.size(),
                    ", got ",
                    outputs.size(),
                    ")");

    plan.inner_stride = 1;
    for (size_t d = static_cast<size_t>(plan.axis_norm) + 1; d < input_shape.size(); ++d) {
        plan.inner_stride *= input_shape[d];
    }

    return plan;
}

bool try_alias_contiguous_split_outputs(const std::shared_ptr<const ov::Node>& node,
                                        GpuTensor* input,
                                        const std::vector<GpuTensor*>& outputs,
                                        const char* stage_name) {
    if (!node || !input || !input->buf.valid() || input->shape.empty() || outputs.empty()) {
        return false;
    }
    for (const auto* out : outputs) {
        if (!out || !out->prefer_private) {
            return false;
        }
    }

    const auto plan = make_split_plan(node, input->shape, outputs);
    size_t outer = 1;
    for (size_t d = 0; d < static_cast<size_t>(plan.axis_norm); ++d) {
        outer *= input->shape[d];
    }
    if (outer != 1) {
        return false;
    }

    ov::element::Type element_type =
        input->expected_type == ov::element::dynamic ? input->buf.type : input->expected_type;
    if (element_type == ov::element::dynamic) {
        element_type = node->get_input_element_type(0);
    }
    OPENVINO_ASSERT(element_type != ov::element::dynamic,
                    "GFX MLIR: Split input element type is unknown for stage ",
                    stage_name);
    const size_t elem_size = element_type.size();
    OPENVINO_ASSERT(elem_size != 0, "GFX MLIR: Split element size is zero for stage ", stage_name);

    size_t axis_offset = 0;
    for (size_t oi = 0; oi < outputs.size(); ++oi) {
        auto* out = outputs[oi];
        if (!out) {
            axis_offset += plan.split_sizes[oi];
            continue;
        }

        ov::Shape out_shape = input->shape;
        out_shape[static_cast<size_t>(plan.axis_norm)] = plan.split_sizes[oi];
        const size_t byte_offset = axis_offset * static_cast<size_t>(plan.inner_stride) * elem_size;
        const size_t bytes = ov::shape_size(out_shape) * elem_size;
        OPENVINO_ASSERT(byte_offset + bytes <= input->buf.size,
                        "GFX MLIR: Split view exceeds input buffer for stage ",
                        stage_name,
                        " (offset=",
                        byte_offset,
                        ", bytes=",
                        bytes,
                        ", input=",
                        input->buf.size,
                        ")");

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

void propagate_view_metadata(GpuTensor& dst, const GpuTensor& src) {
    dst.gqa_broadcast_view = src.gqa_broadcast_view;
    dst.gqa_storage_shape = src.gqa_storage_shape;
    dst.gqa_kv_heads = src.gqa_kv_heads;
}

bool alias_tensor_view(GpuTensor& dst, const GpuTensor& src, const ov::Shape& shape, ov::element::Type element_type) {
    if (!src.buf.valid()) {
        return false;
    }
    dst.buf = src.buf;
    dst.buf.external = true;
    dst.buf.owned = false;
    dst.shape = shape;
    dst.expected_type = element_type == ov::element::dynamic
                            ? (src.expected_type == ov::element::dynamic ? src.buf.type : src.expected_type)
                            : element_type;
    propagate_view_metadata(dst, src);
    if (!src.i64_values.empty() && src.i64_values.size() == ov::shape_size(shape)) {
        dst.i64_values = src.i64_values;
    }
    return true;
}

bool try_alias_same_shape_unary_view(GpuTensor* input,
                                     const std::vector<GpuTensor*>& outputs,
                                     const ov::Shape& out_shape,
                                     ov::element::Type output_type) {
    if (!input || outputs.size() != 1 || !outputs[0] || input->shape != out_shape) {
        return false;
    }
    return alias_tensor_view(*outputs[0], *input, out_shape, output_type);
}

bool output_consumers_are_reshape_or_sdpa(const std::shared_ptr<const ov::Node>& node) {
    if (!node || node->get_output_size() != 1) {
        return false;
    }
    bool has_consumer = false;
    for (const auto& target : node->output(0).get_target_inputs()) {
        has_consumer = true;
        const auto* consumer = target.get_node();
        if (dynamic_cast<const ov::op::v1::Reshape*>(consumer) ||
            dynamic_cast<const ov::op::v13::ScaledDotProductAttention*>(consumer)) {
            continue;
        }
        return false;
    }
    return has_consumer;
}

bool try_alias_gqa_broadcast_view(const std::shared_ptr<const ov::Node>& node,
                                  GpuTensor* input,
                                  const std::vector<GpuTensor*>& outputs) {
    if (!node || !input || !input->buf.valid() || input->shape.size() != 5 ||
        outputs.size() != 1 || !outputs[0] || outputs[0]->shape.size() != 5 ||
        !output_consumers_are_reshape_or_sdpa(node)) {
        return false;
    }
    if (!ov::as_type_ptr<const ov::op::v3::Broadcast>(node) &&
        !ov::as_type_ptr<const ov::op::v1::Broadcast>(node)) {
        return false;
    }

    const auto& in_shape = input->shape;
    const auto& out_shape = outputs[0]->shape;
    if (in_shape[2] != 1 || out_shape[2] <= 1 ||
        in_shape[0] != out_shape[0] ||
        in_shape[1] != out_shape[1] ||
        in_shape[3] != out_shape[3] ||
        in_shape[4] != out_shape[4]) {
        return false;
    }

    auto* out = outputs[0];
    out->buf = input->buf;
    out->buf.external = true;
    out->buf.owned = false;
    out->expected_type = input->expected_type == ov::element::dynamic ? input->buf.type : input->expected_type;
    out->gqa_broadcast_view = true;
    out->gqa_storage_shape = ov::Shape{in_shape[0], in_shape[1], in_shape[3], in_shape[4]};
    out->gqa_kv_heads = in_shape[1];
    return true;
}

}  // namespace

MlirStage::MlirStage(const std::shared_ptr<const ov::Node>& node)
    : m_node(node),
      m_name(node ? node->get_friendly_name() : std::string("mlir_stage")),
      m_type(node ? node->get_type_name() : std::string("Unknown")) {
    if (node && node->get_output_partial_shape(0).is_static()) {
        m_output_shape = node->get_output_shape(0);
    }
    m_is_view_op = select_tensor_layout_plan(m_type, m_node).view_only;
}

const std::vector<int64_t>& MlirStage::cached_constant_i64_input(size_t input_idx, const char* what) {
    OPENVINO_ASSERT(m_node && input_idx < m_node->get_input_size(),
                    "GFX MLIR: constant i64 input index is out of range for stage ",
                    m_name);
    if (m_cached_constant_i64_inputs.size() <= input_idx) {
        m_cached_constant_i64_inputs.resize(input_idx + 1);
        m_cached_constant_i64_input_valid.resize(input_idx + 1, false);
    }
    if (!m_cached_constant_i64_input_valid[input_idx]) {
        m_cached_constant_i64_inputs[input_idx] =
            evaluate_constant_source_i64(m_node->input_value(input_idx), what);
        m_cached_constant_i64_input_valid[input_idx] = true;
    }
    return m_cached_constant_i64_inputs[input_idx];
}

void MlirStage::apply_kernel_metadata(const KernelRuntimeMetadata& meta,
                                      size_t scalar_inputs) {
    if (!meta.valid) {
        return;
    }
    m_parallel_cfg = meta.dispatch;
    if (meta.force_single_dispatch) {
        m_force_single_dispatch = true;
    }
    apply_kernel_runtime_binding_state(
        make_kernel_runtime_binding_state({},
                                          meta.kernel_input_arg_count,
                                          std::move(meta.operands.operand_kinds),
                                          std::move(meta.operands.operand_arg_indices),
                                          std::move(meta.operands.scalar_args)));
    if (m_kernel_binding.operand_kinds.empty() && scalar_inputs != 0) {
        OPENVINO_ASSERT(m_kernel_binding.scalar_args.size() == scalar_inputs,
                        "GFX MLIR: kernel scalar args mismatch for ",
                        m_name,
                        " (expected ",
                        scalar_inputs,
                        ", got ",
                        m_kernel_binding.scalar_args.size(),
                        ")");
    }
}

void MlirStage::apply_kernel_runtime_binding_state(KernelRuntimeBindingState binding) {
    m_kernel_binding = std::move(binding);
    m_kernel_binding_owned_by_source_plan = false;
}

void MlirStage::apply_source_plan_kernel_runtime_binding_state(KernelRuntimeBindingState binding) {
    m_kernel_binding = std::move(binding);
    m_kernel_binding_owned_by_source_plan = true;
}

KernelRuntimeBindingState MlirStage::resolved_kernel_runtime_binding_state() const {
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

void MlirStage::compile_prebuilt_kernel_source(const KernelSource& source,
                                               KernelRuntimeBindingState binding,
                                               std::string_view stage_kind) {
    std::string log;
    m_last_compiled_kernel_entry_point = source.entry_point;
    try {
        m_kernel = compile_kernel(source, &log);
    } catch (const std::exception& e) {
        OPENVINO_THROW("GFX MLIR: failed to compile ",
                       stage_kind,
                       " stage ",
                       m_name,
                       " (",
                       m_type,
                       "): ",
                       e.what());
    }
    OPENVINO_ASSERT(m_kernel,
                    "GFX MLIR: failed to compile ",
                    stage_kind,
                    " stage ",
                    m_name,
                    " (",
                    m_type,
                    "): ",
                    log);
    m_kernel->prepare_runtime_artifacts();
    apply_kernel_runtime_binding_state(std::move(binding));
}

void MlirStage::annotate_msl_custom_kernel_binding_plan(mlir::ModuleOp module,
                                                        std::string_view stage_type,
                                                        std::string_view entry_point,
                                                        std::vector<size_t> tensor_input_indices) {
    auto plan = make_msl_runtime_binding_plan_for_custom_kernel(stage_type,
                                                                entry_point,
                                                                std::move(tensor_input_indices));
    OPENVINO_ASSERT(plan.valid(),
                    "GFX MLIR: custom-kernel ABI manifest is invalid for stage ",
                    m_name,
                    " (",
                    stage_type,
                    ", ",
                    entry_point,
                    ")");
    OPENVINO_ASSERT(annotate_msl_module_with_runtime_binding_plan(module, plan),
                    "GFX MLIR: failed to annotate MSL binding manifest for stage ",
                    m_name,
                    " (",
                    stage_type,
                    ", ",
                    entry_point,
                    ")");
}

void MlirStage::annotate_msl_custom_kernel_binding_plan(mlir::ModuleOp module,
                                                        std::string_view stage_type,
                                                        std::string_view entry_point,
                                                        std::vector<size_t> tensor_input_indices,
                                                        std::vector<int32_t> scalar_args) {
    auto plan = make_msl_runtime_binding_plan_for_custom_kernel(stage_type,
                                                                entry_point,
                                                                std::move(tensor_input_indices),
                                                                std::move(scalar_args));
    OPENVINO_ASSERT(plan.valid(),
                    "GFX MLIR: custom-kernel scalar ABI manifest is invalid for stage ",
                    m_name,
                    " (",
                    stage_type,
                    ", ",
                    entry_point,
                    ")");
    OPENVINO_ASSERT(annotate_msl_module_with_runtime_binding_plan(module, plan),
                    "GFX MLIR: failed to annotate MSL scalar binding manifest for stage ",
                    m_name,
                    " (",
                    stage_type,
                    ", ",
                    entry_point,
                    ")");
}

void MlirStage::compile_from_plan(MlirKernelPlanContext& plan_ctx,
                                  mlir::ModuleOp module,
                                  const char* stage_kind) {
    auto& build_info = plan_ctx.build_info;
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
        plan_ctx.output_args != 0 ? plan_ctx.output_args : (m_node ? m_node->get_output_size() : 0);
    apply_kernel_runtime_binding_state(
        make_kernel_runtime_binding_state(std::move(build_info.mapping.mapping.kernel_inputs),
                                          buffer_inputs,
                                          {},
                                          {}));
    KernelSource src = build_info.plan.to_source();
    src.signature.output_arg_count = static_cast<uint32_t>(output_arg_count);
    if (backend_kind() == GpuBackend::Metal && is_conv_like()) {
        gfx_attach_mpsrt_conv_const_tensors(src, m_node);
    }
    if (src.module) {
        normalize_operand_segment_sizes(src.module);
        if (is_vulkan_backend()) {
            auto fixed_arg_count = src.module->getAttrOfType<mlir::IntegerAttr>("gfx.fixed_arg_count");
            auto func = get_entry_func(src.module);
            if (fixed_arg_count) {
                const size_t total_buffer_args = static_cast<size_t>(std::max<int64_t>(fixed_arg_count.getInt(), 0));
                const size_t output_arg_count = plan_ctx.output_args != 0 ? plan_ctx.output_args
                                                                          : (m_node ? m_node->get_output_size() : 0);
                const size_t non_output_buffer_args =
                    total_buffer_args >= output_arg_count ? (total_buffer_args - output_arg_count) : 0;
                if (func) {
                    const size_t annotate_count =
                        std::min(non_output_buffer_args, static_cast<size_t>(func.getNumArguments()));
                    for (size_t arg_idx = 0; arg_idx < annotate_count; ++arg_idx) {
                        func.setArgAttr(static_cast<unsigned>(arg_idx),
                                        "gfx.kernel_runtime_arg_index",
                                        mlir::IntegerAttr::get(mlir::IntegerType::get(src.module.getContext(), 32),
                                                               static_cast<int32_t>(arg_idx)));
                    }
                }
                src.module->setAttr("gfx.kernel_output_arg_count",
                                    mlir::IntegerAttr::get(mlir::IntegerType::get(src.module.getContext(), 32),
                                                           static_cast<int32_t>(output_arg_count)));
            }
        }
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
    } catch (const std::exception& e) {
        OPENVINO_THROW("GFX MLIR: failed to compile ",
                       stage_kind,
                       " stage ",
                       m_name,
                       " (",
                       m_type,
                       "): ",
                       e.what());
    }
    OPENVINO_ASSERT(m_kernel,
                    "GFX MLIR: failed to compile ",
                    stage_kind,
                    " stage ",
                    m_name,
                    " (",
                    m_type,
                    "): ",
                    log);
    m_kernel->prepare_runtime_artifacts();
    const std::string runtime_metadata_entry_point =
        m_last_compiled_kernel_entry_point.empty() ? src.entry_point : m_last_compiled_kernel_entry_point;
    if (m_kernel_binding_owned_by_source_plan) {
        return;
    }
    if (src.module) {
        auto runtime_meta = extract_kernel_runtime_metadata(src.module,
                                                           output_arg_count,
                                                           buffer_inputs,
                                                           runtime_metadata_entry_point);
        apply_kernel_metadata(runtime_meta, scalar_inputs);
    } else if (module) {
        auto runtime_meta = build_info.runtime_metadata(m_node, plan_ctx.output_args);
        apply_kernel_metadata(runtime_meta, scalar_inputs);
    }
}

void MlirStage::init(GpuBufferManager* buffer_manager) {
    m_buffer_manager = buffer_manager;
}

void MlirStage::compile(GpuBufferManager* buffer_manager) {
    auto& ctx = gfx_mlir_context();
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
    auto apply_spirv_kernel_binding_adapter = [&](mlir::ModuleOp module,
                                                  KernelRuntimeBindingState binding) {
        OPENVINO_ASSERT(is_vulkan_backend(),
                        "GFX MLIR: SPIR-V kernel operand adapter used on non-Vulkan backend");
        if (module) {
            annotate_kernel_operand_abi_attrs_for_spirv_adapter(module, binding);
        }
        apply_kernel_runtime_binding_state(std::move(binding));
    };
    if (auto desc = static_matmul_desc_for_node(m_node)) {
        desc->has_activation = m_has_activation;
        desc->activation = m_activation;
        desc->alpha = m_activation_alpha;
        m_matmul_reduction_threads = gfx_matmul_parallel_reduction_threads(*desc);
    }
    if (m_type == "RMS" && m_node && m_node->get_input_partial_shape(0).rank().is_static()) {
        const auto pshape = m_node->get_input_partial_shape(0);
        const auto rank = pshape.rank().get_length();
        if (rank > 0 && pshape[rank - 1].is_static()) {
            m_rms_hidden = static_cast<uint32_t>(pshape[rank - 1].get_length());
            m_rms_reduction_threads = gfx_rms_parallel_reduction_threads(m_rms_hidden);
        }
    }
    std::optional<CompressedMatMulInfo> compressed_matmul_info;
    if (!is_vulkan_backend()) {
        compressed_matmul_info = detect_compressed_matmul_weights(m_node);
    }
    if (m_node) {
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
            if (compressed_matmul_info && i == 1) {
                continue;
            }
            auto const_tensor = gfx_evaluate_constant_source_tensor(m_node->input_value(i));
            if (!const_tensor.has_value()) {
                continue;
            }
            if (!const_cache_checked) {
                OPENVINO_ASSERT(m_buffer_manager,
                                "GFX MLIR: const buffer manager is required for constants (stage ",
                                m_name,
                                ")");
                OPENVINO_ASSERT(m_buffer_manager->supports_const_cache(),
                                "GFX MLIR: const cache must be supported for stage ",
                                m_name);
                const_cache_checked = true;
            }
            if (m_const_buffers->present[i] && m_const_buffers->buffers[i].buf.valid()) {
                continue;
            }
            std::vector<ov::float16> packed_f16;
            std::vector<uint8_t> packed_conv_weights;
            const void* const_data = const_tensor->data();
            size_t bytes = const_tensor->get_byte_size();
            auto et = const_tensor->get_element_type();
            if (backend_kind() == GpuBackend::Metal && should_pack_matmul_const_input_as_f16(m_node, i, *const_tensor)) {
                const size_t original_bytes = bytes;
                packed_f16 = pack_f32_tensor_as_f16(*const_tensor);
                const_data = packed_f16.data();
                bytes = packed_f16.size() * sizeof(ov::float16);
                et = ov::element::f16;
                increment_compile_counter("matmul_const_f32_to_f16_pack_count");
                increment_compile_counter("matmul_const_f32_to_f16_original_bytes", original_bytes);
                increment_compile_counter("matmul_const_f32_to_f16_packed_bytes", bytes);
            }
            if (backend_kind() == GpuBackend::Metal && should_pack_conv2d_const_weights_oc4(m_node, i, *const_tensor)) {
                const size_t original_bytes = bytes;
                packed_conv_weights = pack_conv2d_weights_oc4(*const_tensor);
                et = const_tensor->get_element_type();
                m_conv_weight_storage_type = et;
                const_data = packed_conv_weights.data();
                bytes = packed_conv_weights.size();
                m_conv_weights_packed_oc4 = true;
                increment_compile_counter("conv2d_const_oc4_pack_count");
                increment_compile_counter("conv2d_const_oc4_original_bytes", original_bytes);
                increment_compile_counter("conv2d_const_oc4_packed_bytes", bytes);
            }
            if (gfx_log_debug_enabled() && et == ov::element::f32 && bytes >= sizeof(float)) {
                const float* vals = const_tensor->data<const float>();
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
                key << m_name
                    << "/const/"
                    << i
                    << "/"
                    << et.get_type_name()
                    << "/"
                    << bytes
                    << "/"
                    << hash;
                GpuBuffer buf = m_buffer_manager->wrap_const(key.str(), const_data, bytes, et);
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
    if (compressed_matmul_info) {
        OPENVINO_ASSERT(m_buffer_manager,
                        "GFX MLIR: const buffer manager is required for compressed MatMul stage ",
                        m_name);
        const auto caps = query_parallelism_caps(m_buffer_manager);
        m_matmul_reduction_threads = compressed_matmul_parallel_reduction_threads(*compressed_matmul_info, caps);
        m_compressed_matmul_output_block =
            compressed_matmul_output_block(*compressed_matmul_info, caps, m_matmul_reduction_threads);
        m_compressed_matmul_n = static_cast<uint32_t>(compressed_matmul_info->n);
        const std::vector<uint8_t> packed_weights =
            pack_compressed_matmul_weights_for_output_block(*compressed_matmul_info, m_compressed_matmul_output_block);
        const std::vector<uint8_t> packed_scales = pack_compressed_matmul_scales(*compressed_matmul_info);
        if (gfx_log_debug_enabled()) {
            std::ostringstream oss;
            oss << "compressed MatMul reduction threads=" << m_matmul_reduction_threads
                << " output_block=" << m_compressed_matmul_output_block
                << " K=" << compressed_matmul_info->k
                << " N=" << compressed_matmul_info->n
                << " parts=" << compressed_matmul_info->parts.size()
                << " packed_weight_bytes=" << packed_weights.size()
                << " packed_scale_bytes=" << packed_scales.size()
                << " max_threads=" << caps.max_total_threads_per_group
                << " simd=" << std::max(caps.subgroup_size, caps.preferred_simd_width);
            gfx_log_debug("MLIRConst") << oss.str();
        }

        auto compressed_source_plan =
            make_compressed_matmul_msl_kernel_source_plan(*compressed_matmul_info,
                                                          m_matmul_reduction_threads,
                                                          m_compressed_matmul_output_block);
        OPENVINO_ASSERT(compressed_source_plan.valid(),
                        "GFX Metal compressed MatMul: source plan is invalid for stage ",
                        m_name);
        compile_prebuilt_kernel_source(compressed_source_plan.source,
                                       compressed_source_plan.binding.kernel_runtime_binding(),
                                       "compressed MatMul");

        m_kernel_extra_inputs.clear();
        std::ostringstream weight_suffix;
        weight_suffix << "compressed_matmul/packed_weights/"
                      << compressed_matmul_info->weights->get_element_type().get_type_name()
                      << "/block"
                      << m_compressed_matmul_output_block;
        m_kernel_extra_inputs.push_back(make_hashed_kernel_bytes_param_tensor(*m_buffer_manager,
                                                                             m_name,
                                                                             weight_suffix.str(),
                                                                             packed_weights.data(),
                                                                             packed_weights.size(),
                                                                             ov::element::u8,
                                                                             ov::Shape{packed_weights.size()}));
        m_kernel_extra_inputs.push_back(make_hashed_kernel_bytes_param_tensor(*m_buffer_manager,
                                                                             m_name,
                                                                             "compressed_matmul/packed_scale",
                                                                             packed_scales.data(),
                                                                             packed_scales.size(),
                                                                             compressed_matmul_info->scale
                                                                                 ->get_element_type(),
                                                                             ov::Shape{compressed_matmul_info->n,
                                                                                       compressed_matmul_info->groups,
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
            stride0 = compute_broadcast_element_strides(m_node->get_input_shape(0), out_shape);
        }
        if (stride1.empty() && m_node) {
            stride1 = compute_broadcast_element_strides(m_node->get_input_shape(1), out_shape);
        }

        auto broadcast_payload = make_binary_broadcast_runtime_param_payload(*m_buffer_manager,
                                                                            m_name,
                                                                            out_shape,
                                                                            std::move(stride0),
                                                                            std::move(stride1));
        m_kernel_extra_inputs.insert(m_kernel_extra_inputs.end(),
                                     std::make_move_iterator(broadcast_payload.extra_inputs.begin()),
                                     std::make_move_iterator(broadcast_payload.extra_inputs.end()));
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MLIRExec") << "Prepared eltwise broadcast meta: dims=" << out_shape.size()
                                      << " extras=" << m_kernel_extra_inputs.size();
        }
    };
    // Only for classic binary eltwise ops; keep list tight to avoid surprises.
    const bool uses_compact_bias_add_kernel =
        is_vulkan_backend() && m_type == "Add" && m_node && is_bias_broadcast_add(m_node) &&
        !has_absorbed_input_transpose();
    if (!uses_compact_bias_add_kernel &&
        ((m_type == "Add" || m_type == "Subtract" || m_type == "Multiply" ||
         m_type == "Divide") ||
        m_type == "Power" || m_type == "Mod" || m_type == "FloorMod" || m_type == "Minimum" ||
        m_type == "Maximum" || m_type == "Equal" || m_type == "NotEqual" || m_type == "Less" ||
        m_type == "Greater" || m_type == "LessEqual" || m_type == "GreaterEqual" ||
        m_type == "LogicalAnd" || m_type == "LogicalOr" || m_type == "LogicalXor" ||
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
            const bool conv_like = (m_type == "Convolution" || m_type == "GroupConvolution");
            size_t out_rank = m_bias_params.shape.size();
            if (m_node) {
                const auto& pshape = m_node->get_output_partial_shape(0);
                if (pshape.rank().is_static()) {
                    out_rank = static_cast<size_t>(pshape.rank().get_length());
                }
            }
            GpuTensor tensor = make_bias_runtime_param_tensor(*m_buffer_manager,
                                                              m_name,
                                                              m_bias_params.values,
                                                              m_bias_params.shape,
                                                              m_bias_params.element_type,
                                                              out_et,
                                                              out_rank,
                                                              conv_like);
            m_kernel_extra_inputs.push_back(std::move(tensor));
        }
    }
    if (m_has_bn) {
        const size_t channels = m_bn_params.gamma.size();
        if (channels) {
            auto bn_payload = make_batchnorm_scale_bias_runtime_param_payload(*m_buffer_manager,
                                                                             m_name,
                                                                             m_bn_params.gamma,
                                                                             m_bn_params.beta,
                                                                             m_bn_params.mean,
                                                                             m_bn_params.var,
                                                                             m_bn_params.epsilon,
                                                                             out_et);
            m_kernel_extra_inputs.insert(m_kernel_extra_inputs.end(),
                                         std::make_move_iterator(bn_payload.extra_inputs.begin()),
                                         std::make_move_iterator(bn_payload.extra_inputs.end()));
        }
    }
    // Convolution: build fixed extra inputs (bias + BN + params) to align with
    // MSL/Spir-V kernel signatures and avoid runtime CPU-side packing.
    if (m_node && is_conv_like()) {
        m_kernel_extra_inputs.clear();
        const auto& out_shape = m_node->get_output_shape(0);
        const auto& in_shape = m_node->get_input_shape(0);
        const size_t in_rank = in_shape.size();
        if (in_rank == 5) {
            // 3D convolution (NCDHW) — params only.
            OPENVINO_ASSERT(out_shape.size() == 5, "GFX MLIR: Conv3D expects NCDHW output");
            const auto& w = m_node->get_input_shape(1);
            auto conv = std::dynamic_pointer_cast<const ov::op::v1::Convolution>(m_node);
            OPENVINO_ASSERT(conv, "GFX MLIR: Conv3D node cast failed");
            auto conv3d_payload = make_conv3d_runtime_param_payload(*m_buffer_manager,
                                                                     m_name,
                                                                     in_shape,
                                                                     out_shape,
                                                                     w,
                                                                     conv->get_strides(),
                                                                     conv->get_dilations(),
                                                                     conv->get_pads_begin(),
                                                                     conv->get_pads_end());
            m_kernel_extra_inputs = std::move(conv3d_payload.extra_inputs);
            // No bias/BN for Conv3D path yet. The custom-kernel manifest owns
            // output-before-params argument ordering for Metal.
        } else {
            OPENVINO_ASSERT(out_shape.size() == 4, "GFX MLIR: Conv expects NCHW output");
            OPENVINO_ASSERT(in_shape.size() == 4, "GFX MLIR: Conv expects NCHW input");
            m_conv_output_channels_per_thread = 1;
            m_conv_output_width_per_thread = 1;
            ov::element::Type et = out_et == ov::element::dynamic ? m_node->get_output_element_type(0) : out_et;
        if (et == ov::element::dynamic) {
            et = ov::element::f32;
        }
        uint32_t groups = 1;
        uint32_t C_out = 0, C_in_pg = 0, C_out_pg = 0, kH = 0, kW = 0;
        uint32_t strideH = 1, strideW = 1, dilationH = 1, dilationW = 1;
        uint32_t padTop = 0, padLeft = 0, padBottom = 0, padRight = 0;
        float epsilon = 0.f;
        if (auto conv = std::dynamic_pointer_cast<const ov::op::v1::Convolution>(m_node)) {
            const auto& w = conv->get_input_shape(1);
            C_out = static_cast<uint32_t>(w.at(0));
            C_in_pg = static_cast<uint32_t>(w.at(1));
            C_out_pg = C_out;
            const uint32_t C_in = static_cast<uint32_t>(in_shape.at(1));
            if (C_in_pg != 0 && C_in % C_in_pg == 0) {
                groups = C_in / C_in_pg;
                if (groups == 0) groups = 1;
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
        } else if (auto gconv = std::dynamic_pointer_cast<const ov::op::v1::GroupConvolution>(m_node)) {
            const auto& w = gconv->get_input_shape(1);  // [G, O_pg, I_pg, kH, kW]
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
            OPENVINO_THROW("GFX MLIR: unsupported conv-like op ", m_type);
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
            desc.outW = static_cast<uint32_t>(out_shape.at(3));
            m_conv_output_channels_per_thread = gfx_conv2d_output_channel_block(desc);
            m_conv_output_width_per_thread = gfx_conv2d_output_width_block(desc);
        }
        // Prepare bias + BN buffers even if unused to keep argument order stable.
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
        auto conv_payload = make_conv2d_runtime_param_payload(*m_buffer_manager,
                                                              m_name,
                                                              in_shape,
                                                              out_shape,
                                                              et,
                                                              C_out,
                                                              groups,
                                                              C_in_pg,
                                                              C_out_pg,
                                                              kH,
                                                              kW,
                                                              strideH,
                                                              strideW,
                                                              dilationH,
                                                              dilationW,
                                                              padTop,
                                                              padLeft,
                                                              padBottom,
                                                              padRight,
                                                              m_has_bias,
                                                              m_has_bn,
                                                              m_has_activation ? kernel_activation_code(m_activation) : 0u,
                                                              m_activation_alpha,
                                                              epsilon,
                                                              bias,
                                                              gamma,
                                                              beta,
                                                              mean,
                                                              var);
        OPENVINO_ASSERT(conv_payload.extra_inputs.size() == 6,
                        "GFX MLIR: Conv params payload must contain bias, BN and metadata buffers for ",
                        m_name);
        m_kernel_extra_inputs = std::move(conv_payload.extra_inputs);
        }
    }
    if (m_type == "MaxPool" || m_type == "AvgPool") {
        m_kernel_extra_inputs.clear();
        auto maxpool = std::dynamic_pointer_cast<const ov::op::util::MaxPoolBase>(m_node);
        auto avgpool = std::dynamic_pointer_cast<const ov::op::v1::AvgPool>(m_node);
        OPENVINO_ASSERT(maxpool || avgpool, "GFX MLIR: pool node cast failed");
        const auto in = m_node->get_input_shape(0);
        const auto out = m_node->get_output_shape(0);
        OPENVINO_ASSERT(in.size() == 4 && out.size() == 4, "GFX MLIR: pool expects NCHW");
        const auto& kernel = maxpool ? maxpool->get_kernel() : avgpool->get_kernel();
        const auto& strides = maxpool ? maxpool->get_strides() : avgpool->get_strides();
        const auto& pads_begin = maxpool ? maxpool->get_pads_begin() : avgpool->get_pads_begin();
        const auto& pads_end = maxpool ? maxpool->get_pads_end() : avgpool->get_pads_end();
        ov::Strides dilations(kernel.size(), 1);
        if (maxpool) {
            if (auto p = std::dynamic_pointer_cast<const ov::op::v8::MaxPool>(m_node)) {
                dilations = p->get_dilations();
            } else if (auto p = std::dynamic_pointer_cast<const ov::op::v14::MaxPool>(m_node)) {
                dilations = p->get_dilations();
            }
        }
        const bool is_avg = avgpool != nullptr;
        const bool exclude_pad = is_avg ? avgpool->get_exclude_pad() : true;
        auto pool_payload = make_pool_runtime_param_payload(*m_buffer_manager,
                                                            m_name,
                                                            in,
                                                            out,
                                                            kernel,
                                                            strides,
                                                            pads_begin,
                                                            pads_end,
                                                            dilations,
                                                            is_avg,
                                                            exclude_pad);
        m_kernel_extra_inputs = std::move(pool_payload.extra_inputs);
    }
    if (m_type == "ShapeOf") {
        OPENVINO_ASSERT(m_node, "GFX MLIR: ShapeOf stage requires node");
        const auto input_pshape = m_node->get_input_partial_shape(0);
        OPENVINO_ASSERT(input_pshape.rank().is_static(), "GFX MLIR: ShapeOf input rank must be static");
        const size_t rank = static_cast<size_t>(input_pshape.rank().get_length());
        const auto output_et = m_node->get_output_element_type(0);
        OPENVINO_ASSERT(output_et == ov::element::i32 || output_et == ov::element::i64,
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

        auto shapeof_payload =
            make_shapeof_runtime_param_payload(*m_buffer_manager, m_name, compile_shape, output_et);
        const auto shapeof_scalars = shapeof_payload.scalar_args;
        m_kernel_extra_inputs = std::move(shapeof_payload.extra_inputs);
        if (!is_vulkan_backend()) {
            auto shapeof_plan = make_kernel_runtime_binding_plan_for_custom_kernel(
                "ShapeOf",
                "shapeof_kernel",
                {0},
                shapeof_scalars);
            OPENVINO_ASSERT(shapeof_plan.valid,
                            "GFX MLIR: ShapeOf custom-kernel ABI manifest is invalid for stage ",
                            m_name);
            apply_kernel_runtime_binding_state(shapeof_plan.runtime_binding);
        } else {
            auto shapeof_plan = make_kernel_runtime_binding_plan_for_custom_kernel(
                "ShapeOf",
                "shapeof_kernel",
                {0},
                shapeof_scalars,
                GfxKernelBackendDomain::Spirv,
                GfxKernelStorageKind::Buffer,
                "spirv:buffer:");
            OPENVINO_ASSERT(shapeof_plan.valid,
                            "GFX MLIR: ShapeOf SPIR-V custom-kernel ABI manifest is invalid for stage ",
                            m_name);
            apply_kernel_runtime_binding_state(shapeof_plan.runtime_binding);
        }
    }
    if (m_type == "GfxSDPAWithCausalMask") {
        if (is_vulkan_backend()) {
            OPENVINO_THROW("GFX Vulkan SDPA causal mask fusion is not enabled yet");
        }
        const auto et = m_node ? m_node->get_output_element_type(0) : ov::element::dynamic;
        auto sdpa_source_plan = make_causal_sdpa_msl_kernel_source_plan(et);
        OPENVINO_ASSERT(sdpa_source_plan.valid(),
                        "GFX Metal SDPA causal mask source plan is invalid for stage ",
                        m_name);
        compile_prebuilt_kernel_source(sdpa_source_plan.source,
                                       sdpa_source_plan.binding.kernel_runtime_binding(),
                                       "SDPA causal mask");
        m_force_single_dispatch = false;
        return;
    }
    if (m_type == "ScaledDotProductAttention") {
        if (is_vulkan_backend()) {
            OPENVINO_THROW("GFX Vulkan SDPA: native ScaledDotProductAttention is not enabled yet");
        }
        const auto et = m_node ? m_node->get_output_element_type(0) : ov::element::dynamic;
        const bool has_mask = m_node && m_node->get_input_size() >= 4;
        auto sdpa_source_plan = make_sdpa_msl_kernel_source_plan(et, has_mask);
        OPENVINO_ASSERT(sdpa_source_plan.valid(), "GFX Metal SDPA source plan is invalid for stage ", m_name);
        compile_prebuilt_kernel_source(sdpa_source_plan.source,
                                       sdpa_source_plan.binding.kernel_runtime_binding(),
                                       "SDPA");
        m_force_single_dispatch = false;
        return;
    }
    if (m_type == "Softmax" || m_type == "LogSoftmax" ||
        m_type == "Split" || m_type == "VariadicSplit") {
        return;
    }
    const auto optimization_plan = stage_optimization_plan();
    if (should_skip_generic_kernel_compile(optimization_plan)) {
        return;
    }
    KernelPlan plan = [&]() {
        if (is_vulkan_backend() && m_type == "Add" && is_bias_broadcast_add(m_node) &&
            !has_absorbed_input_transpose()) {
            auto module = build_mlir_add_from_node(m_node, ctx, m_input_transforms);
            return KernelPlan(module, "binary_bias_add", 3);
        }
        if (m_type == "Add" && has_absorbed_input_transpose()) {
            auto module = build_mlir_add_from_node(m_node, ctx, m_input_transforms);
            return KernelPlan(module, resolve_entry_point(module, {}, "gfx_kernel"), 0);
        }
        if (m_type == "Convolution" && has_absorbed_input_transpose()) {
            auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
            OPENVINO_ASSERT(conv, "GFX MLIR: expected Convolution node for absorbed transpose");
            auto module = build_mlir_conv2d_from_node(conv, ctx, input_transform(0));
            return KernelPlan(module, resolve_entry_point(module, {}, "conv2d_main"), 0);
        }
        if (m_type == "GroupConvolution" && has_absorbed_input_transpose()) {
            auto gconv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(m_node);
            OPENVINO_ASSERT(gconv, "GFX MLIR: expected GroupConvolution node for absorbed transpose");
            auto module = build_mlir_group_conv2d_from_node(gconv, ctx, input_transform(0));
            return KernelPlan(module, resolve_entry_point(module, {}, "group_conv2d_main"), 0);
        }
        MlirKernelPlanBuilder plan_builder;
        KernelSpec spec(m_node, 0);
        return plan_builder.build_plan(spec, ctx);
    }();
    auto module = plan.module();
    auto should_skip_vulkan_conv_parallel = [&]() {
        if (!is_vulkan_backend() || m_type != "Convolution" || has_absorbed_input_transpose() || !m_node) {
            return false;
        }
        if (optimization_plan.conv.kind != GfxConvRouteKind::None) {
            return false;
        }
        auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
        if (!conv || conv->get_input_size() != 2 || conv->get_output_size() != 1) {
            return false;
        }
        const auto& in_shape = conv->get_input_shape(0);
        const auto& w_shape = conv->get_input_shape(1);
        const auto& out_shape = conv->get_output_shape(0);
        if (in_shape.size() != 4 || w_shape.size() != 4 || out_shape.size() != 4) {
            return false;
        }
        return out_shape[1] == 1 &&
               w_shape[2] == 1 && w_shape[3] == 1 &&
               conv->get_strides().at(0) == 1 && conv->get_strides().at(1) == 1 &&
               conv->get_dilations().at(0) == 1 && conv->get_dilations().at(1) == 1 &&
               conv->get_pads_begin().at(0) == 0 && conv->get_pads_begin().at(1) == 0 &&
               conv->get_pads_end().at(0) == 0 && conv->get_pads_end().at(1) == 0 &&
               in_shape[2] == out_shape[2] && in_shape[3] == out_shape[3];
    };
    const bool force_vulkan_conv_serial_fallback =
        is_vulkan_backend() &&
        m_vulkan_conv_serial_retry_attempted &&
        is_conv_like() &&
        !has_absorbed_input_transpose() &&
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
        module->setAttr("gfx.skip_conv_parallel", mlir::BoolAttr::get(module.getContext(), true));
        module->setAttr("gfx.prefer_parallel", mlir::BoolAttr::get(module.getContext(), false));
        // This path keeps the shared serial loop nest instead of rewriting the
        // convolution to scf.parallel. Launch it as a single kernel instance;
        // default element-wise dispatch would otherwise execute the full serial
        // loop body once per output element and corrupt the accumulation.
        m_force_single_dispatch = true;
    }
    if (is_vulkan_backend() && m_type == "GroupConvolution" && has_absorbed_input_transpose()) {
        // The shared transformed depthwise GroupConvolution currently stays on
        // a serial MLIR loop nest. It must execute as a single kernel launch;
        // default element-wise dispatch would replay the full serial body per
        // output element and corrupt the output.
        m_force_single_dispatch = true;
    }
    apply_fused_operations(module);
    if (m_type == "ShapeOf" && module) {
        if (!is_vulkan_backend()) {
            annotate_msl_custom_kernel_binding_plan(module,
                                                              "ShapeOf",
                                                              "shapeof_kernel",
                                                              {0},
                                                              m_kernel_binding.scalar_args);
        } else {
            auto shapeof_plan = make_kernel_runtime_binding_plan_for_custom_kernel(
                "ShapeOf",
                "shapeof_kernel",
                {0},
                m_kernel_binding.scalar_args,
                GfxKernelBackendDomain::Spirv,
                GfxKernelStorageKind::Buffer,
                "spirv:buffer:");
            OPENVINO_ASSERT(shapeof_plan.valid,
                            "GFX MLIR: ShapeOf SPIR-V custom-kernel ABI manifest is invalid for stage ",
                            m_name);
            apply_spirv_kernel_binding_adapter(
                module,
                shapeof_plan.runtime_binding);
        }
    }
    if (m_type == "Tile" && m_node && module) {
        OPENVINO_ASSERT(m_node->get_input_partial_shape(0).is_static() &&
                            m_node->get_output_partial_shape(0).is_static(),
                        "GFX MLIR: Tile requires static input/output shapes for stage ",
                        m_name);
        const ov::Shape in_shape = m_node->get_input_shape(0);
        const ov::Shape out_shape = m_node->get_output_shape(0);
        auto tile_payload = make_tile_runtime_param_payload(*m_buffer_manager, m_name, in_shape, out_shape);
        const auto tile_scalars = tile_payload.scalar_args;
        m_kernel_extra_inputs = std::move(tile_payload.extra_inputs);
        if (!is_vulkan_backend()) {
            annotate_msl_custom_kernel_binding_plan(module,
                                                              "Tile",
                                                              "tile_kernel",
                                                              {0},
                                                              tile_scalars);
        } else {
            auto tile_plan = make_kernel_runtime_binding_plan_for_custom_kernel(
                "Tile",
                "tile_kernel",
                {0},
                tile_scalars,
                GfxKernelBackendDomain::Spirv,
                GfxKernelStorageKind::Buffer,
                "spirv:buffer:");
            OPENVINO_ASSERT(tile_plan.valid,
                            "GFX MLIR: Tile SPIR-V custom-kernel ABI manifest is invalid for stage ",
                            m_name);
            apply_spirv_kernel_binding_adapter(
                module,
                tile_plan.runtime_binding);
        }
    }
    if (auto gather_elements = ov::as_type_ptr<const ov::op::v6::GatherElements>(m_node)) {
        OPENVINO_ASSERT(gather_elements->get_input_partial_shape(0).is_static() &&
                            gather_elements->get_input_partial_shape(1).is_static() &&
                            gather_elements->get_output_partial_shape(0).is_static(),
                        "GFX MLIR: GatherElements requires static shapes for stage ",
                        m_name);
        const ov::Shape data_shape = gather_elements->get_input_shape(0);
        const ov::Shape out_shape = gather_elements->get_output_shape(0);
        const int64_t axis_norm = normalize_axis(gather_elements->get_axis(),
                                                 out_shape.size(),
                                                 "GFX MLIR: GatherElements");
        auto gather_elements_payload = make_gather_elements_runtime_param_payload(*m_buffer_manager,
                                                                                  m_name,
                                                                                  data_shape,
                                                                                  out_shape,
                                                                                  static_cast<uint32_t>(axis_norm));
        m_kernel_extra_inputs = std::move(gather_elements_payload.extra_inputs);
        if (!is_vulkan_backend()) {
            annotate_msl_custom_kernel_binding_plan(module,
                                                              "GatherElements",
                                                              "gather_elements_kernel",
                                                              {0, 1});
        } else {
            auto gather_elements_plan = make_kernel_runtime_binding_plan_for_custom_kernel(
                "GatherElements",
                "gather_elements_kernel",
                {0, 1},
                {},
                GfxKernelBackendDomain::Spirv,
                GfxKernelStorageKind::Buffer,
                "spirv:buffer:");
            OPENVINO_ASSERT(gather_elements_plan.valid,
                            "GFX MLIR: GatherElements SPIR-V custom-kernel ABI manifest is invalid for stage ",
                            m_name);
            apply_spirv_kernel_binding_adapter(
                module,
                gather_elements_plan.runtime_binding);
        }
    }
    auto plan_ctx = build_mlir_kernel_plan(
        module,
        plan.entry_point(),
        m_node,
        /*output_args_override=*/0,
        m_kernel_extra_inputs.size(),
        m_name.c_str(),
        "gfx_kernel",
        [&](const KernelArgMappingInfo& info) -> size_t {
            size_t func_results = info.func_results;
            if (m_node && func_results == 0) {
                func_results = m_node->get_output_size();
            }
            const auto sig = info.signature;
            return sig.total() ? sig.total() : (info.func_inputs + func_results);
        });
    if (m_type == "ShapeOf") {
        plan_ctx.build_info.plan = KernelPlan(module, plan_ctx.build_info.plan.entry_point(), 4);
    }
    if (m_type == "Tile") {
        plan_ctx.build_info.plan = KernelPlan(module, plan_ctx.build_info.plan.entry_point(), 8);
    }
    if (m_type == "GatherElements") {
        plan_ctx.build_info.plan = KernelPlan(module, plan_ctx.build_info.plan.entry_point(), 4);
    }
    if (m_type == "MaxPool" || m_type == "AvgPool") {
        plan_ctx = build_mlir_kernel_plan(
            module,
            plan.entry_point(),
            m_node,
            /*output_args_override=*/1,
            m_kernel_extra_inputs.size(),
            m_name.c_str(),
            "gfx_kernel",
            [&](const KernelArgMappingInfo&) -> size_t {
                return 3;
            });
        plan_ctx.build_info.plan = KernelPlan(module, plan_ctx.build_info.plan.entry_point(), 3);
    }
    if (is_vulkan_backend() && is_matmul_like() && module) {
        // Keep Vulkan MatMul on an explicit compact buffer ABI. Lowering will
        // rebuild the final interleaved scalar+buffer metadata from the actual
        // gpu.launch_func operands; pre-seeding buffer-only operand kinds here
        // makes stage/runtime metadata drift from the lowered SPIR-V ABI.
        module->setAttr("gfx.fixed_arg_count",
                        mlir::IntegerAttr::get(mlir::IntegerType::get(module.getContext(), 32), 3));
        plan_ctx = build_mlir_kernel_plan(
            module,
            plan.entry_point(),
            m_node,
            /*output_args_override=*/0,
            m_kernel_extra_inputs.size(),
            m_name.c_str(),
            "gfx_kernel",
            [&](const KernelArgMappingInfo&) -> size_t {
                return 3;
            });
        plan_ctx.build_info.plan = KernelPlan(module, plan_ctx.build_info.plan.entry_point(), 3);
    }
    auto& build_info = plan_ctx.build_info;
    const auto signature = build_info.mapping.signature;
    const size_t scalar_inputs = plan_ctx.scalar_inputs;
    size_t output_args = plan_ctx.output_args;
    const size_t buffer_inputs = plan_ctx.buffer_inputs;
    const size_t kernel_inputs_size = plan_ctx.kernel_inputs_size;
    const size_t node_inputs = plan_ctx.node_inputs;
    const size_t extra_inputs_for_mapping = plan_ctx.extra_inputs_for_mapping;
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("MLIRExec") << "Kernel signature: entry=" << build_info.plan.entry_point()
                                  << " func_inputs=" << signature.inputs
                                  << " func_results=" << signature.results
                                  << " scalar_inputs=" << scalar_inputs
                                  << " output_args=" << output_args
                                  << " buffer_inputs=" << buffer_inputs
                                  << " extra_inputs=" << m_kernel_extra_inputs.size()
                                  << " extra_inputs_map=" << extra_inputs_for_mapping
                                  << " kernel_inputs=" << kernel_inputs_size
                                  << " node_inputs=" << node_inputs;
    }
    if (m_type == "Concat") {
        if (is_vulkan_backend()) {
            module->setAttr("gfx.prefer_parallel", mlir::BoolAttr::get(module.getContext(), false));
            m_force_single_dispatch = true;
        } else {
            plan_ctx.build_info.plan = KernelPlan(module, plan_ctx.build_info.plan.entry_point(), 3);
        }
    }
    if (is_vulkan_backend() && m_type == "Add" && is_bias_broadcast_add(m_node)) {
        plan_ctx = build_mlir_kernel_plan(
            module,
            "binary_bias_add",
            m_node,
            /*output_args_override=*/0,
            /*extra_inputs=*/0,
            m_name.c_str(),
            "binary_bias_add",
            [&](const KernelArgMappingInfo&) -> size_t {
                return 3;
            });
    }
    if (is_vulkan_backend() && (m_type == "Interpolate" || m_type == "Transpose")) {
        m_force_single_dispatch = true;
    }
    const bool use_manual_conv2d_vulkan =
        (is_vulkan_backend() &&
         m_type == "Convolution" &&
         !has_absorbed_input_transpose() &&
         !m_has_bias &&
         !m_has_activation &&
         !m_has_bn &&
         m_node &&
         m_node->get_input_size() == 2 &&
         m_node->get_output_size() == 1 &&
         optimization_plan.conv.kind != GfxConvRouteKind::None &&
         optimization_plan.conv.kind != GfxConvRouteKind::Direct1x1 &&
         optimization_plan.conv.algorithm.kind != GfxConvAlgorithmKind::Im2ColMatMul &&
         optimization_plan.conv.algorithm.kind != GfxConvAlgorithmKind::Indirect &&
         m_node->get_input_partial_shape(0).rank().is_static() &&
         m_node->get_input_partial_shape(0).rank().get_length() == 4);
    const bool use_manual_group_conv2d_vulkan =
        (is_vulkan_backend() &&
         m_type == "GroupConvolution" &&
         !m_has_bias &&
         !m_has_activation &&
         !m_has_bn &&
         m_node &&
         m_node->get_input_size() == 2 &&
         m_node->get_output_size() == 1 &&
         optimization_plan.conv.kind != GfxConvRouteKind::None &&
         m_node->get_input_partial_shape(0).rank().is_static() &&
         m_node->get_input_partial_shape(0).rank().get_length() == 4);
    if (gfx_log_debug_enabled() && is_vulkan_backend() &&
        (m_type == "Convolution" || m_type == "GroupConvolution")) {
        gfx_log_debug("MLIRExec") << "Vulkan conv lowering: stage=" << m_name
                                  << " type=" << m_type
                                  << " route=" << conv_route_kind_attr(optimization_plan.conv.kind)
                                  << " algorithm="
                                  << conv_algorithm_kind_attr(optimization_plan.conv.algorithm.kind)
                                  << " lowering="
                                  << (use_manual_conv2d_vulkan || use_manual_group_conv2d_vulkan
                                          ? "manual_vulkan"
                                          : "shared_mlir");
    }
    if (use_manual_conv2d_vulkan) {
        if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node)) {
            ParallelDispatchConfig manual_dispatch_cfg{};
            bool use_parallel_manual_vulkan_conv = false;
            const auto& in_shape = conv->get_input_shape(0);
            const auto& w_shape = conv->get_input_shape(1);
            const auto& out_shape = conv->get_output_shape(0);
            if (in_shape.size() == 4 && w_shape.size() == 4 && out_shape.size() == 4 && out_shape[0] == 1) {
                const auto caps = query_parallelism_caps(m_buffer_manager);
                const uint64_t input_channels = static_cast<uint64_t>(std::max<size_t>(1, in_shape[1]));
                const uint64_t output_channels = static_cast<uint64_t>(std::max<size_t>(1, w_shape[0]));
                const uint64_t kernel_work =
                    input_channels * static_cast<uint64_t>(std::max<size_t>(1, w_shape[2])) *
                    static_cast<uint64_t>(std::max<size_t>(1, w_shape[3]));
                const bool stride2 = conv->get_strides().at(0) > 1 || conv->get_strides().at(1) > 1;
                const auto parallel_plan = select_conv_parallelism(caps,
                                                                   out_shape,
                                                                   input_channels,
                                                                   output_channels,
                                                                   kernel_work,
                                                                   stride2,
                                                                   /*depthwise=*/false);
                if (parallel_plan.prefer_parallel &&
                    parallel_plan.dispatch.threads_h > 0 &&
                    parallel_plan.dispatch.threads_w > 0) {
                    manual_dispatch_cfg = parallel_plan.dispatch;
                    use_parallel_manual_vulkan_conv = true;
                }
            }
            module = build_mlir_conv2d_vulkan(conv,
                                              ctx,
                                              use_parallel_manual_vulkan_conv ? &manual_dispatch_cfg : nullptr);
            m_force_single_dispatch = !use_parallel_manual_vulkan_conv;
            mlir::OpBuilder b(module.getContext());
            apply_stage_optimization_attrs(module, optimization_plan);
            module->setAttr("gfx.skip_conv_parallel", mlir::BoolAttr::get(module.getContext(), true));
            module->setAttr("gfx.prefer_parallel",
                            mlir::BoolAttr::get(module.getContext(), use_parallel_manual_vulkan_conv));
            if (use_parallel_manual_vulkan_conv) {
                module->setAttr("gfx.parallel_dispatch", mlir::BoolAttr::get(module.getContext(), true));
                module->setAttr("gfx.dispatch_tile_h",
                                mlir::IntegerAttr::get(mlir::IndexType::get(module.getContext()),
                                                       manual_dispatch_cfg.tile_h));
                module->setAttr("gfx.dispatch_tile_w",
                                mlir::IntegerAttr::get(mlir::IndexType::get(module.getContext()),
                                                       manual_dispatch_cfg.tile_w));
                module->setAttr("gfx.dispatch_threads_h",
                                mlir::IntegerAttr::get(mlir::IndexType::get(module.getContext()),
                                                       manual_dispatch_cfg.threads_h));
                module->setAttr("gfx.dispatch_threads_w",
                                mlir::IntegerAttr::get(mlir::IndexType::get(module.getContext()),
                                                       manual_dispatch_cfg.threads_w));
            }
            m_kernel_extra_inputs.clear();
            apply_spirv_kernel_binding_adapter(
                module,
                make_kernel_runtime_binding_state({0, 1},
                                                  2,
                                                  {1, 1, 1},
                                                  {0, 1, 2}));
            const std::string manual_entry =
                use_parallel_manual_vulkan_conv ? "conv2d_kernel" : "conv2d_main";
            plan_ctx = build_mlir_kernel_plan(
                module,
                manual_entry,
                m_node,
                /*output_args_override=*/0,
                /*extra_inputs=*/0,
                m_name.c_str(),
                manual_entry,
                [&](const KernelArgMappingInfo& info) -> size_t {
                    size_t func_results = info.func_results;
                    if (m_node && func_results == 0) {
                        func_results = m_node->get_output_size();
                    }
                    const auto sig = info.signature;
                    return sig.total() ? sig.total() : (info.func_inputs + func_results);
                });
            plan_ctx.build_info.plan = KernelPlan(module, plan_ctx.build_info.plan.entry_point(), 3);
        }
    } else if (use_manual_group_conv2d_vulkan) {
        if (auto gconv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(m_node)) {
            module = build_mlir_group_conv2d_vulkan(gconv, ctx, input_transform(0));
            m_force_single_dispatch = true;
            mlir::OpBuilder b(module->getContext());
            apply_stage_optimization_attrs(module, optimization_plan);
            module->setAttr("gfx.skip_conv_parallel", mlir::BoolAttr::get(module.getContext(), true));
            module->setAttr("gfx.prefer_parallel", mlir::BoolAttr::get(module.getContext(), false));
            m_kernel_extra_inputs.clear();
            apply_spirv_kernel_binding_adapter(
                module,
                make_kernel_runtime_binding_state({0, 1},
                                                  2,
                                                  {1, 1, 1},
                                                  {0, 1, 2}));
            plan_ctx = build_mlir_kernel_plan(
                module,
                "group_conv2d_main",
                m_node,
                /*output_args_override=*/0,
                /*extra_inputs=*/0,
                m_name.c_str(),
                "group_conv2d_main",
                [&](const KernelArgMappingInfo& info) -> size_t {
                    size_t func_results = info.func_results;
                    if (m_node && func_results == 0) {
                        func_results = m_node->get_output_size();
                    }
                    const auto sig = info.signature;
                    return sig.total() ? sig.total() : (info.func_inputs + func_results);
                });
        }
    }
    if (m_has_residual_add && module) {
        module->setAttr("gfx.fused_residual_add", mlir::BoolAttr::get(module.getContext(), true));
    }
    if (!is_vulkan_backend() && (m_type == "MaxPool" || m_type == "AvgPool") && module) {
        annotate_msl_custom_kernel_binding_plan(module, m_type, "pool2d_kernel", {0});
    }
    if (!is_vulkan_backend() && m_has_residual_add && m_type == "RMS" && module) {
        annotate_msl_custom_kernel_binding_plan(module, "RMSResidual", "rms_kernel", {0, 1, 2});
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
                make_kernel_runtime_binding_state({0},
                                                  1,
                                                  {1, 1, 1},
                                                  {0, 1, 2}));
        }
    }
    if (m_has_residual_add && m_type == "RMS") {
        if (is_vulkan_backend()) {
            apply_kernel_runtime_binding_state(
                make_kernel_runtime_binding_state({0, 1, 2},
                                                  3,
                                                  {1, 1, 1, 1},
                                                  {0, 1, 2, 3}));
        }
    }
    if (module) {
        if (gfx_log_debug_enabled() && !m_kernel_binding.scalar_args.empty()) {
            std::ostringstream oss;
            oss << "Kernel scalar args: ";
            const size_t dump_n = std::min<size_t>(m_kernel_binding.scalar_args.size(), 8);
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
            gfx_log_debug("MLIRExec") << "Kernel attrs: operand_kinds=" << (has_kinds ? "yes" : "no")
                                      << " scalar_values=" << (has_scalars ? "yes" : "no");
            gfx_log_debug("MLIRExec") << "Kernel operand kinds size=" << m_kernel_binding.operand_kinds.size();
            if (!m_kernel_binding.operand_arg_indices.empty()) {
                std::ostringstream idxs;
                idxs << "Kernel operand arg indices: ";
                const size_t dump_n = std::min<size_t>(m_kernel_binding.operand_arg_indices.size(), 8);
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
                gfx_log_debug("MLIRExec") << "operand_kinds isa ArrayAttr="
                                          << (llvm::isa<mlir::ArrayAttr>(attr) ? "yes" : "no")
                                          << " DenseI32ArrayAttr="
                                          << (llvm::isa<mlir::DenseI32ArrayAttr>(attr) ? "yes" : "no")
                                          << " DenseIntElementsAttr="
                                          << (llvm::isa<mlir::DenseIntElementsAttr>(attr) ? "yes" : "no");
            }
        }
    }
}

std::vector<KernelArg> MlirStage::materialize_bound_kernel_args(const std::vector<GpuTensor*>& outputs) const {
    OPENVINO_ASSERT(m_kernel, "GFX MLIR: kernel is not compiled for stage ", m_name);
    OPENVINO_ASSERT(m_buffer_manager, "GFX MLIR: buffer manager is not initialized for stage ", m_name);
    const KernelRuntimeBindingState binding = resolved_kernel_runtime_binding_state();

    const auto resolve_input_tensor = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
        }
        if (m_const_buffers &&
            input_idx < m_const_buffers->buffers.size() &&
            input_idx < m_const_buffers->present.size() &&
            m_const_buffers->present[input_idx] &&
            m_const_buffers->buffers[input_idx].buf.valid()) {
            return const_cast<GpuTensor*>(&m_const_buffers->buffers[input_idx]);
        }
        return nullptr;
    };

    const size_t expected_inputs = binding.input_arg_count ? binding.input_arg_count : binding.inputs.size();
    const std::vector<GpuTensor> empty_extras;
    const std::vector<GpuTensor>* extras = &m_kernel_extra_inputs;
    if (binding.operand_kinds.empty() && expected_inputs <= binding.inputs.size()) {
        extras = &empty_extras;
    }

    auto bundle = build_kernel_args_from_metadata(
        binding.operand_kinds,
        binding.operand_arg_indices,
        binding.scalar_args,
        binding.inputs,
        binding.input_arg_count,
        *extras,
        outputs,
        [&](size_t input_idx) { return resolve_input_tensor(input_idx); },
        m_name.c_str(),
        nullptr);
    return materialize_kernel_bytes_args(bundle.args, *m_buffer_manager, m_name.c_str());
}

void MlirStage::prewarm_runtime_state() {
    if (m_is_view_op || !m_kernel) {
        return;
    }

    std::vector<GpuTensor*> outputs;
    if (!m_outputs.empty()) {
        outputs = m_outputs;
    } else if (m_output) {
        outputs.push_back(m_output);
    }
    outputs.erase(std::remove(outputs.begin(), outputs.end(), nullptr), outputs.end());
    if (outputs.empty()) {
        return;
    }

    try {
        const auto bound_args = materialize_bound_kernel_args(outputs);
        if (bound_args.empty()) {
            return;
        }
        m_kernel->prewarm_bindings(bound_args);
    } catch (...) {
        // Keep runtime-state prewarm best-effort. Stages with custom launch
        // paths continue to use their validated lazy execution path.
    }
}

void MlirStage::execute(GpuCommandBufferHandle command_buffer) {
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("MLIRExec") << "Execute stage " << m_name << " (" << m_type << ")";
    }
    std::vector<GpuTensor*> outputs = m_outputs;
    if (outputs.empty() && m_output) {
        outputs.push_back(m_output);
    }
    if (outputs.empty()) {
        OPENVINO_THROW("GFX MLIR: output tensor is not bound for stage ", m_name);
    }
    auto resolve_input_shape = [&](size_t idx) -> ov::Shape {
        if (idx < m_inputs.size() && m_inputs[idx] && !m_inputs[idx]->shape.empty()) {
            return m_inputs[idx]->shape;
        }
        if (m_node && m_node->get_input_partial_shape(idx).is_static()) {
            return m_node->get_input_shape(idx);
        }
        return {};
    };
    auto resolve_input_shape_known = [&](size_t idx, ov::Shape& shape) -> bool {
        if (idx < m_inputs.size() && m_inputs[idx] && !m_inputs[idx]->shape.empty()) {
            shape = m_inputs[idx]->shape;
            return true;
        }
        if (!m_node || idx >= m_node->get_input_size()) {
            return false;
        }
        if (m_node->get_input_partial_shape(idx).is_static()) {
            shape = m_node->get_input_shape(idx);
            return true;
        }
        if (auto input_const = ov::util::get_constant_from_source(m_node->input_value(idx))) {
            shape = input_const->get_shape();
            return true;
        }
        return false;
    };

    std::optional<KernelRuntimeBindingState> kernel_binding_override;
    std::optional<std::vector<int32_t>> kernel_scalar_args_override;

    auto set_kernel_binding_override = [&](KernelRuntimeBindingState binding) {
        kernel_binding_override = std::move(binding);
    };
    auto set_spirv_kernel_binding_override = [&](mlir::ModuleOp module,
                                                 KernelRuntimeBindingState binding) {
        OPENVINO_ASSERT(is_vulkan_backend(),
                        "GFX MLIR: SPIR-V execute binding override used on non-Vulkan backend");
        if (module) {
            annotate_kernel_operand_abi_attrs_for_spirv_adapter(module, binding);
        }
        set_kernel_binding_override(std::move(binding));
    };

    auto resolve_input_tensor = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* t = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (t && t->buf.valid()) {
            return t;
        }
        if (m_const_buffers &&
            input_idx < m_const_buffers->buffers.size() &&
            input_idx < m_const_buffers->present.size() &&
            m_const_buffers->present[input_idx] &&
            m_const_buffers->buffers[input_idx].buf.valid()) {
            return &m_const_buffers->buffers[input_idx];
        }
        return nullptr;
    };
    auto resolve_i64_values = [&](size_t input_idx) -> std::optional<std::vector<int64_t>> {
        if (input_idx < m_inputs.size() && m_inputs[input_idx] && !m_inputs[input_idx]->i64_values.empty()) {
            return m_inputs[input_idx]->i64_values;
        }
        if (!m_node || input_idx >= m_node->get_input_size()) {
            return std::nullopt;
        }
        if (auto input_const = ov::util::get_constant_from_source(m_node->input_value(input_idx))) {
            return input_const->cast_vector<int64_t>();
        }
        return std::nullopt;
    };
    auto assign_i64_values = [](GpuTensor* out, const std::vector<int64_t>& values, const ov::Shape& shape) {
        if (!out || values.size() != ov::shape_size(shape)) {
            return;
        }
        out->i64_values = values;
    };
    auto compute_row_major_strides = [](const ov::Shape& shape) {
        std::vector<size_t> strides(shape.size(), 1);
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[static_cast<size_t>(i)] =
                strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
        }
        return strides;
    };
    auto compute_reduce_i64_values = [&](const std::vector<int64_t>& input_values,
                                         const ov::Shape& input_shape,
                                         const RuntimeReduceInfo& reduce_info,
                                         const ov::Shape& output_shape) -> std::optional<std::vector<int64_t>> {
        if (input_values.size() != ov::shape_size(input_shape)) {
            return std::nullopt;
        }

        const size_t output_count = ov::shape_size(output_shape);
        std::vector<int64_t> output_values(output_count, 0);
        std::vector<bool> initialized(output_count, false);
        auto in_strides = compute_row_major_strides(input_shape);
        auto out_strides = compute_row_major_strides(output_shape);
        std::vector<size_t> input_coord(input_shape.size(), 0);

        for (size_t linear = 0; linear < input_values.size(); ++linear) {
            size_t rem = linear;
            for (size_t axis = 0; axis < input_shape.size(); ++axis) {
                const size_t stride = in_strides[axis];
                input_coord[axis] = stride == 0 ? 0 : rem / stride;
                rem = stride == 0 ? 0 : rem - input_coord[axis] * stride;
            }

            size_t out_axis = 0;
            size_t out_linear = 0;
            for (size_t axis = 0; axis < input_shape.size(); ++axis) {
                const bool reduced = reduce_info.axes.count(axis) != 0;
                if (reduced && !reduce_info.keep_dims) {
                    continue;
                }
                const size_t coord = reduced ? 0 : input_coord[axis];
                if (out_axis < out_strides.size()) {
                    out_linear += coord * out_strides[out_axis];
                }
                ++out_axis;
            }
            if (output_shape.empty()) {
                out_linear = 0;
            }

            const int64_t value = input_values[linear];
            if (!initialized[out_linear]) {
                initialized[out_linear] = true;
                if (m_type == "ReduceSum" || m_type == "ReduceProd" ||
                    m_type == "ReduceMax" || m_type == "ReduceMin") {
                    output_values[out_linear] = value;
                } else {
                    return std::nullopt;
                }
                continue;
            }
            if (m_type == "ReduceSum") {
                output_values[out_linear] += value;
            } else if (m_type == "ReduceProd") {
                output_values[out_linear] *= value;
            } else if (m_type == "ReduceMax") {
                output_values[out_linear] = std::max(output_values[out_linear], value);
            } else if (m_type == "ReduceMin") {
                output_values[out_linear] = std::min(output_values[out_linear], value);
            } else {
                return std::nullopt;
            }
        }

        return output_values;
    };
    auto range_length_i64 = [&](int64_t start, int64_t stop, int64_t step) -> size_t {
        OPENVINO_ASSERT(step != 0, "GFX MLIR: Range step must be non-zero for stage ", m_name);
        const bool forward = step > 0;
        if ((forward && start >= stop) || (!forward && start <= stop)) {
            return 0;
        }
        const uint64_t distance = forward ? static_cast<uint64_t>(stop - start) : static_cast<uint64_t>(start - stop);
        const uint64_t stride = static_cast<uint64_t>(forward ? step : -step);
        return static_cast<size_t>((distance + stride - 1) / stride);
    };

    auto resolve_known_input_shape = [&](size_t idx, ov::Shape& shape) -> bool {
        if (idx < m_inputs.size() && m_inputs[idx] && !m_inputs[idx]->shape.empty()) {
            shape = m_inputs[idx]->shape;
            return true;
        }
        if (m_node && idx < m_node->get_input_size() && m_node->get_input_partial_shape(idx).is_static()) {
            shape = m_node->get_input_shape(idx);
            return true;
        }
        return false;
    };

    auto ensure_output_shape = [&](size_t oi, GpuTensor* out) {
        if (!out) {
            return;
        }
        if (out->shape.empty() && m_node && m_node->get_output_partial_shape(oi).is_static()) {
            out->shape = m_node->get_output_shape(oi);
        }
    };

    auto compute_binary_broadcast_shape = [&](const ov::Shape& lhs, const ov::Shape& rhs) {
        const size_t rank = std::max(lhs.size(), rhs.size());
        ov::Shape out(rank, 1);
        for (size_t i = 0; i < rank; ++i) {
            const size_t lhs_dim = lhs.size() > i ? lhs[lhs.size() - 1 - i] : 1;
            const size_t rhs_dim = rhs.size() > i ? rhs[rhs.size() - 1 - i] : 1;
            OPENVINO_ASSERT(lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1,
                            "GFX MLIR: incompatible binary broadcast dims for stage ",
                            m_name,
                            " axis_from_back=",
                            i,
                            " lhs=",
                            lhs,
                            " rhs=",
                            rhs);
            out[rank - 1 - i] = std::max(lhs_dim, rhs_dim);
        }
        return out;
    };

    auto bind_small_i64_const_outputs = [&](const std::string& suffix) -> bool {
        if (!m_buffer_manager || outputs.empty()) {
            return false;
        }
        std::vector<ov::element::Type> resolved_types;
        resolved_types.reserve(outputs.size());
        for (auto* out : outputs) {
            if (!out || out->i64_values.empty() || out->i64_values.size() != ov::shape_size(out->shape) ||
                out->i64_values.size() > 16) {
                return false;
            }
            const auto type = out->expected_type == ov::element::dynamic && m_node && m_node->get_output_size() > 0
                                  ? m_node->get_output_element_type(0)
                                  : out->expected_type;
            if (type != ov::element::i32 && type != ov::element::i64) {
                return false;
            }
            resolved_types.push_back(type);
        }

        if (m_small_i64_const_output_cache.size() == outputs.size()) {
            bool cache_hit = true;
            for (size_t oi = 0; oi < outputs.size(); ++oi) {
                const auto* out = outputs[oi];
                const auto& cached = m_small_i64_const_output_cache[oi];
                if (!cached.buf.valid() ||
                    cached.expected_type != resolved_types[oi] ||
                    cached.shape != out->shape ||
                    cached.i64_values != out->i64_values) {
                    cache_hit = false;
                    break;
                }
            }
            if (cache_hit) {
                for (size_t oi = 0; oi < outputs.size(); ++oi) {
                    auto* out = outputs[oi];
                    const auto& cached = m_small_i64_const_output_cache[oi];
                    out->buf = cached.buf;
                    out->buf.external = true;
                    out->expected_type = cached.expected_type;
                    out->shape = cached.shape;
                    out->i64_values = cached.i64_values;
                }
                if (auto* profiler = static_cast<GfxProfiler*>(m_profiler)) {
                    if (m_profiling_enabled) {
                        profiler->increment_counter("small_i64_const_cache_hit_count");
                    }
                }
                return true;
            }
        }

        m_small_i64_const_output_cache.clear();
        m_small_i64_const_output_cache.reserve(outputs.size());
        for (size_t oi = 0; oi < outputs.size(); ++oi) {
            auto* out = outputs[oi];
            const auto type = resolved_types[oi];
            std::ostringstream key;
            key << m_name << "/" << suffix << "/" << type << "/";
            for (auto value : out->i64_values) {
                key << value << ",";
            }
            GpuBuffer buf;
            if (type == ov::element::i32) {
                std::vector<int32_t> values;
                values.reserve(out->i64_values.size());
                for (auto value : out->i64_values) {
                    values.push_back(static_cast<int32_t>(value));
                }
                buf = m_buffer_manager->wrap_const(key.str(),
                                                   values.data(),
                                                   values.size() * sizeof(int32_t),
                                                   ov::element::i32);
            } else {
                buf = m_buffer_manager->wrap_const(key.str(),
                                                   out->i64_values.data(),
                                                   out->i64_values.size() * sizeof(int64_t),
                                                   ov::element::i64);
            }
            OPENVINO_ASSERT(buf.valid(), "GFX MLIR: failed to bind small const output for stage ", m_name);
            buf.owned = false;
            out->buf = buf;
            out->buf.external = true;
            out->expected_type = type;

            GpuTensor cached;
            cached.buf = out->buf;
            cached.shape = out->shape;
            cached.i64_values = out->i64_values;
            cached.expected_type = type;
            m_small_i64_const_output_cache.push_back(std::move(cached));
        }
        if (auto* profiler = static_cast<GfxProfiler*>(m_profiler)) {
            if (m_profiling_enabled) {
                profiler->increment_counter("small_i64_const_stage_count");
            }
        }
        return true;
    };

    if (m_type == "GfxSDPAWithCausalMask" && m_node) {
        OPENVINO_ASSERT(ov::as_type_ptr<const ov::gfx_plugin::op::GfxSDPAWithCausalMask>(m_node),
                        "GFX MLIR: expected GfxSDPAWithCausalMask node for stage ",
                        m_name);
        ov::Shape q_shape = resolve_input_shape(0);
        ov::Shape k_shape = resolve_input_shape(1);
        ov::Shape v_shape = resolve_input_shape(2);
        ov::Shape mask_shape = resolve_input_shape(3);
        ov::Shape pos_shape = resolve_input_shape(4);
        OPENVINO_ASSERT(q_shape.size() == 4 && k_shape.size() == 4 && v_shape.size() == 4,
                        "GFX MLIR: fused causal-mask SDPA expects rank-4 Q/K/V for stage ",
                        m_name);
        OPENVINO_ASSERT(mask_shape.size() == 2 && pos_shape.size() == 1,
                        "GFX MLIR: fused causal-mask SDPA expects attention_mask[B,K] and cache_positions[Q] for stage ",
                        m_name);
        const bool k_heads_match = q_shape[1] == k_shape[1] ||
                                   (k_shape[1] > 0 && (q_shape[1] % k_shape[1]) == 0);
        const bool v_heads_match = q_shape[1] == v_shape[1] ||
                                   (v_shape[1] > 0 && (q_shape[1] % v_shape[1]) == 0);
        OPENVINO_ASSERT(q_shape[0] == k_shape[0] && q_shape[0] == v_shape[0] &&
                        k_heads_match && v_heads_match &&
                        k_shape[2] == v_shape[2] && q_shape[3] == k_shape[3] &&
                        q_shape[0] == mask_shape[0] && q_shape[2] == pos_shape[0],
                        "GFX MLIR: incompatible fused causal-mask SDPA shapes for stage ",
                        m_name,
                        " q=",
                        q_shape,
                        " k=",
                        k_shape,
                        " v=",
                        v_shape,
                        " mask=",
                        mask_shape,
                        " positions=",
                        pos_shape);
        ov::Shape out_shape{q_shape[0], q_shape[1], q_shape[2], v_shape[3]};
        for (auto* out : outputs) {
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
        if (auto scale_const = ov::util::get_constant_from_source(m_node->input_value(5))) {
            const auto vals = scale_const->cast_vector<float>();
            if (!vals.empty()) {
                scale = vals[0];
            }
        }
        const auto* k_tensor = resolve_input_tensor(1);
        const auto* v_tensor = resolve_input_tensor(2);
        const bool k_view_gqa = k_tensor && k_tensor->gqa_broadcast_view && k_tensor->gqa_kv_heads > 0;
        const bool v_view_gqa = v_tensor && v_tensor->gqa_broadcast_view && v_tensor->gqa_kv_heads > 0;
        const bool k_gqa = k_view_gqa || k_shape[1] != q_shape[1];
        const bool v_gqa = v_view_gqa || v_shape[1] != q_shape[1];
        auto sdpa_plan = make_causal_sdpa_msl_runtime_params_plan(q_shape,
                                                                  k_shape,
                                                                  v_shape,
                                                                  mask_shape,
                                                                  scale,
                                                                  k_gqa,
                                                                  k_view_gqa ? k_tensor->gqa_kv_heads : k_shape[1],
                                                                  v_gqa,
                                                                  v_view_gqa ? v_tensor->gqa_kv_heads : v_shape[1]);
        OPENVINO_ASSERT(sdpa_plan.valid(), "GFX MLIR: invalid causal SDPA MSL runtime params for stage ", m_name);
        m_kernel_extra_inputs.clear();
        m_kernel_extra_inputs.push_back(make_kernel_i32_param_tensor(*m_buffer_manager,
                                                                     m_name,
                                                                     "sdpa_causal_mask_params",
                                                                     sdpa_plan.params));
        set_kernel_binding_override(sdpa_plan.binding.kernel_runtime_binding());
    }

    if (m_type == "ScaledDotProductAttention" && m_node) {
        auto sdpa = ov::as_type_ptr<const ov::op::v13::ScaledDotProductAttention>(m_node);
        OPENVINO_ASSERT(sdpa, "GFX MLIR: expected ScaledDotProductAttention node for stage ", m_name);
        ov::Shape q_shape = resolve_input_shape(0);
        ov::Shape k_shape = resolve_input_shape(1);
        ov::Shape v_shape = resolve_input_shape(2);
        OPENVINO_ASSERT(q_shape.size() == 4 && k_shape.size() == 4 && v_shape.size() == 4,
                        "GFX MLIR: SDPA expects rank-4 Q/K/V for stage ",
                        m_name);
        OPENVINO_ASSERT(q_shape[0] == k_shape[0] && q_shape[0] == v_shape[0] &&
                        q_shape[1] == k_shape[1] && q_shape[1] == v_shape[1] &&
                        k_shape[2] == v_shape[2] && q_shape[3] == k_shape[3],
                        "GFX MLIR: incompatible SDPA Q/K/V shapes for stage ",
                        m_name,
                        " q=",
                        q_shape,
                        " k=",
                        k_shape,
                        " v=",
                        v_shape);
        ov::Shape out_shape{q_shape[0], q_shape[1], q_shape[2], v_shape[3]};
        for (auto* out : outputs) {
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
        if (m_node->get_input_size() >= 5) {
            if (auto scale_const = ov::util::get_constant_from_source(m_node->input_value(4))) {
                const auto vals = scale_const->cast_vector<float>();
                if (!vals.empty()) {
                    scale = vals[0];
                }
            }
        }
        const auto* k_tensor = resolve_input_tensor(1);
        const auto* v_tensor = resolve_input_tensor(2);
        const bool k_gqa = k_tensor && k_tensor->gqa_broadcast_view && k_tensor->gqa_kv_heads > 0;
        const bool v_gqa = v_tensor && v_tensor->gqa_broadcast_view && v_tensor->gqa_kv_heads > 0;

        ov::Shape mask_shape{1, 1, 1, 1};
        const bool has_mask = m_node->get_input_size() >= 4;
        if (has_mask) {
            mask_shape = resolve_input_shape(3);
            OPENVINO_ASSERT(mask_shape.size() == 4,
                            "GFX MLIR: SDPA mask expects rank-4 shape for stage ",
                            m_name);
        }
        auto sdpa_plan = make_sdpa_msl_runtime_params_plan(q_shape,
                                                           k_shape,
                                                           v_shape,
                                                           mask_shape,
                                                           has_mask,
                                                           scale,
                                                           k_gqa,
                                                           k_gqa ? k_tensor->gqa_kv_heads : k_shape[1],
                                                           v_gqa,
                                                           v_gqa ? v_tensor->gqa_kv_heads : v_shape[1]);
        OPENVINO_ASSERT(sdpa_plan.valid(), "GFX MLIR: invalid SDPA MSL runtime params for stage ", m_name);
        m_kernel_extra_inputs.clear();
        m_kernel_extra_inputs.push_back(make_kernel_i32_param_tensor(*m_buffer_manager,
                                                                     m_name,
                                                                     "sdpa_params",
                                                                     sdpa_plan.params));
        set_kernel_binding_override(sdpa_plan.binding.kernel_runtime_binding());
    }

    if (auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(m_node)) {
        OPENVINO_ASSERT(!outputs.empty(), "GFX MLIR: missing concat outputs for stage ", m_name);
        ov::Shape out_shape;
        bool out_shape_ready = false;
        for (size_t input_idx = 0; input_idx < concat->get_input_size(); ++input_idx) {
            ov::Shape input_shape = resolve_input_shape(input_idx);
            if (input_shape.empty() && m_node->get_input_partial_shape(input_idx).is_static()) {
                input_shape = m_node->get_input_shape(input_idx);
            }
            if (input_shape.empty()) {
                OPENVINO_THROW("GFX MLIR: Concat input shape is unknown for stage ", m_name);
            }
            if (!out_shape_ready) {
                out_shape = input_shape;
                out_shape_ready = true;
                continue;
            }
            OPENVINO_ASSERT(input_shape.size() == out_shape.size(),
                            "GFX MLIR: Concat rank mismatch for stage ",
                            m_name);
        }
        OPENVINO_ASSERT(out_shape_ready, "GFX MLIR: Concat has no resolved inputs for stage ", m_name);
        const int64_t axis_norm = normalize_axis(concat->get_axis(), out_shape.size(), "GFX MLIR: Concat");
        size_t axis_total = 0;
        for (size_t input_idx = 0; input_idx < concat->get_input_size(); ++input_idx) {
            ov::Shape input_shape = resolve_input_shape(input_idx);
            if (input_shape.empty() && m_node->get_input_partial_shape(input_idx).is_static()) {
                input_shape = m_node->get_input_shape(input_idx);
            }
            OPENVINO_ASSERT(input_shape.size() == out_shape.size(),
                            "GFX MLIR: Concat rank mismatch for stage ",
                            m_name);
            for (size_t dim = 0; dim < out_shape.size(); ++dim) {
                if (static_cast<int64_t>(dim) == axis_norm) {
                    continue;
                }
                OPENVINO_ASSERT(input_shape[dim] == out_shape[dim],
                                "GFX MLIR: Concat non-axis dim mismatch for stage ",
                                m_name);
            }
            axis_total += input_shape[static_cast<size_t>(axis_norm)];
        }
        out_shape[static_cast<size_t>(axis_norm)] = axis_total;
        for (auto* out : outputs) {
            if (!out) {
                continue;
            }
            out->shape = out_shape;
            if (out->expected_type == ov::element::dynamic) {
                out->expected_type = m_node->get_output_element_type(0);
            }
        }
        std::vector<std::vector<int64_t>> concat_values;
        concat_values.reserve(concat->get_input_size());
        bool all_values_resolved = true;
        for (size_t input_idx = 0; input_idx < concat->get_input_size(); ++input_idx) {
            auto values = resolve_i64_values(input_idx);
            if (!values.has_value()) {
                all_values_resolved = false;
                break;
            }
            concat_values.push_back(std::move(*values));
        }
        if (all_values_resolved) {
            const size_t inner = std::accumulate(out_shape.begin() + axis_norm + 1,
                                                 out_shape.end(),
                                                 size_t{1},
                                                 std::multiplies<size_t>());
            const size_t outer = std::accumulate(out_shape.begin(),
                                                 out_shape.begin() + axis_norm,
                                                 size_t{1},
                                                 std::multiplies<size_t>());
            std::vector<int64_t> values;
            values.reserve(ov::shape_size(out_shape));
            for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
                for (size_t input_idx = 0; input_idx < concat->get_input_size(); ++input_idx) {
                    ov::Shape input_shape = resolve_input_shape(input_idx);
                    if (input_shape.empty() && m_node->get_input_partial_shape(input_idx).is_static()) {
                        input_shape = m_node->get_input_shape(input_idx);
                    }
                    const size_t chunk = input_shape[static_cast<size_t>(axis_norm)] * inner;
                    const size_t offset = outer_idx * chunk;
                    if (offset + chunk > concat_values[input_idx].size()) {
                        all_values_resolved = false;
                        break;
                    }
                    values.insert(values.end(),
                                  concat_values[input_idx].begin() + static_cast<std::ptrdiff_t>(offset),
                                  concat_values[input_idx].begin() + static_cast<std::ptrdiff_t>(offset + chunk));
                }
                if (!all_values_resolved) {
                    break;
                }
            }
        if (all_values_resolved && values.size() == ov::shape_size(out_shape)) {
            for (auto* out : outputs) {
                assign_i64_values(out, values, out_shape);
            }
            if (bind_small_i64_const_outputs("concat_i64_const")) {
                return;
            }
        }
    }
        m_output_shape = out_shape;
        GpuTensor* single_live_input = nullptr;
        size_t live_input_count = 0;
        for (size_t input_idx = 0; input_idx < concat->get_input_size(); ++input_idx) {
            ov::Shape input_shape = resolve_input_shape(input_idx);
            if (input_shape.empty() && m_node->get_input_partial_shape(input_idx).is_static()) {
                input_shape = m_node->get_input_shape(input_idx);
            }
            if (ov::shape_size(input_shape) == 0) {
                continue;
            }
            ++live_input_count;
            single_live_input = resolve_input_tensor(input_idx);
        }
        if (live_input_count == 1 && single_live_input && single_live_input->shape == out_shape) {
            const auto output_type = m_node->get_output_element_type(0);
            bool all_aliased = true;
            for (auto* out : outputs) {
                if (!out || !alias_tensor_view(*out, *single_live_input, out_shape, output_type)) {
                    all_aliased = false;
                    break;
                }
            }
            if (all_aliased) {
                return;
            }
        }
    }

    if (!is_vulkan_backend()) {
        auto gather_v1 = ov::as_type_ptr<const ov::op::v1::Gather>(m_node);
        auto gather_v7 = ov::as_type_ptr<const ov::op::v7::Gather>(m_node);
        auto gather_v8 = ov::as_type_ptr<const ov::op::v8::Gather>(m_node);
        if (gather_v1 || gather_v7 || gather_v8) {
            if (gather_v7) {
                OPENVINO_ASSERT(gather_v7->get_batch_dims() == 0,
                                "GFX MLIR: Gather v7 batch_dims not supported for stage ",
                                m_name);
            }
            if (gather_v8) {
                OPENVINO_ASSERT(gather_v8->get_batch_dims() == 0,
                                "GFX MLIR: Gather v8 batch_dims not supported for stage ",
                                m_name);
            }
            ov::Shape data_shape;
            ov::Shape idx_shape;
            const bool data_shape_known = resolve_input_shape_known(0, data_shape);
            const bool idx_shape_known = resolve_input_shape_known(1, idx_shape);
            if (!data_shape_known) {
                OPENVINO_THROW("GFX MLIR: Gather data shape is unknown for stage ", m_name);
            }
            if (!idx_shape_known) {
                OPENVINO_THROW("GFX MLIR: Gather indices shape is unknown for stage ", m_name);
            }
            auto axis_const = ov::util::get_constant_from_source(m_node->input_value(2));
            OPENVINO_ASSERT(axis_const, "GFX MLIR: Gather axis must be constant for stage ", m_name);
            const auto axis_values = axis_const->cast_vector<int64_t>();
            OPENVINO_ASSERT(axis_values.size() == 1, "GFX MLIR: Gather axis must be scalar for stage ", m_name);
            const int64_t axis_norm = normalize_axis(axis_values[0], data_shape.size(), "GFX MLIR: Gather");

            ov::Shape out_shape;
            out_shape.reserve(data_shape.size() + idx_shape.size());
            out_shape.insert(out_shape.end(),
                             data_shape.begin(),
                             data_shape.begin() + static_cast<ptrdiff_t>(axis_norm));
            out_shape.insert(out_shape.end(), idx_shape.begin(), idx_shape.end());
            out_shape.insert(out_shape.end(),
                             data_shape.begin() + static_cast<ptrdiff_t>(axis_norm) + 1,
                             data_shape.end());

            const auto output_et = m_node->get_output_element_type(0);
            for (auto* out : outputs) {
                if (!out) {
                    continue;
                }
                out->shape = out_shape;
                out->expected_type = output_et;
            }
            m_output_shape = out_shape;
            if (auto data_values = resolve_i64_values(0)) {
                if (auto idx_values = resolve_i64_values(1)) {
                    if (axis_norm == 0 && data_shape.size() == 1 && !data_values->empty()) {
                        std::vector<int64_t> gathered_values;
                        gathered_values.reserve(idx_values->size());
                        for (auto idx : *idx_values) {
                            int64_t normalized = idx < 0 ? idx + static_cast<int64_t>(data_values->size()) : idx;
                            normalized = std::clamp<int64_t>(normalized,
                                                             0,
                                                             static_cast<int64_t>(data_values->empty()
                                                                                      ? 0
                                                                                      : data_values->size() - 1));
                            gathered_values.push_back((*data_values)[static_cast<size_t>(normalized)]);
                        }
                        for (auto* out : outputs) {
                            assign_i64_values(out, gathered_values, out_shape);
                        }
                        if (bind_small_i64_const_outputs("gather_i64_const")) {
                            return;
                        }
                    }
                }
            }

            const auto axis_dim = static_cast<uint32_t>(data_shape[static_cast<size_t>(axis_norm)]);
            const auto indices_count = static_cast<uint32_t>(ov::shape_size(idx_shape));
            if (axis_dim == 1 && indices_count == 1 && out_shape == data_shape) {
                GpuTensor* input = resolve_input_tensor(0);
                OPENVINO_ASSERT(input && input->buf.valid(),
                                "GFX MLIR: missing input buffer for Gather identity view ",
                                m_name);
                for (auto* out : outputs) {
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

            auto gather_payload = make_gather_runtime_param_payload(*m_buffer_manager,
                                                                    m_name,
                                                                    data_shape,
                                                                    idx_shape,
                                                                    static_cast<uint32_t>(axis_norm));
            m_kernel_extra_inputs = std::move(gather_payload.extra_inputs);
            if (is_vulkan_backend()) {
                set_kernel_binding_override(
                    make_kernel_runtime_binding_state({0, 1},
                                                      2,
                                                      {1, 1, 1, 1},
                                                      {0, 1, 2, 3}));
            }
        }

        if (auto transpose = ov::as_type_ptr<const ov::op::v1::Transpose>(m_node)) {
            ov::Shape in_shape = resolve_input_shape(0);
            if (in_shape.empty()) {
                OPENVINO_THROW("GFX MLIR: Transpose input shape is unknown for stage ", m_name);
            }
            auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
            OPENVINO_ASSERT(perm_const, "GFX MLIR: Transpose perm must be constant for stage ", m_name);
            const auto perm_i64 = perm_const->cast_vector<int64_t>();
            OPENVINO_ASSERT(perm_i64.size() == in_shape.size(),
                            "GFX MLIR: Transpose perm rank mismatch for stage ",
                            m_name);

            ov::Shape out_shape(in_shape.size(), 0);
            for (size_t i = 0; i < perm_i64.size(); ++i) {
                const int64_t axis = perm_i64[i];
                OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < in_shape.size(),
                                "GFX MLIR: Transpose perm out of range for stage ",
                                m_name);
                out_shape[i] = in_shape[static_cast<size_t>(axis)];
            }
            const auto in_stride_u32 = gfx_shape_strides_u32(in_shape);
            auto is_runtime_linear_view = [&]() {
                std::vector<size_t> out_stride(out_shape.size(), 1);
                for (int i = static_cast<int>(out_shape.size()) - 2; i >= 0; --i) {
                    out_stride[static_cast<size_t>(i)] =
                        out_stride[static_cast<size_t>(i + 1)] * out_shape[static_cast<size_t>(i + 1)];
                }
                for (size_t i = 0; i < out_shape.size(); ++i) {
                    if (out_shape[i] <= 1) {
                        continue;
                    }
                    if (out_stride[i] != static_cast<size_t>(in_stride_u32[static_cast<size_t>(perm_i64[i])])) {
                        return false;
                    }
                }
                return true;
            };

            const auto output_et = m_node->get_output_element_type(0);
            for (auto* out : outputs) {
                if (!out) {
                    continue;
                }
                out->shape = out_shape;
                out->expected_type = output_et;
            }
            m_output_shape = out_shape;

            if (is_runtime_linear_view()) {
                GpuTensor* input = resolve_input_tensor(0);
                OPENVINO_ASSERT(input && input->buf.valid(),
                                "GFX MLIR: missing input buffer for runtime Transpose view ",
                                m_name);
                for (auto* out : outputs) {
                    if (!out) {
                        continue;
                    }
                    out->buf = input->buf;
                    out->buf.external = true;
                    out->buf.owned = false;
                    out->expected_type = output_et;
                }
                return;
            }

            auto transpose_payload =
                make_transpose_runtime_param_payload(*m_buffer_manager, m_name, in_shape, out_shape, perm_i64);
            m_kernel_extra_inputs = std::move(transpose_payload.extra_inputs);
            if (is_vulkan_backend()) {
                set_kernel_binding_override(
                    make_kernel_runtime_binding_state({0},
                                                      1,
                                                      {1, 1, 1, 1, 1, 1, 1},
                                                      {0, 1, 2, 3, 4, 5, 6}));
            }
        }
    }

    if (m_type == "ShapeOf") {
        ov::Shape in_shape = resolve_input_shape(0);
        if (in_shape.empty()) {
            OPENVINO_THROW("GFX MLIR: ShapeOf input shape is unknown for stage ", m_name);
        }
        const auto output_et = m_node ? m_node->get_output_element_type(0) : ov::element::i64;
        OPENVINO_ASSERT(output_et == ov::element::i32 || output_et == ov::element::i64,
                        "GFX MLIR: ShapeOf output must be i32/i64");
        const size_t rank = in_shape.size();
        const ov::Shape out_shape{rank};
        std::vector<int64_t> shape_values;
        shape_values.reserve(rank);
        for (auto dim : in_shape) {
            shape_values.push_back(static_cast<int64_t>(dim));
        }
        for (auto* out : outputs) {
            if (out) {
                out->shape = out_shape;
                out->expected_type = output_et;
                out->i64_values = shape_values;
            }
        }
        if (bind_small_i64_const_outputs("shapeof_i64_const")) {
            return;
        }

        auto shapeof_payload =
            make_shapeof_runtime_param_payload(*m_buffer_manager, m_name, in_shape, output_et);
        const auto shapeof_scalars = shapeof_payload.scalar_args;
        m_kernel_extra_inputs = std::move(shapeof_payload.extra_inputs);
        auto shapeof_plan = make_kernel_runtime_binding_plan_for_custom_kernel(
            "ShapeOf",
            "shapeof_kernel",
            {0},
            shapeof_scalars,
            is_vulkan_backend() ? GfxKernelBackendDomain::Spirv : GfxKernelBackendDomain::AppleMsl,
            GfxKernelStorageKind::Buffer,
            is_vulkan_backend() ? "spirv:buffer:" : "apple_msl:buffer:");
        OPENVINO_ASSERT(shapeof_plan.valid,
                        "GFX MLIR: ShapeOf custom-kernel runtime binding manifest is invalid for stage ",
                        m_name);
        set_kernel_binding_override(shapeof_plan.runtime_binding);
    } else if (auto matmul = ov::as_type_ptr<const ov::op::v0::MatMul>(m_node)) {
        ov::Shape a_shape;
        ov::Shape b_shape;
        const bool a_known = resolve_known_input_shape(0, a_shape);
        bool b_known = resolve_known_input_shape(1, b_shape);
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
            OPENVINO_ASSERT(K == Kb,
                            "GFX MLIR: MatMul K mismatch for stage ",
                            m_name,
                            " (",
                            K,
                            " vs ",
                            Kb,
                            ")");
            const ov::Shape batch_prefix = broadcast_batch_prefix(a_shape, b_shape, "GFX MLIR: MatMul");
            const size_t batch = static_cast<size_t>(shape_product(batch_prefix, 0, batch_prefix.size()));
            const size_t batch_a = a_rank > 2 ? static_cast<size_t>(shape_product(a_shape, 0, a_rank - 2)) : 1;
            const size_t batch_b = b_rank > 2 ? static_cast<size_t>(shape_product(b_shape, 0, b_rank - 2)) : 1;
            OPENVINO_ASSERT(batch_a == batch_b || batch_a == 1 || batch_b == 1,
                            "GFX MLIR: MatMul mixed batch-prefix broadcast is not supported for stage ",
                            m_name);

            ov::Shape out_shape = batch_prefix;
            out_shape.push_back(M);
            out_shape.push_back(N);
            for (auto* out : outputs) {
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
            if (!m_is_compressed_matmul && (m_last_input_shape != matmul_key || !m_kernel)) {
                auto runtime_input_type = [&](size_t input_idx, const ov::element::Type& fallback) {
                    if (auto* tensor = resolve_input_tensor(input_idx)) {
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
                desc.input_a_type = runtime_input_type(0, matmul->get_input_element_type(0));
                desc.input_b_type = runtime_input_type(1, matmul->get_input_element_type(1));
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
                m_matmul_reduction_threads = gfx_matmul_parallel_reduction_threads(desc);

                std::string log;
                if (!is_vulkan_backend()) {
                    KernelSource src = make_apple_metal_runtime_matmul_kernel_source(gfx_mlir_context(),
                                                                                     m_buffer_manager,
                                                                                     m_node,
                                                                                     desc,
                                                                                     a_shape,
                                                                                     b_shape,
                                                                                     m_name);
                    OPENVINO_ASSERT(src.msl_generator || !src.msl_source.empty() || src.module,
                                    "GFX MLIR: failed to build runtime MatMul source for stage ",
                                    m_name);
                    try {
                        m_kernel = compile_kernel(src, &log);
                    } catch (const std::exception& e) {
                        OPENVINO_THROW("GFX MLIR: failed to compile MatMul stage ",
                                       m_name,
                                       ": ",
                                       e.what());
                    }
                    OPENVINO_ASSERT(m_kernel, "GFX MLIR: failed to compile MatMul stage ", m_name, ": ", log);
                    auto runtime_meta = extract_kernel_runtime_metadata(src.module,
                                                                        src.signature.output_arg_count,
                                                                        m_node->get_input_size(),
                                                                        src.entry_point);
                    apply_kernel_metadata(runtime_meta, /*scalar_inputs=*/0);
                }
                if (is_vulkan_backend()) {
                    set_kernel_binding_override(
                        make_kernel_runtime_binding_state({0, 1},
                                                          2,
                                                          {1, 1, 1},
                                                          {0, 1, 2}));
                }
                m_kernel_extra_inputs.clear();
                m_parallel_cfg = ParallelDispatchConfig{};
                m_force_single_dispatch = false;
                m_last_input_shape = std::move(matmul_key);
            }
        }
    } else if (ov::as_type_ptr<const ov::op::v1::Select>(m_node)) {
        ov::Shape cond_shape;
        ov::Shape true_shape;
        ov::Shape false_shape;
        if (resolve_known_input_shape(0, cond_shape) &&
            resolve_known_input_shape(1, true_shape) &&
            resolve_known_input_shape(2, false_shape)) {
            const ov::Shape data_shape = compute_binary_broadcast_shape(true_shape, false_shape);
            const ov::Shape out_shape = compute_binary_broadcast_shape(cond_shape, data_shape);
            for (auto* out : outputs) {
                if (!out) {
                    continue;
                }
                out->shape = out_shape;
                if (out->expected_type == ov::element::dynamic) {
                    out->expected_type = m_node->get_output_element_type(0);
                }
            }
            m_output_shape = out_shape;

            auto select_payload = make_select_runtime_param_payload(*m_buffer_manager,
                                                                    m_name,
                                                                    cond_shape,
                                                                    true_shape,
                                                                    false_shape,
                                                                    out_shape);
            const auto select_scalars = select_payload.scalar_args;
            m_kernel_extra_inputs = std::move(select_payload.extra_inputs);
            auto select_plan = make_kernel_runtime_binding_plan_for_custom_kernel(
                "Select",
                "select_kernel",
                {0, 1, 2},
                select_scalars,
                is_vulkan_backend() ? GfxKernelBackendDomain::Spirv : GfxKernelBackendDomain::AppleMsl,
                GfxKernelStorageKind::Buffer,
                is_vulkan_backend() ? "spirv:buffer:" : "apple_msl:buffer:");
            OPENVINO_ASSERT(select_plan.valid,
                            "GFX MLIR: Select custom-kernel runtime binding manifest is invalid for stage ",
                            m_name);
            set_kernel_binding_override(select_plan.runtime_binding);
        }
    } else if (auto scatter = ov::as_type_ptr<const ov::op::v3::ScatterUpdate>(m_node)) {
        ov::Shape data_shape;
        ov::Shape idx_shape;
        ov::Shape upd_shape;
        if (resolve_known_input_shape(0, data_shape) &&
            resolve_known_input_shape(1, idx_shape) &&
            resolve_known_input_shape(2, upd_shape)) {
            auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(scatter->input_value(3).get_node_shared_ptr());
            OPENVINO_ASSERT(axis_const, "GFX MLIR: ScatterUpdate axis must be constant for stage ", m_name);
            const auto axis_values = axis_const->cast_vector<int64_t>();
            OPENVINO_ASSERT(axis_values.size() == 1, "GFX MLIR: ScatterUpdate axis must be scalar for stage ", m_name);
            int64_t axis = axis_values[0];
            if (axis < 0) {
                axis += static_cast<int64_t>(data_shape.size());
            }
            OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < data_shape.size(),
                            "GFX MLIR: ScatterUpdate axis out of range for stage ",
                            m_name);
            OPENVINO_ASSERT(data_shape.size() <= 8 && idx_shape.size() <= 8 && upd_shape.size() <= 16,
                            "GFX MLIR: ScatterUpdate rank exceeds Metal params capacity for stage ",
                            m_name);

            for (auto* out : outputs) {
                if (!out) {
                    continue;
                }
                out->shape = data_shape;
                if (out->expected_type == ov::element::dynamic) {
                    out->expected_type = m_node->get_output_element_type(0);
                }
            }
            m_output_shape = data_shape;

            auto scatter_update_payload = make_scatter_update_runtime_param_payload(*m_buffer_manager,
                                                                                    m_name,
                                                                                    data_shape,
                                                                                    idx_shape,
                                                                                    upd_shape,
                                                                                    static_cast<uint32_t>(axis));
            m_kernel_extra_inputs = std::move(scatter_update_payload.extra_inputs);
            if (is_vulkan_backend()) {
                set_kernel_binding_override(
                    make_kernel_runtime_binding_state({0, 1, 2},
                                                      4,
                                                      {1, 1, 1, 1, 1},
                                                      {0, 1, 2, 4, 3}));
            }
        }
    } else if (m_type == "Slice" || m_type == "StridedSlice") {
        const bool use_runtime_slice_args = is_vulkan_backend() ||
                                            (m_node &&
                                             (!m_node->get_input_partial_shape(0).is_static() ||
                                              !m_node->get_output_partial_shape(0).is_static() ||
                                              slice_requires_runtime_indexing(m_node)));
        ov::Shape in_shape = resolve_input_shape(0);
        if (in_shape.empty()) {
            OPENVINO_THROW("GFX MLIR: Slice/StridedSlice input shape is unknown for stage ", m_name);
        }
        ov::Shape out_shape = outputs.front() && !outputs.front()->shape.empty()
                                  ? outputs.front()->shape
                                  : ov::Shape{};
        if (out_shape.empty() && m_node && m_node->get_output_partial_shape(0).is_static()) {
            out_shape = m_node->get_output_shape(0);
        }
        const size_t rank = in_shape.size();
        if (out_shape.empty()) {
            if (auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(m_node)) {
                auto starts = evaluate_optional_constant_source_i64(slice->input_value(1));
                auto ends = evaluate_optional_constant_source_i64(slice->input_value(2));
                auto steps = evaluate_optional_constant_source_i64(slice->input_value(3));
                std::optional<std::vector<int64_t>> axes;
                if (slice->get_input_size() > 4) {
                    axes = evaluate_optional_constant_source_i64(slice->input_value(4));
                }
                if (starts && ends && steps) {
                    out_shape = in_shape;
                    std::vector<int64_t> axes_values;
                    if (axes) {
                        axes_values = *axes;
                    } else {
                        axes_values.resize(starts->size());
                        std::iota(axes_values.begin(), axes_values.end(), 0);
                    }
                    OPENVINO_ASSERT(starts->size() == ends->size() &&
                                        starts->size() == steps->size() &&
                                        starts->size() == axes_values.size(),
                                    "GFX MLIR: Slice starts/ends/steps/axes size mismatch for stage ",
                                    m_name);
                    for (size_t i = 0; i < axes_values.size(); ++i) {
                        int64_t axis = axes_values[i];
                        if (axis < 0) {
                            axis += static_cast<int64_t>(rank);
                        }
                        OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank,
                                        "GFX MLIR: Slice axis out of range for stage ",
                                        m_name);
                        const auto dim = static_cast<int64_t>(in_shape[static_cast<size_t>(axis)]);
                        const int64_t step = (*steps)[i];
                        OPENVINO_ASSERT(step > 0, "GFX MLIR: Slice only supports positive steps for stage ", m_name);
                        const int64_t start = normalize_slice_index((*starts)[i], dim, true);
                        const int64_t finish = normalize_slice_index((*ends)[i], dim, false);
                        const int64_t extent = std::max<int64_t>(0, finish - start);
                        out_shape[static_cast<size_t>(axis)] =
                            static_cast<size_t>((extent + step - 1) / step);
                    }
                }
            }
        }
        if (out_shape.empty() && m_node && m_node->get_output_partial_shape(0).rank().is_static()) {
            const auto pshape = m_node->get_output_partial_shape(0);
            out_shape.reserve(static_cast<size_t>(pshape.rank().get_length()));
            for (size_t i = 0; i < static_cast<size_t>(pshape.rank().get_length()); ++i) {
                out_shape.push_back(pshape[i].is_static() ? static_cast<size_t>(pshape[i].get_length())
                                                          : (i < in_shape.size() ? in_shape[i] : 1));
            }
        }
        OPENVINO_ASSERT(!out_shape.empty(), "GFX MLIR: Slice/StridedSlice output shape is unknown for stage ", m_name);
        OPENVINO_ASSERT(rank == out_shape.size(), "GFX MLIR: Slice/StridedSlice rank mismatch for stage ", m_name);

        std::vector<int32_t> starts_full(rank, 0);
        std::vector<int32_t> steps_full(rank, 1);
        const auto in_stride = gfx_shape_strides_u32(in_shape);
        build_slice_runtime_spec(m_node, in_shape, out_shape, starts_full, steps_full);

        for (auto* out : outputs) {
            if (out) {
                out->shape = out_shape;
                if (m_node && m_node->get_output_element_type(0) != ov::element::dynamic) {
                    out->expected_type = m_node->get_output_element_type(0);
                }
            }
        }

        auto is_runtime_linear_view = [&]() {
            std::vector<uint32_t> out_stride(rank, 1);
            for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                out_stride[static_cast<size_t>(i)] =
                    out_stride[static_cast<size_t>(i + 1)] *
                    static_cast<uint32_t>(out_shape[static_cast<size_t>(i + 1)]);
            }
            for (size_t i = 0; i < rank; ++i) {
                if (starts_full[i] != 0 || steps_full[i] != 1) {
                    return false;
                }
                if (out_shape[i] <= 1) {
                    continue;
                }
                if (out_stride[i] != in_stride[i]) {
                    return false;
                }
            }
            return true;
        };

        if (is_runtime_linear_view()) {
            GpuTensor* input = resolve_input_tensor(0);
            OPENVINO_ASSERT(input && input->buf.valid(),
                            "GFX MLIR: missing input buffer for runtime Slice view ",
                            m_name);
            for (auto* out : outputs) {
                if (!out) {
                    continue;
                }
                out->buf = input->buf;
                out->buf.external = true;
                out->buf.owned = false;
                if (m_node && m_node->get_output_element_type(0) != ov::element::dynamic) {
                    out->expected_type = m_node->get_output_element_type(0);
                }
            }
            m_output_shape = out_shape;
            return;
        }

        m_kernel_extra_inputs.clear();
        if (use_runtime_slice_args) {
            auto slice_payload = make_slice_runtime_param_payload(*m_buffer_manager,
                                                                  m_name,
                                                                  in_shape,
                                                                  out_shape,
                                                                  starts_full,
                                                                  steps_full);
            m_kernel_extra_inputs = std::move(slice_payload.extra_inputs);
        }

        const bool needs_slice_kernel_compile =
            !m_kernel ||
            m_last_input_shape.empty() ||
            (!use_runtime_slice_args && m_last_input_shape != in_shape);
        if (m_node && needs_slice_kernel_compile) {
            if (!use_runtime_slice_args) {
                if (gfx_log_debug_enabled() && !m_inputs.empty() && m_inputs.front() && !outputs.empty() && outputs.front()) {
                    gfx_log_debug("MLIRExec") << "Slice types in_expected="
                                              << m_inputs.front()->expected_type
                                              << " in_buf="
                                              << m_inputs.front()->buf.type
                                              << " out_expected="
                                              << outputs.front()->expected_type
                                              << " out_buf="
                                              << outputs.front()->buf.type;
                    std::ostringstream buf_info;
                    buf_info << "Slice buffers in_handle=" << m_inputs.front()->buf.buffer
                             << " out_handle=" << outputs.front()->buf.buffer
                             << " in_size=" << m_inputs.front()->buf.size
                             << " out_size=" << outputs.front()->buf.size;
                    gfx_log_debug("MLIRExec") << buf_info.str();
                }
            }
            auto& ctx = gfx_mlir_context();
            auto module = build_mlir_for_node(m_node, ctx);
            if (module) {
                if (!is_vulkan_backend() && use_runtime_slice_args) {
                    annotate_msl_custom_kernel_binding_plan(module,
                                                                      "Slice",
                                                                      "slice_kernel",
                                                                      {0});
                } else if (is_vulkan_backend()) {
                    set_spirv_kernel_binding_override(
                        module,
                        make_kernel_runtime_binding_state(
                            {0},
                            1,
                            use_runtime_slice_args ? std::vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1}
                                                   : std::vector<int32_t>{1, 1},
                            use_runtime_slice_args ? std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}
                                                   : std::vector<int32_t>{0, 1}));
                }
            }
            auto plan_ctx = build_mlir_kernel_plan(
                module,
                {},
                m_node,
                outputs.size(),
                m_kernel_extra_inputs.size(),
                m_name.c_str(),
                "slice_main",
                [&](const KernelArgMappingInfo& info) -> size_t {
                    const auto sig = info.signature;
                    return sig.total()
                               ? sig.total()
                               : (info.mapping.kernel_inputs.size() + outputs.size() + m_kernel_extra_inputs.size());
                });
            plan_ctx.build_info.plan = KernelPlan(module,
                                                  plan_ctx.build_info.plan.entry_point(),
                                                  use_runtime_slice_args ? 8u : 2u);
            compile_from_plan(plan_ctx, module, "slice");
            if (is_vulkan_backend()) {
                set_kernel_binding_override(
                    make_kernel_runtime_binding_state(
                        {0},
                        1,
                        use_runtime_slice_args ? std::vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1}
                                               : std::vector<int32_t>{1, 1},
                        use_runtime_slice_args ? std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}
                                               : std::vector<int32_t>{0, 1}));
            } else {
                set_kernel_binding_override(make_kernel_runtime_binding_state({0}, 1, {}, {}));
            }
            m_last_input_shape = in_shape;
        }
    } else if (m_type == "Interpolate") {
        ov::Shape in_shape = resolve_input_shape(0);
        if (in_shape.empty()) {
            OPENVINO_THROW("GFX MLIR: Interpolate input shape is unknown for stage ", m_name);
        }
        ov::Shape out_shape = outputs.front() && !outputs.front()->shape.empty()
                                  ? outputs.front()->shape
                                  : ov::Shape{};
        if (out_shape.empty() && m_node && m_node->get_output_partial_shape(0).is_static()) {
            out_shape = m_node->get_output_shape(0);
        }
        OPENVINO_ASSERT(in_shape.size() == 4 && out_shape.size() == 4,
                        "GFX MLIR: Interpolate expects NCHW rank4");
        bool align_corners = false;
        bool use_half_pixel = true;
        uint32_t nearest_mode = 0;
        if (auto v0 = ov::as_type_ptr<const ov::op::v0::Interpolate>(m_node)) {
            align_corners = v0->get_attrs().align_corners;
            use_half_pixel = !align_corners;
        } else if (auto v4 = ov::as_type_ptr<const ov::op::v4::Interpolate>(m_node)) {
            using Base = ov::op::util::InterpolateBase;
            align_corners =
                v4->get_attrs().coordinate_transformation_mode == Base::CoordinateTransformMode::ALIGN_CORNERS;
            use_half_pixel =
                v4->get_attrs().coordinate_transformation_mode == Base::CoordinateTransformMode::HALF_PIXEL;
            switch (v4->get_attrs().nearest_mode) {
                case Base::NearestMode::FLOOR:
                case Base::NearestMode::ROUND_PREFER_FLOOR:
                    nearest_mode = 1;
                    break;
                case Base::NearestMode::CEIL:
                case Base::NearestMode::ROUND_PREFER_CEIL:
                    nearest_mode = 2;
                    break;
                case Base::NearestMode::SIMPLE:
                default:
                    nearest_mode = 0;
                    break;
            }
        } else if (auto v11 = ov::as_type_ptr<const ov::op::v11::Interpolate>(m_node)) {
            using Base = ov::op::util::InterpolateBase;
            align_corners =
                v11->get_attrs().coordinate_transformation_mode == Base::CoordinateTransformMode::ALIGN_CORNERS;
            use_half_pixel =
                v11->get_attrs().coordinate_transformation_mode == Base::CoordinateTransformMode::HALF_PIXEL;
            switch (v11->get_attrs().nearest_mode) {
                case Base::NearestMode::FLOOR:
                case Base::NearestMode::ROUND_PREFER_FLOOR:
                    nearest_mode = 1;
                    break;
                case Base::NearestMode::CEIL:
                case Base::NearestMode::ROUND_PREFER_CEIL:
                    nearest_mode = 2;
                    break;
                case Base::NearestMode::SIMPLE:
                default:
                    nearest_mode = 0;
                    break;
            }
        } else {
            OPENVINO_THROW("GFX MLIR: unsupported Interpolate op kind");
        }
        for (auto* out : outputs) {
            if (out) {
                out->shape = out_shape;
            }
        }
        const bool use_runtime_interpolate_params = !is_vulkan_backend();
        m_kernel_extra_inputs.clear();
        if (use_runtime_interpolate_params) {
            auto interpolate_payload = make_interpolate_runtime_param_payload(*m_buffer_manager,
                                                                              m_name,
                                                                              in_shape,
                                                                              out_shape,
                                                                              align_corners,
                                                                              use_half_pixel,
                                                                              nearest_mode);
            m_kernel_extra_inputs = std::move(interpolate_payload.extra_inputs);
        }
        if (m_node && (!m_kernel || m_last_input_shape.empty())) {
            auto& ctx = gfx_mlir_context();
            auto module = build_mlir_for_node(m_node, ctx);
            if (module) {
                if (!is_vulkan_backend() && use_runtime_interpolate_params) {
                    annotate_msl_custom_kernel_binding_plan(module,
                                                                      "Interpolate",
                                                                     "interpolate_kernel",
                                                                     {0});
                } else if (is_vulkan_backend()) {
                    set_spirv_kernel_binding_override(
                        module,
                        make_kernel_runtime_binding_state(
                            {0},
                            1,
                            use_runtime_interpolate_params ? std::vector<int32_t>{1, 1, 1}
                                                           : std::vector<int32_t>{1, 1},
                            use_runtime_interpolate_params ? std::vector<int32_t>{0, 1, 2}
                                                           : std::vector<int32_t>{0, 1}));
                }
            }
            auto plan_ctx = build_mlir_kernel_plan(
                module,
                {},
                m_node,
                outputs.size(),
                m_kernel_extra_inputs.size(),
                m_name.c_str(),
                "interpolate_main",
                [&](const KernelArgMappingInfo& info) -> size_t {
                    const auto sig = info.signature;
                    return sig.total()
                               ? sig.total()
                               : (info.mapping.kernel_inputs.size() + outputs.size() + m_kernel_extra_inputs.size());
                });
            plan_ctx.build_info.plan =
                KernelPlan(module, plan_ctx.build_info.plan.entry_point(), use_runtime_interpolate_params ? 3u : 2u);
            compile_from_plan(plan_ctx, module, "interpolate");
            if (is_vulkan_backend()) {
                set_kernel_binding_override(
                    make_kernel_runtime_binding_state(
                        {0},
                        1,
                        use_runtime_interpolate_params ? std::vector<int32_t>{1, 1, 1}
                                                       : std::vector<int32_t>{1, 1},
                        use_runtime_interpolate_params ? std::vector<int32_t>{0, 1, 2}
                                                       : std::vector<int32_t>{0, 1}));
            }
            m_last_input_shape = in_shape;
        }
    } else if (m_type == "Softmax" || m_type == "LogSoftmax") {
        ov::Shape in_shape = resolve_input_shape(0);
        if (in_shape.empty()) {
            OPENVINO_THROW("GFX MLIR: Softmax input shape is unknown for stage ", m_name);
        }
        int64_t axis = -1;
        if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(m_node)) axis = s1->get_axis();
        else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(m_node)) axis = s8->get_axis();
        else if (auto ls = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(m_node)) axis = ls->get_axis();
        else OPENVINO_THROW("GFX MLIR: unsupported softmax op kind");
        const auto dims = compute_softmax_dims(in_shape, axis, "GFX MLIR: Softmax");
        const uint64_t total_work = dims.rows;
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MLIRSoftmax") << "shape_rank=" << in_shape.size()
                                         << " axis=" << dims.axis
                                         << " rows*inner=" << total_work
                                         << " tiled=0";
        }
        for (auto* out : outputs) {
            if (out) {
                out->shape = in_shape;
            }
        }
        auto softmax_payload = make_softmax_runtime_param_payload(*m_buffer_manager,
                                                                  m_name,
                                                                  dims.rows,
                                                                  dims.axis_len,
                                                                  dims.inner);
        m_kernel_extra_inputs = std::move(softmax_payload.extra_inputs);
        if (m_node && (!m_kernel || m_last_input_shape.empty())) {
            auto& ctx = gfx_mlir_context();
            const bool log_softmax = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(m_node) != nullptr;
            auto module = log_softmax
                              ? build_mlir_logsoftmax_from_node(m_node, ctx, in_shape)
                              : build_mlir_softmax_from_node(m_node, ctx, in_shape);
            if (module) {
                module->setAttr("gfx.prefer_parallel",
                                mlir::BoolAttr::get(module.getContext(), false));
                if (!is_vulkan_backend()) {
                    annotate_msl_custom_kernel_binding_plan(module,
                                                                      m_type,
                                                                      "softmax_kernel",
                                                                      {0});
                } else {
                    // Vulkan keeps a straightforward 0,1,2 mapping to avoid
                    // backend-specific descriptor ordering issues.
                    set_spirv_kernel_binding_override(
                        module,
                        make_kernel_runtime_binding_state({0},
                                                          1,
                                                          {1, 1, 1},
                                                          {0, 1, 2}));
                }
            }
            auto plan_ctx = build_mlir_kernel_plan(
                module,
                {},
                m_node,
                outputs.size(),
                m_kernel_extra_inputs.size(),
                m_name.c_str(),
                "softmax_main",
                [&](const KernelArgMappingInfo& info) -> size_t {
                    const auto sig = info.signature;
                    return sig.total()
                               ? sig.total()
                               : (info.mapping.kernel_inputs.size() + outputs.size());
                });
            // Softmax MSL kernel expects 3 buffers (input, output, params).
            plan_ctx.build_info.plan = KernelPlan(module, plan_ctx.build_info.plan.entry_point(), 3);
            compile_from_plan(plan_ctx, module, "softmax");
            if (is_vulkan_backend()) {
                set_kernel_binding_override(
                    make_kernel_runtime_binding_state({0},
                                                      1,
                                                      {1, 1, 1},
                                                      {0, 1, 2}));
            }
            m_last_input_shape = in_shape;
        }
    } else if (m_type == "Split" || m_type == "VariadicSplit") {
        ov::Shape stage_input_shape = resolve_input_shape(0);
        if (stage_input_shape.empty()) {
            OPENVINO_THROW("GFX MLIR: Split input shape is unknown for stage ", m_name);
        }
        ov::Shape logical_input_shape = stage_input_shape;
        if (m_node && m_node->get_input_partial_shape(0).is_static()) {
            logical_input_shape = m_node->get_input_shape(0);
        }
        const auto plan = make_split_plan(m_node, logical_input_shape, outputs);
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
        const bool needs_split_compile = (m_last_input_shape != stage_input_shape || !m_kernel);
        if (needs_split_compile) {
            auto& ctx = gfx_mlir_context();
            auto module = build_mlir_split_from_node(m_node,
                                                     ctx,
                                                     stage_input_shape,
                                                     input_transform(0));
            if (!has_input_transform && m_node) {
                const ov::element::Type out_et = m_node ? m_node->get_output_element_type(0)
                                                        : (outputs.empty() ? ov::element::f32
                                                                            : outputs.front()->expected_type);
                auto split_source_plan = make_direct_split_msl_kernel_source_plan(m_type,
                                                                                  out_et,
                                                                                  logical_input_shape,
                                                                                  plan.split_sizes,
                                                                                  plan.axis_len,
                                                                                  plan.inner_stride,
                                                                                  module);
                OPENVINO_ASSERT(split_source_plan.valid(),
                                "GFX MLIR: direct Split MSL source plan is invalid for stage ",
                                m_name);
                compile_prebuilt_kernel_source(split_source_plan.source,
                                               split_source_plan.binding.kernel_runtime_binding(),
                                               "direct Split");
            } else {
                auto plan_ctx = build_mlir_kernel_plan(
                    module,
                    "split_main",
                    m_node,
                    outputs.size(),
                    /*extra_inputs=*/0,
                    m_name.c_str(),
                    "split_main",
                    [&](const KernelArgMappingInfo& info) -> size_t {
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
            ensure_output_shape(i, outputs[i]);
        }
    }

    auto is_unary_eltwise_stage = [&]() {
        return m_type == "Relu" || m_type == "Sigmoid" || m_type == "Tanh" || m_type == "Elu" ||
               m_type == "Gelu" || m_type == "Swish" || m_type == "HSwish" || m_type == "HSigmoid" ||
               m_type == "SoftPlus" || m_type == "Mish" || m_type == "SoftSign" || m_type == "Abs" ||
               m_type == "Sign" || m_type == "Clamp" || m_type == "Exp" || m_type == "Log" ||
               m_type == "Sqrt" || m_type == "Floor" || m_type == "Ceiling" || m_type == "Negative" ||
               m_type == "Sin" || m_type == "Cos" || m_type == "Tan" || m_type == "Erf" ||
               m_type == "Asin" || m_type == "Acos" || m_type == "Atan" || m_type == "Asinh" ||
               m_type == "Acosh" || m_type == "Atanh" || m_type == "Sinh" || m_type == "Cosh" ||
               m_type == "Round";
    };

    if (m_node && is_unary_eltwise_stage() && m_node->get_input_size() >= 1 && !outputs.empty()) {
        ov::Shape input_shape;
        if (resolve_known_input_shape(0, input_shape)) {
            for (auto* out : outputs) {
                if (!out) {
                    continue;
                }
                out->shape = input_shape;
                if (out->expected_type == ov::element::dynamic) {
                    out->expected_type = m_node->get_output_element_type(0);
                }
            }
            m_output_shape = input_shape;
            kernel_scalar_args_override = std::vector<int32_t>{
                static_cast<int32_t>(ov::shape_size(input_shape))};
        }
    }

    auto is_binary_eltwise_stage = [&]() {
        return m_type == "Add" || m_type == "Subtract" || m_type == "Multiply" || m_type == "Divide" ||
               m_type == "Power" || m_type == "Mod" || m_type == "FloorMod" || m_type == "Minimum" ||
               m_type == "Maximum" || m_type == "Equal" || m_type == "NotEqual" || m_type == "Less" ||
               m_type == "Greater" || m_type == "LessEqual" || m_type == "GreaterEqual" ||
               m_type == "LogicalAnd" || m_type == "LogicalOr" || m_type == "LogicalXor" ||
               m_type == "SquaredDifference" || m_type == "PRelu";
    };
    const bool uses_compact_bias_add_kernel =
        is_vulkan_backend() && m_type == "Add" && m_node && is_bias_broadcast_add(m_node) &&
        !has_absorbed_input_transpose();

    if (m_node && is_binary_eltwise_stage() && m_node->get_input_size() >= 2 && !outputs.empty()) {
        ov::Shape lhs_shape;
        ov::Shape rhs_shape;
        const bool lhs_known = resolve_known_input_shape(0, lhs_shape);
        const bool rhs_known = resolve_known_input_shape(1, rhs_shape);
        if (lhs_known && rhs_known) {
            const ov::Shape out_shape = compute_binary_broadcast_shape(lhs_shape, rhs_shape);
            for (auto* out : outputs) {
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
                auto lhs_values = resolve_i64_values(0);
                auto rhs_values = resolve_i64_values(1);
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
                        for (auto* out : outputs) {
                            assign_i64_values(out, values, out_shape);
                        }
                        if (bind_small_i64_const_outputs("add_i64_const")) {
                            return;
                        }
                    }
                }
            }

            if (uses_compact_bias_add_kernel) {
                m_kernel_extra_inputs.clear();
                kernel_scalar_args_override = std::vector<int32_t>{};
            } else {
                ov::Shape meta_shape = out_shape.empty() ? ov::Shape{1} : out_shape;
                auto stride0 = compute_broadcast_element_strides(lhs_shape, meta_shape);
                auto stride1 = compute_broadcast_element_strides(rhs_shape, meta_shape);
                auto broadcast_payload = make_binary_broadcast_runtime_param_payload(*m_buffer_manager,
                                                                                    m_name,
                                                                                    out_shape,
                                                                                    std::move(stride0),
                                                                                    std::move(stride1));
                m_kernel_extra_inputs = std::move(broadcast_payload.extra_inputs);
                kernel_scalar_args_override = std::move(broadcast_payload.scalar_args);
            }
        }
    }

    if (m_is_view_op && m_type == "ReadValue" && !outputs.empty() && !m_inputs.empty() && m_inputs[0]) {
        outputs.front()->shape = m_inputs[0]->shape;
        if (outputs.front()->expected_type == ov::element::dynamic) {
            outputs.front()->expected_type =
                m_inputs[0]->expected_type == ov::element::dynamic ? m_inputs[0]->buf.type : m_inputs[0]->expected_type;
        }
    }

    if (m_is_view_op && m_type == "Reshape" && !outputs.empty() && !m_inputs.empty() && m_inputs[0]) {
        ov::Shape in_shape;
        bool in_shape_known = resolve_input_shape_known(0, in_shape);
        if (!in_shape_known && m_node && m_node->get_input_partial_shape(0).rank().is_static()) {
            const auto pshape = m_node->get_input_partial_shape(0);
            in_shape.reserve(static_cast<size_t>(pshape.rank().get_length()));
            for (size_t i = 0; i < static_cast<size_t>(pshape.rank().get_length()); ++i) {
                in_shape.push_back(pshape[i].is_static() ? static_cast<size_t>(pshape[i].get_length()) : 1);
            }
            in_shape_known = true;
        }
        OPENVINO_ASSERT(in_shape_known, "GFX MLIR: Reshape input shape is unknown for stage ", m_name);
        ov::Shape out_shape;
        if (auto reshape = ov::as_type_ptr<const ov::op::v1::Reshape>(m_node)) {
            auto pattern = resolve_i64_values(1);
            if (!pattern) {
                pattern = evaluate_optional_constant_source_i64(reshape->input_value(1));
            }
            if (pattern) {
                out_shape.reserve(pattern->size());
                int64_t infer_pos = -1;
                size_t known_product = 1;
                for (size_t i = 0; i < pattern->size(); ++i) {
                    const int64_t dim = (*pattern)[i];
                    if (dim == 0 && reshape->get_special_zero()) {
                        OPENVINO_ASSERT(i < in_shape.size(),
                                        "GFX MLIR: Reshape special_zero axis out of range for stage ",
                                        m_name);
                        out_shape.push_back(in_shape[i]);
                        known_product *= in_shape[i];
                    } else if (dim == -1) {
                        OPENVINO_ASSERT(infer_pos < 0, "GFX MLIR: Reshape has multiple -1 dims for stage ", m_name);
                        infer_pos = static_cast<int64_t>(out_shape.size());
                        out_shape.push_back(1);
                    } else {
                        OPENVINO_ASSERT(dim >= 0, "GFX MLIR: Reshape target dim is invalid for stage ", m_name);
                        out_shape.push_back(static_cast<size_t>(dim));
                        known_product *= static_cast<size_t>(dim);
                    }
                }
                if (infer_pos >= 0) {
                    const size_t input_elems = ov::shape_size(in_shape);
                    OPENVINO_ASSERT(known_product != 0 && input_elems % known_product == 0,
                                    "GFX MLIR: Reshape cannot infer -1 dim for stage ",
                                    m_name);
                    out_shape[static_cast<size_t>(infer_pos)] = input_elems / known_product;
                }
            }
        }
        if (out_shape.empty() && m_node && m_node->get_output_partial_shape(0).rank().is_static()) {
            const auto pshape = m_node->get_output_partial_shape(0);
            out_shape.reserve(static_cast<size_t>(pshape.rank().get_length()));
            for (size_t i = 0; i < static_cast<size_t>(pshape.rank().get_length()); ++i) {
                out_shape.push_back(pshape[i].is_static() ? static_cast<size_t>(pshape[i].get_length()) : 1);
            }
        }
        for (auto* out : outputs) {
            if (!out) {
                continue;
            }
            out->shape = out_shape;
            if (out->expected_type == ov::element::dynamic) {
                out->expected_type =
                    m_inputs[0]->expected_type == ov::element::dynamic ? m_inputs[0]->buf.type : m_inputs[0]->expected_type;
            }
            if (auto input_values = resolve_i64_values(0)) {
                assign_i64_values(out, *input_values, out_shape);
            }
        }
        if (bind_small_i64_const_outputs("reshape_i64_const")) {
            return;
        }
    }

    if (m_is_view_op && (m_type == "Squeeze" || m_type == "Unsqueeze") &&
        !outputs.empty() && !m_inputs.empty() && m_inputs[0]) {
        ov::Shape in_shape = resolve_input_shape(0);
        OPENVINO_ASSERT(!in_shape.empty(), "GFX MLIR: ", m_type, " input shape is unknown for stage ", m_name);
        ov::Shape out_shape;
        if (auto squeeze = ov::as_type_ptr<const ov::op::v0::Squeeze>(m_node)) {
            std::vector<int64_t> axes;
            if (squeeze->get_input_size() > 1) {
                axes = cached_constant_i64_input(1, "Squeeze axes");
                ov::util::normalize_axes(axes, static_cast<int64_t>(in_shape.size()));
            }
            for (size_t i = 0; i < in_shape.size(); ++i) {
                const bool remove_axis = axes.empty()
                                             ? (in_shape[i] == 1)
                                             : (std::find(axes.begin(), axes.end(), static_cast<int64_t>(i)) != axes.end());
                if (!remove_axis) {
                    out_shape.push_back(in_shape[i]);
                }
            }
        } else if (auto unsqueeze = ov::as_type_ptr<const ov::op::v0::Unsqueeze>(m_node)) {
            auto axes = cached_constant_i64_input(1, "Unsqueeze axes");
            ov::util::normalize_axes(axes, static_cast<int64_t>(in_shape.size() + axes.size()));
            std::sort(axes.begin(), axes.end());
            out_shape = in_shape;
            for (size_t i = 0; i < axes.size(); ++i) {
                out_shape.insert(out_shape.begin() + static_cast<std::ptrdiff_t>(axes[i]), 1);
            }
        }
        if (out_shape.empty()) {
            out_shape = ov::Shape{1};
        }
        for (auto* out : outputs) {
            if (!out) {
                continue;
            }
            out->shape = out_shape;
            if (out->expected_type == ov::element::dynamic) {
                out->expected_type =
                    m_inputs[0]->expected_type == ov::element::dynamic ? m_inputs[0]->buf.type : m_inputs[0]->expected_type;
            }
            if (auto input_values = resolve_i64_values(0)) {
                assign_i64_values(out, *input_values, out_shape);
            }
        }
        if (bind_small_i64_const_outputs("shape_view_i64_const")) {
            return;
        }
    }

    if (auto reduce_info = get_runtime_reduce_info(m_node)) {
        ov::Shape in_shape = resolve_input_shape(0);
        OPENVINO_ASSERT(!in_shape.empty(), "GFX MLIR: Reduce input shape is unknown for stage ", m_name);
        const size_t rank = in_shape.size();
        OPENVINO_ASSERT(rank <= 8, "GFX MLIR: Reduce rank exceeds kernel metadata capacity for stage ", m_name);

        ov::Shape out_shape;
        out_shape.reserve(rank);
        for (size_t i = 0; i < rank; ++i) {
            const bool reduced = reduce_info->axes.count(i) != 0;
            if (reduced) {
                if (reduce_info->keep_dims) {
                    out_shape.push_back(1);
                }
            } else {
                out_shape.push_back(in_shape[i]);
            }
        }
        if (out_shape.empty()) {
            out_shape = ov::Shape{1};
        }

        for (auto* out : outputs) {
            if (!out) {
                continue;
            }
            out->shape = out_shape;
            if (out->expected_type == ov::element::dynamic && m_node) {
                out->expected_type = m_node->get_output_element_type(0);
            }
            if (auto input_values = resolve_i64_values(0)) {
                if (auto reduced_values = compute_reduce_i64_values(*input_values, in_shape, *reduce_info, out_shape)) {
                    assign_i64_values(out, *reduced_values, out_shape);
                }
            }
        }
        if (bind_small_i64_const_outputs("reduce_i64_const")) {
            return;
        }
        m_output_shape = out_shape;
        auto reduce_payload = make_reduce_runtime_param_payload(*m_buffer_manager,
                                                                m_name,
                                                                in_shape,
                                                                reduce_info->axes,
                                                                reduce_info->keep_dims,
                                                                out_shape);
        const auto reduce_scalars = reduce_payload.scalar_args;
        m_kernel_extra_inputs = std::move(reduce_payload.extra_inputs);
        auto reduce_plan = make_kernel_runtime_binding_plan_for_custom_kernel(
            m_type,
            "reduce_kernel",
            {0},
            reduce_scalars,
            is_vulkan_backend() ? GfxKernelBackendDomain::Spirv : GfxKernelBackendDomain::AppleMsl,
            GfxKernelStorageKind::Buffer,
            is_vulkan_backend() ? "spirv:buffer:" : "apple_msl:buffer:");
        OPENVINO_ASSERT(reduce_plan.valid,
                        "GFX MLIR: Reduce custom-kernel runtime binding manifest is invalid for stage ",
                        m_name);
        set_kernel_binding_override(reduce_plan.runtime_binding);
    }

    if (ov::as_type_ptr<const ov::op::v1::Broadcast>(m_node) ||
        ov::as_type_ptr<const ov::op::v3::Broadcast>(m_node)) {
        ov::Shape in_shape = resolve_input_shape(0);
        bool bidirectional_broadcast = false;
        if (auto broadcast_v3 = ov::as_type_ptr<const ov::op::v3::Broadcast>(m_node)) {
            bidirectional_broadcast =
                broadcast_v3->get_broadcast_spec().m_type == ov::op::BroadcastType::BIDIRECTIONAL;
        }
        const auto out_pshape = m_node->get_output_partial_shape(0);
        OPENVINO_ASSERT(out_pshape.rank().is_static(), "GFX MLIR: Broadcast output rank is dynamic for stage ", m_name);
        const size_t out_rank = static_cast<size_t>(out_pshape.rank().get_length());
        OPENVINO_ASSERT(out_rank <= 8, "GFX MLIR: Broadcast rank exceeds kernel metadata capacity for stage ", m_name);
        const size_t in_rank = in_shape.size();
        OPENVINO_ASSERT(in_rank <= out_rank, "GFX MLIR: Broadcast input rank exceeds output rank for stage ", m_name);

        ov::Shape out_shape;
        if (m_node->get_input_size() > 1) {
            auto target = resolve_i64_values(1);
            if (!target) {
                target = evaluate_optional_constant_source_i64(m_node->input_value(1));
            }
            if (target) {
                ov::Shape target_shape;
                target_shape.reserve(target->size());
                out_shape.reserve(target->size());
                for (auto dim : *target) {
                    target_shape.push_back(static_cast<size_t>(std::max<int64_t>(dim, 0)));
                }
                out_shape = bidirectional_broadcast ? compute_binary_broadcast_shape(in_shape, target_shape)
                                                    : target_shape;
                if (bidirectional_broadcast) {
                    OPENVINO_ASSERT(out_shape.size() == out_rank,
                                    "GFX MLIR: Broadcast bidirectional output rank mismatch for stage ",
                                    m_name,
                                    " input=",
                                    in_shape,
                                    " target=",
                                    target_shape,
                                    " output=",
                                    out_shape);
                }
            }
        }
        if (out_shape.empty()) {
            out_shape.resize(out_rank, 1);
            for (size_t i = 0; i < out_rank; ++i) {
                const auto& dim = out_pshape[i];
                if (dim.is_static()) {
                    out_shape[i] = static_cast<size_t>(dim.get_length());
                    continue;
                }
                const size_t aligned_input_axis = out_rank - in_rank;
                if (i >= aligned_input_axis && !in_shape.empty()) {
                    const size_t input_axis = i - aligned_input_axis;
                    if (input_axis < in_shape.size() && in_shape[input_axis] != 1) {
                        out_shape[i] = in_shape[input_axis];
                    }
                }
            }
        }
        OPENVINO_ASSERT(out_shape.size() == out_rank,
                        "GFX MLIR: Broadcast target shape rank mismatch for stage ",
                        m_name);

        for (auto* out : outputs) {
            if (!out) {
                continue;
            }
            out->shape = out_shape;
            if (out->expected_type == ov::element::dynamic && m_node) {
                out->expected_type = m_node->get_output_element_type(0);
            }
            if (auto input_values = resolve_i64_values(0)) {
                assign_i64_values(out, *input_values, in_shape);
            }
        }
        m_output_shape = out_shape;
        if (bind_small_i64_const_outputs("broadcast_i64_const")) {
            return;
        }
        if (try_alias_same_shape_unary_view(resolve_input_tensor(0),
                                            outputs,
                                            out_shape,
                                            m_node->get_output_element_type(0))) {
            return;
        }
        if (try_alias_gqa_broadcast_view(m_node, resolve_input_tensor(0), outputs)) {
            return;
        }
        auto broadcast_payload =
            make_broadcast_runtime_param_payload(*m_buffer_manager, m_name, in_shape, out_shape);
        const auto broadcast_scalars = broadcast_payload.scalar_args;
        m_kernel_extra_inputs = std::move(broadcast_payload.extra_inputs);
        auto broadcast_plan = make_kernel_runtime_binding_plan_for_custom_kernel(
            "Broadcast",
            "broadcast_kernel",
            {0},
            broadcast_scalars,
            is_vulkan_backend() ? GfxKernelBackendDomain::Spirv : GfxKernelBackendDomain::AppleMsl,
            GfxKernelStorageKind::Buffer,
            is_vulkan_backend() ? "spirv:buffer:" : "apple_msl:buffer:");
        OPENVINO_ASSERT(broadcast_plan.valid,
                        "GFX MLIR: Broadcast custom-kernel runtime binding manifest is invalid for stage ",
                        m_name);
        set_kernel_binding_override(broadcast_plan.runtime_binding);
    }

    if (m_type == "Convert") {
        ov::Shape in_shape = resolve_input_shape(0);
        OPENVINO_ASSERT(!in_shape.empty(), "GFX MLIR: Convert input shape is unknown for stage ", m_name);
        for (auto* out : outputs) {
            if (!out) {
                continue;
            }
            out->shape = in_shape;
            if (out->expected_type == ov::element::dynamic && m_node) {
                out->expected_type = m_node->get_output_element_type(0);
            }
            if (auto input_values = resolve_i64_values(0)) {
                assign_i64_values(out, *input_values, in_shape);
            }
        }
        m_output_shape = in_shape;
        if (bind_small_i64_const_outputs("convert_i64_const")) {
            return;
        }
        const std::vector<int32_t> convert_scalars = {static_cast<int32_t>(ov::shape_size(in_shape))};
        auto convert_plan = make_kernel_runtime_binding_plan_for_custom_kernel(
            "Convert",
            "convert_kernel",
            {0},
            convert_scalars,
            is_vulkan_backend() ? GfxKernelBackendDomain::Spirv : GfxKernelBackendDomain::AppleMsl,
            GfxKernelStorageKind::Buffer,
            is_vulkan_backend() ? "spirv:buffer:" : "apple_msl:buffer:");
        OPENVINO_ASSERT(convert_plan.valid,
                        "GFX MLIR: Convert custom-kernel runtime binding manifest is invalid for stage ",
                        m_name);
        set_kernel_binding_override(convert_plan.runtime_binding);
    }

    if (m_type == "Range") {
        auto start_values = resolve_i64_values(0);
        auto stop_values = resolve_i64_values(1);
        auto step_values = resolve_i64_values(2);
        ov::Shape out_shape;
        std::vector<int64_t> range_values;
        if (start_values && stop_values && step_values &&
            start_values->size() == 1 && stop_values->size() == 1 && step_values->size() == 1) {
            const int64_t start = (*start_values)[0];
            const int64_t stop = (*stop_values)[0];
            const int64_t step = (*step_values)[0];
            const size_t len = range_length_i64(start, stop, step);
            out_shape = ov::Shape{len};
            range_values.reserve(len);
            for (size_t ri = 0; ri < len; ++ri) {
                range_values.push_back(start + static_cast<int64_t>(ri) * step);
            }
        } else if (m_node && m_node->get_output_partial_shape(0).is_static()) {
            out_shape = m_node->get_output_shape(0);
        }
        OPENVINO_ASSERT(!out_shape.empty(), "GFX MLIR: Range output shape is unknown for stage ", m_name);
        const auto output_et = m_node ? m_node->get_output_element_type(0) : ov::element::i64;
        for (auto* out : outputs) {
            if (!out) {
                continue;
            }
            out->shape = out_shape;
            out->expected_type = output_et;
            if (!range_values.empty() || ov::shape_size(out_shape) == 0) {
                assign_i64_values(out, range_values, out_shape);
            }
        }
        m_output_shape = out_shape;
        if (bind_small_i64_const_outputs("range_i64_const")) {
            return;
        }
        const std::vector<int32_t> range_scalars = {static_cast<int32_t>(ov::shape_size(out_shape))};
        auto range_plan = make_kernel_runtime_binding_plan_for_custom_kernel(
            "Range",
            "range_kernel",
            {0, 1, 2},
            range_scalars,
            is_vulkan_backend() ? GfxKernelBackendDomain::Spirv : GfxKernelBackendDomain::AppleMsl,
            GfxKernelStorageKind::Buffer,
            is_vulkan_backend() ? "spirv:buffer:" : "apple_msl:buffer:");
        OPENVINO_ASSERT(range_plan.valid,
                        "GFX MLIR: Range custom-kernel runtime binding manifest is invalid for stage ",
                        m_name);
        set_kernel_binding_override(range_plan.runtime_binding);
    }

    if (m_type == "RMS" && m_node && !outputs.empty()) {
        ov::Shape out_shape;
        if (!resolve_known_input_shape(0, out_shape)) {
            out_shape = resolve_input_shape(0);
        }
        if (!out_shape.empty()) {
            for (auto* out : outputs) {
                if (!out) {
                    continue;
                }
                out->shape = out_shape;
                if (out->expected_type == ov::element::dynamic) {
                    out->expected_type = m_node->get_output_element_type(0);
                }
            }
            m_output_shape = out_shape;
        }
    }

    if (m_type == "RoPE" && m_node && !outputs.empty()) {
        ov::Shape out_shape;
        if (!resolve_known_input_shape(0, out_shape)) {
            out_shape = resolve_input_shape(0);
        }
        if (!out_shape.empty()) {
            for (auto* out : outputs) {
                if (!out) {
                    continue;
                }
                out->shape = out_shape;
                if (out->expected_type == ov::element::dynamic) {
                    out->expected_type = m_node->get_output_element_type(0);
                }
            }
            m_output_shape = out_shape;
        }
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        GpuTensor* out = outputs[i];
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
        const auto elem_size = out_type == ov::element::dynamic ? out->buf.type.size() : out_type.size();
        const size_t out_bytes = ov::shape_size(out->shape) * elem_size;
        if (m_is_view_op) {
            out->expected_type = out_type;
            continue;
        }
        auto allocate_output_buffer = [&](size_t bytes) {
            OPENVINO_ASSERT(m_buffer_manager, "GFX MLIR: buffer manager is not set for stage ", m_name);
            GpuBufferDesc desc{};
            desc.bytes = bytes;
            desc.type = out_type;
            desc.usage = out->prefer_private ? BufferUsage::Intermediate : BufferUsage::IO;
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
                GpuTensor* src = resolve_input_tensor(input_idx);
                if (src && same_gpu_allocation(out->buf, src->buf)) {
                    return true;
                }
            }
            return false;
        };
        if (!out->buf.valid()) {
            out->buf = allocate_output_buffer(out_bytes);
            OPENVINO_ASSERT(out->buf.valid(), "GFX MLIR: output buffer is not allocated for stage ", m_name);
        }
        const bool aliases_input = output_aliases_input();
        if (out->buf.size < out_bytes || aliases_input) {
            if (out->buf.owned && !out->buf.external) {
                if (!out->buf.from_handle && !aliases_input) {
                    m_buffer_manager->release_temp(std::move(out->buf));
                }
                out->buf = allocate_output_buffer(out_bytes);
                OPENVINO_ASSERT(out->buf.valid(), "GFX MLIR: output buffer is not allocated for stage ", m_name);
            } else if (aliases_input) {
                out->buf = allocate_output_buffer(out_bytes);
                OPENVINO_ASSERT(out->buf.valid(), "GFX MLIR: output buffer is not allocated for stage ", m_name);
            }
        }
        if (out->buf.size < out_bytes) {
            OPENVINO_THROW("GFX MLIR: output buffer too small for stage ",
                           m_name,
                           " (need ",
                           out_bytes,
                           ", have ",
                           out->buf.size,
                           ", owned=",
                           out->buf.owned,
                           ", external=",
                           out->buf.external,
                           ", from_handle=",
                           out->buf.from_handle,
                           ", prefer_private=",
                           out->prefer_private,
                           ")");
        }
        out->expected_type = out_type;
    }

    if ((m_type == "Split" || m_type == "VariadicSplit") &&
        try_alias_contiguous_split_outputs(m_node, resolve_input_tensor(0), outputs, m_name.c_str())) {
        return;
    }

    if (m_type == "Broadcast" && try_alias_gqa_broadcast_view(m_node, resolve_input_tensor(0), outputs)) {
        return;
    }

    if (m_is_view_op) {
        if (m_inputs.empty() || !m_inputs[0] || !m_inputs[0]->buf.valid()) {
            OPENVINO_THROW("GFX MLIR: missing input buffer for view op ", m_name);
        }
        auto* in = m_inputs[0];
        auto* out = outputs.front();
        const auto in_type = in->expected_type == ov::element::dynamic ? in->buf.type : in->expected_type;
        const auto out_type = out->expected_type == ov::element::dynamic ? in_type : out->expected_type;
        const size_t in_bytes = ov::shape_size(in->shape) * in_type.size();
        const size_t out_bytes = ov::shape_size(out->shape) * out_type.size();
        OPENVINO_ASSERT(in_bytes == out_bytes,
                        "GFX MLIR: view op byte size mismatch for ",
                        m_name,
                        " (",
                        in_bytes,
                        " vs ",
                        out_bytes,
                        ")");
        out->buf = in->buf;
        out->buf.external = true;
        out->buf.owned = false;
        out->expected_type = out_type;
        propagate_view_metadata(*out, *in);
        return;
    }

    if (m_type == "Concat" &&
        !has_absorbed_input_transpose() &&
        !prefer_specialized_concat_execution()) {
        auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(m_node);
        OPENVINO_ASSERT(concat, "GFX MLIR: expected v0::Concat for stage ", m_name);
        auto* output = outputs.front();
        OPENVINO_ASSERT(output && output->buf.valid(), "GFX MLIR: missing concat output buffer for stage ", m_name);

        const ov::Shape& out_shape = output->shape;
        OPENVINO_ASSERT(!out_shape.empty(), "GFX MLIR: concat output shape unknown for stage ", m_name);
        const size_t rank = out_shape.size();
        const int64_t axis_norm = normalize_axis(concat->get_axis(), rank, "GFX MLIR: Concat");
        size_t outer = 1;
        for (size_t dim = 0; dim < static_cast<size_t>(axis_norm); ++dim) {
            outer *= out_shape[dim];
        }
        size_t inner = 1;
        for (size_t dim = static_cast<size_t>(axis_norm) + 1; dim < rank; ++dim) {
            inner *= out_shape[dim];
        }
        const size_t axis_total = out_shape[static_cast<size_t>(axis_norm)];
        const auto out_type = output->expected_type == ov::element::dynamic ? output->buf.type : output->expected_type;
        const size_t elem_bytes = out_type.size();

        struct CopyBatch {
            const GpuBuffer* src = nullptr;
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
        for (size_t input_idx = 0; input_idx < concat->get_input_size(); ++input_idx) {
            GpuTensor* src = resolve_input_tensor(input_idx);
            OPENVINO_ASSERT(src && src->buf.valid(), "GFX MLIR: missing concat input buffer for stage ", m_name);
            ov::Shape src_shape = !src->shape.empty() ? src->shape : ov::Shape{};
            if (src_shape.empty() && m_node->get_input_partial_shape(input_idx).is_static()) {
                src_shape = m_node->get_input_shape(input_idx);
            }
            OPENVINO_ASSERT(src_shape.size() == rank, "GFX MLIR: concat rank mismatch for stage ", m_name);
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
                region.dst_offset = ((outer_idx * axis_total + axis_offset) * inner) * elem_bytes;
                region.bytes = region_bytes;
                if (src->buf.buffer == output->buf.buffer &&
                    src->buf.offset + region.src_offset == output->buf.offset + region.dst_offset) {
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
            batch.total_bytes = static_cast<uint64_t>(batch.regions.size()) * static_cast<uint64_t>(region_bytes);
            copied_bytes += batch.total_bytes;
            copied_regions += static_cast<uint64_t>(batch.regions.size());
            axis_offset += axis_len;
            batches.push_back(std::move(batch));
        }

        auto* profiler = static_cast<GfxProfiler*>(m_profiler);
        const bool profiling = m_profiling_enabled && profiler;
        if (batches.empty()) {
            if (profiling) {
                profiler->increment_counter("concat_self_copy_skip_bytes", skipped_self_copy_bytes);
                profiler->increment_counter("concat_self_copy_skip_region_count", skipped_self_copy_regions);
            }
            return;
        }

        if (!batches.empty()) {
            const auto copy_start = profiling ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
            for (const auto& batch : batches) {
                gpu_copy_buffer_regions(reinterpret_cast<GpuCommandQueueHandle>(command_buffer),
                                        *batch.src,
                                        output->buf,
                                        batch.regions.data(),
                                        batch.regions.size());
            }
            if (profiling) {
                const auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - copy_start);
                profiler->record_segment("transfer",
                                         "concat_buffer_copy",
                                         cpu_us,
                                         0,
                                         0,
                                         copied_bytes,
                                         copied_bytes,
                                         0,
                                         0,
                                         -1,
                                         0,
                                         reinterpret_cast<uint64_t>(command_buffer));
                profiler->increment_counter("concat_copy_input_count",
                                            static_cast<uint64_t>(batches.size()));
                profiler->increment_counter("concat_copy_region_count", copied_regions);
                profiler->increment_counter("concat_self_copy_skip_bytes", skipped_self_copy_bytes);
                profiler->increment_counter("concat_self_copy_skip_region_count", skipped_self_copy_regions);
            }
            return;
        }
    }

    if (!m_kernel) {
        OPENVINO_THROW("GFX MLIR: kernel was not compiled for stage ", m_name);
    }

    ProfileState profile_state{};
    KernelExecutionHooks hooks;
    KernelExecutionHooks* hooks_ptr = nullptr;
    if (m_profiling_enabled && m_profiler) {
        profile_state.enabled = true;
        hooks_ptr = prepare_profiling(profile_state, hooks);
    }

    const bool use_generic_vulkan_concat_kernel = is_vulkan_backend() && m_type == "Concat";

    if (m_type == "Concat" && !use_generic_vulkan_concat_kernel) {
        auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(m_node);
        OPENVINO_ASSERT(concat, "GFX MLIR: expected v0::Concat for stage ", m_name);
        OPENVINO_ASSERT(!outputs.empty() && outputs.front() && outputs.front()->buf.valid(),
                        "GFX MLIR: missing concat output buffer for stage ",
                        m_name);

        const ov::Shape& out_shape = outputs.front()->shape;
        OPENVINO_ASSERT(!out_shape.empty(), "GFX MLIR: concat output shape unknown for stage ", m_name);
        const int64_t axis_norm = normalize_axis(concat->get_axis(), out_shape.size(), "GFX MLIR: Concat");
        uint64_t outer = 1;
        for (size_t i = 0; i < static_cast<size_t>(axis_norm); ++i) {
            outer *= out_shape[i];
        }
        uint64_t inner = 1;
        for (size_t i = static_cast<size_t>(axis_norm) + 1; i < out_shape.size(); ++i) {
            inner *= out_shape[i];
        }
        const uint32_t axis_total = static_cast<uint32_t>(out_shape[static_cast<size_t>(axis_norm)]);
        uint64_t axis_offset = 0;
        auto ensure_concat_output_not_aliased = [&]() {
            auto* dst = outputs.front();
            if (!dst || !dst->buf.valid()) {
                return;
            }
            bool aliases_input = false;
            for (size_t input_idx = 0; input_idx < m_inputs.size(); ++input_idx) {
                GpuTensor* src = resolve_input_tensor(input_idx);
                if (src && same_gpu_allocation(dst->buf, src->buf)) {
                    aliases_input = true;
                    break;
                }
            }
            if (!aliases_input) {
                return;
            }
            ov::element::Type out_type = dst->expected_type;
            if (out_type == ov::element::dynamic) {
                out_type = m_node->get_output_element_type(0);
            }
            GpuBufferDesc desc{};
            desc.bytes = ov::shape_size(out_shape) * out_type.size();
            desc.type = out_type;
            desc.usage = dst->prefer_private ? BufferUsage::Intermediate : BufferUsage::IO;
            desc.cpu_read = !dst->prefer_private;
            desc.cpu_write = !dst->prefer_private;
            desc.prefer_device_local = dst->prefer_private;
            desc.label = m_name.c_str();
            OPENVINO_ASSERT(m_buffer_manager, "GFX MLIR: buffer manager is not set for concat stage ", m_name);
            if (dst->buf.owned && !dst->buf.external && !dst->buf.from_handle) {
                m_buffer_manager->release_temp(std::move(dst->buf));
            }
            dst->buf = m_buffer_manager->allocate_temp(desc);
            OPENVINO_ASSERT(dst->buf.valid(), "GFX MLIR: failed to allocate non-aliased concat output for stage ", m_name);
        };
        ensure_concat_output_not_aliased();

        for (size_t i = 0; i < m_inputs.size(); ++i) {
            GpuTensor* src = resolve_input_tensor(i);
            if (!src || !src->buf.valid()) {
                continue;
            }
            ov::Shape src_shape = src->shape;
            if (src_shape.empty() && m_node->get_input_partial_shape(i).is_static()) {
                src_shape = m_node->get_input_shape(i);
            }
            OPENVINO_ASSERT(!src_shape.empty(), "GFX MLIR: concat input shape unknown for stage ", m_name);
            const uint32_t axis_len = static_cast<uint32_t>(src_shape[static_cast<size_t>(axis_norm)]);
            const uint64_t total = outer * axis_len * inner;
            if (total == 0) {
                axis_offset += axis_len;
                continue;
            }

            GpuTensor params_tensor = make_concat_runtime_param_tensor(*m_buffer_manager,
                                                                       m_name,
                                                                       i,
                                                                       static_cast<uint32_t>(outer),
                                                                       static_cast<uint32_t>(inner),
                                                                       static_cast<uint32_t>(axis_offset),
                                                                       axis_len,
                                                                       axis_total);

            std::vector<size_t> kernel_inputs{0};
            std::vector<int32_t> operand_kinds{1, 1, 1};
            std::vector<int32_t> operand_arg_indices{0, 1, 2};
            std::vector<GpuTensor> extras{params_tensor};
            auto bundle = build_kernel_args_from_metadata(
                operand_kinds,
                operand_arg_indices,
                {},
                kernel_inputs,
                1,
                extras,
                outputs,
                [&](size_t) { return src; },
                m_name.c_str(),
                nullptr);
            auto bound_args = materialize_kernel_bytes_args(bundle.args, *m_buffer_manager, m_name.c_str());
            KernelDispatch dispatch = make_1d_dispatch(static_cast<size_t>(total), m_kernel->clamp_threadgroup_size(256));
            m_kernel->execute(command_buffer, dispatch, bound_args, hooks_ptr);
            axis_offset += axis_len;
        }

        if (profile_state.enabled) {
            finalize_profiling(profile_state);
        }
        return;
    }

    if (m_type == "Tile" && !outputs.empty() && outputs.front() && !outputs.front()->shape.empty()) {
        const ov::Shape tile_input_shape = resolve_input_shape(0);
        const ov::Shape tile_output_shape = outputs.front()->shape;
        std::vector<int32_t> tile_scalars = {
            static_cast<int32_t>(ov::shape_size(tile_output_shape)),
            static_cast<int32_t>(std::max<size_t>(tile_output_shape.size(), 1))};
        if (!tile_input_shape.empty()) {
            auto tile_payload =
                make_tile_runtime_param_payload(*m_buffer_manager, m_name, tile_input_shape, tile_output_shape);
            tile_scalars = tile_payload.scalar_args;
            m_kernel_extra_inputs = std::move(tile_payload.extra_inputs);
        }
        auto tile_plan = make_kernel_runtime_binding_plan_for_custom_kernel(
            "Tile",
            "tile_kernel",
            {0},
            tile_scalars,
            is_vulkan_backend() ? GfxKernelBackendDomain::Spirv : GfxKernelBackendDomain::AppleMsl,
            GfxKernelStorageKind::Buffer,
            is_vulkan_backend() ? "spirv:buffer:" : "apple_msl:buffer:");
        OPENVINO_ASSERT(tile_plan.valid,
                        "GFX MLIR: Tile custom-kernel runtime binding manifest is invalid for stage ",
                        m_name);
        set_kernel_binding_override(tile_plan.runtime_binding);
    }

    KernelRuntimeBindingState execution_binding =
        kernel_binding_override ? *kernel_binding_override : resolved_kernel_runtime_binding_state();
    if (kernel_scalar_args_override) {
        if (!execution_binding.operand_kinds.empty()) {
            const size_t scalar_slots =
                static_cast<size_t>(std::count(execution_binding.operand_kinds.begin(),
                                               execution_binding.operand_kinds.end(),
                                               0));
            OPENVINO_ASSERT(scalar_slots == kernel_scalar_args_override->size(),
                            "GFX MLIR: scalar runtime arg count mismatch for stage ",
                            m_name,
                            " (",
                            m_type,
                            "): manifest expects ",
                            scalar_slots,
                            ", runtime supplied ",
                            kernel_scalar_args_override->size());
        }
        execution_binding.scalar_args = *kernel_scalar_args_override;
    }
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("MLIRExec") << "Kernel args prep: scalars=" << execution_binding.scalar_args.size()
                                  << " inputs=" << execution_binding.inputs.size()
                                  << " outputs=" << outputs.size()
                                  << " kinds=" << execution_binding.operand_kinds.size();
        if (m_type == "MatMul") {
            for (size_t input_idx = 0; input_idx < m_inputs.size(); ++input_idx) {
                auto* tensor = resolve_input_tensor(input_idx);
                if (!tensor || !tensor->buf.valid()) {
                    gfx_log_debug("MLIRExec") << "MatMul input[" << input_idx << "] <missing>";
                    continue;
                }
                std::ostringstream oss;
                oss << "MatMul input[" << input_idx << "]"
                    << " buf=" << tensor->buf.buffer
                    << " uid=" << tensor->buf.allocation_uid
                    << " off=" << tensor->buf.offset
                    << " bytes=" << tensor->buf.size
                    << " shape=" << tensor->shape
                    << " type=" << tensor->expected_type;
                gfx_log_debug("MLIRExec") << oss.str();
            }
            for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
                auto* tensor = outputs[output_idx];
                if (!tensor || !tensor->buf.valid()) {
                    gfx_log_debug("MLIRExec") << "MatMul output[" << output_idx << "] <missing>";
                    continue;
                }
                std::ostringstream oss;
                oss << "MatMul output[" << output_idx << "]"
                    << " buf=" << tensor->buf.buffer
                    << " uid=" << tensor->buf.allocation_uid
                    << " off=" << tensor->buf.offset
                    << " bytes=" << tensor->buf.size
                    << " shape=" << tensor->shape
                    << " type=" << tensor->expected_type;
                gfx_log_debug("MLIRExec") << oss.str();
            }
        }
    }

    std::string arg_map;
    std::vector<GpuTensor> empty_extras;
    const std::vector<GpuTensor>* extras = &m_kernel_extra_inputs;
    if (execution_binding.operand_kinds.empty()) {
        const size_t expected_inputs =
            execution_binding.input_arg_count ? execution_binding.input_arg_count : execution_binding.inputs.size();
        if (expected_inputs <= execution_binding.inputs.size()) {
            extras = &empty_extras;  // Kernel does not expect extra buffers; drop them.
        }
    }

    KernelArgsBundle bundle = build_kernel_args_from_metadata(
        execution_binding.operand_kinds,
        execution_binding.operand_arg_indices,
        execution_binding.scalar_args,
        execution_binding.inputs,
        execution_binding.input_arg_count,
        *extras,
        outputs,
        [&](size_t input_idx) { return resolve_input_tensor(input_idx); },
        m_name.c_str(),
        gfx_log_debug_enabled() ? &arg_map : nullptr);
    if (gfx_log_debug_enabled() && !arg_map.empty()) {
        gfx_log_debug("MLIRExec") << arg_map;
    }
    auto bound_args = materialize_kernel_bytes_args(bundle.args, *m_buffer_manager, m_name.c_str());

    KernelDispatch dispatch{};
    ov::Shape dispatch_shape = (outputs.front() && !outputs.front()->shape.empty())
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
    if ((m_type == "Split" || m_type == "VariadicSplit") && !m_output_shape.empty()) {
        dispatch_shape = m_output_shape;
    }
    if (m_parallel_cfg.enabled) {
        dispatch = make_parallel_dispatch(dispatch_shape, m_parallel_cfg, m_kernel.get());
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MLIRExec") << "Dispatch grid=(" << dispatch.grid[0] << ", "
                                      << dispatch.grid[1] << ", "
                                      << dispatch.grid[2] << ")"
                                      << " tpg=(" << dispatch.threads_per_group[0] << ", "
                                      << dispatch.threads_per_group[1] << ", "
                                      << dispatch.threads_per_group[2] << ")"
                                      << " loops=" << m_parallel_cfg.loop_dims;
        }
    } else if ((m_type == "ScaledDotProductAttention" || m_type == "GfxSDPAWithCausalMask") &&
               dispatch_shape.size() == 4 &&
               dispatch_shape[3] <= 256 &&
               !m_inputs.empty() &&
               m_inputs[0] &&
               !m_inputs[0]->shape.empty() &&
               m_inputs[0]->shape.size() == 4 &&
               m_inputs[0]->shape[3] <= 256) {
        const uint64_t vectors = static_cast<uint64_t>(dispatch_shape[0]) *
                                 static_cast<uint64_t>(dispatch_shape[1]) *
                                 static_cast<uint64_t>(dispatch_shape[2]);
        const bool simdgroup_path = dispatch_shape[3] <= 64 && m_inputs[0]->shape[3] <= 64;
        const size_t threads = m_kernel->clamp_threadgroup_size(simdgroup_path ? 32 : 256);
        dispatch = make_1d_dispatch(static_cast<size_t>(vectors * static_cast<uint64_t>(threads)), threads);
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MLIRExec") << "SDPA streaming dispatch grid=(" << dispatch.grid[0] << ", "
                                      << dispatch.grid[1] << ", "
                                      << dispatch.grid[2] << ")"
                                      << " tpg=(" << dispatch.threads_per_group[0] << ", "
                                      << dispatch.threads_per_group[1] << ", "
                                      << dispatch.threads_per_group[2] << ")";
        }
    } else if (m_force_single_dispatch) {
        dispatch = make_1d_dispatch(1, 1);
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MLIRExec") << "Single dispatch grid=(" << dispatch.grid[0] << ", "
                                      << dispatch.grid[1] << ", "
                                      << dispatch.grid[2] << ")"
                                      << " tpg=(" << dispatch.threads_per_group[0] << ", "
                                      << dispatch.threads_per_group[1] << ", "
                                      << dispatch.threads_per_group[2] << ")";
        }
    } else if (m_type == "MatMul" && m_matmul_reduction_threads > 1) {
        const uint64_t outputs = static_cast<uint64_t>(ov::shape_size(dispatch_shape));
        uint64_t work_groups = outputs;
        if (m_is_compressed_matmul && m_compressed_matmul_output_block > 1 && m_compressed_matmul_n > 0) {
            const uint64_t n = static_cast<uint64_t>(m_compressed_matmul_n);
            const uint64_t cols_per_group = static_cast<uint64_t>(m_compressed_matmul_output_block);
            const uint64_t rows = (outputs + n - 1) / n;
            work_groups = rows * ((n + cols_per_group - 1) / cols_per_group);
        }
        dispatch = make_1d_dispatch(static_cast<size_t>(work_groups * m_matmul_reduction_threads),
                                    m_kernel->clamp_threadgroup_size(m_matmul_reduction_threads));
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MLIRExec") << "MatMul reduction dispatch grid=(" << dispatch.grid[0] << ", "
                                      << dispatch.grid[1] << ", "
                                      << dispatch.grid[2] << ")"
                                      << " tpg=(" << dispatch.threads_per_group[0] << ", "
                                      << dispatch.threads_per_group[1] << ", "
                                      << dispatch.threads_per_group[2] << ")";
        }
    } else if (m_type == "Convolution" &&
               m_conv_output_channels_per_thread > 1 &&
               dispatch_shape.size() == 4) {
        const uint64_t work_items = gfx_conv2d_dispatch_items(static_cast<uint64_t>(dispatch_shape[0]),
                                                              static_cast<uint64_t>(dispatch_shape[1]),
                                                              static_cast<uint64_t>(dispatch_shape[2]),
                                                              static_cast<uint64_t>(dispatch_shape[3]),
                                                              m_conv_output_channels_per_thread,
                                                              m_conv_output_width_per_thread);
        dispatch = make_1d_dispatch(static_cast<size_t>(work_items),
                                    m_kernel->clamp_threadgroup_size(256));
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MLIRExec") << "Conv channel-block dispatch grid=(" << dispatch.grid[0] << ", "
                                      << dispatch.grid[1] << ", "
                                      << dispatch.grid[2] << ")"
                                      << " tpg=(" << dispatch.threads_per_group[0] << ", "
                                      << dispatch.threads_per_group[1] << ", "
                                      << dispatch.threads_per_group[2] << ")"
                                      << " channels_per_thread=" << m_conv_output_channels_per_thread
                                      << " width_per_thread=" << m_conv_output_width_per_thread;
        }
    } else if (m_type == "RMS" && m_rms_reduction_threads > 1 && m_rms_hidden > 0) {
        const uint64_t total = static_cast<uint64_t>(ov::shape_size(dispatch_shape));
        const uint64_t rows = (total + m_rms_hidden - 1) / m_rms_hidden;
        dispatch = make_1d_dispatch(static_cast<size_t>(rows * m_rms_reduction_threads),
                                    m_kernel->clamp_threadgroup_size(m_rms_reduction_threads));
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MLIRExec") << "RMS reduction dispatch grid=(" << dispatch.grid[0] << ", "
                                      << dispatch.grid[1] << ", "
                                      << dispatch.grid[2] << ")"
                                      << " tpg=(" << dispatch.threads_per_group[0] << ", "
                                      << dispatch.threads_per_group[1] << ", "
                                      << dispatch.threads_per_group[2] << ")";
        }
    } else {
        // Fallback: linear dispatch over total elements.
        dispatch = KernelPlan::make_default_dispatch(dispatch_shape, *m_kernel);
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MLIRExec") << "Default dispatch grid=(" << dispatch.grid[0] << ", "
                                      << dispatch.grid[1] << ", "
                                      << dispatch.grid[2] << ")"
                                      << " tpg=(" << dispatch.threads_per_group[0] << ", "
                                      << dispatch.threads_per_group[1] << ", "
                                      << dispatch.threads_per_group[2] << ")";
        }
    }

    try {
        m_kernel->execute(command_buffer, dispatch, bound_args, hooks_ptr);
    } catch (const std::exception& ex) {
        const auto opt_plan = stage_optimization_plan();
        const bool is_im2col_matmul =
            is_conv_like() && opt_plan.conv.algorithm.kind == GfxConvAlgorithmKind::Im2ColMatMul;
        const bool is_shared_vulkan_conv =
            is_vulkan_backend() &&
            is_conv_like() &&
            opt_plan.conv.kind == GfxConvRouteKind::None &&
            !m_vulkan_conv_serial_retry_attempted;
        if (!is_vulkan_backend() || !is_vulkan_pipeline_creation_failure(ex) ||
            (!is_matmul_like() && !is_im2col_matmul && !is_shared_vulkan_conv) || !m_buffer_manager ||
            ((is_matmul_like() || is_im2col_matmul) && m_matmul_serial_retry_attempted)) {
            throw;
        }

        ov::Shape tuning_shape = m_output_shape;
        if (is_im2col_matmul && tuning_shape.size() == 4) {
            const uint64_t batch = static_cast<uint64_t>(std::max<size_t>(1, tuning_shape[0]));
            const uint64_t spatial =
                static_cast<uint64_t>(std::max<size_t>(1, tuning_shape[2])) *
                static_cast<uint64_t>(std::max<size_t>(1, tuning_shape[3]));
            const uint64_t channels = static_cast<uint64_t>(std::max<size_t>(1, tuning_shape[1]));
            tuning_shape = batch == 1 ? ov::Shape{channels, spatial} : ov::Shape{batch, spatial, channels};
        }

        const auto caps = query_parallelism_caps(m_buffer_manager);
        if (!m_matmul_safe_retry_attempted) {
            auto safe_plan = select_safe_matmul_fallback_plan(caps, tuning_shape);
            if (safe_plan.has_value()) {
                gfx_log_info("MLIRExec") << "Retrying " << m_name << " with safe matmul variant "
                                         << safe_plan->variant << " after pipeline creation failure: " << ex.what();
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
            gfx_log_info("MLIRExec") << "Retrying " << m_name
                                     << " with serial matmul fallback after pipeline creation failure: "
                                     << ex.what();
            remember_matmul_parallelism(caps, tuning_shape, make_serial_matmul_fallback_plan());
            m_matmul_serial_retry_attempted = true;
            m_kernel.reset();
            m_last_input_shape = {};
            compile(m_buffer_manager);
            execute(command_buffer);
            return;
        }

        if (is_shared_vulkan_conv) {
            gfx_log_info("MLIRExec") << "Retrying " << m_name
                                     << " with serial Vulkan convolution fallback after pipeline creation failure: "
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

void MlirStage::set_inputs(const std::vector<GpuTensor*>& inputs) {
    m_inputs = inputs;
    if (!m_const_buffers) {
        m_const_buffers = std::make_shared<ConstBufferSet>();
        m_const_buffers->buffers.resize(inputs.size());
        m_const_buffers->present.assign(inputs.size(), false);
    }
}

void MlirStage::set_output(GpuTensor* output) {
    m_output = output;
    m_outputs.clear();
    if (output) {
        m_outputs.push_back(output);
    }
}

void MlirStage::set_output_refs(const std::vector<GpuTensor*>& outputs) {
    m_outputs = outputs;
    m_output = m_outputs.empty() ? nullptr : m_outputs.front();
}

void MlirStage::set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) {
    std::vector<GpuTensor*> refs;
    refs.reserve(outputs.size());
    for (const auto& output : outputs) {
        refs.push_back(output.get());
    }
    set_output_refs(refs);
}

void MlirStage::set_input_transform(size_t input_idx, const GfxInputTransform& transform) {
    if (m_input_transforms.size() <= input_idx) {
        m_input_transforms.resize(input_idx + 1);
    }
    m_input_transforms[input_idx] = transform;
}

void MlirStage::enable_profiling(bool enable) {
    m_profiling_enabled = enable;
}

void MlirStage::set_profiler(void* profiler,
                             uint32_t node_id,
                             const std::string& node_name,
                             const std::string& node_type) {
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
    OPENVINO_ASSERT(!m_kernel, "MlirStage: cannot fuse activation after compilation");
    m_has_activation = true;
    m_activation = kind;
    m_activation_alpha = alpha;
    return true;
}

bool MlirStage::fuse_input_activation(size_t input_idx, ActivationKind kind, float alpha) {
    if (m_has_input_activation || m_has_activation || m_type != "Multiply" || input_idx >= 2) {
        return false;
    }
    if (kind != ActivationKind::Relu &&
        kind != ActivationKind::Sigmoid &&
        kind != ActivationKind::Tanh &&
        kind != ActivationKind::Gelu &&
        kind != ActivationKind::Swish &&
        kind != ActivationKind::HSwish &&
        kind != ActivationKind::HSigmoid) {
        return false;
    }
    if (!m_node || m_node->get_output_element_type(0).is_integral_number() ||
        m_node->get_output_element_type(0) == ov::element::boolean) {
        return false;
    }
    OPENVINO_ASSERT(!m_kernel, "MlirStage: cannot fuse input activation after compilation");
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
    OPENVINO_ASSERT(!m_kernel, "MlirStage: cannot fuse residual add after compilation");
    m_has_residual_add = true;
    return true;
}

bool MlirStage::fuse_batchnorm(const BatchNormParams& params) {
    OPENVINO_ASSERT(!m_kernel, "MlirStage: cannot fuse batchnorm after compilation");
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
    const auto& pshape = m_node->get_output_partial_shape(0);
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

bool MlirStage::fuse_bias(const BiasParams& params) {
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

const GfxInputTransform* MlirStage::input_transform(size_t input_idx) const {
    if (input_idx >= m_input_transforms.size() || !m_input_transforms[input_idx].has_transpose()) {
        return nullptr;
    }
    return &m_input_transforms[input_idx];
}

ov::Shape MlirStage::compile_time_input_shape(size_t input_idx) const {
    if (const auto* transform = input_transform(input_idx)) {
        return transform->source_shape;
    }
    if (m_node) {
        try {
            return m_node->get_input_shape(input_idx);
        } catch (const std::exception&) {
        }
    }
    return {};
}

std::vector<int32_t> MlirStage::compile_time_broadcast_strides(size_t input_idx, const ov::Shape& out_shape) const {
    const auto in_shape = compile_time_input_shape(input_idx);
    if (in_shape.empty()) {
        return {};
    }
    if (const auto* transform = input_transform(input_idx)) {
        OPENVINO_ASSERT(m_node && m_node->get_input_partial_shape(input_idx).is_static(),
                        "GFX MLIR: absorbed transpose requires static consumer input shape for stage ",
                        m_name);
        return compute_permuted_broadcast_element_strides(transform->source_shape,
                                                          m_node->get_input_shape(input_idx),
                                                          transform->transpose_permutation,
                                                          out_shape,
                                                          "GFX MLIR");
    }
    return compute_broadcast_element_strides(in_shape, out_shape);
}

bool MlirStage::has_absorbed_input_transpose() const {
    return std::any_of(m_input_transforms.begin(),
                       m_input_transforms.end(),
                       [](const GfxInputTransform& transform) { return transform.has_transpose(); });
}

KernelExecutionHooks* MlirStage::prepare_profiling(ProfileState&, KernelExecutionHooks&) {
    return nullptr;
}

void MlirStage::finalize_profiling(const ProfileState&) {
}

GfxStageOptimizationPlan MlirStage::stage_optimization_plan() const {
    return select_stage_optimization_plan(m_buffer_manager,
                                          backend_kind(),
                                          m_type,
                                          m_node,
                                          m_node ? m_node->get_output_element_type(0) : ov::element::dynamic,
                                          m_has_bias,
                                          m_has_activation,
                                          m_has_bn,
                                          GfxStageRuntimeTraits{});
}

void MlirStage::clone_into(MlirStage& dst) const {
    dst.m_kernel = m_kernel;
    dst.m_is_compressed_matmul = m_is_compressed_matmul;
    dst.m_output_shape = m_output_shape;
    dst.m_last_input_shape = m_last_input_shape;
    dst.m_input_transforms = m_input_transforms;
    dst.m_kernel_binding = m_kernel_binding;
    dst.m_kernel_binding_owned_by_source_plan = m_kernel_binding_owned_by_source_plan;
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
    dst.m_vulkan_conv_serial_retry_attempted = m_vulkan_conv_serial_retry_attempted;
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
    dst.m_kernel_extra_inputs = m_kernel_extra_inputs;
}

bool MlirStage::is_conv_like() const {
    return m_type == "Convolution" || m_type == "GroupConvolution" || m_type == "GroupConv2D";
}

bool MlirStage::is_matmul_like() const {
    return m_type == "MatMul";
}

void MlirStage::apply_stage_optimization_attrs(mlir::ModuleOp module,
                                               const GfxStageOptimizationPlan& plan) {
    if (!module) {
        return;
    }
    auto* ctx = module.getContext();
    module->setAttr("gfx.stage_archetype",
                    mlir::StringAttr::get(ctx, stage_archetype_attr(plan.archetype)));
    const bool conv_mpsrt_annotated =
        is_conv_like() &&
        annotate_module_with_conv_mpsrt_plan(module, plan, m_node, m_type) != GfxConvMpsrtLoweringKind::None;
    if (!conv_mpsrt_annotated) {
        (void)run_gfx_apple_stage_pipeline(module, plan, m_type);
    }
    module->setAttr("gfx.tensor_layout_kind",
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
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                           static_cast<int32_t>(plan.execution.submit.weight)));
    module->setAttr("gfx.conv_route_kind",
                    mlir::StringAttr::get(ctx, conv_route_kind_attr(plan.conv.kind)));
    module->setAttr("gfx.conv_family",
                    mlir::StringAttr::get(ctx, conv_family_attr(plan.conv.family)));
    module->setAttr("gfx.conv_algorithm_kind",
                    mlir::StringAttr::get(ctx, conv_algorithm_kind_attr(plan.conv.algorithm.kind)));
    module->setAttr("gfx.conv_variant",
                    mlir::StringAttr::get(ctx, plan.conv.algorithm.variant));
}

void MlirStage::apply_input_transform_attrs(mlir::ModuleOp module) const {
    if (!module) {
        return;
    }
    auto* ctx = module.getContext();
    mlir::OpBuilder b(ctx);
    for (size_t input_idx = 0; input_idx < m_input_transforms.size(); ++input_idx) {
        const auto* transform = input_transform(input_idx);
        if (!transform) {
            continue;
        }
        llvm::SmallVector<mlir::Attribute> attrs;
        attrs.reserve(transform->transpose_permutation.size());
        for (int64_t axis : transform->transpose_permutation) {
            attrs.push_back(b.getI64IntegerAttr(axis));
        }
        const std::string attr_name = "gfx.absorbed_input" + std::to_string(input_idx) + "_perm";
        module->setAttr(attr_name, b.getArrayAttr(attrs));
    }
}

void MlirStage::set_parallel_preference(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    auto* ctx = module.getContext();
    if (m_type == "GroupConvolution" && has_absorbed_input_transpose()) {
        // The transformed depthwise GroupConvolution path is lowered through a
        // shared linalg.generic kernel. Keep it on the serial MLIR pipeline
        // until the Vulkan parallel-dispatch ABI for this shape-preserving
        // layout transform is fixed.
        module->setAttr("gfx.prefer_parallel", mlir::BoolAttr::get(ctx, false));
        return;
    }
    bool conv2d = false;
    if (m_node && is_conv_like()) {
        auto in_shape = m_node->get_input_partial_shape(0);
        if (in_shape.rank().is_static() &&
            (in_shape.rank().get_length() == 4 || in_shape.rank().get_length() == 5)) {
            conv2d = true;
        }
    }
    bool prefer_parallel = conv2d;
    const auto optimization_plan = stage_optimization_plan();
    if (conv2d && optimization_plan.conv.algorithm.kind == GfxConvAlgorithmKind::Im2ColMatMul) {
        auto matmul_shape = m_output_shape;
        if (matmul_shape.size() == 4) {
            const uint64_t batch = static_cast<uint64_t>(std::max<size_t>(1, matmul_shape[0]));
            const uint64_t spatial =
                static_cast<uint64_t>(std::max<size_t>(1, matmul_shape[2])) *
                static_cast<uint64_t>(std::max<size_t>(1, matmul_shape[3]));
            const uint64_t channels = static_cast<uint64_t>(std::max<size_t>(1, matmul_shape[1]));
            if (batch == 1) {
                matmul_shape = ov::Shape{channels, spatial};
            } else {
                matmul_shape = ov::Shape{batch, spatial, channels};
            }
        }
        const auto caps = query_parallelism_caps(m_buffer_manager);
        const auto plan = select_matmul_parallelism(caps, matmul_shape);
        prefer_parallel = prefer_parallel || plan.prefer_parallel;
        if (plan.prefer_parallel) {
            module->setAttr("gfx.dispatch_tile_h",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), plan.dispatch.tile_h));
            module->setAttr("gfx.dispatch_tile_w",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), plan.dispatch.tile_w));
            module->setAttr("gfx.dispatch_threads_h",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), plan.dispatch.threads_h));
            module->setAttr("gfx.dispatch_threads_w",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), plan.dispatch.threads_w));
        }
    } else if (conv2d && m_type == "Convolution") {
        auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
        if (conv && conv->get_input_size() == 2 && conv->get_output_size() == 1) {
            const auto& in_shape = conv->get_input_shape(0);
            const auto& w_shape = conv->get_input_shape(1);
            if (in_shape.size() == 4 && w_shape.size() == 4) {
                const auto caps = query_parallelism_caps(m_buffer_manager);
                const uint64_t input_channels = static_cast<uint64_t>(std::max<size_t>(1, in_shape[1]));
                const uint64_t output_channels = static_cast<uint64_t>(std::max<size_t>(1, w_shape[0]));
                const uint64_t kernel_work =
                    input_channels * static_cast<uint64_t>(std::max<size_t>(1, w_shape[2])) *
                    static_cast<uint64_t>(std::max<size_t>(1, w_shape[3]));
                const bool stride2 = conv->get_strides().at(0) > 1 || conv->get_strides().at(1) > 1;
                const bool depthwise =
                    optimization_plan.conv.algorithm.kind == GfxConvAlgorithmKind::DepthwiseDirect;
                const auto plan = select_conv_parallelism(caps,
                                                          m_output_shape,
                                                          input_channels,
                                                          output_channels,
                                                          kernel_work,
                                                          stride2,
                                                          depthwise);
                prefer_parallel = prefer_parallel || plan.prefer_parallel;
                if (plan.prefer_parallel) {
                    module->setAttr("gfx.dispatch_tile_h",
                                    mlir::IntegerAttr::get(mlir::IndexType::get(ctx), plan.dispatch.tile_h));
                    module->setAttr("gfx.dispatch_tile_w",
                                    mlir::IntegerAttr::get(mlir::IndexType::get(ctx), plan.dispatch.tile_w));
                    module->setAttr("gfx.dispatch_threads_h",
                                    mlir::IntegerAttr::get(mlir::IndexType::get(ctx), plan.dispatch.threads_h));
                    module->setAttr("gfx.dispatch_threads_w",
                                    mlir::IntegerAttr::get(mlir::IndexType::get(ctx), plan.dispatch.threads_w));
                }
            }
        }
    }
    if (is_matmul_like()) {
        const auto caps = query_parallelism_caps(m_buffer_manager);
        const auto plan = select_matmul_parallelism(caps, m_output_shape);
        prefer_parallel = prefer_parallel || plan.prefer_parallel;
        if (plan.prefer_parallel) {
            module->setAttr("gfx.dispatch_tile_h",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), plan.dispatch.tile_h));
            module->setAttr("gfx.dispatch_tile_w",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), plan.dispatch.tile_w));
            module->setAttr("gfx.dispatch_threads_h",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), plan.dispatch.threads_h));
            module->setAttr("gfx.dispatch_threads_w",
                            mlir::IntegerAttr::get(mlir::IndexType::get(ctx), plan.dispatch.threads_w));
        }
    }
    module->setAttr("gfx.prefer_parallel", mlir::BoolAttr::get(ctx, prefer_parallel));
}

void MlirStage::apply_fused_operations(mlir::ModuleOp module) {
    if (!module) {
        return;
    }
    if (m_has_bn) {
        const bool applied = apply_fused_batchnorm(module, m_bn_params);
        OPENVINO_ASSERT(applied, "GFX MLIR: failed to apply fused batchnorm for stage ", m_name);
    }
    if (m_has_bias) {
        const bool applied = apply_fused_bias(module, m_bias_params);
        OPENVINO_ASSERT(applied, "GFX MLIR: failed to apply fused bias for stage ", m_name);
    }
    if (m_has_activation) {
        if (!is_conv_like()) {
            module->setAttr("gfx.post_activation_only",
                            mlir::BoolAttr::get(module.getContext(), true));
        }
        const bool applied = apply_fused_activation(module, m_activation, m_activation_alpha);
        OPENVINO_ASSERT(applied, "GFX MLIR: failed to apply fused activation for stage ", m_name);
    }
    if (m_has_input_activation) {
        const bool applied = apply_fused_input_activation(module,
                                                         m_input_activation_index,
                                                         m_input_activation,
                                                         m_input_activation_alpha);
        OPENVINO_ASSERT(applied, "GFX MLIR: failed to apply fused input activation for stage ", m_name);
    }
}

}  // namespace gfx_plugin
}  // namespace ov
