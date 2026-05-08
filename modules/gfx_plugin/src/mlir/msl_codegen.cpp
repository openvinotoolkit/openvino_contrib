// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/msl_codegen.hpp"

#include "mlir/gfx_apple_stage_pipeline.hpp"
#include "mlir/gfx_apple_vendor_descriptors.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mpsrt_const_tensor_sources.hpp"
#include "mlir/gfx_mpsrt_conv_metadata.hpp"
#include "mlir/gfx_mpsrt_metadata.hpp"
#include "mlir/mlir_kernel_plan_utils.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "transforms/mlir_fused_ops.hpp"

#include "mlir/IR/Builders.h"

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/reverse.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/space_to_depth.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <utility>

namespace ov {
namespace gfx_plugin {
namespace {

bool is_msl_ident_char(char c) {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ||
           c == '_';
}

bool replace_kernel_entry_name(std::string& source,
                               std::string_view current_entry_point,
                               std::string_view required_entry_point) {
    if (current_entry_point.empty() ||
        required_entry_point.empty() ||
        current_entry_point == required_entry_point) {
        return false;
    }

    const std::string needle = "kernel void " + std::string(current_entry_point);
    size_t pos = source.find(needle);
    while (pos != std::string::npos) {
        const size_t name_pos = pos + std::string("kernel void ").size();
        const size_t after_name = name_pos + current_entry_point.size();
        if (after_name < source.size() && !is_msl_ident_char(source[after_name])) {
            source.replace(name_pos, current_entry_point.size(), required_entry_point);
            return true;
        }
        pos = source.find(needle, pos + 1);
    }
    return false;
}

ov::element::Type resolve_matmul_buffer_type(const ov::element::Type& type,
                                             const ov::element::Type& fallback) {
    if (type != ov::element::dynamic) {
        return type;
    }
    return fallback == ov::element::dynamic ? ov::element::f32 : fallback;
}

void force_apple_msl_buffer_placement(GfxStageOptimizationPlan& plan,
                                      std::string_view stage_type) {
    plan.placement.domain = GfxStageBackendDomain::AppleMsl;
    plan.placement.storage = GfxStageStorageKind::Buffer;
    plan.placement.uses_vendor_primitive = false;
    plan.placement.uses_custom_kernel = true;
    plan.placement.specialization_key = std::string("apple_msl:buffer:") + std::string(stage_type);
}

GfxKernelExternalBufferAbiSpec make_matmul_bias_external_buffer_abi() {
    return make_gfx_kernel_roles_abi({GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorInput,
                                      GfxKernelBufferRole::TensorOutput});
}

uint32_t resolve_matmul_source_arg_count(mlir::ModuleOp module,
                                         uint32_t arg_count,
                                         const KernelArgMappingInfo& info) {
    const uint32_t inferred_total =
        static_cast<uint32_t>(infer_kernel_arg_count_from_module(module, info.signature.total()));
    if (arg_count != 0) {
        return arg_count;
    }
    return inferred_total;
}

ov::Shape static_shape_or_placeholder(const ov::PartialShape& pshape) {
    OPENVINO_ASSERT(pshape.rank().is_static(), "GFX Metal: tensor rank must be static for MSL codegen");
    ov::Shape shape;
    shape.reserve(static_cast<size_t>(pshape.rank().get_length()));
    for (const auto& dim : pshape) {
        shape.push_back(dim.is_static() ? static_cast<size_t>(dim.get_length()) : 1);
    }
    return shape;
}

std::vector<int64_t> to_i64_shape(const ov::Shape& shape) {
    std::vector<int64_t> values;
    values.reserve(shape.size());
    for (auto dim : shape) {
        values.push_back(static_cast<int64_t>(dim));
    }
    return values;
}

std::vector<int64_t> make_strides(const ov::Shape& shape) {
    const size_t rank = shape.size();
    std::vector<int64_t> strides(rank, 1);
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] =
            strides[static_cast<size_t>(i + 1)] * static_cast<int64_t>(shape[static_cast<size_t>(i + 1)]);
    }
    return strides;
}

void fill_broadcast_strides(const ov::Shape& output_shape,
                            const ov::Shape& input_shape,
                            std::vector<int64_t>& strides) {
    const size_t output_rank = output_shape.size();
    const size_t input_rank = input_shape.size();
    strides.assign(output_rank, 0);
    auto input_strides = make_strides(input_shape);
    for (size_t i = 0; i < output_rank; ++i) {
        const size_t output_dim = output_shape[output_rank - 1 - i];
        const size_t input_dim = (i < input_rank) ? input_shape[input_rank - 1 - i] : 1;
        const size_t input_stride = (i < input_rank) ? input_strides[input_rank - 1 - i] : 0;
        if (input_dim == output_dim) {
            strides[output_rank - 1 - i] = static_cast<int64_t>(input_stride);
        } else if (input_dim == 1) {
            strides[output_rank - 1 - i] = 0;
        } else {
            OPENVINO_THROW("GFX Metal: incompatible broadcast dims");
        }
    }
}

ov::Shape shape_from_entry_argument(mlir::ModuleOp module, size_t arg_idx, const ov::Shape& fallback) {
    if (!module) {
        return fallback;
    }
    auto func = get_entry_func(module);
    if (!func || arg_idx >= func.getNumArguments()) {
        return fallback;
    }
    auto type = func.getArgument(arg_idx).getType();
    if (auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
        if (!ranked.hasStaticShape()) {
            return fallback;
        }
        ov::Shape shape;
        shape.reserve(ranked.getRank());
        for (int64_t dim : ranked.getShape()) {
            shape.push_back(static_cast<size_t>(dim));
        }
        return shape;
    }
    if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type)) {
        if (!memref.hasStaticShape()) {
            return fallback;
        }
        ov::Shape shape;
        shape.reserve(memref.getRank());
        for (int64_t dim : memref.getShape()) {
            shape.push_back(static_cast<size_t>(dim));
        }
        return shape;
    }
    return fallback;
}

ov::Shape shape_from_entry_argument_or_partial(mlir::ModuleOp module,
                                               size_t arg_idx,
                                               const ov::PartialShape& fallback) {
    return shape_from_entry_argument(module, arg_idx, static_shape_or_placeholder(fallback));
}

ov::Shape output_shape_for_codegen(mlir::ModuleOp module, const std::shared_ptr<const ov::Node>& node) {
    OPENVINO_ASSERT(node, "GFX Metal: output shape requested for null node");
    if (node->get_output_partial_shape(0).is_static()) {
        return node->get_output_shape(0);
    }
    if (module) {
        auto func = get_entry_func(module);
        if (func && func.getFunctionType().getNumResults() > 0) {
            auto type = func.getFunctionType().getResult(0);
            if (auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(type)) {
                ov::Shape shape;
                shape.reserve(ranked.getRank());
                for (int64_t dim : ranked.getShape()) {
                    shape.push_back(dim == mlir::ShapedType::kDynamic ? 1 : static_cast<size_t>(dim));
                }
                return shape;
            }
            if (auto memref = llvm::dyn_cast<mlir::MemRefType>(type)) {
                ov::Shape shape;
                shape.reserve(memref.getRank());
                for (int64_t dim : memref.getShape()) {
                    shape.push_back(dim == mlir::ShapedType::kDynamic ? 1 : static_cast<size_t>(dim));
                }
                return shape;
            }
        }
    }
    return static_shape_or_placeholder(node->get_output_partial_shape(0));
}

std::vector<int64_t> read_absorbed_input_permutation(mlir::ModuleOp module, size_t input_idx) {
    std::vector<int64_t> permutation;
    if (!module) {
        return permutation;
    }
    auto attr = module->getAttrOfType<mlir::ArrayAttr>("gfx.absorbed_input" + std::to_string(input_idx) + "_perm");
    if (!attr) {
        return permutation;
    }
    permutation.reserve(attr.size());
    for (auto value : attr) {
        auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(value);
        OPENVINO_ASSERT(int_attr, "GFX Metal: absorbed input permutation attr must be integer");
        permutation.push_back(int_attr.getInt());
    }
    return permutation;
}

std::optional<EltwiseKind> eltwise_kind_from_node(const ov::Node& node) {
    const std::string type = node.get_type_name();
    if (type == "Add") return EltwiseKind::Add;
    if (type == "Subtract") return EltwiseKind::Sub;
    if (type == "Multiply") return EltwiseKind::Mul;
    if (type == "Divide") return EltwiseKind::Div;
    if (type == "Power") return EltwiseKind::Pow;
    if (type == "Mod") return EltwiseKind::Mod;
    if (type == "FloorMod") return EltwiseKind::FloorMod;
    if (type == "PRelu") return EltwiseKind::Prelu;
    if (type == "SquaredDifference") return EltwiseKind::SquaredDiff;
    if (type == "Minimum") return EltwiseKind::Min;
    if (type == "Maximum") return EltwiseKind::Max;
    if (type == "LogicalAnd") return EltwiseKind::LogicalAnd;
    if (type == "LogicalOr") return EltwiseKind::LogicalOr;
    if (type == "LogicalXor") return EltwiseKind::LogicalXor;
    if (type == "Equal") return EltwiseKind::Equal;
    if (type == "NotEqual") return EltwiseKind::NotEqual;
    if (type == "Less") return EltwiseKind::Less;
    if (type == "Greater") return EltwiseKind::Greater;
    if (type == "LessEqual") return EltwiseKind::LessEqual;
    if (type == "GreaterEqual") return EltwiseKind::GreaterEqual;
    return std::nullopt;
}

std::optional<ReduceKind> reduce_kind_from_node(const ov::Node& node) {
    const std::string type = node.get_type_name();
    if (type == "ReduceSum") return ReduceKind::Sum;
    if (type == "ReduceMean") return ReduceKind::Mean;
    if (type == "ReduceMax") return ReduceKind::Max;
    if (type == "ReduceMin") return ReduceKind::Min;
    if (type == "ReduceProd") return ReduceKind::Prod;
    if (type == "ReduceL1") return ReduceKind::L1;
    if (type == "ReduceL2") return ReduceKind::L2;
    return std::nullopt;
}

std::optional<ActivationKind> unary_activation_kind_from_node(const ov::Node& node) {
    const std::string type = node.get_type_name();
    if (type == "Relu") return ActivationKind::Relu;
    if (type == "Sigmoid") return ActivationKind::Sigmoid;
    if (type == "Tanh") return ActivationKind::Tanh;
    if (type == "Elu") return ActivationKind::Elu;
    if (type == "Gelu") return ActivationKind::Gelu;
    if (type == "Swish") return ActivationKind::Swish;
    if (type == "HSwish") return ActivationKind::HSwish;
    if (type == "HSigmoid") return ActivationKind::HSigmoid;
    if (type == "SoftPlus") return ActivationKind::SoftPlus;
    if (type == "Mish") return ActivationKind::Mish;
    if (type == "SoftSign") return ActivationKind::SoftSign;
    if (type == "Abs") return ActivationKind::Abs;
    if (type == "Sign") return ActivationKind::Sign;
    if (type == "Clamp") return ActivationKind::Clamp;
    if (type == "Exp") return ActivationKind::Exp;
    if (type == "Log") return ActivationKind::Log;
    if (type == "Sqrt") return ActivationKind::Sqrt;
    if (type == "Floor") return ActivationKind::Floor;
    if (type == "Ceiling" || type == "Ceil") return ActivationKind::Ceil;
    if (type == "Negative") return ActivationKind::Negative;
    if (type == "Sin") return ActivationKind::Sin;
    if (type == "Cos") return ActivationKind::Cos;
    if (type == "Tan") return ActivationKind::Tan;
    if (type == "Erf") return ActivationKind::Erf;
    if (type == "Asin") return ActivationKind::Asin;
    if (type == "Acos") return ActivationKind::Acos;
    if (type == "Atan") return ActivationKind::Atan;
    if (type == "Asinh") return ActivationKind::Asinh;
    if (type == "Acosh") return ActivationKind::Acosh;
    if (type == "Atanh") return ActivationKind::Atanh;
    if (type == "Sinh") return ActivationKind::Sinh;
    if (type == "Cosh") return ActivationKind::Cosh;
    if (type == "Round") return ActivationKind::RoundAway;
    return std::nullopt;
}

std::optional<ActivationKind> activation_kind_from_module_attr(mlir::ModuleOp module,
                                                               llvm::StringRef attr_name) {
    if (!module) {
        return std::nullopt;
    }
    auto attr = module->getAttrOfType<mlir::StringAttr>(attr_name);
    if (!attr) {
        return std::nullopt;
    }
    const auto value = attr.getValue();
    if (value == "Relu") return ActivationKind::Relu;
    if (value == "Sigmoid") return ActivationKind::Sigmoid;
    if (value == "Tanh") return ActivationKind::Tanh;
    if (value == "Gelu") return ActivationKind::Gelu;
    if (value == "Swish") return ActivationKind::Swish;
    if (value == "HSwish") return ActivationKind::HSwish;
    if (value == "HSigmoid") return ActivationKind::HSigmoid;
    return std::nullopt;
}

std::vector<int64_t> get_slice_const_i64(const ov::Output<ov::Node>& source, const char* what) {
    auto c = ov::util::get_constant_from_source(source);
    OPENVINO_ASSERT(c, "GFX Metal Slice: ", what, " must be Constant");
    return c->cast_vector<int64_t>();
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

struct StaticSliceMeta {
    std::vector<uint32_t> out_shape;
    std::vector<uint32_t> in_stride;
    std::vector<int32_t> starts;
    std::vector<int32_t> steps;
    uint32_t total = 0;
};

StaticSliceMeta build_static_slice_meta(const std::shared_ptr<const ov::Node>& node) {
    OPENVINO_ASSERT(node, "GFX Metal Slice: node is null");
    const auto in_shape = node->get_input_shape(0);
    const auto out_shape = node->get_output_shape(0);
    const size_t rank = in_shape.size();
    OPENVINO_ASSERT(rank == out_shape.size(),
                    "GFX Metal Slice: rank-changing Slice/StridedSlice is not supported");

    StaticSliceMeta meta;
    meta.out_shape.reserve(rank);
    meta.starts.assign(rank, 0);
    meta.steps.assign(rank, 1);
    meta.in_stride.assign(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        meta.out_shape.push_back(static_cast<uint32_t>(out_shape[i]));
    }
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        meta.in_stride[static_cast<size_t>(i)] =
            meta.in_stride[static_cast<size_t>(i + 1)] * static_cast<uint32_t>(in_shape[static_cast<size_t>(i + 1)]);
    }
    meta.total = static_cast<uint32_t>(ov::shape_size(out_shape));

    if (auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node)) {
        auto starts = get_slice_const_i64(slice->input_value(1), "Slice starts");
        auto ends = get_slice_const_i64(slice->input_value(2), "Slice ends");
        auto steps = get_slice_const_i64(slice->input_value(3), "Slice steps");
        std::vector<int64_t> axes;
        if (slice->get_input_size() > 4) {
            axes = get_slice_const_i64(slice->input_value(4), "Slice axes");
        } else {
            axes.resize(starts.size());
            std::iota(axes.begin(), axes.end(), 0);
        }
        OPENVINO_ASSERT(starts.size() == ends.size() && starts.size() == steps.size() && starts.size() == axes.size(),
                        "GFX Metal Slice: starts/ends/steps/axes size mismatch");
        for (size_t i = 0; i < axes.size(); ++i) {
            int64_t axis = axes[i];
            if (axis < 0) {
                axis += static_cast<int64_t>(rank);
            }
            OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank,
                            "GFX Metal Slice: axis out of range");
            OPENVINO_ASSERT(steps[i] != 0, "GFX Metal Slice: zero step is not supported");
            const auto dim = static_cast<int64_t>(in_shape[static_cast<size_t>(axis)]);
            meta.starts[static_cast<size_t>(axis)] =
                static_cast<int32_t>(normalize_slice_index(starts[i], dim, true));
            meta.steps[static_cast<size_t>(axis)] = static_cast<int32_t>(steps[i]);
        }
        return meta;
    }

    auto slice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node);
    OPENVINO_ASSERT(slice, "GFX Metal Slice: expected Slice/StridedSlice node");
    OPENVINO_ASSERT(std::all_of(slice->get_new_axis_mask().begin(),
                                slice->get_new_axis_mask().end(),
                                [](int64_t v) { return v == 0; }),
                    "GFX Metal Slice: StridedSlice new_axis_mask is not supported");
    OPENVINO_ASSERT(std::all_of(slice->get_shrink_axis_mask().begin(),
                                slice->get_shrink_axis_mask().end(),
                                [](int64_t v) { return v == 0; }),
                    "GFX Metal Slice: StridedSlice shrink_axis_mask is not supported");
    OPENVINO_ASSERT(std::all_of(slice->get_ellipsis_mask().begin(),
                                slice->get_ellipsis_mask().end(),
                                [](int64_t v) { return v == 0; }),
                    "GFX Metal Slice: StridedSlice ellipsis_mask is not supported");

    auto begin = get_slice_const_i64(slice->input_value(1), "StridedSlice begin");
    auto end = get_slice_const_i64(slice->input_value(2), "StridedSlice end");
    std::vector<int64_t> strides(rank, 1);
    if (slice->get_input_size() > 3) {
        auto values = get_slice_const_i64(slice->input_value(3), "StridedSlice strides");
        OPENVINO_ASSERT(values.size() <= rank, "GFX Metal Slice: StridedSlice strides rank mismatch");
        std::copy(values.begin(), values.end(), strides.begin());
    }
    const auto& begin_mask = slice->get_begin_mask();
    const auto& end_mask = slice->get_end_mask();
    for (size_t axis = 0; axis < rank; ++axis) {
        const auto dim = static_cast<int64_t>(in_shape[axis]);
        const bool masked_begin = axis < begin_mask.size() && begin_mask[axis] != 0;
        const bool masked_end = axis < end_mask.size() && end_mask[axis] != 0;
        const int64_t step = strides[axis];
        OPENVINO_ASSERT(step != 0, "GFX Metal Slice: StridedSlice zero step is not supported");
        int64_t start = axis < begin.size() ? begin[axis] : 0;
        int64_t finish = axis < end.size() ? end[axis] : dim;
        start = masked_begin ? (step < 0 ? dim - 1 : 0) : normalize_slice_index(start, dim, true);
        finish = masked_end ? (step < 0 ? -1 : dim) : normalize_slice_index(finish, dim, false);
        (void)finish;
        meta.starts[axis] = static_cast<int32_t>(start);
        meta.steps[axis] = static_cast<int32_t>(step);
    }
    return meta;
}

std::string generate_static_msl_for_slice(const std::shared_ptr<const ov::Node>& node,
                                          const ov::element::Type& storage_type) {
    const auto meta = build_static_slice_meta(node);
    const auto scalar_t = msl_type_from_element(storage_type == ov::element::dynamic ? ov::element::f32 : storage_type);
    const uint32_t rank = static_cast<uint32_t>(meta.out_shape.size());
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "using scalar_t = " << scalar_t << ";\n";
    ss << "constant uint TOTAL_C = " << meta.total << ";\n";
    ss << "constant uint RANK_C = " << rank << ";\n";
    ss << "constant uint OUT_SHAPE_C[" << rank << "] = {";
    for (size_t i = 0; i < meta.out_shape.size(); ++i) {
        if (i) ss << ", ";
        ss << meta.out_shape[i];
    }
    ss << "};\n";
    ss << "constant uint IN_STRIDE_C[" << rank << "] = {";
    for (size_t i = 0; i < meta.in_stride.size(); ++i) {
        if (i) ss << ", ";
        ss << meta.in_stride[i];
    }
    ss << "};\n";
    ss << "constant int STARTS_C[" << rank << "] = {";
    for (size_t i = 0; i < meta.starts.size(); ++i) {
        if (i) ss << ", ";
        ss << meta.starts[i];
    }
    ss << "};\n";
    ss << "constant int STEPS_C[" << rank << "] = {";
    for (size_t i = 0; i < meta.steps.size(); ++i) {
        if (i) ss << ", ";
        ss << meta.steps[i];
    }
    ss << "};\n";
    ss << "kernel void slice_kernel(\n";
    ss << "  device const scalar_t* A [[buffer(0)]],\n";
    ss << "  device scalar_t* C [[buffer(1)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    if (gid >= TOTAL_C) return;\n";
    ss << "    uint idx = gid;\n";
    ss << "    int in_off = 0;\n";
    ss << "    for (int d = (int)RANK_C - 1; d >= 0; --d) {\n";
    ss << "        uint coord = idx % OUT_SHAPE_C[d];\n";
    ss << "        idx /= OUT_SHAPE_C[d];\n";
    ss << "        in_off += (STARTS_C[d] + int(coord) * STEPS_C[d]) * int(IN_STRIDE_C[d]);\n";
    ss << "    }\n";
    ss << "    C[gid] = A[in_off];\n";
    ss << "}\n";
    return ss.str();
}

std::string matmul_epilogue_activation_expr(ActivationKind activation) {
    switch (activation) {
        case ActivationKind::Relu:
            return "max(x, 0.0f)";
        case ActivationKind::Sigmoid:
            return "1.0f / (1.0f + exp(-x))";
        case ActivationKind::Tanh:
            return "tanh(x)";
        case ActivationKind::Gelu:
            return "0.5f * x * (1.0f + tanh(0.79788456f * (x + 0.044715f * x * x * x)))";
        case ActivationKind::Swish:
            return "(x >= 0.0f) ? (x / (1.0f + exp(-x))) : (x * exp(x) / (1.0f + exp(x)))";
        case ActivationKind::HSwish:
            return "x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
        case ActivationKind::HSigmoid:
            return "clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f";
        case ActivationKind::Abs:
            return "fabs(x)";
        case ActivationKind::Sign:
            return "(x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f)";
        case ActivationKind::Identity:
        default:
            return "x";
    }
}

}  // namespace

std::string normalize_msl_source_for_kernel_plan(std::string source,
                                                 std::string_view current_entry_point,
                                                 const GfxCustomKernelStagePlan& plan) {
    const auto& custom_kernel = plan.stage_manifest.custom_kernel;
    if (!plan.valid || !custom_kernel.valid || custom_kernel.entry_point.empty()) {
        return source;
    }
    (void)replace_kernel_entry_name(source, current_entry_point, custom_kernel.entry_point);
    return source;
}

GfxAppleMslStageLoweringPlan materialize_apple_msl_stage_manifest(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const std::string& stage_type,
    std::string_view kernel_entry_point) {
    GfxAppleMslStageLoweringPlan lowering_plan{};
    GfxAppleStagePipelineOptions options{};
    options.plan = plan;
    options.stage_type = stage_type;
    options.kernel_entry_point = std::string(kernel_entry_point);
    options.materialize_typed_program = false;
    const auto pipeline_result = run_gfx_apple_stage_pipeline(module, options);
    if (!pipeline_result.valid) {
        return lowering_plan;
    }
    lowering_plan.stage_plan = pipeline_result.stage_plan;
    if (lowering_plan.stage_plan.stage.kind != GfxMpsrtStageKind::MSLDispatch) {
        return {};
    }

    lowering_plan.custom_kernel_plan =
        make_gfx_custom_kernel_stage_plan(stage_type,
                                          lowering_plan.stage_plan.stage.dispatch_entry_point);
    if (!lowering_plan.custom_kernel_plan.valid) {
        return {};
    }
    lowering_plan.valid = true;
    return lowering_plan;
}

bool materialize_apple_msl_typed_program(
    mlir::ModuleOp module,
    const GfxAppleMslStageLoweringPlan& lowering_plan,
    const GfxMpsrtExternalBufferAbiPlan& external_buffer_abi) {
    if (!module || !lowering_plan.valid) {
        return false;
    }
    return materialize_module_mpsrt_ops_from_stage_plan(module,
                                                        lowering_plan.stage_plan,
                                                        external_buffer_abi);
}

void configure_msl_kernel_source_for_plan(KernelSource& source,
                                          std::string_view stage_type) {
    if (!source.module) {
        return;
    }

    GfxMpsrtModuleStagePlan stage_plan;
    if (!read_module_mpsrt_stage_plan(source.module, stage_plan) ||
        stage_plan.stage.kind != GfxMpsrtStageKind::MSLDispatch) {
        return;
    }

    auto custom_kernel_plan = make_gfx_custom_kernel_stage_plan(stage_type, source.entry_point);
    if (!custom_kernel_plan.valid) {
        custom_kernel_plan = make_gfx_custom_kernel_stage_plan(stage_plan.stage.stage_type, source.entry_point);
    }
    const auto& custom_kernel = custom_kernel_plan.stage_manifest.custom_kernel;
    if (!custom_kernel_plan.valid || !custom_kernel.valid || custom_kernel.entry_point.empty()) {
        return;
    }

    const std::string source_entry =
        source.entry_point.empty() ? stage_plan.stage.kernel_name : source.entry_point;
    const std::string required_entry = custom_kernel.entry_point;
    if (!source.msl_source.empty()) {
        source.msl_source = normalize_msl_source_for_kernel_plan(std::move(source.msl_source),
                                                                 source_entry,
                                                                 custom_kernel_plan);
    }
    if (source.msl_generator) {
        auto generator = std::move(source.msl_generator);
        source.msl_generator =
            [generator = std::move(generator), source_entry, custom_kernel_plan](mlir::ModuleOp module) mutable {
                return normalize_msl_source_for_kernel_plan(generator(module),
                                                            source_entry,
                                                            custom_kernel_plan);
        };
    }
    source.entry_point = required_entry;
    GfxMpsrtExternalBufferAbiPlan external_buffer_abi{};
    if (gfx_mpsrt_external_buffer_abi_from_kernel_manifest(source.module,
                                                          external_buffer_abi,
                                                          source.signature.arg_count,
                                                          source.signature.output_arg_count) &&
        read_module_mpsrt_stage_plan(source.module, stage_plan)) {
        GfxAppleMslStageLoweringPlan lowering_plan{};
        lowering_plan.valid = true;
        lowering_plan.stage_plan = std::move(stage_plan);
        lowering_plan.custom_kernel_plan = std::move(custom_kernel_plan);
        (void)materialize_apple_msl_typed_program(source.module, lowering_plan, external_buffer_abi);
    }
}

GfxMpsrtKernelSourcePlan configure_msl_kernel_source_plan(KernelSource source,
                                                          std::string_view stage_type) {
    configure_msl_kernel_source_for_plan(source, stage_type);
    return make_mpsrt_kernel_source_plan_from_configured_source(std::move(source));
}

GfxMpsrtKernelSourcePlan configure_msl_kernel_source_plan_for_node(
    KernelSource source,
    const std::shared_ptr<const ov::Node>& node,
    const GpuBufferManager* buffer_manager,
    std::string_view stage_type,
    bool has_bias,
    bool has_activation,
    bool has_batchnorm) {
    if (!source.module || !node) {
        return {};
    }

    const auto msl_kernel_plan = make_gfx_custom_kernel_stage_plan(stage_type, source.entry_point);
    if (!msl_kernel_plan.valid) {
        return {};
    }

    auto plan = select_stage_optimization_plan(buffer_manager,
                                               GpuBackend::Metal,
                                               std::string(stage_type),
                                               node,
                                               node->get_output_element_type(0),
                                               has_bias,
                                               has_activation,
                                               has_batchnorm,
                                               GfxStageRuntimeTraits{});
    if (plan.placement.domain != GfxStageBackendDomain::AppleMsl) {
        force_apple_msl_buffer_placement(plan, stage_type);
    }

    annotate_msl_module_with_stage_plan(source.module, plan, std::string(stage_type), source.entry_point);
    auto source_plan = configure_msl_kernel_source_plan(source, stage_type);
    if (source_plan.valid()) {
        return source_plan;
    }
    configure_msl_kernel_source_for_plan(source, stage_type);
    return make_mpsrt_kernel_source_plan_from_configured_source(std::move(source));
}

static bool configure_apple_metal_slice_kernel_source(KernelSource& source,
                                                      const std::shared_ptr<const ov::Node>& node,
                                                      const ov::element::Type& storage_type,
                                                      bool has_runtime_slice_params);
static bool configure_apple_metal_compute_kernel_source(KernelSource& source,
                                                        const std::shared_ptr<const ov::Node>& node);
static bool configure_apple_metal_softmax_kernel_source(
    KernelSource& source,
    const std::shared_ptr<const ov::Node>& node,
    const std::optional<ov::Shape>& runtime_input_shape = std::nullopt);
static bool configure_apple_metal_pool2d_kernel_source(KernelSource& source,
                                                       const std::shared_ptr<const ov::Node>& node);
static bool configure_apple_metal_unary_kernel_source(KernelSource& source,
                                                      const std::shared_ptr<const ov::Node>& node);
static bool configure_apple_metal_elementwise_kernel_source(KernelSource& source,
                                                            const std::shared_ptr<const ov::Node>& node);
static bool configure_apple_metal_structural_kernel_source(KernelSource& source,
                                                           const std::shared_ptr<const ov::Node>& node);
static bool configure_apple_metal_data_movement_kernel_source(KernelSource& source,
                                                              const std::shared_ptr<const ov::Node>& node);
static bool configure_apple_metal_msl_kernel_source(
    KernelSource& source,
    const std::shared_ptr<const ov::Node>& node,
    std::string_view stage_type,
    const ov::element::Type& storage_type,
    bool has_runtime_slice_params,
    const std::optional<ov::Shape>& runtime_input_shape = std::nullopt);

static bool has_apple_msl_custom_kernel_manifest(mlir::ModuleOp module) {
    GfxKernelStageManifest manifest{};
    return module &&
           detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest) &&
           manifest.valid &&
           manifest.backend_domain == GfxKernelBackendDomain::AppleMsl &&
           manifest.execution_kind == GfxKernelExecutionKind::CustomKernel;
}

static GfxMpsrtKernelSourcePlan try_configure_apple_mps_vendor_kernel_source_plan_for_node(
    KernelSource source,
    const std::shared_ptr<const ov::Node>& node,
    const GpuBufferManager* buffer_manager,
    std::string_view stage_type,
    bool has_bias,
    bool has_activation,
    bool has_batchnorm,
    ActivationKind activation) {
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
        const auto plan = select_stage_optimization_plan(buffer_manager,
                                                         GpuBackend::Metal,
                                                         canonical_stage_type,
                                                         node,
                                                         node->get_output_element_type(0),
                                                         /*has_bias=*/false,
                                                         /*has_activation=*/false,
                                                         /*has_batchnorm=*/false,
                                                         GfxStageRuntimeTraits{});
        const auto lowering = annotate_module_with_conv_mpsrt_plan(source.module,
                                                                   plan,
                                                                   node,
                                                                   fallback_stage_type,
                                                                   has_activation,
                                                                   activation);
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
            const auto plan = select_stage_optimization_plan(buffer_manager,
                                                             GpuBackend::Metal,
                                                             std::string(stage_type),
                                                             node,
                                                             node->get_output_element_type(0),
                                                             /*has_bias=*/false,
                                                             /*has_activation=*/false,
                                                             /*has_batchnorm=*/false,
                                                             GfxStageRuntimeTraits{});
            if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
                plan.placement.storage == GfxStageStorageKind::Image) {
                std::vector<GfxMpsrtTensorDesc> input_descs;
                std::vector<GfxMpsrtTensorDesc> output_descs;
                if (!gfx_apple_make_mps_io_tensor_descs_for_node(node,
                                                                 GfxStageStorageKind::Image,
                                                                 input_descs,
                                                                 output_descs) ||
                    input_descs.front().image_feature_channels % 4u != 0) {
                    return {};
                }
                auto source_module = source.module;
                const auto materialized = materialize_apple_mps_pool2d_program(source_module,
                                                                               plan,
                                                                               std::string(stage_type),
                                                                               pool_desc,
                                                                               {},
                                                                               input_descs,
                                                                               output_descs);
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
            const auto plan = select_stage_optimization_plan(buffer_manager,
                                                             GpuBackend::Metal,
                                                             "Interpolate",
                                                             node,
                                                             node->get_output_element_type(0),
                                                             /*has_bias=*/false,
                                                             /*has_activation=*/false,
                                                             /*has_batchnorm=*/false,
                                                             GfxStageRuntimeTraits{});
            if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
                plan.placement.storage == GfxStageStorageKind::Image) {
                std::vector<GfxMpsrtTensorDesc> input_descs;
                std::vector<GfxMpsrtTensorDesc> output_descs;
                if (!gfx_apple_make_mps_io_tensor_descs_for_node(node,
                                                                 GfxStageStorageKind::Image,
                                                                 input_descs,
                                                                 output_descs)) {
                    return {};
                }
                auto source_module = source.module;
                const auto materialized = materialize_apple_mps_resize2d_program(source_module,
                                                                                 plan,
                                                                                 "Interpolate",
                                                                                 resize_desc,
                                                                                 {},
                                                                                 input_descs,
                                                                                 output_descs);
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
            const auto plan = select_stage_optimization_plan(buffer_manager,
                                                             GpuBackend::Metal,
                                                             "Softmax",
                                                             node,
                                                             node->get_output_element_type(0),
                                                             /*has_bias=*/false,
                                                             /*has_activation=*/false,
                                                             /*has_batchnorm=*/false,
                                                             GfxStageRuntimeTraits{});
            if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
                plan.placement.storage == GfxStageStorageKind::Matrix) {
                std::vector<GfxMpsrtTensorDesc> input_descs;
                std::vector<GfxMpsrtTensorDesc> output_descs;
                if (!gfx_apple_make_mps_io_tensor_descs_for_node(node,
                                                                 GfxStageStorageKind::Matrix,
                                                                 input_descs,
                                                                 output_descs)) {
                    return {};
                }
                auto source_module = source.module;
                const auto materialized = materialize_apple_mps_softmax_program(source_module,
                                                                                plan,
                                                                                "Softmax",
                                                                                softmax_desc,
                                                                                {},
                                                                                input_descs,
                                                                                output_descs);
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
            const auto plan = select_stage_optimization_plan(buffer_manager,
                                                             GpuBackend::Metal,
                                                             "TopK",
                                                             node,
                                                             node->get_output_element_type(0),
                                                             /*has_bias=*/false,
                                                             /*has_activation=*/false,
                                                             /*has_batchnorm=*/false,
                                                             GfxStageRuntimeTraits{});
            if (plan.placement.domain == GfxStageBackendDomain::AppleMps &&
                plan.placement.storage == GfxStageStorageKind::Matrix) {
                std::vector<GfxMpsrtTensorDesc> input_descs;
                std::vector<GfxMpsrtTensorDesc> output_descs;
                if (!gfx_apple_make_mps_io_tensor_descs_for_node(node,
                                                                 GfxStageStorageKind::Matrix,
                                                                 input_descs,
                                                                 output_descs)) {
                    return {};
                }
                auto source_module = source.module;
                const auto materialized = materialize_apple_mps_topk_program(source_module,
                                                                             plan,
                                                                             "TopK",
                                                                             topk_desc,
                                                                             {},
                                                                             input_descs,
                                                                             output_descs);
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

static GfxMpsrtKernelSourcePlan try_configure_clean_apple_mps_vendor_kernel_source_plan_for_node(
    const std::shared_ptr<const ov::Node>& node,
    const GpuBufferManager* buffer_manager,
    std::string_view stage_type,
    bool has_bias,
    bool has_activation,
    bool has_batchnorm,
    ActivationKind activation) {
    if (!node) {
        return {};
    }

    KernelSource clean_source;
    clean_source.module = build_mlir_for_node(node, gfx_mlir_context());
    if (!clean_source.module) {
        return {};
    }
    return try_configure_apple_mps_vendor_kernel_source_plan_for_node(clean_source,
                                                                      node,
                                                                      buffer_manager,
                                                                      stage_type,
                                                                      has_bias,
                                                                      has_activation,
                                                                      has_batchnorm,
                                                                      activation);
}

static GfxKernelStageFamily resolve_apple_metal_msl_stage_family(const KernelSource& source,
                                                                 std::string_view stage_type) {
    GfxKernelStageManifest manifest{};
    if (source.module &&
        detail::gfx_mpsrt_read_stage_manifest_attrs(source.module, manifest) &&
        manifest.valid &&
        manifest.backend_domain == GfxKernelBackendDomain::AppleMsl &&
        manifest.execution_kind == GfxKernelExecutionKind::CustomKernel) {
        return manifest.stage_family;
    }
    const auto kernel_family = classify_gfx_custom_kernel_family(stage_type, source.entry_point);
    return gfx_kernel_stage_family_from_kernel_family(kernel_family);
}

static bool configure_apple_metal_msl_kernel_source_for_stage_type(
    KernelSource& source,
    const std::shared_ptr<const ov::Node>& node,
    std::string_view stage_type,
    const std::optional<ov::Shape>& runtime_input_shape) {
    switch (resolve_apple_metal_msl_stage_family(source, stage_type)) {
        case GfxKernelStageFamily::Convolution:
        case GfxKernelStageFamily::Conv3D:
        case GfxKernelStageFamily::Gemm:
        case GfxKernelStageFamily::RmsnormRope:
            return configure_apple_metal_compute_kernel_source(source, node);
        case GfxKernelStageFamily::Pooling:
            return configure_apple_metal_pool2d_kernel_source(source, node);
        case GfxKernelStageFamily::Softmax:
        case GfxKernelStageFamily::AttentionSoftmax:
            return configure_apple_metal_softmax_kernel_source(source, node, runtime_input_shape);
        case GfxKernelStageFamily::Eltwise:
            return configure_apple_metal_elementwise_kernel_source(source, node) ||
                   configure_apple_metal_unary_kernel_source(source, node) ||
                   configure_apple_metal_structural_kernel_source(source, node);
        case GfxKernelStageFamily::Transpose:
            return configure_apple_metal_data_movement_kernel_source(source, node) ||
                   configure_apple_metal_structural_kernel_source(source, node);
        case GfxKernelStageFamily::ConcatSplit:
        case GfxKernelStageFamily::Reduction:
        case GfxKernelStageFamily::TopK:
        case GfxKernelStageFamily::Convert:
        case GfxKernelStageFamily::Layout:
            return configure_apple_metal_structural_kernel_source(source, node);
        case GfxKernelStageFamily::GatherScatter:
            return configure_apple_metal_data_movement_kernel_source(source, node) ||
                   configure_apple_metal_structural_kernel_source(source, node);
        case GfxKernelStageFamily::GroupConvolution:
        case GfxKernelStageFamily::Resize:
        case GfxKernelStageFamily::KvCache:
        case GfxKernelStageFamily::Unknown:
        default:
            return false;
    }
}

static bool configure_apple_metal_msl_kernel_source(KernelSource& source,
                                                    const std::shared_ptr<const ov::Node>& node,
                                                    std::string_view stage_type,
                                                    const ov::element::Type& storage_type,
                                                    bool has_runtime_slice_params,
                                                    const std::optional<ov::Shape>& runtime_input_shape) {
    if (!source.msl_generator && source.msl_source.empty()) {
        (void)configure_apple_metal_msl_kernel_source_for_stage_type(source,
                                                                     node,
                                                                     stage_type,
                                                                     runtime_input_shape);
    }

    (void)configure_apple_metal_slice_kernel_source(source,
                                                    node,
                                                    storage_type,
                                                    has_runtime_slice_params);
    return source.msl_generator || !source.msl_source.empty();
}

GfxMpsrtKernelSourcePlan configure_apple_metal_kernel_source_plan_for_stage(
    KernelSource& source,
    const std::shared_ptr<const ov::Node>& node,
    const GpuBufferManager* buffer_manager,
    std::string_view stage_type,
    bool has_bias,
    bool has_activation,
    bool has_batchnorm,
    ActivationKind activation,
    const ov::element::Type& storage_type,
    bool has_runtime_slice_params,
    const std::optional<ov::Shape>& runtime_input_shape) {
    if (source.module) {
        auto vendor_source_plan =
            try_configure_apple_mps_vendor_kernel_source_plan_for_node(source,
                                                                       node,
                                                                       buffer_manager,
                                                                       stage_type,
                                                                       has_bias,
                                                                       has_activation,
                                                                       has_batchnorm,
                                                                       activation);
        if (vendor_source_plan.valid()) {
            return vendor_source_plan;
        }
        if (has_apple_msl_custom_kernel_manifest(source.module)) {
            auto clean_vendor_source_plan =
                try_configure_clean_apple_mps_vendor_kernel_source_plan_for_node(node,
                                                                                 buffer_manager,
                                                                                 stage_type,
                                                                                 has_bias,
                                                                                 has_activation,
                                                                                 has_batchnorm,
                                                                                 activation);
            if (clean_vendor_source_plan.valid()) {
                return clean_vendor_source_plan;
            }
        }
    }

    configure_apple_metal_msl_kernel_source(source,
                                            node,
                                            stage_type,
                                            storage_type,
                                            has_runtime_slice_params,
                                            runtime_input_shape);
    if (!source.module) {
        return {};
    }
    return configure_msl_kernel_source_plan_for_node(source,
                                                     node,
                                                     buffer_manager,
                                                     stage_type,
                                                     has_bias,
                                                     has_activation,
                                                     has_batchnorm);
}

static bool configure_apple_metal_slice_kernel_source(KernelSource& source,
                                                      const std::shared_ptr<const ov::Node>& node,
                                                      const ov::element::Type& storage_type,
                                                      bool has_runtime_slice_params) {
    if (!node ||
        (!ov::is_type<const ov::op::v8::Slice>(node) &&
         !ov::is_type<const ov::op::v1::StridedSlice>(node))) {
        return false;
    }

    const ov::element::Type effective_type =
        storage_type == ov::element::dynamic ? node->get_output_element_type(0) : storage_type;
    ConvertCodegenDesc desc{};
    desc.element_type = effective_type;
    desc.dst_type = effective_type;
    source.entry_point = "slice_kernel";
    const bool dynamic_slice_shape = !node->get_input_partial_shape(0).is_static() ||
                                     !node->get_output_partial_shape(0).is_static();
    if (!has_runtime_slice_params && !dynamic_slice_shape) {
        source.msl_source = generate_static_msl_for_slice(node, desc.dst_type);
        source.msl_generator = {};
        source.module = {};
        return true;
    }

    source.msl_source.clear();
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
        return generate_msl_for_slice_generic(desc, module);
    };
    return true;
}

static bool annotate_apple_msl_custom_kernel_binding(mlir::ModuleOp module,
                                                     std::string_view stage_type,
                                                     std::string_view entry_point,
                                                     std::vector<size_t> tensor_input_indices,
                                                     std::vector<int32_t> scalar_args = {}) {
    if (!module) {
        return false;
    }

    auto plan = scalar_args.empty()
                    ? make_msl_runtime_binding_plan_for_custom_kernel(stage_type,
                                                                      entry_point,
                                                                      std::move(tensor_input_indices))
                    : make_msl_runtime_binding_plan_for_custom_kernel(stage_type,
                                                                      entry_point,
                                                                      std::move(tensor_input_indices),
                                                                      std::move(scalar_args));
    return annotate_msl_module_with_runtime_binding_plan(module, plan);
}

static void require_apple_msl_custom_kernel_binding(mlir::ModuleOp module,
                                                    std::string_view stage_type,
                                                    std::string_view entry_point,
                                                    std::vector<size_t> tensor_input_indices,
                                                    std::vector<int32_t> scalar_args = {}) {
    OPENVINO_ASSERT(annotate_apple_msl_custom_kernel_binding(module,
                                                            stage_type,
                                                            entry_point,
                                                            std::move(tensor_input_indices),
                                                            std::move(scalar_args)),
                    "GFX Metal MSL: failed to derive runtime binding from stage manifest for ",
                    stage_type,
                    " / ",
                    entry_point);
}

static bool configure_apple_metal_compute_kernel_source(KernelSource& source,
                                                        const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }

    auto set_desc = [&](auto&& desc, const char* entry) {
        source.entry_point = entry;
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
    };

    if (auto conv = std::dynamic_pointer_cast<const ov::op::v1::Convolution>(node)) {
        const auto in_shape = conv->get_input_shape(0);
        if (in_shape.size() == 5) {
            Conv3DCodegenDesc desc{};
            const auto weights_shape = conv->get_input_shape(1);
            desc.element_type = conv->get_output_element_type(0);
            desc.N = static_cast<uint32_t>(in_shape.at(0));
            desc.C_in = static_cast<uint32_t>(in_shape.at(1));
            desc.D = static_cast<uint32_t>(in_shape.at(2));
            desc.H = static_cast<uint32_t>(in_shape.at(3));
            desc.W = static_cast<uint32_t>(in_shape.at(4));
            desc.C_out = static_cast<uint32_t>(weights_shape.at(0));
            desc.kD = static_cast<uint32_t>(weights_shape.at(2));
            desc.kH = static_cast<uint32_t>(weights_shape.at(3));
            desc.kW = static_cast<uint32_t>(weights_shape.at(4));
            desc.strideD = static_cast<uint32_t>(conv->get_strides().at(0));
            desc.strideH = static_cast<uint32_t>(conv->get_strides().at(1));
            desc.strideW = static_cast<uint32_t>(conv->get_strides().at(2));
            desc.dilationD = static_cast<uint32_t>(conv->get_dilations().at(0));
            desc.dilationH = static_cast<uint32_t>(conv->get_dilations().at(1));
            desc.dilationW = static_cast<uint32_t>(conv->get_dilations().at(2));
            desc.padFront = static_cast<uint32_t>(conv->get_pads_begin().at(0));
            desc.padTop = static_cast<uint32_t>(conv->get_pads_begin().at(1));
            desc.padLeft = static_cast<uint32_t>(conv->get_pads_begin().at(2));
            desc.padBack = static_cast<uint32_t>(conv->get_pads_end().at(0));
            desc.padBottom = static_cast<uint32_t>(conv->get_pads_end().at(1));
            desc.padRight = static_cast<uint32_t>(conv->get_pads_end().at(2));
            const auto out_shape = conv->get_output_shape(0);
            desc.outD = static_cast<uint32_t>(out_shape.at(2));
            desc.outH = static_cast<uint32_t>(out_shape.at(3));
            desc.outW = static_cast<uint32_t>(out_shape.at(4));
            set_desc(desc, "conv3d_kernel");
            if (source.module) {
                require_apple_msl_custom_kernel_binding(source.module,
                                                        "Convolution",
                                                        "conv3d_kernel",
                                                        {0});
            }
            return true;
        }

        Conv2DCodegenDesc desc{};
        const auto weights_shape = conv->get_input_shape(1);
        desc.element_type = conv->get_output_element_type(0);
        desc.input_type = conv->get_input_element_type(0);
        desc.weight_type = conv->get_input_element_type(1);
        desc.output_type = conv->get_output_element_type(0);
        desc.N = static_cast<uint32_t>(in_shape.at(0));
        desc.C_in = static_cast<uint32_t>(in_shape.at(1));
        desc.H = static_cast<uint32_t>(in_shape.at(2));
        desc.W = static_cast<uint32_t>(in_shape.at(3));
        desc.C_out = static_cast<uint32_t>(weights_shape.at(0));
        const uint32_t cin_pg = static_cast<uint32_t>(weights_shape.at(1));
        desc.groups = (cin_pg && desc.C_in % cin_pg == 0) ? desc.C_in / cin_pg : 1;
        desc.C_in_pg = cin_pg;
        desc.C_out_pg = desc.groups ? desc.C_out / desc.groups : desc.C_out;
        desc.kH = static_cast<uint32_t>(weights_shape.at(2));
        desc.kW = static_cast<uint32_t>(weights_shape.at(3));
        desc.strideH = static_cast<uint32_t>(conv->get_strides().at(0));
        desc.strideW = static_cast<uint32_t>(conv->get_strides().at(1));
        desc.dilationH = static_cast<uint32_t>(conv->get_dilations().at(0));
        desc.dilationW = static_cast<uint32_t>(conv->get_dilations().at(1));
        desc.padTop = static_cast<uint32_t>(conv->get_pads_begin().at(0));
        desc.padLeft = static_cast<uint32_t>(conv->get_pads_begin().at(1));
        desc.padBottom = static_cast<uint32_t>(conv->get_pads_end().at(0));
        desc.padRight = static_cast<uint32_t>(conv->get_pads_end().at(1));
        const auto out_shape = conv->get_output_shape(0);
        desc.outH = static_cast<uint32_t>(out_shape.at(2));
        desc.outW = static_cast<uint32_t>(out_shape.at(3));
        desc.output_channels_per_thread = gfx_conv2d_output_channel_block(desc);
        desc.output_width_per_thread = gfx_conv2d_output_width_block(desc);
        set_desc(desc, "conv2d_kernel");
        if (source.module) {
            require_apple_msl_custom_kernel_binding(source.module,
                                                    "Convolution",
                                                    "conv2d_kernel",
                                                    {0});
        }
        return true;
    }

    if (auto group_conv = std::dynamic_pointer_cast<const ov::op::v1::GroupConvolution>(node)) {
        Conv2DCodegenDesc desc{};
        const auto in_shape = group_conv->get_input_shape(0);
        const auto weights_shape = group_conv->get_input_shape(1);
        desc.element_type = group_conv->get_output_element_type(0);
        desc.input_type = group_conv->get_input_element_type(0);
        desc.weight_type = group_conv->get_input_element_type(1);
        desc.output_type = group_conv->get_output_element_type(0);
        desc.N = static_cast<uint32_t>(in_shape.at(0));
        desc.C_in = static_cast<uint32_t>(in_shape.at(1));
        desc.H = static_cast<uint32_t>(in_shape.at(2));
        desc.W = static_cast<uint32_t>(in_shape.at(3));
        desc.groups = static_cast<uint32_t>(weights_shape.at(0));
        desc.C_out_pg = static_cast<uint32_t>(weights_shape.at(1));
        desc.C_in_pg = static_cast<uint32_t>(weights_shape.at(2));
        desc.C_out = desc.groups * desc.C_out_pg;
        desc.kH = static_cast<uint32_t>(weights_shape.at(3));
        desc.kW = static_cast<uint32_t>(weights_shape.at(4));
        desc.strideH = static_cast<uint32_t>(group_conv->get_strides().at(0));
        desc.strideW = static_cast<uint32_t>(group_conv->get_strides().at(1));
        desc.dilationH = static_cast<uint32_t>(group_conv->get_dilations().at(0));
        desc.dilationW = static_cast<uint32_t>(group_conv->get_dilations().at(1));
        desc.padTop = static_cast<uint32_t>(group_conv->get_pads_begin().at(0));
        desc.padLeft = static_cast<uint32_t>(group_conv->get_pads_begin().at(1));
        desc.padBottom = static_cast<uint32_t>(group_conv->get_pads_end().at(0));
        desc.padRight = static_cast<uint32_t>(group_conv->get_pads_end().at(1));
        set_desc(desc, "conv2d_kernel");
        if (source.module) {
            require_apple_msl_custom_kernel_binding(source.module,
                                                    "GroupConvolution",
                                                    "conv2d_kernel",
                                                    {0});
        }
        return true;
    }

    if (auto matmul = std::dynamic_pointer_cast<const ov::op::v0::MatMul>(node)) {
        MatMulCodegenDesc desc{};
        const auto out_shape = output_shape_for_codegen(source.module, node);
        const auto a_shape = static_shape_or_placeholder(matmul->get_input_partial_shape(0));
        const auto b_shape = static_shape_or_placeholder(matmul->get_input_partial_shape(1));
        const size_t a_rank = a_shape.size();
        const size_t b_rank = b_shape.size();
        const size_t out_rank = out_shape.size();
        OPENVINO_ASSERT(a_rank >= 2 && b_rank >= 2 && out_rank >= 2,
                        "GFX Metal MatMul: ranks must be at least 2");
        desc.element_type = matmul->get_output_element_type(0);
        desc.input_a_type = matmul->get_input_element_type(0);
        desc.input_b_type = matmul->get_input_element_type(1);
        desc.output_type = matmul->get_output_element_type(0);
        desc.a_transpose = matmul->get_transpose_a();
        desc.b_transpose = matmul->get_transpose_b();
        desc.M = static_cast<int64_t>(out_shape[out_rank - 2]);
        desc.N = static_cast<int64_t>(out_shape[out_rank - 1]);
        desc.K = static_cast<int64_t>(desc.a_transpose ? a_shape[a_rank - 2] : a_shape[a_rank - 1]);
        desc.batch_a = static_cast<int64_t>(ov::shape_size(a_shape) / static_cast<uint64_t>(desc.M * desc.K));
        desc.batch_b = static_cast<int64_t>(ov::shape_size(b_shape) / static_cast<uint64_t>(desc.K * desc.N));
        desc.b_is_nk_layout = desc.b_transpose;
        desc.batch = static_cast<int64_t>(ov::shape_size(out_shape) / (desc.M * desc.N));
        set_desc(desc, "matmul_kernel");
        if (source.module) {
            require_apple_msl_custom_kernel_binding(source.module,
                                                    "MatMul",
                                                    "matmul_kernel",
                                                    {0, 1});
        }
        return true;
    }

    if (auto rms = std::dynamic_pointer_cast<const ov::op::internal::RMS>(node)) {
        const auto data_shape = static_shape_or_placeholder(rms->get_input_partial_shape(0));
        const auto gamma_shape = static_shape_or_placeholder(rms->get_input_partial_shape(1));
        OPENVINO_ASSERT(!data_shape.empty() && data_shape.back() > 0,
                        "GFX Metal RMS: hidden dimension must be static");
        RmsCodegenDesc desc{};
        desc.element_type = rms->get_output_element_type(0);
        desc.input_type = rms->get_input_element_type(0);
        desc.gamma_type = rms->get_input_element_type(1);
        desc.output_type = rms->get_output_element_type(0);
        desc.hidden = static_cast<uint32_t>(data_shape.back());
        desc.gamma_size = static_cast<uint32_t>(std::max<uint64_t>(1, ov::shape_size(gamma_shape)));
        desc.reduction_threads = gfx_rms_parallel_reduction_threads(desc.hidden);
        desc.epsilon = static_cast<float>(rms->get_epsilon());
        desc.has_residual_add = source.module && source.module->hasAttr("gfx.fused_residual_add");
        set_desc(desc, "rms_kernel");
        if (source.module) {
            require_apple_msl_custom_kernel_binding(source.module,
                                                    desc.has_residual_add ? "RMSResidual" : "RMS",
                                                    "rms_kernel",
                                                    desc.has_residual_add ? std::vector<size_t>{0, 1, 2}
                                                                          : std::vector<size_t>{0, 1});
        }
        return true;
    }

    if (auto rope = std::dynamic_pointer_cast<const ov::op::internal::RoPE>(node)) {
        const auto& cfg = rope->get_config();
        OPENVINO_ASSERT(!cfg.input_trans0213 && !cfg.output_trans0213,
                        "GFX Metal RoPE: transposed layouts are not supported yet");
        OPENVINO_ASSERT(!cfg.is_chatglm && !cfg.is_qwen,
                        "GFX Metal RoPE: ChatGLM/Qwen-special layouts are not supported yet");
        OPENVINO_ASSERT(cfg.slice_start == 0 && cfg.slice_stop == 0,
                        "GFX Metal RoPE: sliced input layout is not supported yet");
        OPENVINO_ASSERT(rope->get_input_size() >= 3 && rope->get_input_size() <= 4,
                        "GFX Metal RoPE: expected data, cos, sin and optional position inputs");
        OPENVINO_ASSERT(cfg.gather_position_arg_id == 0 || cfg.gather_position_arg_id == 3,
                        "GFX Metal RoPE: position gather must use input 3");
        const auto data_shape = static_shape_or_placeholder(rope->get_input_partial_shape(0));
        const auto cos_shape = static_shape_or_placeholder(rope->get_input_partial_shape(1));
        OPENVINO_ASSERT(data_shape.size() == 4 || data_shape.size() == 3,
                        "GFX Metal RoPE: expected rank-3 or rank-4 data tensor");
        OPENVINO_ASSERT(!data_shape.empty() && data_shape.back() > 0,
                        "GFX Metal RoPE: head size must be static");
        OPENVINO_ASSERT(cos_shape.size() >= 2 && cos_shape.size() <= 4,
                        "GFX Metal RoPE: expected rank-2/3/4 cos/sin tensors");
        RopeCodegenDesc desc{};
        desc.element_type = rope->get_output_element_type(0);
        desc.input_type = rope->get_input_element_type(0);
        desc.cos_type = rope->get_input_element_type(1);
        desc.sin_type = rope->get_input_element_type(2);
        desc.output_type = rope->get_output_element_type(0);
        desc.position_type = rope->get_input_size() > 3 ? rope->get_input_element_type(3) : ov::element::dynamic;
        desc.rank = static_cast<uint32_t>(data_shape.size());
        desc.batch = static_cast<uint32_t>(data_shape.size() == 4 ? data_shape[0] : 1);
        desc.heads = static_cast<uint32_t>(data_shape.size() == 4 ? data_shape[1] : data_shape[1]);
        desc.head_size = static_cast<uint32_t>(data_shape.back());
        desc.rotary_dims = static_cast<uint32_t>(cfg.rotary_ndims ? cfg.rotary_ndims : desc.head_size);
        desc.cos_sin_dims = static_cast<uint32_t>(cfg.cos_sin_ndims ? cfg.cos_sin_ndims : desc.rotary_dims);
        desc.cos_rank = static_cast<uint32_t>(cos_shape.size());
        const auto cos_pshape = rope->get_input_partial_shape(1);
        auto mark_dynamic = [&](size_t logical_dim, size_t source_dim) {
            if (source_dim < static_cast<size_t>(cos_pshape.rank().get_length()) && cos_pshape[source_dim].is_dynamic()) {
                desc.cos_dynamic_mask |= (1u << logical_dim);
            }
        };
        if (cos_shape.size() == 2) {
            desc.cos_dims = {{1, 1, static_cast<uint32_t>(cos_shape[0]), static_cast<uint32_t>(cos_shape[1])}};
            mark_dynamic(2, 0);
            mark_dynamic(3, 1);
        } else if (cos_shape.size() == 3) {
            desc.cos_dims = {{1,
                              static_cast<uint32_t>(cos_shape[0]),
                              static_cast<uint32_t>(cos_shape[1]),
                              static_cast<uint32_t>(cos_shape[2])}};
            mark_dynamic(1, 0);
            mark_dynamic(2, 1);
            mark_dynamic(3, 2);
        } else {
            desc.cos_dims = {{static_cast<uint32_t>(cos_shape[0]),
                              static_cast<uint32_t>(cos_shape[1]),
                              static_cast<uint32_t>(cos_shape[2]),
                              static_cast<uint32_t>(cos_shape[3])}};
            mark_dynamic(0, 0);
            mark_dynamic(1, 1);
            mark_dynamic(2, 2);
            mark_dynamic(3, 3);
        }
        desc.is_interleaved = cfg.is_interleaved;
        desc.input_trans0213 = cfg.input_trans0213;
        desc.output_trans0213 = cfg.output_trans0213;
        desc.has_position = cfg.gather_position_arg_id == 3 && rope->get_input_size() > 3;
        set_desc(desc, "rope_kernel");
        if (source.module) {
            require_apple_msl_custom_kernel_binding(source.module,
                                                    desc.has_position ? "RoPEWithPosition" : "RoPE",
                                                    "rope_kernel",
                                                    desc.has_position ? std::vector<size_t>{0, 1, 2, 3}
                                                                      : std::vector<size_t>{0, 1, 2});
        }
        return true;
    }

    return false;
}

static bool configure_apple_metal_softmax_kernel_source(KernelSource& source,
                                                        const std::shared_ptr<const ov::Node>& node,
                                                        const std::optional<ov::Shape>& runtime_input_shape) {
    if (!node) {
        return false;
    }

    int64_t axis = -1;
    bool log_softmax = false;
    if (auto sm1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
        axis = sm1->get_axis();
    } else if (auto sm8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
        axis = sm8->get_axis();
    } else if (auto ls = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(node)) {
        axis = ls->get_axis();
        log_softmax = true;
    } else {
        return false;
    }

    const ov::Shape input_shape =
        runtime_input_shape && !runtime_input_shape->empty()
            ? *runtime_input_shape
            : node->get_input_shape(0);
    OPENVINO_ASSERT(!input_shape.empty(), "GFX Metal Softmax: input tensor shape is unknown");

    SoftmaxCodegenDesc desc{};
    desc.element_type = node->get_output_element_type(0);
    const auto dims = compute_softmax_dims(input_shape, axis, "GFX Metal");
    desc.rows = static_cast<int64_t>(dims.rows);
    desc.cols = static_cast<int64_t>(dims.axis_len);
    desc.inner = static_cast<int64_t>(dims.inner);
    desc.log_softmax = log_softmax;
    source.entry_point = "softmax_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
        return generate_msl_from_mlir(module, desc);
    };
    return true;
}

static bool configure_apple_metal_pool2d_kernel_source(KernelSource& source,
                                                       const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }

    Pool2DCodegenDesc desc{};
    if (auto pool = std::dynamic_pointer_cast<const ov::op::util::MaxPoolBase>(node)) {
        const auto in = pool->get_input_shape(0);
        const auto out = pool->get_output_shape(0);
        ov::Strides dilations(pool->get_kernel().size(), 1);
        if (auto p = std::dynamic_pointer_cast<const ov::op::v8::MaxPool>(node)) {
            dilations = p->get_dilations();
        } else if (auto p = std::dynamic_pointer_cast<const ov::op::v14::MaxPool>(node)) {
            dilations = p->get_dilations();
        }
        desc.element_type = pool->get_output_element_type(0);
        desc.N = static_cast<uint32_t>(in.at(0));
        desc.C = static_cast<uint32_t>(in.at(1));
        desc.H = static_cast<uint32_t>(in.at(2));
        desc.W = static_cast<uint32_t>(in.at(3));
        desc.kH = static_cast<uint32_t>(pool->get_kernel().at(0));
        desc.kW = static_cast<uint32_t>(pool->get_kernel().at(1));
        desc.strideH = static_cast<uint32_t>(pool->get_strides().at(0));
        desc.strideW = static_cast<uint32_t>(pool->get_strides().at(1));
        desc.dilationH = static_cast<uint32_t>(dilations.at(0));
        desc.dilationW = static_cast<uint32_t>(dilations.at(1));
        desc.padTop = static_cast<uint32_t>(pool->get_pads_begin().at(0));
        desc.padLeft = static_cast<uint32_t>(pool->get_pads_begin().at(1));
        desc.padBottom = static_cast<uint32_t>(pool->get_pads_end().at(0));
        desc.padRight = static_cast<uint32_t>(pool->get_pads_end().at(1));
        desc.outH = static_cast<uint32_t>(out.at(2));
        desc.outW = static_cast<uint32_t>(out.at(3));
        desc.is_avg = false;
        desc.exclude_pad = true;
    } else if (auto pool = std::dynamic_pointer_cast<const ov::op::v1::AvgPool>(node)) {
        const auto in = pool->get_input_shape(0);
        const auto out = pool->get_output_shape(0);
        desc.element_type = pool->get_output_element_type(0);
        desc.N = static_cast<uint32_t>(in.at(0));
        desc.C = static_cast<uint32_t>(in.at(1));
        desc.H = static_cast<uint32_t>(in.at(2));
        desc.W = static_cast<uint32_t>(in.at(3));
        desc.kH = static_cast<uint32_t>(pool->get_kernel().at(0));
        desc.kW = static_cast<uint32_t>(pool->get_kernel().at(1));
        desc.strideH = static_cast<uint32_t>(pool->get_strides().at(0));
        desc.strideW = static_cast<uint32_t>(pool->get_strides().at(1));
        desc.padTop = static_cast<uint32_t>(pool->get_pads_begin().at(0));
        desc.padLeft = static_cast<uint32_t>(pool->get_pads_begin().at(1));
        desc.padBottom = static_cast<uint32_t>(pool->get_pads_end().at(0));
        desc.padRight = static_cast<uint32_t>(pool->get_pads_end().at(1));
        desc.outH = static_cast<uint32_t>(out.at(2));
        desc.outW = static_cast<uint32_t>(out.at(3));
        desc.is_avg = true;
        desc.exclude_pad = pool->get_exclude_pad();
    } else {
        return false;
    }

    source.entry_point = "pool2d_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
        return generate_msl_from_mlir(module, desc);
    };
    if (source.module) {
        require_apple_msl_custom_kernel_binding(source.module,
                                                node->get_type_name(),
                                                "pool2d_kernel",
                                                {0});
    }
    return true;
}

static bool configure_apple_metal_unary_kernel_source(KernelSource& source,
                                                      const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }

    const auto activation = unary_activation_kind_from_node(*node);
    if (!activation) {
        return false;
    }

    UnaryCodegenDesc desc{};
    desc.element_type = node->get_output_element_type(0);
    desc.activation = *activation;
    desc.alpha = 0.0f;
    if (auto elu = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
        desc.alpha = static_cast<float>(elu->get_alpha());
    }
    if (auto clamp = ov::as_type_ptr<const ov::op::v0::Clamp>(node)) {
        desc.clamp_min = clamp->get_min();
        desc.clamp_max = clamp->get_max();
    }

    source.entry_point = "unary_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
        return generate_msl_from_mlir(module, desc);
    };
    const auto out_shape = output_shape_for_codegen(source.module, node);
    const int32_t num_elements = static_cast<int32_t>(ov::shape_size(out_shape));
    if (source.module) {
        require_apple_msl_custom_kernel_binding(source.module,
                                                node->get_type_name(),
                                                "unary_kernel",
                                                {0},
                                                {num_elements});
    }
    return true;
}

static bool configure_apple_metal_elementwise_kernel_source(KernelSource& source,
                                                            const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }

    const std::string type = node->get_type_name();
    if (type == "Select") {
        const auto out_shape = output_shape_for_codegen(source.module, node);
        source.entry_point = "select_kernel";
        source.msl_generator = [element_type = node->get_output_element_type(0)](mlir::ModuleOp module) {
            return generate_msl_for_select(module, element_type);
        };
        if (source.module) {
            const std::vector<int32_t> scalars{static_cast<int32_t>(ov::shape_size(out_shape)),
                                               static_cast<int32_t>(out_shape.empty() ? 1 : out_shape.size())};
            require_apple_msl_custom_kernel_binding(source.module,
                                                    "Select",
                                                    "select_kernel",
                                                    {0, 1, 2},
                                                    scalars);
        }
        return true;
    }

    auto kind = eltwise_kind_from_node(*node);
    if (!kind) {
        return false;
    }

    EltwiseCodegenDesc desc{};
    desc.element_type = node->get_output_element_type(0);
    desc.input0_type = node->get_input_element_type(0);
    desc.input1_type = node->get_input_element_type(1);
    desc.output_type = node->get_output_element_type(0);
    desc.eltwise_kind = *kind;
    if (auto input_activation = activation_kind_from_module_attr(source.module, "gfx.input_activation_kind")) {
        desc.has_input_activation = true;
        desc.input_activation = *input_activation;
        if (auto input_attr = source.module->getAttrOfType<mlir::IntegerAttr>("gfx.input_activation_input")) {
            desc.input_activation_index = static_cast<uint32_t>(std::max<int64_t>(input_attr.getInt(), 0));
        }
        if (auto alpha_attr = source.module->getAttrOfType<mlir::FloatAttr>("gfx.input_activation_alpha")) {
            desc.input_activation_alpha = static_cast<float>(alpha_attr.getValueAsDouble());
        }
    }

    const bool dynamic_shape = !node->get_output_partial_shape(0).is_static() ||
                               !node->get_input_partial_shape(0).is_static() ||
                               !node->get_input_partial_shape(1).is_static();
    const auto out_shape = output_shape_for_codegen(source.module, node);
    desc.out_shape = to_i64_shape(out_shape);
    desc.num_elements = static_cast<uint32_t>(ov::shape_size(out_shape));
    const auto input0_shape = shape_from_entry_argument_or_partial(source.module, 0, node->get_input_partial_shape(0));
    const auto input1_shape = shape_from_entry_argument_or_partial(source.module, 1, node->get_input_partial_shape(1));
    const auto perm0 = read_absorbed_input_permutation(source.module, 0);
    const auto perm1 = read_absorbed_input_permutation(source.module, 1);
    desc.is_broadcast = dynamic_shape || !perm0.empty() || !perm1.empty() || (input0_shape != input1_shape) ||
                        (input0_shape != out_shape) || (input1_shape != out_shape);
    if (!perm0.empty()) {
        auto strides = compute_permuted_broadcast_element_strides(input0_shape,
                                                                  static_shape_or_placeholder(node->get_input_partial_shape(0)),
                                                                  perm0,
                                                                  out_shape,
                                                                  "GFX Metal");
        desc.stride0.assign(strides.begin(), strides.end());
    } else {
        fill_broadcast_strides(out_shape, input0_shape, desc.stride0);
    }
    if (!perm1.empty()) {
        auto strides = compute_permuted_broadcast_element_strides(input1_shape,
                                                                  static_shape_or_placeholder(node->get_input_partial_shape(1)),
                                                                  perm1,
                                                                  out_shape,
                                                                  "GFX Metal");
        desc.stride1.assign(strides.begin(), strides.end());
    } else {
        fill_broadcast_strides(out_shape, input1_shape, desc.stride1);
    }

    source.entry_point = "eltwise_kernel";
    source.msl_generator = [desc](mlir::ModuleOp module) mutable {
        return generate_msl_from_mlir(module, desc);
    };
    if (source.module) {
        std::vector<int32_t> scalars = {static_cast<int32_t>(desc.num_elements),
                                        static_cast<int32_t>(out_shape.size())};
        require_apple_msl_custom_kernel_binding(source.module,
                                                type,
                                                "eltwise_kernel",
                                                {0, 1},
                                                scalars);
    }
    return true;
}

static bool configure_apple_metal_structural_kernel_source(KernelSource& source,
                                                           const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }

    const auto reduce_kind = reduce_kind_from_node(*node);
    if (reduce_kind) {
        ReduceCodegenDesc desc{};
        desc.element_type = node->get_output_element_type(0);
        desc.kind = *reduce_kind;
        source.entry_point = "reduce_kernel";
        source.signature.output_arg_count = 1;
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        if (source.module) {
            require_apple_msl_custom_kernel_binding(source.module,
                                                    node->get_type_name(),
                                                    "reduce_kernel",
                                                    {0});
        }
        return true;
    }

    if (auto concat = std::dynamic_pointer_cast<const ov::op::v0::Concat>(node)) {
        ConcatCodegenDesc desc{};
        desc.element_type = concat->get_output_element_type(0);
        const auto out_pshape = concat->get_output_partial_shape(0);
        OPENVINO_ASSERT(out_pshape.rank().is_static(), "GFX Metal Concat: output rank must be static");
        const size_t rank = static_cast<size_t>(out_pshape.rank().get_length());
        OPENVINO_ASSERT(rank > 0, "GFX Metal Concat: output rank must be positive");
        const size_t axis = normalize_axis(concat->get_axis(), rank, "GFX Metal Concat");
        (void)axis;
        desc.inner = 1;
        desc.outer = 1;
        desc.axis_total = 1;
        source.entry_point = "concat_kernel";
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        return true;
    }

    if (auto split = std::dynamic_pointer_cast<const ov::op::v1::Split>(node)) {
        SplitCodegenDesc desc{};
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(split->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "Split axis must be constant");
        desc.axis = axis_const->cast_vector<int64_t>().at(0);
        const auto in_shape = split->get_input_shape(0);
        const auto out_shape = split->get_output_shape(0);
        desc.input_shape = to_i64_shape(in_shape);
        desc.source_input_shape = to_i64_shape(shape_from_entry_argument(source.module, 0, in_shape));
        desc.input_permutation = read_absorbed_input_permutation(source.module, 0);
        const size_t axis = static_cast<size_t>(desc.axis < 0 ? desc.axis + in_shape.size() : desc.axis);
        desc.split_sizes.assign(split->get_output_size(), out_shape[axis]);
        uint64_t inner = 1;
        uint64_t outer = 1;
        for (size_t i = axis + 1; i < in_shape.size(); ++i) inner *= in_shape[i];
        for (size_t i = 0; i < axis; ++i) outer *= in_shape[i];
        desc.inner = inner;
        desc.outer = outer;
        source.entry_point = "split_kernel";
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        return true;
    }

    if (auto split = std::dynamic_pointer_cast<const ov::op::v1::VariadicSplit>(node)) {
        SplitCodegenDesc desc{};
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(split->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "VariadicSplit axis must be constant");
        desc.axis = axis_const->cast_vector<int64_t>().at(0);
        const auto in_shape = split->get_input_shape(0);
        desc.input_shape = to_i64_shape(in_shape);
        desc.source_input_shape = to_i64_shape(shape_from_entry_argument(source.module, 0, in_shape));
        desc.input_permutation = read_absorbed_input_permutation(source.module, 0);
        auto lengths_const = ov::as_type_ptr<const ov::op::v0::Constant>(split->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(lengths_const, "VariadicSplit lengths must be constant");
        auto lengths = lengths_const->cast_vector<int64_t>();
        desc.split_sizes.assign(lengths.begin(), lengths.end());
        const size_t axis = static_cast<size_t>(desc.axis < 0 ? desc.axis + in_shape.size() : desc.axis);
        uint64_t inner = 1;
        uint64_t outer = 1;
        for (size_t i = axis + 1; i < in_shape.size(); ++i) inner *= in_shape[i];
        for (size_t i = 0; i < axis; ++i) outer *= in_shape[i];
        desc.inner = inner;
        desc.outer = outer;
        source.entry_point = "split_kernel";
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        return true;
    }

    if (auto d2s = std::dynamic_pointer_cast<const ov::op::v0::DepthToSpace>(node)) {
        DepthToSpaceCodegenDesc desc{};
        const auto in = d2s->get_input_shape(0);
        const auto out = d2s->get_output_shape(0);
        desc.element_type = d2s->get_output_element_type(0);
        desc.N = in[0];
        desc.C = in[1];
        desc.H = in[2];
        desc.W = in[3];
        desc.block = static_cast<uint32_t>(d2s->get_block_size());
        desc.C_out = out[1];
        desc.H_out = out[2];
        desc.W_out = out[3];
        desc.mode = d2s->get_mode() == ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST ? 0 : 1;
        desc.total = static_cast<uint32_t>(ov::shape_size(out));
        source.entry_point = "depth_to_space_kernel";
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        return true;
    }

    if (auto s2d = std::dynamic_pointer_cast<const ov::op::v0::SpaceToDepth>(node)) {
        SpaceToDepthCodegenDesc desc{};
        const auto in = s2d->get_input_shape(0);
        const auto out = s2d->get_output_shape(0);
        desc.element_type = s2d->get_output_element_type(0);
        desc.N = in[0];
        desc.C = in[1];
        desc.H = in[2];
        desc.W = in[3];
        desc.block = static_cast<uint32_t>(s2d->get_block_size());
        desc.C_out = out[1];
        desc.H_out = out[2];
        desc.W_out = out[3];
        desc.mode = s2d->get_mode() == ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST ? 0 : 1;
        desc.total = static_cast<uint32_t>(ov::shape_size(out));
        source.entry_point = "space_to_depth_kernel";
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        return true;
    }

    if (auto transpose = std::dynamic_pointer_cast<const ov::op::v1::Transpose>(node)) {
        TransposeCodegenDesc desc{};
        desc.element_type = transpose->get_output_element_type(0);
        auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(perm_const, "Transpose perm must be constant");
        auto perm = perm_const->cast_vector<int64_t>();
        for (auto value : perm) {
            desc.perm.push_back(static_cast<uint32_t>(value));
        }
        source.entry_point = "transpose_kernel";
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        if (source.module) {
            require_apple_msl_custom_kernel_binding(source.module,
                                                    "Transpose",
                                                    "transpose_kernel",
                                                    {0});
        }
        return true;
    }

    if (auto convert = std::dynamic_pointer_cast<const ov::op::v0::Convert>(node)) {
        ConvertCodegenDesc desc{};
        desc.src_type = convert->get_input_element_type(0);
        desc.dst_type = convert->get_output_element_type(0);
        desc.element_type = desc.dst_type == ov::element::dynamic ? ov::element::f32 : desc.dst_type;
        source.entry_point = "convert_kernel";
        source.signature.output_arg_count = 1;
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        if (source.module) {
            require_apple_msl_custom_kernel_binding(source.module,
                                                    "Convert",
                                                    "convert_kernel",
                                                    {0});
        }
        return true;
    }

    if (std::dynamic_pointer_cast<const ov::op::v0::ShapeOf>(node) ||
        std::dynamic_pointer_cast<const ov::op::v3::ShapeOf>(node)) {
        ShapeOfCodegenDesc desc{};
        const auto input_pshape = node->get_input_partial_shape(0);
        OPENVINO_ASSERT(input_pshape.rank().is_static(), "ShapeOf: input rank must be static");
        desc.rank = static_cast<uint32_t>(input_pshape.rank().get_length());
        desc.element_type = node->get_output_element_type(0);
        source.entry_point = "shapeof_kernel";
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        return true;
    }

    if (auto pad = std::dynamic_pointer_cast<const ov::op::v1::Pad>(node)) {
        PadCodegenDesc desc{};
        desc.element_type = pad->get_output_element_type(0);
        if (auto c = ov::as_type_ptr<const ov::op::v0::Constant>(pad->input_value(3).get_node_shared_ptr())) {
            if (c->get_element_type().is_real()) {
                desc.pad_value = c->cast_vector<double>()[0];
            } else if (c->get_element_type().is_integral_number()) {
                desc.pad_value = c->cast_vector<int64_t>()[0];
            }
        }
        source.entry_point = "pad_kernel";
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        return true;
    }

    if (std::dynamic_pointer_cast<const ov::op::v0::Tile>(node)) {
        TileCodegenDesc desc{};
        desc.element_type = node->get_output_element_type(0);
        source.entry_point = "tile_kernel";
        source.signature.output_arg_count = 1;
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        return true;
    }

    if (std::dynamic_pointer_cast<const ov::op::v1::Broadcast>(node) ||
        std::dynamic_pointer_cast<const ov::op::v3::Broadcast>(node)) {
        BroadcastCodegenDesc desc{};
        desc.element_type = node->get_output_element_type(0);
        source.entry_point = "broadcast_kernel";
        source.signature.output_arg_count = 1;
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        if (source.module) {
            require_apple_msl_custom_kernel_binding(source.module,
                                                    "Broadcast",
                                                    "broadcast_kernel",
                                                    {0});
        }
        return true;
    }

    if (std::dynamic_pointer_cast<const ov::op::v4::Range>(node)) {
        RangeCodegenDesc desc{};
        desc.element_type = node->get_output_element_type(0);
        desc.output_type = node->get_output_element_type(0);
        desc.start_type = node->get_input_element_type(0);
        desc.stop_type = node->get_input_element_type(1);
        desc.step_type = node->get_input_element_type(2);
        source.entry_point = "range_kernel";
        source.signature.output_arg_count = 1;
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        if (source.module) {
            require_apple_msl_custom_kernel_binding(source.module,
                                                    "Range",
                                                    "range_kernel",
                                                    {0, 1, 2});
        }
        return true;
    }

    if (auto reverse = std::dynamic_pointer_cast<const ov::op::v1::Reverse>(node)) {
        ReverseCodegenDesc desc{};
        const auto in = reverse->get_input_shape(0);
        desc.element_type = reverse->get_output_element_type(0);
        desc.rank = static_cast<uint32_t>(in.size());
        desc.total = static_cast<uint32_t>(ov::shape_size(in));
        uint32_t stride = 1;
        for (int i = static_cast<int>(in.size()) - 1; i >= 0; --i) {
            desc.strides[static_cast<size_t>(i)] = stride;
            desc.dims[static_cast<size_t>(i)] = static_cast<uint32_t>(in[static_cast<size_t>(i)]);
            stride *= static_cast<uint32_t>(in[static_cast<size_t>(i)]);
        }
        auto axes_const = ov::as_type_ptr<const ov::op::v0::Constant>(reverse->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axes_const, "Reverse axes must be constant");
        for (auto axis_value : axes_const->cast_vector<int64_t>()) {
            uint32_t axis = static_cast<uint32_t>(axis_value < 0 ? axis_value + in.size() : axis_value);
            desc.axes_mask |= (1u << axis);
        }
        source.entry_point = "reverse_kernel";
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        return true;
    }

    if (auto topk = std::dynamic_pointer_cast<const ov::op::util::TopKBase>(node)) {
        TopKCodegenDesc desc{};
        const auto in = topk->get_input_shape(0);
        const int64_t axis_i64 = normalize_axis(topk->get_axis(), in.size(), "TopK");
        const size_t axis = static_cast<size_t>(axis_i64);
        desc.axis_len = static_cast<uint32_t>(in[axis]);
        desc.k = static_cast<uint32_t>(topk->get_k());
        uint32_t outer = 1;
        uint32_t inner = 1;
        for (size_t i = 0; i < axis; ++i) outer *= static_cast<uint32_t>(in[i]);
        for (size_t i = axis + 1; i < in.size(); ++i) inner *= static_cast<uint32_t>(in[i]);
        desc.outer = outer;
        desc.inner = inner;
        desc.mode_max = topk->get_mode() == ov::op::TopKMode::MAX;
        switch (topk->get_sort_type()) {
            case ov::op::TopKSortType::SORT_INDICES:
                desc.sort_type = TopKSortType::SortIndices;
                break;
            case ov::op::TopKSortType::NONE:
                desc.sort_type = TopKSortType::None;
                break;
            case ov::op::TopKSortType::SORT_VALUES:
            default:
                desc.sort_type = TopKSortType::SortValues;
                break;
        }
        desc.element_type = topk->get_output_element_type(0);
        desc.index_type = topk->get_output_element_type(1);
        source.entry_point = "topk_kernel";
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
        return true;
    }

    return false;
}

static bool configure_apple_metal_data_movement_kernel_source(KernelSource& source,
                                                              const std::shared_ptr<const ov::Node>& node) {
    if (!node) {
        return false;
    }

    auto set_desc = [&](auto&& desc, const char* entry) {
        source.entry_point = entry;
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
    };

    if (auto interp = std::dynamic_pointer_cast<const ov::op::v0::Interpolate>(node)) {
        InterpolateCodegenDesc desc{};
        const auto in = interp->get_input_shape(0);
        const auto out = interp->get_output_shape(0);
        desc.element_type = interp->get_output_element_type(0);
        desc.N = in[0];
        desc.C = in[1];
        desc.H_in = in[2];
        desc.W_in = in[3];
        desc.H_out = out[2];
        desc.W_out = out[3];
        desc.scale_h = desc.H_out ? static_cast<float>(desc.H_in) / static_cast<float>(desc.H_out) : 1.f;
        desc.scale_w = desc.W_out ? static_cast<float>(desc.W_in) / static_cast<float>(desc.W_out) : 1.f;
        desc.align_corners = interp->get_attrs().align_corners;
        desc.nearest = ov::util::to_lower(interp->get_attrs().mode) == "nearest";
        desc.use_half_pixel = !desc.align_corners;
        desc.nearest_mode = 0;
        set_desc(desc, "interpolate_kernel");
        return true;
    }

    if (auto interp = std::dynamic_pointer_cast<const ov::op::v4::Interpolate>(node)) {
        using Base = ov::op::util::InterpolateBase;
        InterpolateCodegenDesc desc{};
        const auto in = interp->get_input_shape(0);
        const auto out = interp->get_output_shape(0);
        desc.element_type = interp->get_output_element_type(0);
        desc.N = in[0];
        desc.C = in[1];
        desc.H_in = in[2];
        desc.W_in = in[3];
        desc.H_out = out[2];
        desc.W_out = out[3];
        desc.scale_h = desc.H_out ? static_cast<float>(desc.H_in) / static_cast<float>(desc.H_out) : 1.f;
        desc.scale_w = desc.W_out ? static_cast<float>(desc.W_in) / static_cast<float>(desc.W_out) : 1.f;
        desc.align_corners = interp->get_attrs().coordinate_transformation_mode ==
                             Base::CoordinateTransformMode::ALIGN_CORNERS;
        desc.nearest = interp->get_attrs().mode == Base::InterpolateMode::NEAREST;
        desc.use_half_pixel = interp->get_attrs().coordinate_transformation_mode ==
                              Base::CoordinateTransformMode::HALF_PIXEL;
        switch (interp->get_attrs().nearest_mode) {
            case Base::NearestMode::FLOOR:
            case Base::NearestMode::ROUND_PREFER_FLOOR:
                desc.nearest_mode = 1;
                break;
            case Base::NearestMode::CEIL:
            case Base::NearestMode::ROUND_PREFER_CEIL:
                desc.nearest_mode = 2;
                break;
            case Base::NearestMode::SIMPLE:
            default:
                desc.nearest_mode = 0;
                break;
        }
        set_desc(desc, "interpolate_kernel");
        return true;
    }

    if (auto interp = std::dynamic_pointer_cast<const ov::op::v11::Interpolate>(node)) {
        using Base = ov::op::util::InterpolateBase;
        InterpolateCodegenDesc desc{};
        const auto in = interp->get_input_shape(0);
        const auto out = interp->get_output_shape(0);
        desc.element_type = interp->get_output_element_type(0);
        desc.N = in[0];
        desc.C = in[1];
        desc.H_in = in[2];
        desc.W_in = in[3];
        desc.H_out = out[2];
        desc.W_out = out[3];
        desc.scale_h = desc.H_out ? static_cast<float>(desc.H_in) / static_cast<float>(desc.H_out) : 1.f;
        desc.scale_w = desc.W_out ? static_cast<float>(desc.W_in) / static_cast<float>(desc.W_out) : 1.f;
        desc.align_corners = interp->get_attrs().coordinate_transformation_mode ==
                             Base::CoordinateTransformMode::ALIGN_CORNERS;
        desc.nearest = interp->get_attrs().mode == Base::InterpolateMode::NEAREST;
        desc.use_half_pixel = interp->get_attrs().coordinate_transformation_mode ==
                              Base::CoordinateTransformMode::HALF_PIXEL;
        switch (interp->get_attrs().nearest_mode) {
            case Base::NearestMode::FLOOR:
            case Base::NearestMode::ROUND_PREFER_FLOOR:
                desc.nearest_mode = 1;
                break;
            case Base::NearestMode::CEIL:
            case Base::NearestMode::ROUND_PREFER_CEIL:
                desc.nearest_mode = 2;
                break;
            case Base::NearestMode::SIMPLE:
            default:
                desc.nearest_mode = 0;
                break;
        }
        set_desc(desc, "interpolate_kernel");
        return true;
    }

    if (std::dynamic_pointer_cast<const ov::op::v8::Slice>(node) ||
        std::dynamic_pointer_cast<const ov::op::v1::StridedSlice>(node)) {
        ConvertCodegenDesc desc{};
        desc.element_type = node->get_output_element_type(0);
        desc.dst_type = desc.element_type;
        source.entry_point = "slice_kernel";
        source.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_for_slice_generic(desc, module);
        };
        if (source.module) {
            require_apple_msl_custom_kernel_binding(source.module,
                                                    node->get_type_name(),
                                                    "slice_kernel",
                                                    {0});
        }
        return true;
    }

    if (auto gather = std::dynamic_pointer_cast<const ov::op::util::GatherBase>(node)) {
        GatherCodegenDesc desc{};
        if (auto g7 = std::dynamic_pointer_cast<const ov::op::v7::Gather>(node)) {
            OPENVINO_ASSERT(g7->get_batch_dims() == 0, "GFX Metal Gather: batch_dims not supported");
        } else if (auto g8 = std::dynamic_pointer_cast<const ov::op::v8::Gather>(node)) {
            OPENVINO_ASSERT(g8->get_batch_dims() == 0, "GFX Metal Gather: batch_dims not supported");
        }
        desc.index_type = gather->get_input_element_type(1);
        desc.element_type = gather->get_output_element_type(0);
        const auto data_pshape = gather->get_input_partial_shape(0);
        OPENVINO_ASSERT(data_pshape.rank().is_static(), "GFX Metal Gather: data rank must be static");
        const size_t rank = static_cast<size_t>(data_pshape.rank().get_length());
        OPENVINO_ASSERT(rank > 0, "GFX Metal Gather: data rank must be positive");
        const size_t axis = normalize_axis(gather->get_axis(), rank, "GFX Metal Gather");
        (void)axis;
        desc.outer = 1;
        desc.inner = 1;
        desc.axis_dim = 1;
        desc.indices_count = 1;
        set_desc(desc, "gather_kernel");
        return true;
    }

    if (auto gather_nd = std::dynamic_pointer_cast<const ov::op::v5::GatherND>(node)) {
        GatherNDCodegenDesc desc{};
        const auto data = gather_nd->get_input_shape(0);
        const auto indices = gather_nd->get_input_shape(1);
        desc.index_type = gather_nd->get_input_element_type(1);
        desc.k = static_cast<uint32_t>(indices.back());
        desc.num_indices = static_cast<uint32_t>(ov::shape_size(indices) / desc.k);
        desc.element_type = gather_nd->get_output_element_type(0);
        uint32_t stride = 1;
        const size_t rank = data.size();
        for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
            desc.dims[static_cast<size_t>(i)] = static_cast<uint32_t>(data[static_cast<size_t>(i)]);
            desc.strides[static_cast<size_t>(i)] = stride;
            stride *= desc.dims[static_cast<size_t>(i)];
        }
        desc.inner = desc.strides[desc.k];
        desc.total = static_cast<uint32_t>(ov::shape_size(data));
        set_desc(desc, "gathernd_kernel");
        return true;
    }

    if (auto gather_elements = std::dynamic_pointer_cast<const ov::op::v6::GatherElements>(node)) {
        GatherElementsCodegenDesc desc{};
        const auto data = gather_elements->get_input_shape(0);
        const auto out = gather_elements->get_output_shape(0);
        desc.index_type = gather_elements->get_input_element_type(1);
        desc.rank = static_cast<uint32_t>(out.size());
        desc.axis = static_cast<uint32_t>(gather_elements->get_axis());
        desc.total = static_cast<uint32_t>(ov::shape_size(out));
        auto data_strides = make_strides(data);
        auto out_strides = make_strides(out);
        for (size_t i = 0; i < out.size() && i < desc.kMaxDims; ++i) {
            desc.out_dims[i] = static_cast<uint32_t>(out[i]);
            desc.out_strides[i] = static_cast<uint32_t>(out_strides[i]);
            desc.data_dims[i] = static_cast<uint32_t>(data[i]);
            desc.data_strides[i] = static_cast<uint32_t>(data_strides[i]);
        }
        set_desc(desc, "gather_elements_kernel");
        source.signature.output_arg_count = 1;
        return true;
    }

    if (auto scatter = std::dynamic_pointer_cast<const ov::op::v3::ScatterUpdate>(node)) {
        ScatterUpdateCodegenDesc desc{};
        desc.element_type = scatter->get_output_element_type(0);
        desc.index_type = scatter->get_input_element_type(1);
        set_desc(desc, "scatter_update_kernel");
        if (source.module) {
            require_apple_msl_custom_kernel_binding(source.module,
                                                    "ScatterUpdate",
                                                    "scatter_update_kernel",
                                                    {0, 1, 2});
        }
        return true;
    }

    if (auto scatter = std::dynamic_pointer_cast<const ov::op::v3::ScatterNDUpdate>(node)) {
        ScatterNDUpdateCodegenDesc desc{};
        const auto data = scatter->get_input_shape(0);
        const auto indices = scatter->get_input_shape(1);
        desc.index_type = scatter->get_input_element_type(1);
        desc.k = static_cast<uint32_t>(indices.back());
        uint32_t stride = 1;
        for (int i = static_cast<int>(data.size()) - 1; i >= 0; --i) {
            desc.dims[static_cast<size_t>(i)] = static_cast<uint32_t>(data[static_cast<size_t>(i)]);
            desc.strides[static_cast<size_t>(i)] = stride;
            stride *= desc.dims[static_cast<size_t>(i)];
        }
        desc.inner = desc.strides[desc.k];
        desc.num_indices = static_cast<uint32_t>(ov::shape_size(indices) / desc.k);
        desc.total_updates = static_cast<uint32_t>(ov::shape_size(scatter->get_input_shape(2)));
        desc.total_data = static_cast<uint32_t>(ov::shape_size(data));
        desc.element_type = scatter->get_output_element_type(0);
        set_desc(desc, "scatter_nd_update");
        return true;
    }

    if (auto scatter = std::dynamic_pointer_cast<const ov::op::v3::ScatterElementsUpdate>(node)) {
        ScatterElementsUpdateCodegenDesc desc{};
        const auto data = scatter->get_input_shape(0);
        const auto indices = scatter->get_input_shape(1);
        desc.index_type = scatter->get_input_element_type(1);
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(scatter->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "ScatterElementsUpdate axis must be constant");
        desc.axis = static_cast<uint32_t>(axis_const->cast_vector<int64_t>()[0]);
        desc.rank = static_cast<uint32_t>(data.size());
        desc.total_updates = static_cast<uint32_t>(ov::shape_size(indices));
        desc.total_data = static_cast<uint32_t>(ov::shape_size(data));
        auto data_strides = make_strides(data);
        auto update_strides = make_strides(indices);
        for (size_t i = 0; i < data.size() && i < desc.kMaxDims; ++i) {
            desc.data_dims[i] = static_cast<uint32_t>(data[i]);
            desc.data_strides[i] = static_cast<uint32_t>(data_strides[i]);
        }
        for (size_t i = 0; i < indices.size() && i < desc.kMaxDims; ++i) {
            desc.update_dims[i] = static_cast<uint32_t>(indices[i]);
            desc.update_strides[i] = static_cast<uint32_t>(update_strides[i]);
        }
        desc.element_type = scatter->get_output_element_type(0);
        set_desc(desc, "scatter_elements_update");
        return true;
    }

    return false;
}

GfxMpsrtKernelSourcePlan configure_msl_kernel_source_plan_for_spec(KernelSource source,
                                                                   const KernelSpec& spec,
                                                                   const GpuBufferManager* buffer_manager,
                                                                   std::string_view entry_point) {
    if (source.entry_point.empty()) {
        source.entry_point = std::string(entry_point);
    }
    return configure_msl_kernel_source_plan_for_node(std::move(source),
                                                     spec.node(),
                                                     buffer_manager,
                                                     spec.type(),
                                                     /*has_bias=*/false,
                                                     /*has_activation=*/false,
                                                     /*has_batchnorm=*/false);
}

void annotate_msl_module_with_stage_plan(mlir::ModuleOp module,
                                         const GfxStageOptimizationPlan& plan,
                                         const std::string& stage_type,
                                         std::string_view kernel_entry_point) {
    GfxAppleStagePipelineOptions options{};
    options.plan = plan;
    options.stage_type = stage_type;
    options.kernel_entry_point = std::string(kernel_entry_point);
    (void)run_gfx_apple_stage_pipeline(module, options);
}

std::string generate_msl_for_matmul_mpsrt_epilogue(const MatMulCodegenDesc& desc) {
    const ov::element::Type output_type = resolve_matmul_buffer_type(desc.output_type, desc.element_type);
    const ov::element::Type bias_type = resolve_matmul_buffer_type(desc.bias_type, output_type);
    const std::string scalar_out = msl_type_from_element(output_type);
    const std::string scalar_bias = msl_type_from_element(bias_type);

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n";
    ss << "using namespace metal;\n";
    ss << "constant uint BATCH = " << desc.batch << ";\n";
    ss << "constant uint M = " << desc.M << ";\n";
    ss << "constant uint N = " << desc.N << ";\n";
    if (desc.has_bias) {
        ss << "constant uint BIAS_B = " << desc.bias_dims[0] << ";\n";
        ss << "constant uint BIAS_M = " << desc.bias_dims[1] << ";\n";
        ss << "constant uint BIAS_N = " << desc.bias_dims[2] << ";\n";
    }
    ss << "kernel void eltwise_fused_buffer(\n";
    ss << "  device const " << scalar_out << "* gemm [[buffer(0)]],\n";
    if (desc.has_bias) {
        ss << "  device const " << scalar_bias << "* bias [[buffer(1)]],\n";
        ss << "  device " << scalar_out << "* output [[buffer(2)]],\n";
    } else {
        ss << "  device " << scalar_out << "* output [[buffer(1)]],\n";
    }
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "    const uint total = BATCH * M * N;\n";
    ss << "    if (gid >= total) return;\n";
    ss << "    const uint batch = gid / (M * N);\n";
    ss << "    const uint idx = gid - batch * M * N;\n";
    ss << "    const uint row = idx / N;\n";
    ss << "    const uint col = idx - row * N;\n";
    ss << "    float x = static_cast<float>(gemm[gid]);\n";
    if (desc.has_bias) {
        ss << "    const uint bb = (BIAS_B == 1) ? 0 : batch;\n";
        ss << "    const uint bm = (BIAS_M == 1) ? 0 : row;\n";
        ss << "    const uint bn = (BIAS_N == 1) ? 0 : col;\n";
        ss << "    const uint bias_idx = (bb * BIAS_M + bm) * BIAS_N + bn;\n";
        ss << "    x += static_cast<float>(bias[bias_idx]);\n";
    }
    if (desc.has_activation) {
        ss << "    x = " << matmul_epilogue_activation_expr(desc.activation) << ";\n";
    }
    ss << "    output[gid] = static_cast<" << scalar_out << ">(x);\n";
    ss << "}\n";
    return ss.str();
}

GfxMslRuntimeBindingPlan make_msl_runtime_binding_plan_from_stage_manifest(
    const GfxKernelStageManifest& manifest,
    std::vector<size_t> tensor_input_indices) {
    GfxMslRuntimeBindingPlan plan{};
    const auto runtime_plan =
        make_kernel_runtime_binding_plan_from_stage_manifest(manifest,
                                                             std::move(tensor_input_indices));
    if (!runtime_plan.valid) {
        return plan;
    }
    plan.stage_manifest = runtime_plan.stage_manifest;
    plan.runtime_binding = runtime_plan.runtime_binding;
    plan.scalar_arg_count = runtime_plan.scalar_arg_count;
    return plan;
}

GfxMslRuntimeBindingPlan make_msl_runtime_binding_plan_for_custom_kernel(
    std::string_view stage_type,
    std::string_view entry_point,
    std::vector<size_t> tensor_input_indices) {
    const auto custom_kernel_plan = make_gfx_custom_kernel_stage_plan(stage_type, entry_point);
    if (!custom_kernel_plan.valid) {
        return {};
    }
    return make_msl_runtime_binding_plan_from_stage_manifest(custom_kernel_plan.stage_manifest,
                                                            std::move(tensor_input_indices));
}

GfxMslRuntimeBindingPlan make_msl_runtime_binding_plan_for_custom_kernel(
    std::string_view stage_type,
    std::string_view entry_point,
    std::vector<size_t> tensor_input_indices,
    std::vector<int32_t> scalar_args) {
    auto plan = make_msl_runtime_binding_plan_for_custom_kernel(stage_type,
                                                                entry_point,
                                                                std::move(tensor_input_indices));
    if (!plan.valid() || plan.scalar_arg_count != scalar_args.size()) {
        return {};
    }
    plan.runtime_binding.scalar_args = std::move(scalar_args);
    plan.stage_manifest.custom_kernel.scalar_args = plan.runtime_binding.scalar_args;
    return plan;
}

GfxMslRuntimeBindingPlan make_msl_runtime_binding_plan_for_direct_io_custom_kernel(
    std::string_view stage_type,
    std::string_view entry_point,
    std::vector<size_t> tensor_input_indices,
    size_t output_count) {
    if (tensor_input_indices.empty() || output_count == 0) {
        return {};
    }

    auto custom_kernel_plan = make_gfx_custom_kernel_stage_plan(stage_type, entry_point);
    if (!custom_kernel_plan.valid || !custom_kernel_plan.stage_manifest.valid) {
        return {};
    }

    auto manifest = custom_kernel_plan.stage_manifest;
    manifest.custom_kernel.external_buffer_abi =
        make_gfx_kernel_direct_io_abi(static_cast<uint32_t>(tensor_input_indices.size()),
                                      static_cast<uint32_t>(output_count));
    return make_msl_runtime_binding_plan_from_stage_manifest(manifest,
                                                            std::move(tensor_input_indices));
}

GfxDirectSplitMslKernelSourcePlan make_direct_split_msl_kernel_source_plan(
    std::string_view stage_type,
    const ov::element::Type& element_type,
    const ov::Shape& input_shape,
    const std::vector<size_t>& split_sizes,
    uint32_t axis_len,
    uint32_t inner_stride,
    mlir::ModuleOp module) {
    GfxDirectSplitMslKernelSourcePlan plan{};
    if (input_shape.empty() || split_sizes.empty() || axis_len == 0 || inner_stride == 0) {
        return plan;
    }

    const auto binding = make_msl_runtime_binding_plan_for_direct_io_custom_kernel(stage_type,
                                                                                  "split_kernel",
                                                                                  {0},
                                                                                  split_sizes.size());
    if (!binding.valid()) {
        return plan;
    }
    mlir::ModuleOp manifest_module;
    if (module) {
        manifest_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(module.getContext()));
        OPENVINO_ASSERT(annotate_msl_module_with_runtime_binding_plan(manifest_module, binding),
                        "GFX MSL: failed to annotate direct Split stage manifest");
    }

    const auto total_elems = ov::shape_size(input_shape);
    const auto scalar = msl_type_from_element(element_type);
    std::ostringstream msl;
    msl << "#include <metal_stdlib>\nusing namespace metal;\n";
    msl << "constant uint OFFSETS[" << (split_sizes.size() + 1) << "] = {0";
    uint64_t prefix = 0;
    for (auto sz : split_sizes) {
        prefix += static_cast<uint64_t>(sz);
        msl << ", " << prefix;
    }
    msl << "};\n";
    msl << "constant uint AXIS_DIM = " << axis_len << ";\n";
    msl << "constant uint STRIDE_AFTER = " << inner_stride << ";\n";
    msl << "constant uint OUTER_STRIDE = AXIS_DIM * STRIDE_AFTER;\n";
    msl << "kernel void split_kernel(\n";
    msl << "  device const " << scalar << "* input [[buffer(0)]],\n";
    for (size_t oi = 0; oi < split_sizes.size(); ++oi) {
        msl << "  device " << scalar << "* out" << oi << " [[buffer(" << (oi + 1) << ")]],\n";
    }
    msl << "  uint gid [[thread_position_in_grid]]) {\n";
    msl << "    uint total = " << static_cast<uint32_t>(total_elems) << ";\n";
    msl << "    if (gid >= total) return;\n";
    msl << "    uint axis_idx = (gid / STRIDE_AFTER) % AXIS_DIM;\n";
    msl << "    uint outer = gid / OUTER_STRIDE;\n";
    msl << "    uint inner = gid % STRIDE_AFTER;\n";
    msl << "    uint o = 0;\n";
    msl << "    while (o + 1 < " << (split_sizes.size() + 1) << " && axis_idx >= OFFSETS[o + 1]) ++o;\n";
    msl << "    uint local_axis = axis_idx - OFFSETS[o];\n";
    msl << "    uint dst_axis_extent = OFFSETS[o + 1] - OFFSETS[o];\n";
    msl << "    uint dst_idx = (outer * dst_axis_extent + local_axis) * STRIDE_AFTER + inner;\n";
    msl << "    switch (o) {\n";
    for (size_t oi = 0; oi < split_sizes.size(); ++oi) {
        msl << "      case " << oi << ": out" << oi << "[dst_idx] = input[gid]; break;\n";
    }
    msl << "      default: break;\n";
    msl << "    }\n";
    msl << "}\n";

    plan.source = make_kernel_source(manifest_module,
                                     "split_kernel",
                                     msl.str(),
                                     static_cast<uint32_t>(1 + split_sizes.size()));
    plan.source.signature.output_arg_count = static_cast<uint32_t>(split_sizes.size());
    plan.binding = binding;
    return plan;
}

bool annotate_msl_module_with_runtime_binding_plan(mlir::ModuleOp module,
                                                   const GfxMslRuntimeBindingPlan& plan) {
    if (!module || !plan.valid()) {
        return false;
    }
    detail::gfx_mpsrt_set_stage_manifest_attrs(module, plan.stage_manifest);
    module->removeAttr("gfx.kernel_operand_kinds");
    module->removeAttr("gfx.kernel_operand_arg_indices");
    module->removeAttr("gfx.kernel_scalar_values");
    return true;
}

enum class GfxMatMulMetalKernelSourcePlanKind {
    None,
    Mpsrt,
    MslFallback,
};

struct GfxMatMulMetalKernelSourcePlan {
    GfxMatMulMetalKernelSourcePlanKind kind = GfxMatMulMetalKernelSourcePlanKind::None;
    GfxMatMulMpsrtLoweringKind mpsrt_lowering = GfxMatMulMpsrtLoweringKind::None;
    GfxMpsrtKernelSourcePlan mpsrt_plan;
    KernelSource source;
    bool requires_mpsrt_model = false;

    bool valid() const {
        return kind != GfxMatMulMetalKernelSourcePlanKind::None && source.module;
    }
};

GfxMatMulMetalKernelSourcePlan make_matmul_msl_fallback_source_plan(
    mlir::ModuleOp module,
    const GpuBufferManager* buffer_manager,
    const std::shared_ptr<const ov::Node>& node,
    const MatMulCodegenDesc& desc) {
    GfxMatMulMetalKernelSourcePlan result{};
    if (!module || !node) {
        return result;
    }

    constexpr const char* kStageType = "MatMul";
    constexpr const char* kEntryPoint = "matmul_kernel";
    const uint32_t arg_count = desc.has_bias ? 4u : 3u;
    auto plan = select_stage_optimization_plan(buffer_manager,
                                               GpuBackend::Metal,
                                               kStageType,
                                               node,
                                               desc.output_type,
                                               desc.has_bias,
                                               desc.has_activation,
                                               /*has_batchnorm=*/false,
                                               GfxStageRuntimeTraits{});
    force_apple_msl_buffer_placement(plan, kStageType);
    annotate_msl_module_with_stage_plan(module, plan, kStageType, kEntryPoint);
    if (desc.has_bias) {
        GfxKernelStageManifest manifest{};
        if (detail::gfx_mpsrt_read_stage_manifest_attrs(module, manifest) &&
            manifest.custom_kernel.valid) {
            manifest.custom_kernel.external_buffer_abi = make_matmul_bias_external_buffer_abi();
            detail::gfx_mpsrt_set_stage_manifest_attrs(module, manifest);
        }
    }

    auto plan_ctx = build_mlir_kernel_plan(
        module,
        kEntryPoint,
        node,
        /*output_args_override=*/0,
        /*extra_inputs=*/0,
        node->get_friendly_name().c_str(),
        "gfx_kernel",
        [&](const KernelArgMappingInfo& info) -> size_t {
            return resolve_matmul_source_arg_count(module, arg_count, info);
        });

    auto source_desc = desc;
    auto source = plan_ctx.build_info.plan.to_source_with_msl_generator(
        [source_desc](mlir::ModuleOp mod) {
            return generate_msl_from_mlir(mod, source_desc);
        });
    source.signature.output_arg_count = 1;

    result.kind = GfxMatMulMetalKernelSourcePlanKind::MslFallback;
    result.requires_mpsrt_model = false;
    auto mpsrt_plan = configure_msl_kernel_source_plan(std::move(source), kStageType);
    if (mpsrt_plan.valid()) {
        result.mpsrt_plan = std::move(mpsrt_plan);
        result.source = result.mpsrt_plan.source;
        result.requires_mpsrt_model = result.mpsrt_plan.requires_mpsrt_model;
    } else {
        result.kind = GfxMatMulMetalKernelSourcePlanKind::None;
    }
    return result;
}

GfxMatMulMpsrtKernelSourcePlan lower_matmul_module_to_mpsrt_kernel_source(
    mlir::ModuleOp module,
    const GfxStageOptimizationPlan& plan,
    const MatMulCodegenDesc& desc,
    const ov::Shape& shape_a,
    const ov::Shape& shape_b) {
    GfxMatMulMpsrtKernelSourcePlan result{};
    result.lowering = annotate_module_with_matmul_mpsrt_plan(module, plan, desc, shape_a, shape_b);
    if (result.lowering == GfxMatMulMpsrtLoweringKind::None) {
        return result;
    }

    switch (result.lowering) {
        case GfxMatMulMpsrtLoweringKind::MpsGemm:
            break;
        case GfxMatMulMpsrtLoweringKind::MpsGemmWithMslEpilogue:
            result.mpsrt_plan =
                make_mpsrt_kernel_source_plan_from_msl_source(module,
                                                              generate_msl_for_matmul_mpsrt_epilogue(desc));
            break;
        case GfxMatMulMpsrtLoweringKind::None:
            break;
    }
    if (result.lowering == GfxMatMulMpsrtLoweringKind::MpsGemm) {
        result.mpsrt_plan = make_mpsrt_kernel_source_plan_from_module(module);
    }
    if (!result.mpsrt_plan.valid()) {
        result.lowering = GfxMatMulMpsrtLoweringKind::None;
        return result;
    }
    result.source = result.mpsrt_plan.source;
    result.requires_mpsrt_model = result.mpsrt_plan.requires_mpsrt_model;
    return result;
}

GfxMatMulMetalKernelSourcePlan lower_matmul_node_to_metal_kernel_source(
    mlir::MLIRContext& ctx,
    const GpuBufferManager* buffer_manager,
    const std::shared_ptr<const ov::Node>& node,
    MatMulCodegenDesc desc,
    const ov::Shape& shape_a,
    const ov::Shape& shape_b) {
    GfxMatMulMetalKernelSourcePlan result{};
    if (!node) {
        return result;
    }

    auto module = build_mlir_for_node(node, ctx);
    if (!module) {
        return result;
    }
    if (desc.has_activation) {
        const bool applied = apply_fused_activation(module, desc.activation, desc.alpha);
        if (!applied) {
            return result;
        }
    }

    const auto output_type = node->get_output_element_type(0);
    desc.element_type = resolve_matmul_buffer_type(desc.element_type, output_type);
    desc.input_a_type = resolve_matmul_buffer_type(desc.input_a_type, desc.element_type);
    desc.input_b_type = resolve_matmul_buffer_type(desc.input_b_type, desc.element_type);
    desc.output_type = output_type;

    const auto placement = select_stage_optimization_plan(buffer_manager,
                                                          GpuBackend::Metal,
                                                          "MatMul",
                                                          node,
                                                          desc.output_type,
                                                          desc.has_bias,
                                                          desc.has_activation,
                                                          /*has_batchnorm=*/false,
                                                          GfxStageRuntimeTraits{});
    auto mpsrt_source = lower_matmul_module_to_mpsrt_kernel_source(module,
                                                                  placement,
                                                                  desc,
                                                                  shape_a,
                                                                  shape_b);
    if (mpsrt_source.valid()) {
        result.kind = GfxMatMulMetalKernelSourcePlanKind::Mpsrt;
        result.mpsrt_lowering = mpsrt_source.lowering;
        result.mpsrt_plan = std::move(mpsrt_source.mpsrt_plan);
        result.source = std::move(mpsrt_source.source);
        result.requires_mpsrt_model = mpsrt_source.requires_mpsrt_model;
        return result;
    }

    return make_matmul_msl_fallback_source_plan(module, buffer_manager, node, desc);
}

KernelSource make_apple_metal_runtime_matmul_kernel_source(mlir::MLIRContext& ctx,
                                                           const GpuBufferManager* buffer_manager,
                                                           const std::shared_ptr<const ov::Node>& node,
                                                           MatMulCodegenDesc desc,
                                                           const ov::Shape& shape_a,
                                                           const ov::Shape& shape_b,
                                                           std::string_view stage_name) {
    auto source_plan = lower_matmul_node_to_metal_kernel_source(ctx,
                                                                buffer_manager,
                                                                node,
                                                                desc,
                                                                shape_a,
                                                                shape_b);
    OPENVINO_ASSERT(source_plan.valid(),
                    "MetalStage: failed to create runtime MatMul source plan for ",
                    stage_name);
    return std::move(source_plan.source);
}

}  // namespace gfx_plugin
}  // namespace ov
