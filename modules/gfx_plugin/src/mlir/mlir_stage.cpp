// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/mlir_stage.hpp"
#include "mlir/mlir_support.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <sstream>

#include "kernel_ir/gfx_kernel_args.hpp"
#include "kernel_ir/gfx_kernel_plan.hpp"
#include "kernel_ir/gfx_kernel_spec.hpp"
#include "mlir/codegen_common.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_kernel_plan_utils.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_parallelism.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "runtime/gfx_stage_policy.hpp"
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
#include "openvino/core/validation_util.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/split.hpp"
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

std::optional<ov::Tensor> evaluate_constant_source_tensor(const ov::Output<ov::Node>& source) {
    auto node = source.get_node_shared_ptr();
    if (!node) {
        return std::nullopt;
    }
    if (auto constant = ov::as_type_ptr<const ov::op::v0::Constant>(node)) {
        return constant->get_tensor_view();
    }
    if (!node->has_evaluate()) {
        return std::nullopt;
    }

    ov::TensorVector inputs;
    inputs.reserve(node->get_input_size());
    for (const auto& input_value : node->input_values()) {
        auto input_tensor = evaluate_constant_source_tensor(input_value);
        if (!input_tensor.has_value()) {
            return std::nullopt;
        }
        inputs.push_back(*input_tensor);
    }

    ov::TensorVector outputs;
    outputs.reserve(node->get_output_size());
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (node->get_output_partial_shape(i).is_dynamic()) {
            return std::nullopt;
        }
        outputs.emplace_back(node->get_output_element_type(i), node->get_output_shape(i));
    }
    if (!node->evaluate(outputs, inputs)) {
        return std::nullopt;
    }
    return outputs.at(source.get_index());
}

std::vector<int64_t> evaluate_constant_source_i64(const ov::Output<ov::Node>& source, const char* what) {
    auto constant = ov::util::get_constant_from_source(source);
    OPENVINO_ASSERT(constant, "GFX MLIR: ", what, " must be Constant");
    return constant->cast_vector<int64_t>();
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
                              std::vector<uint32_t>& steps_full) {
    const size_t rank = in_shape.size();
    OPENVINO_ASSERT(rank == out_shape.size(),
                    "GFX MLIR: rank-changing Slice/StridedSlice is not supported");
    starts_full.assign(rank, 0);
    steps_full.assign(rank, 1);

    if (auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node)) {
        auto starts = evaluate_constant_source_i64(slice->input_value(1), "Slice starts");
        auto ends = evaluate_constant_source_i64(slice->input_value(2), "Slice ends");
        auto steps = evaluate_constant_source_i64(slice->input_value(3), "Slice steps");
        std::vector<int64_t> axes;
        if (slice->get_input_size() > 4) {
            axes = evaluate_constant_source_i64(slice->input_value(4), "Slice axes");
        } else {
            axes.resize(starts.size());
            std::iota(axes.begin(), axes.end(), 0);
        }
        OPENVINO_ASSERT(starts.size() == ends.size() && starts.size() == steps.size() && starts.size() == axes.size(),
                        "GFX MLIR: Slice starts/ends/steps/axes size mismatch for stage ",
                        node->get_friendly_name());
        for (size_t i = 0; i < axes.size(); ++i) {
            int64_t axis = axes[i];
            if (axis < 0) {
                axis += static_cast<int64_t>(rank);
            }
            OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank,
                            "GFX MLIR: Slice axis out of range for stage ",
                            node->get_friendly_name());
            OPENVINO_ASSERT(steps[i] > 0,
                            "GFX MLIR: Slice only supports positive steps for stage ",
                            node->get_friendly_name());
            const auto dim = static_cast<int64_t>(in_shape[static_cast<size_t>(axis)]);
            starts_full[static_cast<size_t>(axis)] =
                static_cast<int32_t>(normalize_slice_index(starts[i], dim, true));
            steps_full[static_cast<size_t>(axis)] = static_cast<uint32_t>(steps[i]);
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
    auto end = evaluate_constant_source_i64(slice->input_value(2), "StridedSlice end");
    std::vector<int64_t> strides(rank, 1);
    if (slice->get_input_size() > 3) {
        auto values = evaluate_constant_source_i64(slice->input_value(3), "StridedSlice strides");
        OPENVINO_ASSERT(values.size() <= rank,
                        "GFX MLIR: StridedSlice strides rank mismatch for stage ",
                        node->get_friendly_name());
        std::copy(values.begin(), values.end(), strides.begin());
    }
    OPENVINO_ASSERT(begin.size() <= rank && end.size() <= rank,
                    "GFX MLIR: StridedSlice begin/end rank mismatch for stage ",
                    node->get_friendly_name());
    const auto& begin_mask = slice->get_begin_mask();
    const auto& end_mask = slice->get_end_mask();
    for (size_t axis = 0; axis < rank; ++axis) {
        const auto dim = static_cast<int64_t>(in_shape[axis]);
        const bool masked_begin = axis < begin_mask.size() && begin_mask[axis] != 0;
        const bool masked_end = axis < end_mask.size() && end_mask[axis] != 0;
        const int64_t step = strides[axis];
        OPENVINO_ASSERT(step > 0,
                        "GFX MLIR: StridedSlice only supports positive steps for stage ",
                        node->get_friendly_name());
        int64_t start = axis < begin.size() ? begin[axis] : 0;
        int64_t finish = axis < end.size() ? end[axis] : dim;
        start = masked_begin ? 0 : normalize_slice_index(start, dim, true);
        finish = masked_end ? dim : normalize_slice_index(finish, dim, false);
        (void)finish;
        starts_full[axis] = static_cast<int32_t>(start);
        steps_full[axis] = static_cast<uint32_t>(step);
    }
}

inline mlir::ArrayAttr make_i32_array_attr(mlir::OpBuilder& b, const std::vector<int32_t>& vals) {
    llvm::SmallVector<mlir::Attribute, 8> attrs;
    attrs.reserve(vals.size());
    for (auto v : vals) {
        attrs.push_back(b.getI32IntegerAttr(v));
    }
    return b.getArrayAttr(attrs);
}

inline void normalize_operand_segment_sizes(mlir::ModuleOp module) {
    (void)module;
}


inline KernelSource make_split_msl_kernel(const ov::element::Type& out_et,
                                          const ov::Shape& in_shape,
                                          const std::vector<size_t>& split_sizes,
                                          uint32_t axis_len,
                                          uint32_t inner_stride) {
    const auto total_elems = ov::shape_size(in_shape);
    const auto scalar = msl_type_from_element(out_et);
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
    return make_kernel_source(nullptr,
                              "split_kernel",
                              msl.str(),
                              static_cast<uint32_t>(1 + split_sizes.size()));
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
        for (auto v : lengths) {
            OPENVINO_ASSERT(v >= 0, "VariadicSplit negative length not supported");
            plan.split_sizes.push_back(static_cast<size_t>(v));
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

void MlirStage::apply_kernel_metadata(const KernelRuntimeMetadata& meta,
                                      size_t scalar_inputs) {
    if (!meta.valid) {
        return;
    }
    m_parallel_cfg = meta.dispatch;
    if (is_vulkan_backend() && m_type == "Convolution" && m_parallel_cfg.enabled &&
        m_parallel_cfg.loop_dims >= 3 &&
        m_parallel_cfg.threads_h <= 1 && m_parallel_cfg.threads_w <= 1) {
        constexpr uint32_t kDefaultConvThreads = 4;
        m_parallel_cfg.tile_h = kDefaultConvThreads;
        m_parallel_cfg.tile_w = kDefaultConvThreads;
        m_parallel_cfg.threads_h = kDefaultConvThreads;
        m_parallel_cfg.threads_w = kDefaultConvThreads;
    }
    m_kernel_operand_kinds = std::move(meta.operands.operand_kinds);
    m_kernel_operand_arg_indices = std::move(meta.operands.operand_arg_indices);
    m_kernel_scalar_args = std::move(meta.operands.scalar_args);
    m_kernel_input_arg_count = meta.kernel_input_arg_count;
    if (m_kernel_operand_kinds.empty() && scalar_inputs != 0) {
        OPENVINO_ASSERT(m_kernel_scalar_args.size() == scalar_inputs,
                        "GFX MLIR: kernel scalar args mismatch for ",
                        m_name,
                        " (expected ",
                        scalar_inputs,
                        ", got ",
                        m_kernel_scalar_args.size(),
                        ")");
    }
}

void MlirStage::compile_from_plan(MlirKernelPlanContext& plan_ctx,
                                  mlir::ModuleOp module,
                                  const char* stage_kind) {
    auto& build_info = plan_ctx.build_info;
    if (module) {
        normalize_operand_segment_sizes(module);
    }
    const size_t scalar_inputs = plan_ctx.scalar_inputs;
    const size_t buffer_inputs = plan_ctx.buffer_inputs;
    m_kernel_input_arg_count = buffer_inputs;
    m_kernel_inputs = std::move(build_info.mapping.mapping.kernel_inputs);
    KernelSource src = build_info.plan.to_source();
    if (src.module) {
        normalize_operand_segment_sizes(src.module);
    }
    std::string log;
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
    if (module) {
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
            auto const_tensor = evaluate_constant_source_tensor(m_node->input_value(i));
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
            const size_t bytes = const_tensor->get_byte_size();
            const auto et = const_tensor->get_element_type();
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
                const uint64_t hash = gfx_hash_bytes(const_tensor->data(), bytes);
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
                GpuBuffer buf = m_buffer_manager->wrap_const(key.str(), const_tensor->data(), bytes, et);
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
        const ov::Shape in0_shape = compile_time_input_shape(0);
        const ov::Shape in1_shape = compile_time_input_shape(1);
        const auto elem_size = m_node->get_output_element_type(0).size();
        const auto dims_i32 = [&]() {
            std::vector<int32_t> v(out_shape.size());
            for (size_t i = 0; i < out_shape.size(); ++i) v[i] = static_cast<int32_t>(out_shape[i]);
            return v;
        }();
        (void)elem_size;
        auto stride0 = compile_time_broadcast_strides(0, out_shape);
        auto stride1 = compile_time_broadcast_strides(1, out_shape);
        if (stride0.empty() && m_node) {
            stride0 = compute_broadcast_element_strides(m_node->get_input_shape(0), out_shape);
        }
        if (stride1.empty() && m_node) {
            stride1 = compute_broadcast_element_strides(m_node->get_input_shape(1), out_shape);
        }

        auto wrap_vec = [&](const char* suffix, const std::vector<int32_t>& vec) -> GpuTensor {
            GpuTensor t;
            const std::string key = m_name + "/" + suffix;
            GpuBuffer buf = m_buffer_manager->wrap_const(key,
                                                         vec.data(),
                                                         vec.size() * sizeof(int32_t),
                                                         ov::element::i32);
            OPENVINO_ASSERT(buf.valid(),
                            "GFX MLIR: failed to wrap broadcast meta buffer for stage ",
                            m_name);
            buf.owned = false;
            t.buf = buf;
            t.expected_type = ov::element::i32;
            t.shape = ov::Shape{vec.size()};
            return t;
        };

        m_kernel_extra_inputs.push_back(wrap_vec("out_dims", dims_i32));
        m_kernel_extra_inputs.push_back(wrap_vec("stride0", stride0));
        m_kernel_extra_inputs.push_back(wrap_vec("stride1", stride1));
        if (gfx_log_debug_enabled()) {
            gfx_log_debug("MLIRExec") << "Prepared eltwise broadcast meta: dims=" << dims_i32.size()
                                      << " extras=" << m_kernel_extra_inputs.size();
        }
    };
    // Only for classic binary eltwise ops; keep list tight to avoid surprises.
    if (m_type == "Add" || m_type == "Subtract" || m_type == "Multiply" || m_type == "Divide" ||
        m_type == "Power" || m_type == "Mod" || m_type == "FloorMod" || m_type == "Minimum" ||
        m_type == "Maximum" || m_type == "Equal" || m_type == "NotEqual" || m_type == "Less" ||
        m_type == "Greater" || m_type == "LessEqual" || m_type == "GreaterEqual" ||
        m_type == "LogicalAnd" || m_type == "LogicalOr" || m_type == "LogicalXor" ||
        m_type == "SquaredDifference" || m_type == "PRelu") {
        prepare_eltwise_broadcast_meta();
    }
    ov::element::Type out_et = ov::element::dynamic;
    if (m_node) {
        out_et = m_node->get_output_element_type(0);
    }
    if (m_has_bias) {
        const size_t count = m_bias_params.values.size();
        if (count) {
            ov::element::Type bias_et = out_et == ov::element::dynamic ? m_bias_params.element_type : out_et;
            if (bias_et == ov::element::dynamic) {
                bias_et = ov::element::f32;
            }
            const std::string key = m_name + "/bias";
            GpuBuffer buf;
            if (bias_et == ov::element::f16) {
                m_bias_f16.resize(count);
                for (size_t i = 0; i < count; ++i) {
                    m_bias_f16[i] = ov::float16(m_bias_params.values[i]);
                }
                buf = m_buffer_manager->wrap_const(key,
                                                   m_bias_f16.data(),
                                                   m_bias_f16.size() * sizeof(ov::float16),
                                                   bias_et);
            } else {
                buf = m_buffer_manager->wrap_const(key,
                                                   m_bias_params.values.data(),
                                                   m_bias_params.values.size() * sizeof(float),
                                                   bias_et);
            }
            OPENVINO_ASSERT(buf.valid(),
                            "GFX MLIR: failed to wrap bias buffer for stage ",
                            m_name);
            buf.owned = false;
            GpuTensor tensor;
            tensor.buf = buf;
            tensor.expected_type = bias_et;
            const bool conv_like = (m_type == "Convolution" || m_type == "GroupConvolution");
            if (conv_like) {
                tensor.shape = ov::Shape{m_bias_params.values.size()};
            } else {
                size_t out_rank = m_bias_params.shape.size();
                if (m_node) {
                    const auto& pshape = m_node->get_output_partial_shape(0);
                    if (pshape.rank().is_static()) {
                        out_rank = static_cast<size_t>(pshape.rank().get_length());
                    }
                }
                std::vector<int64_t> aligned_shape(out_rank, 1);
                if (out_rank >= m_bias_params.shape.size()) {
                    const size_t offset = out_rank - m_bias_params.shape.size();
                    for (size_t i = 0; i < m_bias_params.shape.size(); ++i) {
                        aligned_shape[offset + i] = m_bias_params.shape[i];
                    }
                }
                ov::Shape bias_shape;
                bias_shape.reserve(aligned_shape.size());
                for (auto dim : aligned_shape) {
                    bias_shape.push_back(static_cast<size_t>(dim));
                }
                tensor.shape = std::move(bias_shape);
            }
            m_kernel_extra_inputs.push_back(std::move(tensor));
        }
    }
    if (m_has_bn) {
        const size_t channels = m_bn_params.gamma.size();
        if (channels) {
            ov::element::Type bn_et = out_et == ov::element::dynamic ? ov::element::f32 : out_et;
            if (bn_et == ov::element::dynamic) {
                bn_et = ov::element::f32;
            }
            std::vector<float> scale_vals(channels);
            std::vector<float> bias_vals(channels);
            for (size_t c = 0; c < channels; ++c) {
                const float gamma = m_bn_params.gamma[c];
                const float beta = m_bn_params.beta[c];
                const float mean = m_bn_params.mean[c];
                const float var = m_bn_params.var[c];
                const float inv_std = 1.0f / std::sqrt(var + m_bn_params.epsilon);
                const float scale = gamma * inv_std;
                const float bias = beta - mean * scale;
                scale_vals[c] = scale;
                bias_vals[c] = bias;
            }
            auto wrap_bn = [&](const std::string& suffix,
                               const std::vector<float>& vals) -> GpuTensor {
                GpuTensor tensor;
                const std::string key = m_name + "/" + suffix;
                GpuBuffer buf;
                if (bn_et == ov::element::f16) {
                    std::vector<ov::float16> tmp(vals.size());
                    for (size_t i = 0; i < vals.size(); ++i) {
                        tmp[i] = ov::float16(vals[i]);
                    }
                    buf = m_buffer_manager->wrap_const(key,
                                                       tmp.data(),
                                                       tmp.size() * sizeof(ov::float16),
                                                       bn_et);
                } else {
                    buf = m_buffer_manager->wrap_const(key,
                                                       vals.data(),
                                                       vals.size() * sizeof(float),
                                                       bn_et);
                }
                OPENVINO_ASSERT(buf.valid(),
                                "GFX MLIR: failed to wrap batchnorm buffer for stage ",
                                m_name);
                buf.owned = false;
                tensor.buf = buf;
                tensor.expected_type = bn_et;
                tensor.shape = ov::Shape{channels};
                return tensor;
            };
            m_kernel_extra_inputs.push_back(wrap_bn("bn_scale", scale_vals));
            m_kernel_extra_inputs.push_back(wrap_bn("bn_bias", bias_vals));
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
            struct Conv3DParams {
                uint32_t N, C_in, D, H, W;
                uint32_t C_out;
                uint32_t kD, kH, kW;
                uint32_t strideD, strideH, strideW;
                uint32_t dilationD, dilationH, dilationW;
                uint32_t padFront, padTop, padLeft, padBack, padBottom, padRight;
                uint32_t outD, outH, outW;
            } params{};
            params.N = static_cast<uint32_t>(in_shape[0]);
            params.C_in = static_cast<uint32_t>(in_shape[1]);
            params.D = static_cast<uint32_t>(in_shape[2]);
            params.H = static_cast<uint32_t>(in_shape[3]);
            params.W = static_cast<uint32_t>(in_shape[4]);
            params.C_out = static_cast<uint32_t>(w.at(0));
            params.kD = static_cast<uint32_t>(w.at(2));
            params.kH = static_cast<uint32_t>(w.at(3));
            params.kW = static_cast<uint32_t>(w.at(4));
            params.strideD = static_cast<uint32_t>(conv->get_strides().at(0));
            params.strideH = static_cast<uint32_t>(conv->get_strides().at(1));
            params.strideW = static_cast<uint32_t>(conv->get_strides().at(2));
            params.dilationD = static_cast<uint32_t>(conv->get_dilations().at(0));
            params.dilationH = static_cast<uint32_t>(conv->get_dilations().at(1));
            params.dilationW = static_cast<uint32_t>(conv->get_dilations().at(2));
            params.padFront = static_cast<uint32_t>(conv->get_pads_begin().at(0));
            params.padTop = static_cast<uint32_t>(conv->get_pads_begin().at(1));
            params.padLeft = static_cast<uint32_t>(conv->get_pads_begin().at(2));
            params.padBack = static_cast<uint32_t>(conv->get_pads_end().at(0));
            params.padBottom = static_cast<uint32_t>(conv->get_pads_end().at(1));
            params.padRight = static_cast<uint32_t>(conv->get_pads_end().at(2));
            params.outD = static_cast<uint32_t>(out_shape[2]);
            params.outH = static_cast<uint32_t>(out_shape[3]);
            params.outW = static_cast<uint32_t>(out_shape[4]);
            const std::string key = m_name + "/conv3d_params";
            GpuBuffer buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
            OPENVINO_ASSERT(buf.valid(), "GFX MLIR: failed to wrap conv3d params buffer for ", m_name);
            buf.owned = false;
            GpuTensor t;
            t.buf = buf;
            t.expected_type = ov::element::u8;
            t.shape = ov::Shape{sizeof(params)};
            m_kernel_extra_inputs.push_back(std::move(t));
            // No bias/BN for Conv3D path yet.
            m_kernel_input_arg_count = 2;  // data + weights
        } else {
            OPENVINO_ASSERT(out_shape.size() == 4, "GFX MLIR: Conv expects NCHW output");
            OPENVINO_ASSERT(in_shape.size() == 4, "GFX MLIR: Conv expects NCHW input");
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
        // Prepare bias + BN buffers even if unused to keep argument order stable.
        const size_t channels = static_cast<size_t>(C_out);
        auto wrap_float_vec = [&](const std::string& suffix,
                                  const std::vector<float>& vals) -> GpuTensor {
            GpuTensor t;
            std::vector<ov::float16> tmp_f16;
            const void* data_ptr = vals.data();
            size_t bytes = vals.size() * sizeof(float);
            ov::element::Type buf_et = et;
            if (et == ov::element::f16) {
                tmp_f16.resize(vals.size());
                for (size_t i = 0; i < vals.size(); ++i) tmp_f16[i] = ov::float16(vals[i]);
                data_ptr = tmp_f16.data();
                bytes = tmp_f16.size() * sizeof(ov::float16);
            } else if (!et.is_real()) {
                buf_et = ov::element::f32;
            }
            const std::string key = m_name + "/" + suffix;
            GpuBuffer buf = m_buffer_manager->wrap_const(key, data_ptr, bytes, buf_et);
            OPENVINO_ASSERT(buf.valid(), "GFX MLIR: failed to wrap conv buffer ", suffix, " for ", m_name);
            buf.owned = false;
            t.buf = buf;
            t.expected_type = buf_et;
            t.shape = ov::Shape{channels};
            return t;
        };

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
        m_kernel_extra_inputs.push_back(wrap_float_vec("bias", bias));
        m_kernel_extra_inputs.push_back(wrap_float_vec("gamma", gamma));
        m_kernel_extra_inputs.push_back(wrap_float_vec("beta", beta));
        m_kernel_extra_inputs.push_back(wrap_float_vec("mean", mean));
        m_kernel_extra_inputs.push_back(wrap_float_vec("var", var));
        struct ConvParams {
            uint32_t N, C_in, H, W;
            uint32_t C_out;
            uint32_t groups;
            uint32_t C_in_pg;
            uint32_t C_out_pg;
            uint32_t kH, kW;
            uint32_t strideH, strideW;
            uint32_t dilationH, dilationW;
            uint32_t padTop, padLeft;
            uint32_t padBottom, padRight;
            uint32_t outH, outW;
            uint32_t has_bias;
            uint32_t has_bn;
            uint32_t activation;
            float alpha;
            float epsilon;
            float clamp_min;
            float clamp_max;
        } params{};
        params.N = static_cast<uint32_t>(in_shape[0]);
        params.C_in = static_cast<uint32_t>(in_shape[1]);
        params.H = static_cast<uint32_t>(in_shape[2]);
        params.W = static_cast<uint32_t>(in_shape[3]);
        params.C_out = C_out;
        params.groups = groups;
        params.C_in_pg = C_in_pg ? C_in_pg : params.C_in;
        params.C_out_pg = C_out_pg ? C_out_pg : params.C_out;
        params.kH = kH;
        params.kW = kW;
        params.strideH = strideH;
        params.strideW = strideW;
        params.dilationH = dilationH;
        params.dilationW = dilationW;
        params.padTop = padTop;
        params.padLeft = padLeft;
        params.padBottom = padBottom;
        params.padRight = padRight;
        params.outH = static_cast<uint32_t>(out_shape[2]);
        params.outW = static_cast<uint32_t>(out_shape[3]);
        params.has_bias = m_has_bias ? 1u : 0u;
        params.has_bn = m_has_bn ? 1u : 0u;
        params.activation = m_has_activation ? 1u : 0u;
        params.alpha = m_activation_alpha;
        params.epsilon = epsilon;
        params.clamp_min = 0.0f;
        params.clamp_max = 0.0f;
        {
            const std::string key = m_name + "/conv_params";
            GpuBuffer buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
            OPENVINO_ASSERT(buf.valid(), "GFX MLIR: failed to wrap conv params buffer for ", m_name);
            buf.owned = false;
            GpuTensor t;
            t.buf = buf;
            t.expected_type = ov::element::u8;
            t.shape = ov::Shape{sizeof(params)};
            m_kernel_extra_inputs.push_back(std::move(t));
        }
        }
    }
    if (m_type == "MaxPool" || m_type == "AvgPool") {
        m_kernel_extra_inputs.clear();
        auto maxpool = std::dynamic_pointer_cast<const ov::op::v1::MaxPool>(m_node);
        auto avgpool = std::dynamic_pointer_cast<const ov::op::v1::AvgPool>(m_node);
        OPENVINO_ASSERT(maxpool || avgpool, "GFX MLIR: pool node cast failed");
        const auto in = m_node->get_input_shape(0);
        const auto out = m_node->get_output_shape(0);
        OPENVINO_ASSERT(in.size() == 4 && out.size() == 4, "GFX MLIR: pool expects NCHW");
        const auto& kernel = maxpool ? maxpool->get_kernel() : avgpool->get_kernel();
        const auto& strides = maxpool ? maxpool->get_strides() : avgpool->get_strides();
        const auto& pads_begin = maxpool ? maxpool->get_pads_begin() : avgpool->get_pads_begin();
        const auto& pads_end = maxpool ? maxpool->get_pads_end() : avgpool->get_pads_end();
        const bool is_avg = avgpool != nullptr;
        const bool exclude_pad = is_avg ? avgpool->get_exclude_pad() : true;
        struct PoolParams {
            uint32_t N, C, H, W;
            uint32_t kH, kW;
            uint32_t strideH, strideW;
            uint32_t dilationH, dilationW;
            uint32_t padTop, padLeft, padBottom, padRight;
            uint32_t outH, outW;
            uint32_t is_avg;
            uint32_t exclude_pad;
        } params{};
        params.N = static_cast<uint32_t>(in[0]);
        params.C = static_cast<uint32_t>(in[1]);
        params.H = static_cast<uint32_t>(in[2]);
        params.W = static_cast<uint32_t>(in[3]);
        params.kH = static_cast<uint32_t>(kernel.at(0));
        params.kW = static_cast<uint32_t>(kernel.at(1));
        params.strideH = static_cast<uint32_t>(strides.at(0));
        params.strideW = static_cast<uint32_t>(strides.at(1));
        params.dilationH = 1;
        params.dilationW = 1;
        params.padTop = static_cast<uint32_t>(pads_begin.at(0));
        params.padLeft = static_cast<uint32_t>(pads_begin.at(1));
        params.padBottom = static_cast<uint32_t>(pads_end.at(0));
        params.padRight = static_cast<uint32_t>(pads_end.at(1));
        params.outH = static_cast<uint32_t>(out.at(2));
        params.outW = static_cast<uint32_t>(out.at(3));
        params.is_avg = is_avg ? 1u : 0u;
        params.exclude_pad = exclude_pad ? 1u : 0u;
        const std::string key = m_name + "/pool_params";
        GpuBuffer buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
        OPENVINO_ASSERT(buf.valid(), "GFX MLIR: failed to wrap pool params for ", m_name);
        buf.owned = false;
        GpuTensor t;
        t.buf = buf;
        t.expected_type = ov::element::u8;
        t.shape = ov::Shape{sizeof(params)};
        m_kernel_extra_inputs.push_back(std::move(t));
    }
    if (m_type == "Softmax" || m_type == "LogSoftmax" ||
        m_type == "Split" || m_type == "VariadicSplit") {
        return;
    }
    const auto optimization_plan = stage_optimization_plan();
    KernelPlan plan = [&]() {
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
    apply_stage_optimization_attrs(module, optimization_plan);
    apply_input_transform_attrs(module);
    set_parallel_preference(module);
    apply_fused_operations(module);
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
        plan_ctx.build_info.plan = KernelPlan(module, plan_ctx.build_info.plan.entry_point(), 3);
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
         optimization_plan.conv.algorithm.kind != GfxConvAlgorithmKind::Im2ColMatMul &&
         optimization_plan.conv.algorithm.kind != GfxConvAlgorithmKind::Indirect &&
         m_node->get_input_partial_shape(0).rank().is_static() &&
         m_node->get_input_partial_shape(0).rank().get_length() == 4);
    const bool use_manual_group_conv2d_vulkan =
        (is_vulkan_backend() &&
         m_type == "GroupConvolution" &&
         !has_absorbed_input_transpose() &&
         !m_has_bias &&
         !m_has_activation &&
         !m_has_bn &&
         m_node &&
         m_node->get_input_size() == 2 &&
         m_node->get_output_size() == 1 &&
         m_node->get_input_partial_shape(0).rank().is_static() &&
         m_node->get_input_partial_shape(0).rank().get_length() == 4);
    if (use_manual_conv2d_vulkan) {
        if (auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node)) {
            module = build_mlir_conv2d_vulkan(conv, ctx);
            m_force_single_dispatch = true;
            mlir::OpBuilder b(module->getContext());
            apply_stage_optimization_attrs(module, optimization_plan);
            module->setAttr("gfx.skip_conv_parallel", mlir::BoolAttr::get(module.getContext(), true));
            module->setAttr("gfx.prefer_parallel", mlir::BoolAttr::get(module.getContext(), true));
            m_kernel_extra_inputs.clear();
            std::vector<int32_t> kinds = {1, 1, 1};
            std::vector<int32_t> arg_idx = {0, 1, 2};
            module->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr(b, kinds));
            module->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr(b, arg_idx));
            plan_ctx = build_mlir_kernel_plan(
                module,
                "conv2d_main",
                m_node,
                /*output_args_override=*/0,
                /*extra_inputs=*/0,
                m_name.c_str(),
                "conv2d_main",
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
            module = build_mlir_group_conv2d_vulkan(gconv, ctx);
            m_force_single_dispatch = true;
            mlir::OpBuilder b(module->getContext());
            apply_stage_optimization_attrs(module, optimization_plan);
            module->setAttr("gfx.skip_conv_parallel", mlir::BoolAttr::get(module.getContext(), true));
            module->setAttr("gfx.prefer_parallel", mlir::BoolAttr::get(module.getContext(), false));
            m_kernel_extra_inputs.clear();
            std::vector<int32_t> kinds = {1, 1, 1};
            std::vector<int32_t> arg_idx = {0, 1, 2};
            module->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr(b, kinds));
            module->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr(b, arg_idx));
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
    compile_from_plan(plan_ctx, module, "stage");
    if (module) {
        if (gfx_log_debug_enabled() && !m_kernel_scalar_args.empty()) {
            std::ostringstream oss;
            oss << "Kernel scalar args: ";
            const size_t dump_n = std::min<size_t>(m_kernel_scalar_args.size(), 8);
            for (size_t i = 0; i < dump_n; ++i) {
                if (i) {
                    oss << ", ";
                }
                oss << m_kernel_scalar_args[i];
            }
            if (m_kernel_scalar_args.size() > dump_n) {
                oss << ", ...";
            }
            gfx_log_debug("MLIRExec") << oss.str();
        }
        if (gfx_log_debug_enabled()) {
            const bool has_kinds = module->hasAttr("gfx.kernel_operand_kinds");
            const bool has_scalars = module->hasAttr("gfx.kernel_scalar_values");
            gfx_log_debug("MLIRExec") << "Kernel attrs: operand_kinds=" << (has_kinds ? "yes" : "no")
                                      << " scalar_values=" << (has_scalars ? "yes" : "no");
            gfx_log_debug("MLIRExec") << "Kernel operand kinds size=" << m_kernel_operand_kinds.size();
            if (!m_kernel_operand_arg_indices.empty()) {
                std::ostringstream idxs;
                idxs << "Kernel operand arg indices: ";
                const size_t dump_n = std::min<size_t>(m_kernel_operand_arg_indices.size(), 8);
                for (size_t i = 0; i < dump_n; ++i) {
                    if (i) {
                        idxs << ", ";
                    }
                    idxs << m_kernel_operand_arg_indices[i];
                }
                if (m_kernel_operand_arg_indices.size() > dump_n) {
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

    if (m_kernel_inputs.empty() && m_node) {
        const size_t in_count = m_node->get_input_size();
        m_kernel_inputs.reserve(in_count);
        for (size_t i = 0; i < in_count; ++i) {
            m_kernel_inputs.push_back(i);
        }
    }

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

    auto ensure_output_shape = [&](size_t oi, GpuTensor* out) {
        if (!out) {
            return;
        }
        if (out->shape.empty() && m_node && m_node->get_output_partial_shape(oi).is_static()) {
            out->shape = m_node->get_output_shape(oi);
        }
    };

    if (m_type == "Slice" || m_type == "StridedSlice") {
        const bool use_runtime_slice_args = is_vulkan_backend();
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
        OPENVINO_ASSERT(!out_shape.empty(), "GFX MLIR: Slice/StridedSlice output shape is unknown for stage ", m_name);
        const size_t rank = in_shape.size();
        OPENVINO_ASSERT(rank == out_shape.size(), "GFX MLIR: Slice/StridedSlice rank mismatch for stage ", m_name);

        std::vector<uint32_t> out_shape_u(rank);
        std::vector<uint32_t> in_stride(rank, 1);
        std::vector<int32_t> starts_full(rank, 0);
        std::vector<uint32_t> steps_full(rank, 1);
        for (size_t i = 0; i < rank; ++i) {
            out_shape_u[i] = static_cast<uint32_t>(out_shape[i]);
        }
        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
            in_stride[static_cast<size_t>(i)] =
                in_stride[static_cast<size_t>(i + 1)] * static_cast<uint32_t>(in_shape[static_cast<size_t>(i + 1)]);
        }
        build_slice_runtime_spec(m_node, in_shape, out_shape, starts_full, steps_full);

        for (auto* out : outputs) {
            if (out) {
                out->shape = out_shape;
            }
        }

        auto wrap_bytes_tensor = [&](const std::string& suffix,
                                     const void* data,
                                     size_t bytes,
                                     const ov::element::Type& et,
                                     const ov::Shape& shape) {
            GpuTensor t;
            GpuBuffer buf = m_buffer_manager->wrap_const(m_name + "/" + suffix, data, bytes, et);
            OPENVINO_ASSERT(buf.valid(), "GFX MLIR: failed to wrap slice buffer ", suffix, " for ", m_name);
            buf.owned = false;
            t.buf = buf;
            t.expected_type = et;
            t.shape = shape;
            return t;
        };

        m_kernel_extra_inputs.clear();
        if (use_runtime_slice_args) {
            const uint32_t total = static_cast<uint32_t>(ov::shape_size(out_shape));
            const uint32_t rank_u = static_cast<uint32_t>(rank);
            m_kernel_extra_inputs.push_back(wrap_bytes_tensor("slice_total", &total, sizeof(total), ov::element::u32, ov::Shape{1}));
            m_kernel_extra_inputs.push_back(wrap_bytes_tensor("slice_rank", &rank_u, sizeof(rank_u), ov::element::u32, ov::Shape{1}));
            m_kernel_extra_inputs.push_back(wrap_bytes_tensor("slice_out_shape",
                                                              out_shape_u.data(),
                                                              out_shape_u.size() * sizeof(uint32_t),
                                                              ov::element::u32,
                                                              ov::Shape{out_shape_u.size()}));
            m_kernel_extra_inputs.push_back(wrap_bytes_tensor("slice_in_stride",
                                                              in_stride.data(),
                                                              in_stride.size() * sizeof(uint32_t),
                                                              ov::element::u32,
                                                              ov::Shape{in_stride.size()}));
            m_kernel_extra_inputs.push_back(wrap_bytes_tensor("slice_starts",
                                                              starts_full.data(),
                                                              starts_full.size() * sizeof(int32_t),
                                                              ov::element::i32,
                                                              ov::Shape{starts_full.size()}));
            m_kernel_extra_inputs.push_back(wrap_bytes_tensor("slice_steps",
                                                              steps_full.data(),
                                                              steps_full.size() * sizeof(uint32_t),
                                                              ov::element::u32,
                                                              ov::Shape{steps_full.size()}));
        }

        if (m_node && (m_last_input_shape != in_shape || !m_kernel)) {
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
                KernelSource src;
                src.entry_point = "slice_kernel";
                src.signature.arg_count = 2;
                std::string log;
                try {
                    m_kernel = compile_kernel(src, &log);
                } catch (const std::exception& e) {
                    OPENVINO_THROW("GFX MLIR: failed to compile slice stage ",
                                   m_name,
                                   " (",
                                   m_type,
                                   "): ",
                                   e.what());
                }
                OPENVINO_ASSERT(m_kernel,
                                "GFX MLIR: failed to compile slice stage ",
                                m_name,
                                " (",
                                m_type,
                                "): ",
                                log);
                m_kernel_operand_kinds = {1, 1};
                m_kernel_operand_arg_indices = {0, 1};
                m_kernel_inputs = {0};
                m_kernel_input_arg_count = 1;
                m_last_input_shape = in_shape;
            }
            auto& ctx = gfx_mlir_context();
            auto module = build_mlir_for_node(m_node, ctx);
            if (module) {
                mlir::OpBuilder b(module.getContext());
                std::vector<int32_t> kinds = use_runtime_slice_args ? std::vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1}
                                                                    : std::vector<int32_t>{1, 1};
                std::vector<int32_t> arg_idx = use_runtime_slice_args ? std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}
                                                                      : std::vector<int32_t>{0, 1};
                module->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr(b, kinds));
                module->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr(b, arg_idx));
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
            m_kernel_operand_kinds = use_runtime_slice_args ? std::vector<int32_t>{1, 1, 1, 1, 1, 1, 1, 1}
                                                            : std::vector<int32_t>{1, 1};
            m_kernel_operand_arg_indices = use_runtime_slice_args ? std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7}
                                                                  : std::vector<int32_t>{0, 1};
            m_kernel_inputs = {0};
            m_kernel_input_arg_count = 1;
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
        struct InterpolateParams {
            uint32_t N;
            uint32_t C;
            uint32_t H_in;
            uint32_t W_in;
            uint32_t H_out;
            uint32_t W_out;
            float scale_h;
            float scale_w;
            uint32_t align_corners;
            uint32_t use_half_pixel;
            uint32_t nearest_mode;
        } params{};
        params.N = static_cast<uint32_t>(in_shape[0]);
        params.C = static_cast<uint32_t>(in_shape[1]);
        params.H_in = static_cast<uint32_t>(in_shape[2]);
        params.W_in = static_cast<uint32_t>(in_shape[3]);
        params.H_out = static_cast<uint32_t>(out_shape[2]);
        params.W_out = static_cast<uint32_t>(out_shape[3]);
        params.scale_h = params.H_out ? static_cast<float>(params.H_in) / static_cast<float>(params.H_out) : 1.f;
        params.scale_w = params.W_out ? static_cast<float>(params.W_in) / static_cast<float>(params.W_out) : 1.f;
        params.align_corners = 0;
        params.use_half_pixel = 1;
        params.nearest_mode = 0;
        if (auto v0 = ov::as_type_ptr<const ov::op::v0::Interpolate>(m_node)) {
            params.align_corners = v0->get_attrs().align_corners ? 1u : 0u;
            params.use_half_pixel = params.align_corners ? 0u : 1u;
        } else if (auto v4 = ov::as_type_ptr<const ov::op::v4::Interpolate>(m_node)) {
            using Base = ov::op::util::InterpolateBase;
            params.align_corners =
                v4->get_attrs().coordinate_transformation_mode == Base::CoordinateTransformMode::ALIGN_CORNERS ? 1u : 0u;
            params.use_half_pixel =
                v4->get_attrs().coordinate_transformation_mode == Base::CoordinateTransformMode::HALF_PIXEL ? 1u : 0u;
            switch (v4->get_attrs().nearest_mode) {
                case Base::NearestMode::FLOOR:
                case Base::NearestMode::ROUND_PREFER_FLOOR:
                    params.nearest_mode = 1;
                    break;
                case Base::NearestMode::CEIL:
                case Base::NearestMode::ROUND_PREFER_CEIL:
                    params.nearest_mode = 2;
                    break;
                case Base::NearestMode::SIMPLE:
                default:
                    params.nearest_mode = 0;
                    break;
            }
        } else if (auto v11 = ov::as_type_ptr<const ov::op::v11::Interpolate>(m_node)) {
            using Base = ov::op::util::InterpolateBase;
            params.align_corners =
                v11->get_attrs().coordinate_transformation_mode == Base::CoordinateTransformMode::ALIGN_CORNERS ? 1u : 0u;
            params.use_half_pixel =
                v11->get_attrs().coordinate_transformation_mode == Base::CoordinateTransformMode::HALF_PIXEL ? 1u : 0u;
            switch (v11->get_attrs().nearest_mode) {
                case Base::NearestMode::FLOOR:
                case Base::NearestMode::ROUND_PREFER_FLOOR:
                    params.nearest_mode = 1;
                    break;
                case Base::NearestMode::CEIL:
                case Base::NearestMode::ROUND_PREFER_CEIL:
                    params.nearest_mode = 2;
                    break;
                case Base::NearestMode::SIMPLE:
                default:
                    params.nearest_mode = 0;
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
            const std::string key = m_name + "/interpolate_params";
            GpuBuffer buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
            OPENVINO_ASSERT(buf.valid(), "GFX MLIR: failed to wrap interpolate params buffer for ", m_name);
            buf.owned = false;
            GpuTensor t;
            t.buf = buf;
            t.expected_type = ov::element::u8;
            t.shape = ov::Shape{sizeof(params)};
            m_kernel_extra_inputs.push_back(std::move(t));
        }
        if (m_node && (m_last_input_shape != in_shape || !m_kernel)) {
            auto& ctx = gfx_mlir_context();
            auto module = build_mlir_for_node(m_node, ctx);
            if (module) {
                mlir::OpBuilder b(module.getContext());
                std::vector<int32_t> kinds = use_runtime_interpolate_params ? std::vector<int32_t>{1, 1, 1}
                                                                            : std::vector<int32_t>{1, 1};
                std::vector<int32_t> arg_idx = use_runtime_interpolate_params ? std::vector<int32_t>{0, 1, 2}
                                                                              : std::vector<int32_t>{0, 1};
                module->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr(b, kinds));
                module->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr(b, arg_idx));
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
            m_kernel_operand_kinds = use_runtime_interpolate_params ? std::vector<int32_t>{1, 1, 1}
                                                                   : std::vector<int32_t>{1, 1};
            m_kernel_operand_arg_indices = use_runtime_interpolate_params ? std::vector<int32_t>{0, 1, 2}
                                                                         : std::vector<int32_t>{0, 1};
            m_kernel_inputs = {0};
            m_kernel_input_arg_count = 1;
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
        // Prepare params buffer (rows, cols, inner) for MSL softmax kernel.
        struct SoftmaxParams {
            uint32_t rows;
            uint32_t cols;
            uint32_t inner;
        } params{static_cast<uint32_t>(dims.rows),
                 static_cast<uint32_t>(dims.axis_len),
                 static_cast<uint32_t>(dims.inner)};
        {
            const std::string key = m_name + "/softmax_params";
            GpuBuffer buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::i32);
            OPENVINO_ASSERT(buf.valid(), "GFX MLIR: failed to wrap softmax params buffer for stage ", m_name);
            buf.owned = false;
            GpuTensor t;
            t.buf = buf;
            t.expected_type = ov::element::i32;
            t.shape = ov::Shape{3};
            m_kernel_extra_inputs.clear();
            m_kernel_extra_inputs.push_back(t);
        }
        m_kernel.reset();  // split is small; rebuild specialized kernel every call
        m_last_input_shape.clear();
        if (m_node) {
            auto& ctx = gfx_mlir_context();
            const bool log_softmax = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(m_node) != nullptr;
            auto module = log_softmax
                              ? build_mlir_logsoftmax_from_node(m_node, ctx, in_shape)
                              : build_mlir_softmax_from_node(m_node, ctx, in_shape);
            if (module) {
                module->setAttr("gfx.prefer_parallel",
                                mlir::BoolAttr::get(module.getContext(), false));
                // Operand mapping: buffer0=input, buffer1=output, buffer2=params.
                mlir::OpBuilder b(module.getContext());
                // Argument order for softmax kernels is: input, output, params.
                // Use a straightforward 0,1,2 mapping to avoid backend-specific
                // descriptor ordering issues (Vulkan driver can lose device
                // when bindings are permuted).
                std::vector<int32_t> kinds{1, 1, 1};
                std::vector<int32_t> arg_idx{0, 1, 2};
                module->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr(b, kinds));
                module->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr(b, arg_idx));
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
            // Pin bindings: 0=input, 1=output, 2=params to avoid backend reordering.
            m_kernel_operand_kinds = {1, 1, 1};
            m_kernel_operand_arg_indices = {0, 1, 2};
            m_kernel_inputs = {0};
            m_kernel_input_arg_count = 1;
            m_last_input_shape = in_shape;
        }
    } else if (m_type == "Split" || m_type == "VariadicSplit") {
        ov::Shape stage_input_shape = resolve_input_shape(0);
        if (stage_input_shape.empty()) {
            OPENVINO_THROW("GFX MLIR: Split input shape is unknown for stage ", m_name);
        }
        ov::Shape logical_input_shape = m_node ? m_node->get_input_shape(0) : stage_input_shape;
        if (logical_input_shape.empty()) {
            logical_input_shape = stage_input_shape;
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
        bool compiled_split = false;
        const bool has_input_transform = has_absorbed_input_transpose();
        if (!has_input_transform && m_node && (m_last_input_shape != stage_input_shape || !m_kernel)) {
            const ov::element::Type out_et = m_node ? m_node->get_output_element_type(0)
                                                    : (outputs.empty() ? ov::element::f32
                                                                        : outputs.front()->expected_type);
            KernelSource src = make_split_msl_kernel(out_et,
                                                     logical_input_shape,
                                                     plan.split_sizes,
                                                     plan.axis_len,
                                                     plan.inner_stride);
            m_kernel_input_arg_count = 1;
            m_kernel_inputs.clear();
            m_kernel_inputs.push_back(0);
            m_kernel_operand_kinds.clear();
            m_kernel_operand_arg_indices.clear();
            for (size_t i = 0; i < 1 + plan.split_sizes.size(); ++i) {
                m_kernel_operand_kinds.push_back(1);
                m_kernel_operand_arg_indices.push_back(static_cast<int32_t>(i));
            }
            std::string log;
            try {
                m_kernel = compile_kernel(src, &log);
                compiled_split = static_cast<bool>(m_kernel);
            } catch (const std::exception& e) {
                if (gfx_log_debug_enabled()) {
                    gfx_log_debug("MLIRSplit") << "Specialized split kernel compile failed: " << e.what();
                }
                compiled_split = false;
            }
            if (compiled_split) {
                m_last_input_shape = stage_input_shape;
            } else {
                m_kernel.reset();
                m_kernel_operand_kinds.clear();
                m_kernel_operand_arg_indices.clear();
                m_kernel_inputs.clear();
                m_kernel_input_arg_count = 0;
            }
        }
        if (!compiled_split && (m_last_input_shape != stage_input_shape || !m_kernel)) {
            auto& ctx = gfx_mlir_context();
            auto module = build_mlir_split_from_node(m_node,
                                                     ctx,
                                                     stage_input_shape,
                                                     input_transform(0));
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
            m_last_input_shape = stage_input_shape;
        }
    } else {
        for (size_t i = 0; i < outputs.size(); ++i) {
            ensure_output_shape(i, outputs[i]);
        }
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        GpuTensor* out = outputs[i];
        if (!out) {
            continue;
        }
        if (out->shape.empty()) {
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
        if (!out->buf.valid()) {
            OPENVINO_THROW("GFX MLIR: output buffer is not allocated for stage ", m_name);
        }
        if (out->buf.size < out_bytes) {
            OPENVINO_THROW("GFX MLIR: output buffer too small for stage ",
                           m_name,
                           " (need ",
                           out_bytes,
                           ", have ",
                           out->buf.size,
                           ")");
        }
        out->expected_type = out_type;
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
        return;
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

    if (m_type == "Concat") {
        auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(m_node);
        OPENVINO_ASSERT(concat, "GFX MLIR: expected v0::Concat for stage ", m_name);
        OPENVINO_ASSERT(!outputs.empty() && outputs.front() && outputs.front()->buf.valid(),
                        "GFX MLIR: missing concat output buffer for stage ",
                        m_name);

        struct ConcatParams {
            uint32_t outer = 0;
            uint32_t inner = 0;
            uint32_t axis_offset = 0;
            uint32_t axis_len = 0;
            uint32_t axis_total = 0;
        };

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

            ConcatParams params{};
            params.outer = static_cast<uint32_t>(outer);
            params.inner = static_cast<uint32_t>(inner);
            params.axis_offset = static_cast<uint32_t>(axis_offset);
            params.axis_len = axis_len;
            params.axis_total = axis_total;

            const std::string key = m_name + "/concat_params/" + std::to_string(i);
            GpuBuffer buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
            OPENVINO_ASSERT(buf.valid(), "GFX MLIR: failed to wrap concat params for stage ", m_name);
            buf.owned = false;

            GpuTensor params_tensor;
            params_tensor.buf = buf;
            params_tensor.expected_type = ov::element::u8;
            params_tensor.shape = ov::Shape{sizeof(params)};

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

    if (gfx_log_debug_enabled()) {
        gfx_log_debug("MLIRExec") << "Kernel args prep: scalars=" << m_kernel_scalar_args.size()
                                  << " inputs=" << m_kernel_inputs.size()
                                  << " outputs=" << outputs.size()
                                  << " kinds=" << m_kernel_operand_kinds.size();
    }
    // Conv3D uses params as extra input; keep output before params.
    auto maybe_set_conv3d_inputs = [&]() {
        if (m_type == "Convolution") {
            const auto shape = resolve_input_shape(0);
            if (shape.size() == 5) {
                m_kernel_input_arg_count = 2;  // data + weights
            }
        }
    };
    maybe_set_conv3d_inputs();

    std::string arg_map;
    std::vector<GpuTensor> empty_extras;
    const std::vector<GpuTensor>* extras = &m_kernel_extra_inputs;
    if (m_kernel_operand_kinds.empty()) {
        const size_t expected_inputs = m_kernel_input_arg_count ? m_kernel_input_arg_count : m_kernel_inputs.size();
        if (expected_inputs <= m_kernel_inputs.size()) {
            extras = &empty_extras;  // Kernel does not expect extra buffers; drop them.
        }
    }

    auto bundle = build_kernel_args_from_metadata(
        m_kernel_operand_kinds,
        m_kernel_operand_arg_indices,
        m_kernel_scalar_args,
        m_kernel_inputs,
        m_kernel_input_arg_count,
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

    m_kernel->execute(command_buffer, dispatch, bound_args, hooks_ptr);

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

void MlirStage::set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) {
    m_outputs.clear();
    m_outputs.reserve(outputs.size());
    for (const auto& o : outputs) {
        m_outputs.push_back(o.get());
    }
    if (!m_outputs.empty()) {
        m_output = m_outputs.front();
    }
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
    dst.m_output_shape = m_output_shape;
    dst.m_last_input_shape = m_last_input_shape;
    dst.m_input_transforms = m_input_transforms;
    dst.m_kernel_inputs = m_kernel_inputs;
    dst.m_kernel_input_arg_count = m_kernel_input_arg_count;
    dst.m_const_buffers = m_const_buffers;
    dst.m_parallel_cfg = m_parallel_cfg;
    dst.m_force_single_dispatch = m_force_single_dispatch;
    dst.m_kernel_scalar_args = m_kernel_scalar_args;
    dst.m_kernel_operand_kinds = m_kernel_operand_kinds;
    dst.m_kernel_operand_arg_indices = m_kernel_operand_arg_indices;
    dst.m_has_activation = m_has_activation;
    dst.m_activation = m_activation;
    dst.m_activation_alpha = m_activation_alpha;
    dst.m_has_bn = m_has_bn;
    dst.m_bn_params = m_bn_params;
    dst.m_has_bias = m_has_bias;
    dst.m_bias_params = m_bias_params;
    dst.m_kernel_extra_inputs = m_kernel_extra_inputs;
    dst.m_bias_f16 = m_bias_f16;
}

bool MlirStage::is_conv_like() const {
    return m_type == "Convolution" || m_type == "GroupConvolution";
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
}

}  // namespace gfx_plugin
}  // namespace ov
