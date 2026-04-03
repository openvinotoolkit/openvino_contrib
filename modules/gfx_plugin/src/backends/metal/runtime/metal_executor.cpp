// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/metal/runtime/metal_executor.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <sstream>

#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/profiling/profiler.hpp"
#include "kernel_ir/gfx_codegen_desc.hpp"
#include "mlir/codegen_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/validation_util.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/floor_mod.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reverse.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/space_to_depth.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

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
    std::vector<uint32_t> steps;
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
            OPENVINO_ASSERT(steps[i] > 0, "GFX Metal Slice: only positive steps supported");
            const auto dim = static_cast<int64_t>(in_shape[static_cast<size_t>(axis)]);
            meta.starts[static_cast<size_t>(axis)] =
                static_cast<int32_t>(normalize_slice_index(starts[i], dim, true));
            meta.steps[static_cast<size_t>(axis)] = static_cast<uint32_t>(steps[i]);
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
        OPENVINO_ASSERT(step > 0, "GFX Metal Slice: StridedSlice only positive steps supported");
        int64_t start = axis < begin.size() ? begin[axis] : 0;
        int64_t finish = axis < end.size() ? end[axis] : dim;
        start = masked_begin ? 0 : normalize_slice_index(start, dim, true);
        finish = masked_end ? dim : normalize_slice_index(finish, dim, false);
        (void)finish;
        meta.starts[axis] = static_cast<int32_t>(start);
        meta.steps[axis] = static_cast<uint32_t>(step);
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
    ss << "constant uint STEPS_C[" << rank << "] = {";
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
    ss << "    uint in_off = 0;\n";
    ss << "    for (int d = (int)RANK_C - 1; d >= 0; --d) {\n";
    ss << "        uint coord = idx % OUT_SHAPE_C[d];\n";
    ss << "        idx /= OUT_SHAPE_C[d];\n";
    ss << "        in_off += (uint)((int)STARTS_C[d] + (int)(coord * STEPS_C[d])) * IN_STRIDE_C[d];\n";
    ss << "    }\n";
    ss << "    C[gid] = A[in_off];\n";
    ss << "}\n";
    return ss.str();
}

mlir::ArrayAttr make_i32_array_attr(mlir::OpBuilder& b, const std::vector<int32_t>& vals) {
    llvm::SmallVector<mlir::Attribute, 16> attrs;
    attrs.reserve(vals.size());
    for (auto v : vals) {
        attrs.push_back(b.getI32IntegerAttr(v));
    }
    return b.getArrayAttr(attrs);
}

void annotate_module_operands(mlir::ModuleOp module,
                              const std::vector<int32_t>& kinds,
                              const std::vector<int32_t>& arg_indices,
                              const std::vector<int32_t>& scalar_values) {
    if (!module) {
        return;
    }
    mlir::OpBuilder b(module.getContext());
    module->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr(b, kinds));
    module->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr(b, arg_indices));
    if (!scalar_values.empty()) {
        module->setAttr("gfx.kernel_scalar_values", make_i32_array_attr(b, scalar_values));
    }
}

inline size_t shape_size(const ov::Shape& s) {
    return ov::shape_size(s);
}

std::vector<int64_t> to_i64_shape(const ov::Shape& s) {
    std::vector<int64_t> v;
    v.reserve(s.size());
    for (auto d : s) v.push_back(static_cast<int64_t>(d));
    return v;
}

std::vector<int64_t> make_strides(const ov::Shape& s) {
    const size_t rank = s.size();
    std::vector<int64_t> strides(rank, 1);
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * static_cast<int64_t>(s[i + 1]);
    }
    return strides;
}

std::string generate_bias_broadcast_add_msl(const std::shared_ptr<const ov::Node>& node) {
    OPENVINO_ASSERT(node, "GFX Metal: bias-broadcast add node is null");
    OPENVINO_ASSERT(is_bias_broadcast_add(node), "GFX Metal: expected bias-broadcast Add");
    const auto out_shape = node->get_output_shape(0);
    OPENVINO_ASSERT(out_shape.size() == 4, "GFX Metal: bias-broadcast Add expects rank-4 output");
    const auto scalar_t = msl_type_from_element(node->get_output_element_type(0));
    const uint32_t total = static_cast<uint32_t>(shape_size(out_shape));
    const uint32_t channels = static_cast<uint32_t>(out_shape[1]);
    const uint32_t hw = static_cast<uint32_t>(out_shape[2] * out_shape[3]);

    std::ostringstream ss;
    ss << "#include <metal_stdlib>\nusing namespace metal;\n";
    ss << "kernel void binary_bias_add(\n";
    ss << "  device const " << scalar_t << "* A [[buffer(0)]],\n";
    ss << "  device const " << scalar_t << "* B [[buffer(1)]],\n";
    ss << "  device " << scalar_t << "* C [[buffer(2)]],\n";
    ss << "  uint gid [[thread_position_in_grid]]) {\n";
    ss << "  if (gid >= " << total << "u) return;\n";
    ss << "  uint c = (gid / " << hw << "u) % " << channels << "u;\n";
    ss << "  C[gid] = A[gid] + B[c];\n";
    ss << "}\n";
    return ss.str();
}

void fill_broadcast_strides(const ov::Shape& out,
                            const ov::Shape& in,
                            std::vector<int64_t>& strides) {
    const size_t rank_out = out.size();
    const size_t rank_in = in.size();
    strides.assign(rank_out, 0);
    auto in_strides = make_strides(in);
    for (size_t i = 0; i < rank_out; ++i) {
        const size_t out_dim = out[rank_out - 1 - i];
        const size_t in_dim = (i < rank_in) ? in[rank_in - 1 - i] : 1;
        const size_t in_stride = (i < rank_in) ? in_strides[rank_in - 1 - i] : 0;
        if (in_dim == out_dim) {
            strides[rank_out - 1 - i] = static_cast<int64_t>(in_stride);
        } else if (in_dim == 1) {
            strides[rank_out - 1 - i] = 0;
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

EltwiseKind eltwise_kind_from_node(const ov::Node& node) {
    const std::string t = node.get_type_name();
    if (t == "Add") return EltwiseKind::Add;
    if (t == "Subtract") return EltwiseKind::Sub;
    if (t == "Multiply") return EltwiseKind::Mul;
    if (t == "Divide") return EltwiseKind::Div;
    if (t == "Power") return EltwiseKind::Pow;
    if (t == "Mod") return EltwiseKind::Mod;
    if (t == "FloorMod") return EltwiseKind::FloorMod;
    if (t == "PRelu") return EltwiseKind::Prelu;
    if (t == "SquaredDifference") return EltwiseKind::SquaredDiff;
    if (t == "Minimum") return EltwiseKind::Min;
    if (t == "Maximum") return EltwiseKind::Max;
    if (t == "LogicalAnd") return EltwiseKind::LogicalAnd;
    if (t == "LogicalOr") return EltwiseKind::LogicalOr;
    if (t == "LogicalXor") return EltwiseKind::LogicalXor;
    if (t == "Equal") return EltwiseKind::Equal;
    if (t == "NotEqual") return EltwiseKind::NotEqual;
    if (t == "Less") return EltwiseKind::Less;
    if (t == "Greater") return EltwiseKind::Greater;
    if (t == "LessEqual") return EltwiseKind::LessEqual;
    if (t == "GreaterEqual") return EltwiseKind::GreaterEqual;
    OPENVINO_THROW("GFX Metal: unsupported eltwise op ", t);
}

ActivationKind activation_kind_from_node(const ov::Node& node) {
    const std::string t = node.get_type_name();
    if (t == "Relu") return ActivationKind::Relu;
    if (t == "Sigmoid") return ActivationKind::Sigmoid;
    if (t == "Tanh") return ActivationKind::Tanh;
    if (t == "Elu") return ActivationKind::Elu;
    if (t == "Gelu") return ActivationKind::Gelu;
    if (t == "Swish") return ActivationKind::Swish;
    if (t == "HSwish") return ActivationKind::HSwish;
    if (t == "HSigmoid") return ActivationKind::HSigmoid;
    if (t == "SoftPlus") return ActivationKind::SoftPlus;
    if (t == "Mish") return ActivationKind::Mish;
    if (t == "SoftSign") return ActivationKind::SoftSign;
    if (t == "Abs") return ActivationKind::Abs;
    if (t == "Sign") return ActivationKind::Sign;
    if (t == "LogicalNot") return ActivationKind::LogicalNot;
    if (t == "Clamp") return ActivationKind::Clamp;
    if (t == "Exp") return ActivationKind::Exp;
    if (t == "Log") return ActivationKind::Log;
    if (t == "Sqrt") return ActivationKind::Sqrt;
    if (t == "Floor") return ActivationKind::Floor;
    if (t == "Ceiling" || t == "Ceil") return ActivationKind::Ceil;
    if (t == "Negative") return ActivationKind::Negative;
    if (t == "Sin") return ActivationKind::Sin;
    if (t == "Cos") return ActivationKind::Cos;
    if (t == "Tan") return ActivationKind::Tan;
    if (t == "Erf") return ActivationKind::Erf;
    if (t == "Asin") return ActivationKind::Asin;
    if (t == "Acos") return ActivationKind::Acos;
    if (t == "Atan") return ActivationKind::Atan;
    if (t == "Asinh") return ActivationKind::Asinh;
    if (t == "Acosh") return ActivationKind::Acosh;
    if (t == "Atanh") return ActivationKind::Atanh;
    if (t == "Sinh") return ActivationKind::Sinh;
    if (t == "Cosh") return ActivationKind::Cosh;
    if (t == "Round") return ActivationKind::RoundAway;
    OPENVINO_THROW("GFX Metal: unsupported unary activation op ", t);
}

ReduceKind reduce_kind_from_node(const ov::Node& node) {
    const std::string t = node.get_type_name();
    if (t == "ReduceSum") return ReduceKind::Sum;
    if (t == "ReduceMean") return ReduceKind::Mean;
    if (t == "ReduceMax") return ReduceKind::Max;
    if (t == "ReduceMin") return ReduceKind::Min;
    if (t == "ReduceProd") return ReduceKind::Prod;
    if (t == "ReduceL1") return ReduceKind::L1;
    if (t == "ReduceL2") return ReduceKind::L2;
    OPENVINO_THROW("GFX Metal: unsupported reduce op ", t);
}

void attach_msl_generator(const std::shared_ptr<const ov::Node>& node,
                          KernelSource& src) {
    auto set_desc = [&](auto&& desc, const char* entry = nullptr) {
        using DescT = std::decay_t<decltype(desc)>;
        if (entry && !src.entry_point.empty()) {
            src.entry_point = entry;
        } else if (entry) {
            src.entry_point = entry;
        }
        src.msl_generator = [desc](mlir::ModuleOp module) mutable {
            return generate_msl_from_mlir(module, desc);
        };
    };

    const std::string type = node ? node->get_type_name() : "";

    if (auto conv = std::dynamic_pointer_cast<const ov::op::v1::Convolution>(node)) {
        const auto in_shape = conv->get_input_shape(0);
        if (in_shape.size() == 5) {
            // 3D convolution
            Conv3DCodegenDesc d{};
            const auto w_shape = conv->get_input_shape(1);
            d.element_type = conv->get_output_element_type(0);
            d.N = static_cast<uint32_t>(in_shape.at(0));
            d.C_in = static_cast<uint32_t>(in_shape.at(1));
            d.D = static_cast<uint32_t>(in_shape.at(2));
            d.H = static_cast<uint32_t>(in_shape.at(3));
            d.W = static_cast<uint32_t>(in_shape.at(4));
            d.C_out = static_cast<uint32_t>(w_shape.at(0));
            d.kD = static_cast<uint32_t>(w_shape.at(2));
            d.kH = static_cast<uint32_t>(w_shape.at(3));
            d.kW = static_cast<uint32_t>(w_shape.at(4));
            d.strideD = static_cast<uint32_t>(conv->get_strides().at(0));
            d.strideH = static_cast<uint32_t>(conv->get_strides().at(1));
            d.strideW = static_cast<uint32_t>(conv->get_strides().at(2));
            d.dilationD = static_cast<uint32_t>(conv->get_dilations().at(0));
            d.dilationH = static_cast<uint32_t>(conv->get_dilations().at(1));
            d.dilationW = static_cast<uint32_t>(conv->get_dilations().at(2));
            d.padFront = static_cast<uint32_t>(conv->get_pads_begin().at(0));
            d.padTop = static_cast<uint32_t>(conv->get_pads_begin().at(1));
            d.padLeft = static_cast<uint32_t>(conv->get_pads_begin().at(2));
            d.padBack = static_cast<uint32_t>(conv->get_pads_end().at(0));
            d.padBottom = static_cast<uint32_t>(conv->get_pads_end().at(1));
            d.padRight = static_cast<uint32_t>(conv->get_pads_end().at(2));
            const auto out_shape = conv->get_output_shape(0);
            d.outD = static_cast<uint32_t>(out_shape.at(2));
            d.outH = static_cast<uint32_t>(out_shape.at(3));
            d.outW = static_cast<uint32_t>(out_shape.at(4));
            set_desc(d, "conv3d_kernel");
            src.signature.arg_count = 4;  // in, weights, out, params
            if (src.module) {
                mlir::OpBuilder b(src.module.getContext());
                std::vector<int32_t> kinds{1, 1, 1, 1};
                std::vector<int32_t> arg_idx{0, 1, 2, 3};
                annotate_module_operands(src.module, kinds, arg_idx, {});
            }
            return;
        }
        Conv2DCodegenDesc d{};
        const auto w_shape = conv->get_input_shape(1);
        d.element_type = conv->get_output_element_type(0);
        d.input_type = conv->get_input_element_type(0);
        d.weight_type = conv->get_input_element_type(1);
        d.output_type = conv->get_output_element_type(0);
        d.N = static_cast<uint32_t>(in_shape.at(0));
        d.C_in = static_cast<uint32_t>(in_shape.at(1));
        d.H = static_cast<uint32_t>(in_shape.at(2));
        d.W = static_cast<uint32_t>(in_shape.at(3));
        d.C_out = static_cast<uint32_t>(w_shape.at(0));
        const uint32_t cin_pg = static_cast<uint32_t>(w_shape.at(1));
        d.groups = (cin_pg && d.C_in % cin_pg == 0) ? d.C_in / cin_pg : 1;
        d.C_in_pg = cin_pg;
        d.C_out_pg = d.groups ? d.C_out / d.groups : d.C_out;
        d.kH = static_cast<uint32_t>(w_shape.at(2));
        d.kW = static_cast<uint32_t>(w_shape.at(3));
        d.strideH = static_cast<uint32_t>(conv->get_strides().at(0));
        d.strideW = static_cast<uint32_t>(conv->get_strides().at(1));
        d.dilationH = static_cast<uint32_t>(conv->get_dilations().at(0));
        d.dilationW = static_cast<uint32_t>(conv->get_dilations().at(1));
        d.padTop = static_cast<uint32_t>(conv->get_pads_begin().at(0));
        d.padLeft = static_cast<uint32_t>(conv->get_pads_begin().at(1));
        d.padBottom = static_cast<uint32_t>(conv->get_pads_end().at(0));
        d.padRight = static_cast<uint32_t>(conv->get_pads_end().at(1));
        set_desc(d, "conv2d_kernel");
        src.signature.arg_count = 9;  // in, w, bias, gamma, beta, mean, var, params, out
        if (src.module) {
            mlir::OpBuilder b(src.module.getContext());
            std::vector<int32_t> kinds{1, 1, 1, 1, 1, 1, 1, 1, 1};
            std::vector<int32_t> arg_idx{0, 1, 2, 3, 4, 5, 6, 7, 8};
            annotate_module_operands(src.module, kinds, arg_idx, {});
        }
        return;
    }

    if (auto gconv = std::dynamic_pointer_cast<const ov::op::v1::GroupConvolution>(node)) {
        Conv2DCodegenDesc d{};
        const auto in_shape = gconv->get_input_shape(0);
        const auto w_shape = gconv->get_input_shape(1);  // [G, O_pg, I_pg, kH, kW]
        d.element_type = gconv->get_output_element_type(0);
        d.input_type = gconv->get_input_element_type(0);
        d.weight_type = gconv->get_input_element_type(1);
        d.output_type = gconv->get_output_element_type(0);
        d.N = static_cast<uint32_t>(in_shape.at(0));
        d.C_in = static_cast<uint32_t>(in_shape.at(1));
        d.H = static_cast<uint32_t>(in_shape.at(2));
        d.W = static_cast<uint32_t>(in_shape.at(3));
        d.groups = static_cast<uint32_t>(w_shape.at(0));
        d.C_out_pg = static_cast<uint32_t>(w_shape.at(1));
        d.C_in_pg = static_cast<uint32_t>(w_shape.at(2));
        d.C_out = d.groups * d.C_out_pg;
        d.kH = static_cast<uint32_t>(w_shape.at(3));
        d.kW = static_cast<uint32_t>(w_shape.at(4));
        d.strideH = static_cast<uint32_t>(gconv->get_strides().at(0));
        d.strideW = static_cast<uint32_t>(gconv->get_strides().at(1));
        d.dilationH = static_cast<uint32_t>(gconv->get_dilations().at(0));
        d.dilationW = static_cast<uint32_t>(gconv->get_dilations().at(1));
        d.padTop = static_cast<uint32_t>(gconv->get_pads_begin().at(0));
        d.padLeft = static_cast<uint32_t>(gconv->get_pads_begin().at(1));
        d.padBottom = static_cast<uint32_t>(gconv->get_pads_end().at(0));
        d.padRight = static_cast<uint32_t>(gconv->get_pads_end().at(1));
        set_desc(d, "conv2d_kernel");
        src.signature.arg_count = 9;  // in, w, bias, gamma, beta, mean, var, params, out
        if (src.module) {
            mlir::OpBuilder b(src.module.getContext());
            std::vector<int32_t> kinds{1, 1, 1, 1, 1, 1, 1, 1, 1};
            std::vector<int32_t> arg_idx{0, 1, 2, 3, 4, 5, 6, 7, 8};
            annotate_module_operands(src.module, kinds, arg_idx, {});
        }
        return;
    }

    if (auto mm = std::dynamic_pointer_cast<const ov::op::v0::MatMul>(node)) {
        MatMulCodegenDesc d{};
        const auto out_shape = mm->get_output_shape(0);
        const size_t rank = out_shape.size();
        d.element_type = mm->get_output_element_type(0);
        d.input_a_type = mm->get_input_element_type(0);
        d.input_b_type = mm->get_input_element_type(1);
        d.output_type = mm->get_output_element_type(0);
        d.a_transpose = mm->get_transpose_a();
        d.b_transpose = mm->get_transpose_b();
        d.M = static_cast<int64_t>(out_shape[rank - 2]);
        d.N = static_cast<int64_t>(out_shape[rank - 1]);
        const auto a_shape = mm->get_input_shape(0);
        d.K = static_cast<int64_t>(d.a_transpose ? a_shape[rank - 2] : a_shape[rank - 1]);
        d.batch_a = static_cast<int64_t>(shape_size(a_shape) / static_cast<uint64_t>(d.M * d.K));
        const auto b_shape = mm->get_input_shape(1);
        d.batch_b = static_cast<int64_t>(shape_size(b_shape) / static_cast<uint64_t>(d.K * d.N));
        d.b_is_nk_layout = d.b_transpose;
        d.batch = static_cast<int64_t>(shape_size(out_shape) / (d.M * d.N));
        set_desc(d, "matmul_kernel");
        src.signature.arg_count = 3;  // A, B, C
        if (src.module) {
            mlir::OpBuilder b(src.module.getContext());
            std::vector<int32_t> kinds{1, 1, 1};
            std::vector<int32_t> arg_idx{0, 1, 2};
            annotate_module_operands(src.module, kinds, arg_idx, {});
        }
        return;
    }

    if (auto prelu = std::dynamic_pointer_cast<const ov::op::v0::PRelu>(node)) {
        EltwiseCodegenDesc d{};
        d.element_type = prelu->get_output_element_type(0);
        d.eltwise_kind = EltwiseKind::Prelu;
        const auto out_shape = prelu->get_output_shape(0);
        d.out_shape = to_i64_shape(out_shape);
        d.num_elements = static_cast<uint32_t>(shape_size(out_shape));
        const auto a_shape = prelu->get_input_shape(0);
        const auto b_shape = prelu->get_input_shape(1);
        d.is_broadcast = (a_shape != b_shape);
        fill_broadcast_strides(out_shape, a_shape, d.stride0);
        fill_broadcast_strides(out_shape, b_shape, d.stride1);
        set_desc(d, "eltwise_kernel");
        src.signature.arg_count = 8;
        if (src.module) {
            mlir::OpBuilder b(src.module.getContext());
            std::vector<int32_t> kinds{1, 1, 1, 0, 0, 1, 1, 1};
            std::vector<int32_t> arg_idx{0, 1, 5, -1, -1, 2, 3, 4};
            std::vector<int32_t> scalars{static_cast<int32_t>(d.num_elements),
                                         static_cast<int32_t>(out_shape.size())};
            annotate_module_operands(src.module, kinds, arg_idx, scalars);
        }
        return;
    }

    if (auto sm1 = std::dynamic_pointer_cast<const ov::op::v1::Softmax>(node)) {
        SoftmaxCodegenDesc d{};
        const auto in_shape = sm1->get_input_shape(0);
        d.element_type = sm1->get_output_element_type(0);
        const auto dims = compute_softmax_dims(in_shape, sm1->get_axis(), "GFX Metal");
        d.rows = static_cast<int64_t>(dims.rows);
        d.cols = static_cast<int64_t>(dims.axis_len);
        d.inner = static_cast<int64_t>(dims.inner);
        d.log_softmax = false;
        set_desc(d, "softmax_kernel");
        return;
    }
    if (auto sm8 = std::dynamic_pointer_cast<const ov::op::v8::Softmax>(node)) {
        SoftmaxCodegenDesc d{};
        const auto in_shape = sm8->get_input_shape(0);
        d.element_type = sm8->get_output_element_type(0);
        const auto dims = compute_softmax_dims(in_shape, sm8->get_axis(), "GFX Metal");
        d.rows = static_cast<int64_t>(dims.rows);
        d.cols = static_cast<int64_t>(dims.axis_len);
        d.inner = static_cast<int64_t>(dims.inner);
        d.log_softmax = false;
        set_desc(d, "softmax_kernel");
        return;
    }
    if (auto ls = std::dynamic_pointer_cast<const ov::op::v5::LogSoftmax>(node)) {
        SoftmaxCodegenDesc d{};
        const auto in_shape = ls->get_input_shape(0);
        d.element_type = ls->get_output_element_type(0);
        const auto dims = compute_softmax_dims(in_shape, ls->get_axis(), "GFX Metal");
        d.rows = static_cast<int64_t>(dims.rows);
        d.cols = static_cast<int64_t>(dims.axis_len);
        d.inner = static_cast<int64_t>(dims.inner);
        d.log_softmax = true;
        set_desc(d, "softmax_kernel");
        return;
    }

    // Unary activations
    if (type == "Relu" || type == "Sigmoid" || type == "Tanh" || type == "Elu" || type == "Gelu" ||
        type == "Swish" || type == "HSwish" || type == "HSigmoid" || type == "SoftPlus" ||
        type == "Mish" || type == "SoftSign" || type == "Abs" || type == "Sign" ||
        type == "Clamp" || type == "Exp" || type == "Log" || type == "Sqrt" || type == "Floor" ||
        type == "Ceiling" || type == "Negative" || type == "Sin" || type == "Cos" ||
        type == "Tan" || type == "Erf" || type == "Asin" || type == "Acos" || type == "Atan" ||
        type == "Asinh" || type == "Acosh" || type == "Atanh" || type == "Sinh" ||
        type == "Cosh" || type == "Round") {
        UnaryCodegenDesc d{};
        d.element_type = node->get_output_element_type(0);
        d.activation = activation_kind_from_node(*node);
        d.alpha = 0.0f;
        if (auto elu = std::dynamic_pointer_cast<const ov::op::v0::Elu>(node)) {
            d.alpha = static_cast<float>(elu->get_alpha());
        }
        if (auto clamp = std::dynamic_pointer_cast<const ov::op::v0::Clamp>(node)) {
            d.clamp_min = clamp->get_min();
            d.clamp_max = clamp->get_max();
        }
        set_desc(d, "unary_kernel");
        src.signature.arg_count = 3;
        const auto out_shape = node->get_output_shape(0);
        const int32_t num_elems = static_cast<int32_t>(shape_size(out_shape));
        const std::vector<int32_t> kinds{1, 1, 0};
        const std::vector<int32_t> arg_idx{0, 1, -1};
        const std::vector<int32_t> scalars{num_elems};
        annotate_module_operands(src.module, kinds, arg_idx, scalars);
        return;
    }

    if (auto pool = std::dynamic_pointer_cast<const ov::op::v1::MaxPool>(node)) {
        Pool2DCodegenDesc d{};
        const auto in = pool->get_input_shape(0);
        const auto out = pool->get_output_shape(0);
        d.element_type = pool->get_output_element_type(0);
        d.N = static_cast<uint32_t>(in.at(0));
        d.C = static_cast<uint32_t>(in.at(1));
        d.H = static_cast<uint32_t>(in.at(2));
        d.W = static_cast<uint32_t>(in.at(3));
        d.kH = static_cast<uint32_t>(pool->get_kernel().at(0));
        d.kW = static_cast<uint32_t>(pool->get_kernel().at(1));
        d.strideH = static_cast<uint32_t>(pool->get_strides().at(0));
        d.strideW = static_cast<uint32_t>(pool->get_strides().at(1));
        d.padTop = static_cast<uint32_t>(pool->get_pads_begin().at(0));
        d.padLeft = static_cast<uint32_t>(pool->get_pads_begin().at(1));
        d.padBottom = static_cast<uint32_t>(pool->get_pads_end().at(0));
        d.padRight = static_cast<uint32_t>(pool->get_pads_end().at(1));
        d.outH = static_cast<uint32_t>(out.at(2));
        d.outW = static_cast<uint32_t>(out.at(3));
        d.is_avg = false;
        d.exclude_pad = true;
        set_desc(d, "pool2d_kernel");
        src.signature.arg_count = 3;  // in, params, out
        if (src.module) {
            mlir::OpBuilder b(src.module.getContext());
            std::vector<int32_t> kinds{1, 1, 1};
            std::vector<int32_t> arg_idx{0, 1, 2};
            annotate_module_operands(src.module, kinds, arg_idx, {});
        }
        return;
    }
    if (auto pool = std::dynamic_pointer_cast<const ov::op::v1::AvgPool>(node)) {
        Pool2DCodegenDesc d{};
        const auto in = pool->get_input_shape(0);
        const auto out = pool->get_output_shape(0);
        d.element_type = pool->get_output_element_type(0);
        d.N = static_cast<uint32_t>(in.at(0));
        d.C = static_cast<uint32_t>(in.at(1));
        d.H = static_cast<uint32_t>(in.at(2));
        d.W = static_cast<uint32_t>(in.at(3));
        d.kH = static_cast<uint32_t>(pool->get_kernel().at(0));
        d.kW = static_cast<uint32_t>(pool->get_kernel().at(1));
        d.strideH = static_cast<uint32_t>(pool->get_strides().at(0));
        d.strideW = static_cast<uint32_t>(pool->get_strides().at(1));
        d.padTop = static_cast<uint32_t>(pool->get_pads_begin().at(0));
        d.padLeft = static_cast<uint32_t>(pool->get_pads_begin().at(1));
        d.padBottom = static_cast<uint32_t>(pool->get_pads_end().at(0));
        d.padRight = static_cast<uint32_t>(pool->get_pads_end().at(1));
        d.outH = static_cast<uint32_t>(out.at(2));
        d.outW = static_cast<uint32_t>(out.at(3));
        d.is_avg = true;
        d.exclude_pad = pool->get_exclude_pad();
        set_desc(d, "pool2d_kernel");
        src.signature.arg_count = 3;  // in, params, out
        if (src.module) {
            mlir::OpBuilder b(src.module.getContext());
            std::vector<int32_t> kinds{1, 1, 1};
            std::vector<int32_t> arg_idx{0, 1, 2};
            annotate_module_operands(src.module, kinds, arg_idx, {});
        }
        return;
    }

    if (std::dynamic_pointer_cast<const ov::op::util::BinaryElementwiseArithmetic>(node) ||
        std::dynamic_pointer_cast<const ov::op::util::BinaryElementwiseLogical>(node) ||
        std::dynamic_pointer_cast<const ov::op::util::BinaryElementwiseComparison>(node)) {
        if (type == "Add" && is_bias_broadcast_add(node)) {
            src.entry_point = "binary_bias_add";
            src.signature.arg_count = 3;
            src.msl_generator = [node](mlir::ModuleOp) {
                return generate_bias_broadcast_add_msl(node);
            };
            return;
        }
        EltwiseCodegenDesc d{};
        d.element_type = node->get_output_element_type(0);
        d.eltwise_kind = eltwise_kind_from_node(*node);
        const auto out_shape = node->get_output_shape(0);
        d.out_shape = to_i64_shape(out_shape);
        d.num_elements = static_cast<uint32_t>(shape_size(out_shape));
        const auto a_shape = shape_from_entry_argument(src.module, 0, node->get_input_shape(0));
        const auto b_shape = shape_from_entry_argument(src.module, 1, node->get_input_shape(1));
        const auto perm0 = read_absorbed_input_permutation(src.module, 0);
        const auto perm1 = read_absorbed_input_permutation(src.module, 1);
        d.is_broadcast = !perm0.empty() || !perm1.empty() || (a_shape != b_shape) ||
                         (a_shape != out_shape) || (b_shape != out_shape);
        if (!perm0.empty()) {
            auto strides = compute_permuted_broadcast_element_strides(a_shape,
                                                                      node->get_input_shape(0),
                                                                      perm0,
                                                                      out_shape,
                                                                      "GFX Metal");
            d.stride0.assign(strides.begin(), strides.end());
        } else {
            fill_broadcast_strides(out_shape, a_shape, d.stride0);
        }
        if (!perm1.empty()) {
            auto strides = compute_permuted_broadcast_element_strides(b_shape,
                                                                      node->get_input_shape(1),
                                                                      perm1,
                                                                      out_shape,
                                                                      "GFX Metal");
            d.stride1.assign(strides.begin(), strides.end());
        } else {
            fill_broadcast_strides(out_shape, b_shape, d.stride1);
        }
        // Keep entry point aligned with MLIR metadata and provide operand mapping
        // so kernel args match MSL signature:
        //   buffer0=A, buffer1=B, buffer2=Out, scalar3=NUM_ELEMS, scalar4=RANK,
        //   buffer5=out_dims[], buffer6=stride0[], buffer7=stride1[].
        set_desc(d, "eltwise_kernel");
        src.signature.arg_count = 8;
        if (src.module) {
            std::vector<int32_t> kinds   = {1, 1, 1, 0, 0, 1, 1, 1};
            std::vector<int32_t> arg_idx = {0, 1, 5, -1, -1, 2, 3, 4};
            std::vector<int32_t> scalars = {static_cast<int32_t>(d.num_elements),
                                            static_cast<int32_t>(out_shape.size())};
            annotate_module_operands(src.module, kinds, arg_idx, scalars);
        }
        return;
    }

    if (std::dynamic_pointer_cast<const ov::op::v1::ReduceSum>(node) ||
        std::dynamic_pointer_cast<const ov::op::v1::ReduceMean>(node) ||
        std::dynamic_pointer_cast<const ov::op::v1::ReduceMax>(node) ||
        std::dynamic_pointer_cast<const ov::op::v1::ReduceMin>(node) ||
        std::dynamic_pointer_cast<const ov::op::v1::ReduceProd>(node) ||
        std::dynamic_pointer_cast<const ov::op::v4::ReduceL1>(node) ||
        std::dynamic_pointer_cast<const ov::op::v4::ReduceL2>(node)) {
        ReduceCodegenDesc d{};
        d.element_type = node->get_output_element_type(0);
        d.kind = reduce_kind_from_node(*node);
        set_desc(d, "reduce_kernel");
        return;
    }

    if (auto cat = std::dynamic_pointer_cast<const ov::op::v0::Concat>(node)) {
        ConcatCodegenDesc d{};
        const size_t axis = static_cast<size_t>(cat->get_axis());
        const auto out = cat->get_output_shape(0);
        uint64_t inner = 1, outer = 1;
        for (size_t i = axis + 1; i < out.size(); ++i) inner *= out[i];
        for (size_t i = 0; i < axis; ++i) outer *= out[i];
        d.inner = inner;
        d.outer = outer;
        d.axis_total = out[axis];
        set_desc(d, "concat_kernel");
        return;
    }

    if (auto s = std::dynamic_pointer_cast<const ov::op::v1::Split>(node)) {
        SplitCodegenDesc d{};
        auto axis_c = ov::as_type_ptr<const ov::op::v0::Constant>(s->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_c, "Split axis must be constant");
        d.axis = axis_c->cast_vector<int64_t>().at(0);
        const auto in_shape = s->get_input_shape(0);
        const auto out_shape = s->get_output_shape(0);
        d.input_shape = to_i64_shape(in_shape);
        d.source_input_shape = to_i64_shape(shape_from_entry_argument(src.module, 0, in_shape));
        d.input_permutation = read_absorbed_input_permutation(src.module, 0);
        const size_t axis = static_cast<size_t>(d.axis < 0 ? d.axis + in_shape.size() : d.axis);
        d.split_sizes.assign(s->get_output_size(), out_shape[axis]);
        uint64_t inner = 1, outer = 1;
        for (size_t i = axis + 1; i < in_shape.size(); ++i) inner *= in_shape[i];
        for (size_t i = 0; i < axis; ++i) outer *= in_shape[i];
        d.inner = inner;
        d.outer = outer;
        set_desc(d, "split_kernel");
        return;
    }
    if (auto vs = std::dynamic_pointer_cast<const ov::op::v1::VariadicSplit>(node)) {
        SplitCodegenDesc d{};
        auto axis_c = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_c, "VariadicSplit axis must be constant");
        d.axis = axis_c->cast_vector<int64_t>().at(0);
        const auto in_shape = vs->get_input_shape(0);
        d.input_shape = to_i64_shape(in_shape);
        d.source_input_shape = to_i64_shape(shape_from_entry_argument(src.module, 0, in_shape));
        d.input_permutation = read_absorbed_input_permutation(src.module, 0);
        auto lengths_c = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(lengths_c, "VariadicSplit lengths must be constant");
        auto lengths = lengths_c->cast_vector<int64_t>();
        d.split_sizes.assign(lengths.begin(), lengths.end());
        const size_t axis = static_cast<size_t>(d.axis < 0 ? d.axis + in_shape.size() : d.axis);
        uint64_t inner = 1, outer = 1;
        for (size_t i = axis + 1; i < in_shape.size(); ++i) inner *= in_shape[i];
        for (size_t i = 0; i < axis; ++i) outer *= in_shape[i];
        d.inner = inner;
        d.outer = outer;
        set_desc(d, "split_kernel");
        return;
    }

    if (auto d2s = std::dynamic_pointer_cast<const ov::op::v0::DepthToSpace>(node)) {
        DepthToSpaceCodegenDesc d{};
        const auto in = d2s->get_input_shape(0);
        const auto out = d2s->get_output_shape(0);
        d.element_type = d2s->get_output_element_type(0);
        d.N = in[0];
        d.C = in[1];
        d.H = in[2];
        d.W = in[3];
        d.block = static_cast<uint32_t>(d2s->get_block_size());
        d.C_out = out[1];
        d.H_out = out[2];
        d.W_out = out[3];
        d.mode = d2s->get_mode() == ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST ? 0 : 1;
        d.total = static_cast<uint32_t>(shape_size(out));
        set_desc(d, "depth_to_space_kernel");
        return;
    }
    if (auto s2d = std::dynamic_pointer_cast<const ov::op::v0::SpaceToDepth>(node)) {
        SpaceToDepthCodegenDesc d{};
        const auto in = s2d->get_input_shape(0);
        const auto out = s2d->get_output_shape(0);
        d.element_type = s2d->get_output_element_type(0);
        d.N = in[0];
        d.C = in[1];
        d.H = in[2];
        d.W = in[3];
        d.block = static_cast<uint32_t>(s2d->get_block_size());
        d.C_out = out[1];
        d.H_out = out[2];
        d.W_out = out[3];
        d.mode = s2d->get_mode() == ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST ? 0 : 1;
        d.total = static_cast<uint32_t>(shape_size(out));
        set_desc(d, "space_to_depth_kernel");
        return;
    }

    if (auto t = std::dynamic_pointer_cast<const ov::op::v1::Transpose>(node)) {
        TransposeCodegenDesc d{};
        d.element_type = t->get_output_element_type(0);
        auto perm_c = ov::as_type_ptr<const ov::op::v0::Constant>(t->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(perm_c, "Transpose perm must be constant");
        auto perm = perm_c->cast_vector<int64_t>();
        const auto in = t->get_input_shape(0);
        const auto out = t->get_output_shape(0);
        for (auto v : in) d.in_shape.push_back(static_cast<uint32_t>(v));
        for (auto v : out) d.out_shape.push_back(static_cast<uint32_t>(v));
        for (auto v : perm) d.perm.push_back(static_cast<uint32_t>(v));
        set_desc(d, "transpose_kernel");
        return;
    }

    if (auto cvt = std::dynamic_pointer_cast<const ov::op::v0::Convert>(node)) {
        ConvertCodegenDesc d{};
        d.src_type = cvt->get_input_element_type(0);
        d.dst_type = cvt->get_output_element_type(0);
        d.element_type = d.dst_type == ov::element::dynamic ? ov::element::f32 : d.dst_type;
        set_desc(d, "convert_kernel");
        return;
    }

    if (std::dynamic_pointer_cast<const ov::op::v0::ShapeOf>(node)) {
        ShapeOfCodegenDesc d{};
        d.rank = static_cast<uint32_t>(node->get_input_shape(0).size());
        set_desc(d, "shapeof_kernel");
        return;
    }

    if (auto pad = std::dynamic_pointer_cast<const ov::op::v1::Pad>(node)) {
        PadCodegenDesc d{};
        d.element_type = pad->get_output_element_type(0);
        if (auto c = ov::as_type_ptr<const ov::op::v0::Constant>(pad->input_value(3).get_node_shared_ptr())) {
            if (c->get_element_type().is_real()) d.pad_value = c->cast_vector<double>()[0];
            else if (c->get_element_type().is_integral_number()) d.pad_value = c->cast_vector<int64_t>()[0];
        }
        set_desc(d, "pad_kernel");
        return;
    }

    if (std::dynamic_pointer_cast<const ov::op::v0::Tile>(node)) {
        TileCodegenDesc d{};
        d.element_type = node->get_output_element_type(0);
        set_desc(d, "tile_kernel");
        return;
    }

    if (std::dynamic_pointer_cast<const ov::op::v3::Broadcast>(node)) {
        BroadcastCodegenDesc d{};
        d.element_type = node->get_output_element_type(0);
        set_desc(d, "broadcast_kernel");
        return;
    }

    if (std::dynamic_pointer_cast<const ov::op::v4::Range>(node)) {
        RangeCodegenDesc d{};
        d.element_type = node->get_output_element_type(0);
        set_desc(d, "range_kernel");
        return;
    }

    if (auto rev = std::dynamic_pointer_cast<const ov::op::v1::Reverse>(node)) {
        ReverseCodegenDesc d{};
        const auto in = rev->get_input_shape(0);
        d.element_type = rev->get_output_element_type(0);
        d.rank = static_cast<uint32_t>(in.size());
        d.total = static_cast<uint32_t>(shape_size(in));
        uint32_t stride = 1;
        for (int i = static_cast<int>(in.size()) - 1; i >= 0; --i) {
            d.strides[i] = stride;
            d.dims[i] = static_cast<uint32_t>(in[i]);
            stride *= static_cast<uint32_t>(in[i]);
        }
        auto axes_c = ov::as_type_ptr<const ov::op::v0::Constant>(rev->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axes_c, "Reverse axes must be constant");
        for (auto ax : axes_c->cast_vector<int64_t>()) {
            uint32_t axis = static_cast<uint32_t>(ax < 0 ? ax + in.size() : ax);
            d.axes_mask |= (1u << axis);
        }
        set_desc(d, "reverse_kernel");
        return;
    }

    if (auto tk = std::dynamic_pointer_cast<const ov::op::v1::TopK>(node)) {
        TopKCodegenDesc d{};
        const auto in = tk->get_input_shape(0);
        const size_t axis = static_cast<size_t>(tk->get_axis());
        d.axis_len = static_cast<uint32_t>(in[axis]);
        auto k_c = ov::as_type_ptr<const ov::op::v0::Constant>(tk->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(k_c, "TopK k must be constant");
        d.k = static_cast<uint32_t>(k_c->cast_vector<int64_t>()[0]);
        uint32_t outer = 1, inner = 1;
        for (size_t i = 0; i < axis; ++i) outer *= static_cast<uint32_t>(in[i]);
        for (size_t i = axis + 1; i < in.size(); ++i) inner *= static_cast<uint32_t>(in[i]);
        d.outer = outer;
        d.inner = inner;
        d.mode_max = tk->get_mode() == ov::op::v1::TopK::Mode::MAX;
        d.sort_type = static_cast<TopKSortType>(tk->get_sort_type());
        d.element_type = tk->get_output_element_type(0);
        d.index_type = tk->get_index_element_type();
        set_desc(d, "topk_kernel");
        return;
    }

    if (auto interp = std::dynamic_pointer_cast<const ov::op::v0::Interpolate>(node)) {
        InterpolateCodegenDesc d{};
        const auto in = interp->get_input_shape(0);
        const auto out = interp->get_output_shape(0);
        d.element_type = interp->get_output_element_type(0);
        d.N = in[0];
        d.C = in[1];
        d.H_in = in[2];
        d.W_in = in[3];
        d.H_out = out[2];
        d.W_out = out[3];
        d.scale_h = d.H_out ? static_cast<float>(d.H_in) / static_cast<float>(d.H_out) : 1.f;
        d.scale_w = d.W_out ? static_cast<float>(d.W_in) / static_cast<float>(d.W_out) : 1.f;
        d.align_corners = interp->get_attrs().align_corners;
        d.nearest = ov::util::to_lower(interp->get_attrs().mode) == "nearest";
        d.use_half_pixel = !d.align_corners;
        d.nearest_mode = 0;
        set_desc(d, "interpolate_kernel");
        return;
    }

    if (auto interp = std::dynamic_pointer_cast<const ov::op::v4::Interpolate>(node)) {
        using Base = ov::op::util::InterpolateBase;
        InterpolateCodegenDesc d{};
        const auto in = interp->get_input_shape(0);
        const auto out = interp->get_output_shape(0);
        d.element_type = interp->get_output_element_type(0);
        d.N = in[0];
        d.C = in[1];
        d.H_in = in[2];
        d.W_in = in[3];
        d.H_out = out[2];
        d.W_out = out[3];
        d.scale_h = d.H_out ? static_cast<float>(d.H_in) / static_cast<float>(d.H_out) : 1.f;
        d.scale_w = d.W_out ? static_cast<float>(d.W_in) / static_cast<float>(d.W_out) : 1.f;
        d.align_corners = interp->get_attrs().coordinate_transformation_mode ==
                          Base::CoordinateTransformMode::ALIGN_CORNERS;
        d.nearest = interp->get_attrs().mode == Base::InterpolateMode::NEAREST;
        d.use_half_pixel = interp->get_attrs().coordinate_transformation_mode ==
                           Base::CoordinateTransformMode::HALF_PIXEL;
        switch (interp->get_attrs().nearest_mode) {
            case Base::NearestMode::FLOOR:
            case Base::NearestMode::ROUND_PREFER_FLOOR:
                d.nearest_mode = 1;
                break;
            case Base::NearestMode::CEIL:
            case Base::NearestMode::ROUND_PREFER_CEIL:
                d.nearest_mode = 2;
                break;
            case Base::NearestMode::SIMPLE:
            default:
                d.nearest_mode = 0;
                break;
        }
        set_desc(d, "interpolate_kernel");
        return;
    }

    if (auto interp = std::dynamic_pointer_cast<const ov::op::v11::Interpolate>(node)) {
        using Base = ov::op::util::InterpolateBase;
        InterpolateCodegenDesc d{};
        const auto in = interp->get_input_shape(0);
        const auto out = interp->get_output_shape(0);
        d.element_type = interp->get_output_element_type(0);
        d.N = in[0];
        d.C = in[1];
        d.H_in = in[2];
        d.W_in = in[3];
        d.H_out = out[2];
        d.W_out = out[3];
        d.scale_h = d.H_out ? static_cast<float>(d.H_in) / static_cast<float>(d.H_out) : 1.f;
        d.scale_w = d.W_out ? static_cast<float>(d.W_in) / static_cast<float>(d.W_out) : 1.f;
        d.align_corners = interp->get_attrs().coordinate_transformation_mode ==
                          Base::CoordinateTransformMode::ALIGN_CORNERS;
        d.nearest = interp->get_attrs().mode == Base::InterpolateMode::NEAREST;
        d.use_half_pixel = interp->get_attrs().coordinate_transformation_mode ==
                           Base::CoordinateTransformMode::HALF_PIXEL;
        switch (interp->get_attrs().nearest_mode) {
            case Base::NearestMode::FLOOR:
            case Base::NearestMode::ROUND_PREFER_FLOOR:
                d.nearest_mode = 1;
                break;
            case Base::NearestMode::CEIL:
            case Base::NearestMode::ROUND_PREFER_CEIL:
                d.nearest_mode = 2;
                break;
            case Base::NearestMode::SIMPLE:
            default:
                d.nearest_mode = 0;
                break;
        }
        set_desc(d, "interpolate_kernel");
        return;
    }

    if (std::dynamic_pointer_cast<const ov::op::v8::Slice>(node) ||
        std::dynamic_pointer_cast<const ov::op::v1::StridedSlice>(node)) {
        ConvertCodegenDesc d{};
        d.element_type = node->get_output_element_type(0);
        d.dst_type = d.element_type;
        src.entry_point = "slice_kernel";
        src.msl_generator = [d](mlir::ModuleOp module) mutable {
            return generate_msl_for_slice_generic(d, module);
        };
        src.signature.arg_count = 8;
        if (src.module) {
            mlir::OpBuilder b(src.module.getContext());
            std::vector<int32_t> kinds{1, 1, 1, 1, 1, 1, 1, 1};
            std::vector<int32_t> arg_idx{0, 1, 2, 3, 4, 5, 6, 7};
            annotate_module_operands(src.module, kinds, arg_idx, {});
        }
        return;
    }

    if (auto g = std::dynamic_pointer_cast<const ov::op::v1::Gather>(node)) {
        GatherCodegenDesc d{};
        const auto data = g->get_input_shape(0);
        const auto idx = g->get_input_shape(1);
        const size_t axis = static_cast<size_t>(g->get_axis());
        uint64_t inner = 1, outer = 1;
        for (size_t i = axis + 1; i < data.size(); ++i) inner *= data[i];
        for (size_t i = 0; i < axis; ++i) outer *= data[i];
        d.outer = outer;
        d.inner = inner;
        d.axis_dim = data[axis];
        d.indices_count = shape_size(idx);
        d.index_type = g->get_input_element_type(1);
        d.element_type = g->get_output_element_type(0);
        set_desc(d, "gather_kernel");
        return;
    }

    if (auto gnd = std::dynamic_pointer_cast<const ov::op::v5::GatherND>(node)) {
        GatherNDCodegenDesc d{};
        const auto data = gnd->get_input_shape(0);
        const auto idx = gnd->get_input_shape(1);
        d.index_type = gnd->get_input_element_type(1);
        d.k = static_cast<uint32_t>(idx.back());
        d.num_indices = static_cast<uint32_t>(shape_size(idx) / d.k);
        d.element_type = gnd->get_output_element_type(0);
        uint32_t stride = 1;
        const size_t rank = data.size();
        for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
            d.dims[i] = static_cast<uint32_t>(data[i]);
            d.strides[i] = stride;
            stride *= d.dims[i];
        }
        d.inner = d.strides[d.k];
        d.total = static_cast<uint32_t>(shape_size(data));
        set_desc(d, "gathernd_kernel");
        return;
    }

    if (auto ge = std::dynamic_pointer_cast<const ov::op::v6::GatherElements>(node)) {
        GatherElementsCodegenDesc d{};
        const auto data = ge->get_input_shape(0);
        const auto out = ge->get_output_shape(0);
        d.index_type = ge->get_input_element_type(1);
        d.rank = static_cast<uint32_t>(out.size());
        d.axis = static_cast<uint32_t>(ge->get_axis());
        d.total = static_cast<uint32_t>(shape_size(out));
        auto data_strides = make_strides(data);
        auto out_strides = make_strides(out);
        for (size_t i = 0; i < out.size() && i < d.kMaxDims; ++i) {
            d.out_dims[i] = static_cast<uint32_t>(out[i]);
            d.out_strides[i] = static_cast<uint32_t>(out_strides[i]);
            d.data_dims[i] = static_cast<uint32_t>(data[i]);
            d.data_strides[i] = static_cast<uint32_t>(data_strides[i]);
        }
        set_desc(d, "gather_elements_kernel");
        return;
    }

    if (auto su = std::dynamic_pointer_cast<const ov::op::v3::ScatterNDUpdate>(node)) {
        ScatterNDUpdateCodegenDesc d{};
        const auto data = su->get_input_shape(0);
        const auto idx = su->get_input_shape(1);
        d.index_type = su->get_input_element_type(1);
        d.k = static_cast<uint32_t>(idx.back());
        uint32_t stride = 1;
        for (int i = static_cast<int>(data.size()) - 1; i >= 0; --i) {
            d.dims[i] = static_cast<uint32_t>(data[i]);
            d.strides[i] = stride;
            stride *= d.dims[i];
        }
        d.inner = d.strides[d.k];
        d.num_indices = static_cast<uint32_t>(shape_size(idx) / d.k);
        d.total_updates = static_cast<uint32_t>(shape_size(su->get_input_shape(2)));
        d.total_data = static_cast<uint32_t>(shape_size(data));
        d.element_type = su->get_output_element_type(0);
        set_desc(d, "scatter_nd_update");
        return;
    }

    if (auto seu = std::dynamic_pointer_cast<const ov::op::v3::ScatterElementsUpdate>(node)) {
        ScatterElementsUpdateCodegenDesc d{};
        const auto data = seu->get_input_shape(0);
        const auto idx = seu->get_input_shape(1);
        d.index_type = seu->get_input_element_type(1);
        auto axis_c = ov::as_type_ptr<const ov::op::v0::Constant>(seu->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_c, "ScatterElementsUpdate axis must be constant");
        d.axis = static_cast<uint32_t>(axis_c->cast_vector<int64_t>()[0]);
        d.rank = static_cast<uint32_t>(data.size());
        d.total_updates = static_cast<uint32_t>(shape_size(idx));
        d.total_data = static_cast<uint32_t>(shape_size(data));
        auto data_strides = make_strides(data);
        auto upd_strides = make_strides(idx);
        for (size_t i = 0; i < data.size() && i < d.kMaxDims; ++i) {
            d.data_dims[i] = static_cast<uint32_t>(data[i]);
            d.data_strides[i] = static_cast<uint32_t>(data_strides[i]);
        }
        for (size_t i = 0; i < idx.size() && i < d.kMaxDims; ++i) {
            d.update_dims[i] = static_cast<uint32_t>(idx[i]);
            d.update_strides[i] = static_cast<uint32_t>(upd_strides[i]);
        }
        d.element_type = seu->get_output_element_type(0);
        set_desc(d, "scatter_elements_update");
        return;
    }

    // Fallback: generic MLIR→MSL generation for ops without a specialized descriptor.
    if (!src.msl_generator) {
        const ov::element::Type et = node ? node->get_output_element_type(0) : ov::element::f32;
        src.msl_generator = [et](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, et, 0); };
    }
}

}  // namespace

MetalStage::MetalStage(const std::shared_ptr<const ov::Node>& node,
                       MetalDeviceHandle device,
                       MetalCommandQueueHandle queue)
    : MlirStage(node),
      m_device(device),
      m_queue(queue) {}

namespace {

// Compute row-major byte strides for a tensor shape. When broadcasting to a
// higher-rank output, extra leading dimensions are treated as length 1.
inline std::vector<int32_t> compute_broadcast_strides(const ov::Shape& in_shape,
                                                      const ov::Shape& out_shape,
                                                      size_t elem_size) {
    const size_t out_rank = out_shape.size();
    const size_t in_rank = in_shape.size();
    std::vector<int64_t> aligned(out_rank, 1);
    if (in_rank <= out_rank) {
        const size_t off = out_rank - in_rank;
        for (size_t i = 0; i < in_rank; ++i) {
            aligned[off + i] = static_cast<int64_t>(in_shape[i]);
        }
    }
    std::vector<int32_t> strides(out_rank, 0);
    int64_t stride = static_cast<int64_t>(elem_size);
    for (int64_t i = static_cast<int64_t>(out_rank) - 1; i >= 0; --i) {
        const int64_t dim = aligned[static_cast<size_t>(i)];
        // Broadcasted dim (size 1) uses zero stride.
        strides[static_cast<size_t>(i)] = (dim == 1) ? 0 : static_cast<int32_t>(stride);
        stride *= dim;
    }
    return strides;
}

inline std::vector<int32_t> to_i32_dims(const ov::Shape& shape) {
    std::vector<int32_t> dims(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        dims[i] = static_cast<int32_t>(shape[i]);
    }
    return dims;
}

inline ov::Shape resolve_shape_for_stage(const MlirStage& stage,
                                         const std::shared_ptr<const ov::Node>& node,
                                         GpuTensor* out_tensor) {
    if (out_tensor && !out_tensor->shape.empty()) {
        return out_tensor->shape;
    }
    if (node && node->get_output_partial_shape(0).is_static()) {
        return node->get_output_shape(0);
    }
    return {};
}

}  // namespace

void MetalStage::init(GpuBufferManager* buffer_manager) {
    MlirStage::init(buffer_manager);
    if (!m_device) {
        if (auto* metal_mgr = dynamic_cast<MetalBufferManager*>(buffer_manager)) {
            m_device = metal_mgr->device();
        }
    }
}

void MetalStage::compile(GpuBufferManager* buffer_manager) {
    MlirStage::compile(buffer_manager);
}

void MetalStage::execute(GpuCommandBufferHandle command_buffer) {
    MlirStage::execute(command_buffer);
}

void MetalStage::set_inputs(const std::vector<GpuTensor*>& inputs) {
    MlirStage::set_inputs(inputs);
}

void MetalStage::set_output(GpuTensor* output) {
    MlirStage::set_output(output);
}

void MetalStage::set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) {
    MlirStage::set_outputs(outputs);
}

bool MetalStage::fuse_activation(ActivationKind kind, float alpha) {
    return MlirStage::fuse_activation(kind, alpha);
}

bool MetalStage::fuse_batchnorm(const BatchNormParams& params) {
    return MlirStage::fuse_batchnorm(params);
}

bool MetalStage::fuse_bias(const BiasParams& params) {
    return MlirStage::fuse_bias(params);
}

void MetalStage::enable_profiling(bool enable) {
    MlirStage::enable_profiling(enable);
}

void MetalStage::set_profiler(void* profiler,
                              uint32_t node_id,
                              const std::string& node_name,
                              const std::string& node_type) {
    MlirStage::set_profiler(profiler, node_id, node_name, node_type);
}

std::unique_ptr<GpuStage> MetalStage::clone() const {
    auto stage = std::make_unique<MetalStage>(m_node, m_device, m_queue);
    clone_into(*stage);
    return stage;
}

std::shared_ptr<ICompiledKernel> MetalStage::compile_kernel(const KernelSource& source,
                                                            std::string* log) {
    OPENVINO_ASSERT(m_device, "MetalStage: Metal device handle is null");
    KernelSource src = source;
    attach_msl_generator(m_node, src);
    if (m_node &&
        (ov::is_type<const ov::op::v8::Slice>(m_node) || ov::is_type<const ov::op::v1::StridedSlice>(m_node))) {
        ConvertCodegenDesc d{};
        ov::element::Type storage_type = ov::element::dynamic;
        if (!m_inputs.empty() && m_inputs.front()) {
            storage_type = m_inputs.front()->expected_type;
        }
        if (storage_type == ov::element::dynamic && !m_outputs.empty() && m_outputs.front()) {
            storage_type = m_outputs.front()->expected_type;
        }
        if (storage_type == ov::element::dynamic) {
            storage_type = m_node->get_output_element_type(0);
        }
        d.element_type = storage_type;
        d.dst_type = storage_type;
        src.entry_point = "slice_kernel";
        if (m_kernel_extra_inputs.empty()) {
            src.signature.arg_count = 2;
            src.msl_source = generate_static_msl_for_slice(m_node, d.dst_type);
            src.msl_generator = {};
            src.module = {};
        } else {
            src.msl_generator = [d](mlir::ModuleOp module) mutable {
                return generate_msl_for_slice_generic(d, module);
            };
        }
    }
    OPENVINO_ASSERT(src.msl_generator || !src.msl_source.empty(),
                    "MetalStage: missing MSL source/generator for op ",
                    m_node ? m_node->get_type_name() : "");
    MetalCodegenBackend backend(m_device);
    return backend.compile(src, log);
}

KernelExecutionHooks* MetalStage::prepare_profiling(ProfileState& state,
                                                    KernelExecutionHooks& hooks) {
    auto* profiler = static_cast<MetalProfiler*>(profiler_handle());
    if (!profiler) {
        return nullptr;
    }
    state.cpu_start = std::chrono::steady_clock::now();
    const char* node_name = profile_node_name().empty() ? name().c_str() : profile_node_name().c_str();
    const char* node_type = profile_node_type().empty() ? type().c_str() : profile_node_type().c_str();
    profiler->begin_node(profile_node_id(), node_name, node_type, "GFX");
    hooks.on_begin = [profiler, &state](GpuCommandEncoderHandle enc) {
        state.sample_begin = profiler->gpu_sample_begin(static_cast<MetalCommandEncoderHandle>(enc));
    };
    hooks.on_end = [profiler, &state](GpuCommandEncoderHandle enc) {
        state.sample_end = profiler->gpu_sample_end(static_cast<MetalCommandEncoderHandle>(enc));
    };
    hooks.on_counter = [profiler](std::string_view name, uint64_t delta) {
        profiler->increment_counter(name, delta);
    };
    hooks.on_segment = [profiler](std::string_view phase,
                                  std::string_view name,
                                  std::chrono::microseconds cpu_us,
                                  uint64_t gpu_us,
                                  uint32_t dispatches,
                                  uint64_t bytes_in,
                                  uint64_t bytes_out,
                                  uint64_t macs_est,
                                  uint64_t flops_est,
                                  int64_t inflight_slot,
                                  uint64_t queue_id,
                                  uint64_t cmd_buffer_id) {
        profiler->record_segment(phase,
                                 name,
                                 cpu_us,
                                 gpu_us,
                                 dispatches,
                                 bytes_in,
                                 bytes_out,
                                 macs_est,
                                 flops_est,
                                 inflight_slot,
                                 queue_id,
                                 cmd_buffer_id);
    };
    return &hooks;
}

void MetalStage::finalize_profiling(const ProfileState& state) {
    auto* profiler = static_cast<MetalProfiler*>(profiler_handle());
    if (!profiler) {
        return;
    }
    const auto cpu_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - state.cpu_start);
    profiler->end_node(profile_node_id(), cpu_us, state.sample_begin, state.sample_end);
}

}  // namespace gfx_plugin
}  // namespace ov
