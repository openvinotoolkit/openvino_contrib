// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/vulkan_executor.hpp"

#include <chrono>

#include "backends/vulkan/codegen/vulkan_codegen_backend.hpp"
#include "backends/vulkan/runtime/profiling/profiler.hpp"
#include "kernel_ir/gfx_codegen_desc.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "kernel_ir/gfx_kernel_plan.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/gfx_mlir_type_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/rms.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/util/common_util.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_parallelism.hpp"
#include "runtime/gfx_stage_policy.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "runtime/gfx_shape_utils.hpp"
#include "mlir/mlir_support.hpp"

#include <limits>

namespace ov {
namespace gfx_plugin {

namespace {

constexpr uint32_t kLargeLinearChunkElems = 16384;
constexpr uint32_t kLinearChunkElemsPerDispatch = 65536;
ov::element::Type resolve_stage_element_type(const std::shared_ptr<const ov::Node>& node,
                                             const GpuTensor* tensor) {
    ov::element::Type et = ov::element::dynamic;
    if (tensor) {
        et = tensor->expected_type == ov::element::dynamic ? tensor->buf.type : tensor->expected_type;
    }
    if (et == ov::element::dynamic && node && node->get_output_size() > 0) {
        et = node->get_output_element_type(0);
    }
    return et;
}

ov::element::Type resolve_tensor_element_type(const GpuTensor* tensor) {
    if (!tensor) {
        return ov::element::dynamic;
    }
    return tensor->expected_type == ov::element::dynamic ? tensor->buf.type : tensor->expected_type;
}

ov::element::Type resolve_stage_input_element_type(const std::shared_ptr<const ov::Node>& node,
                                                   size_t input_idx,
                                                   const GpuTensor* tensor) {
    ov::element::Type et = resolve_tensor_element_type(tensor);
    if (et == ov::element::dynamic && node && input_idx < node->get_input_size()) {
        et = node->get_input_element_type(input_idx);
    }
    return et;
}

bool is_supported_linear_elem_type(const ov::element::Type& et) {
    return et == ov::element::f16 || et == ov::element::f32;
}

bool is_supported_arithmetic_binary_elem_type(const ov::element::Type& et) {
    return et == ov::element::f16 || et == ov::element::f32 || et == ov::element::i32 || et == ov::element::i64;
}

bool is_supported_compare_elem_type(const ov::element::Type& et) {
    return et == ov::element::boolean || et == ov::element::i32 || et == ov::element::i64 ||
           et == ov::element::f16 || et == ov::element::f32;
}

bool is_supported_broadcast_elem_type(const ov::element::Type& et) {
    return et == ov::element::boolean || et == ov::element::f16 || et == ov::element::f32 ||
           et == ov::element::i32 || et == ov::element::i64;
}

bool is_supported_linear_convert_type(const ov::element::Type& src_et, const ov::element::Type& dst_et) {
    const auto supported = [](const ov::element::Type& et) {
        return et == ov::element::boolean ||
               et == ov::element::u8 ||
               et == ov::element::i32 ||
               et == ov::element::i64 ||
               et == ov::element::f16 ||
               et == ov::element::f32;
    };
    return supported(src_et) && supported(dst_et);
}

bool is_unsigned_convert_elem_type(const ov::element::Type& et) {
    return et == ov::element::boolean || et == ov::element::u8;
}

bool is_supported_gather_embedding_type(const ov::element::Type& data_et,
                                        const ov::element::Type& idx_et) {
    return (data_et == ov::element::f16 || data_et == ov::element::f32) &&
           (idx_et == ov::element::i32 || idx_et == ov::element::i64);
}

bool is_supported_gather_linear_type(const ov::element::Type& data_et,
                                     const ov::element::Type& idx_et) {
    const bool data_supported = data_et == ov::element::boolean ||
                                data_et == ov::element::f16 ||
                                data_et == ov::element::f32 ||
                                data_et == ov::element::i32 ||
                                data_et == ov::element::i64;
    return data_supported && (idx_et == ov::element::i32 || idx_et == ov::element::i64);
}

bool is_vulkan_pipeline_creation_failure(const std::exception& ex) {
    return std::string(ex.what()).find("vkCreateComputePipelines failed") != std::string::npos;
}

Conv2DDirectPlan make_safe_conv2d_direct_plan(const GfxParallelismCaps& caps) {
    Conv2DDirectPlan plan;
    plan.output_channel_block = 1;
    plan.threads_per_group = std::max<uint32_t>(1u, std::min(std::max<uint32_t>(caps.subgroup_size, caps.preferred_simd_width),
                                                             caps.max_total_threads_per_group));
    plan.variant = "conv2d_direct_oc1_tg" + std::to_string(plan.threads_per_group);
    return plan;
}

std::optional<int32_t> eval_launch_scalar_value(mlir::Value value) {
    if (!value) {
        return std::nullopt;
    }
    if (auto cst = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        auto attr = cst.getValue();
        if (auto iattr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
            return static_cast<int32_t>(iattr.getInt());
        }
        if (auto fattr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
            if (fattr.getType().isF32()) {
                float f = static_cast<float>(fattr.getValueAsDouble());
                int32_t bits = 0;
                static_assert(sizeof(bits) == sizeof(f), "f32 scalar size mismatch");
                std::memcpy(&bits, &f, sizeof(bits));
                return bits;
            }
            if (fattr.getType().isF16()) {
                return static_cast<int32_t>(fattr.getValue().bitcastToAPInt().getZExtValue());
            }
        }
    }
    if (auto cidx = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        return static_cast<int32_t>(cidx.value());
    }
    if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
        return eval_launch_scalar_value(cast.getIn());
    }
    return std::nullopt;
}

LaunchOperandABI extract_launch_operand_abi(mlir::ModuleOp module) {
    LaunchOperandABI abi;
    if (!module) {
        return abi;
    }
    module.walk([&](mlir::gpu::LaunchFuncOp launch) {
        if (abi.valid) {
            return;
        }
        abi.valid = true;
        abi.kinds.reserve(launch.getKernelOperands().size());
        abi.arg_indices.reserve(launch.getKernelOperands().size());
        for (mlir::Value operand : launch.getKernelOperands()) {
            if (mlir::isa<mlir::MemRefType>(operand.getType())) {
                abi.kinds.push_back(1);
                int32_t arg_idx = -1;
                if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                    arg_idx = static_cast<int32_t>(barg.getArgNumber());
                }
                abi.arg_indices.push_back(arg_idx);
                continue;
            }
            abi.kinds.push_back(0);
            abi.arg_indices.push_back(-1);
            auto value = eval_launch_scalar_value(operand);
            abi.scalar_values.push_back(value.value_or(0));
            abi.scalar_known.push_back(value.has_value() ? 1 : 0);
        }
    });
    return abi;
}

uint64_t tensor_elements(const ov::Shape& shape) {
    uint64_t total = 1;
    for (const auto dim : shape) {
        total *= static_cast<uint64_t>(dim);
    }
    return total;
}

std::vector<int32_t> compute_broadcast_element_strides(const ov::Shape& in_shape,
                                                       const ov::Shape& out_shape) {
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
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(out_rank) - 1; i >= 0; --i) {
        const int64_t dim = aligned[static_cast<size_t>(i)];
        strides[static_cast<size_t>(i)] = (dim == 1) ? 0 : static_cast<int32_t>(stride);
        stride *= dim;
    }
    return strides;
}

std::string unary_chunk_key(const std::string& type) {
    if (type == "Swish") {
        return "swish";
    }
    if (type == "Sigmoid") {
        return "sigmoid";
    }
    if (type == "Relu") {
        return "relu";
    }
    if (type == "Tanh") {
        return "tanh";
    }
    if (type == "Sqrt") {
        return "sqrt";
    }
    if (type == "Exp") {
        return "exp";
    }
    if (type == "Log") {
        return "log";
    }
    if (type == "Floor") {
        return "floor";
    }
    if (type == "Ceiling" || type == "Ceil") {
        return "ceil";
    }
    if (type == "Negative") {
        return "neg";
    }
    if (type == "Sin") {
        return "sin";
    }
    if (type == "Cos") {
        return "cos";
    }
    return {};
}

std::string binary_chunk_key(const std::string& type) {
    if (type == "Add") {
        return "add";
    }
    if (type == "Subtract") {
        return "sub";
    }
    if (type == "Multiply") {
        return "mul";
    }
    if (type == "Divide") {
        return "div";
    }
    if (type == "Power") {
        return "pow";
    }
    if (type == "Equal") {
        return "eq";
    }
    if (type == "NotEqual") {
        return "ne";
    }
    if (type == "Less") {
        return "lt";
    }
    if (type == "Greater") {
        return "gt";
    }
    if (type == "LessEqual") {
        return "le";
    }
    if (type == "GreaterEqual") {
        return "ge";
    }
    if (type == "LogicalAnd") {
        return "land";
    }
    if (type == "LogicalOr") {
        return "lor";
    }
    if (type == "LogicalXor") {
        return "lxor";
    }
    return {};
}

bool is_arithmetic_binary_key(const std::string& op_key) {
    return op_key == "add" || op_key == "sub" || op_key == "mul" || op_key == "div" || op_key == "pow";
}

bool is_compare_binary_key(const std::string& op_key) {
    return op_key == "eq" || op_key == "ne" || op_key == "lt" || op_key == "gt" || op_key == "le" || op_key == "ge";
}

bool is_logical_binary_key(const std::string& op_key) {
    return op_key == "land" || op_key == "lor" || op_key == "lxor";
}

bool is_supported_binary_io_types(const ov::element::Type& src0_et,
                                  const ov::element::Type& src1_et,
                                  const ov::element::Type& dst_et,
                                  const std::string& op_key) {
    if (src0_et == ov::element::dynamic || src1_et == ov::element::dynamic || dst_et == ov::element::dynamic) {
        return false;
    }
    if (src0_et != src1_et) {
        return false;
    }
    if (is_arithmetic_binary_key(op_key)) {
        if (op_key == "pow") {
            return src0_et == dst_et && is_supported_linear_elem_type(dst_et);
        }
        return src0_et == dst_et && is_supported_arithmetic_binary_elem_type(dst_et);
    }
    if (is_compare_binary_key(op_key)) {
        return dst_et == ov::element::boolean && is_supported_compare_elem_type(src0_et);
    }
    if (is_logical_binary_key(op_key)) {
        return src0_et == ov::element::boolean && dst_et == ov::element::boolean;
    }
    return false;
}

mlir::Type to_binary_storage_type(mlir::MLIRContext& ctx, const ov::element::Type& et) {
    return to_mlir_type(et,
                        ctx,
                        /*fallback_f32=*/false,
                        /*allow_unsigned=*/false,
                        /*allow_small_ints=*/false,
                        /*allow_bf16=*/false,
                        /*allow_boolean=*/true,
                        /*signless_integers=*/true);
}

std::string reduce_last_axis_key(const std::string& type) {
    if (type == "ReduceMean") {
        return "mean";
    }
    if (type == "ReduceSum") {
        return "sum";
    }
    return {};
}

ov::Shape broadcast_runtime_shape(const ov::Shape& lhs, const ov::Shape& rhs) {
    const size_t rank = std::max(lhs.size(), rhs.size());
    ov::Shape out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const size_t lhs_rev = lhs.size() > i ? lhs[lhs.size() - 1 - i] : 1;
        const size_t rhs_rev = rhs.size() > i ? rhs[rhs.size() - 1 - i] : 1;
        OPENVINO_ASSERT(lhs_rev == rhs_rev || lhs_rev == 1 || rhs_rev == 1,
                        "GFX Vulkan binary chunked: incompatible broadcast dimensions ",
                        lhs_rev,
                        " and ",
                        rhs_rev);
        out[rank - 1 - i] = std::max(lhs_rev, rhs_rev);
    }
    return out;
}

std::vector<int64_t> get_slice_const_i64(const ov::Output<ov::Node>& source, const char* what) {
    auto constant = ov::util::get_constant_from_source(source);
    OPENVINO_ASSERT(constant, "GFX Vulkan Slice: ", what, " must be Constant");
    return constant->cast_vector<int64_t>();
}

int64_t normalize_runtime_slice_index(int64_t index, int64_t dim, bool is_begin) {
    if (index < 0) {
        index += dim;
    }
    if (is_begin) {
        return std::clamp<int64_t>(index, 0, dim);
    }
    return std::clamp<int64_t>(index, -1, dim);
}

struct RuntimeSliceMeta {
    std::vector<uint32_t> out_shape;
    std::vector<uint32_t> in_stride;
    std::vector<int32_t> starts;
    std::vector<int32_t> steps;
    uint32_t total = 0;
    uint32_t rank = 0;
};

RuntimeSliceMeta build_runtime_slice_meta(const std::shared_ptr<const ov::Node>& node,
                                          const ov::Shape& in_shape,
                                          const ov::Shape& out_shape) {
    OPENVINO_ASSERT(node, "GFX Vulkan Slice: node is null");
    const size_t rank = in_shape.size();
    OPENVINO_ASSERT(rank == out_shape.size(), "GFX Vulkan Slice: rank-changing Slice/StridedSlice is not supported");

    RuntimeSliceMeta meta;
    meta.rank = static_cast<uint32_t>(rank);
    meta.total = static_cast<uint32_t>(ov::shape_size(out_shape));
    meta.out_shape.reserve(rank);
    meta.in_stride.assign(rank, 1);
    meta.starts.assign(rank, 0);
    meta.steps.assign(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        meta.out_shape.push_back(static_cast<uint32_t>(out_shape[i]));
    }
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        meta.in_stride[static_cast<size_t>(i)] =
            meta.in_stride[static_cast<size_t>(i + 1)] * static_cast<uint32_t>(in_shape[static_cast<size_t>(i + 1)]);
    }

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
                        "GFX Vulkan Slice: starts/ends/steps/axes size mismatch");
        for (size_t i = 0; i < axes.size(); ++i) {
            int64_t axis = axes[i];
            if (axis < 0) {
                axis += static_cast<int64_t>(rank);
            }
            OPENVINO_ASSERT(axis >= 0 && static_cast<size_t>(axis) < rank, "GFX Vulkan Slice: axis out of range");
            OPENVINO_ASSERT(steps[i] != 0, "GFX Vulkan Slice: zero step is not supported");
            const auto dim = static_cast<int64_t>(in_shape[static_cast<size_t>(axis)]);
            meta.starts[static_cast<size_t>(axis)] =
                static_cast<int32_t>(normalize_runtime_slice_index(starts[i], dim, true));
            meta.steps[static_cast<size_t>(axis)] = static_cast<int32_t>(steps[i]);
        }
        return meta;
    }

    auto slice = ov::as_type_ptr<const ov::op::v1::StridedSlice>(node);
    OPENVINO_ASSERT(slice, "GFX Vulkan Slice: expected Slice/StridedSlice node");
    OPENVINO_ASSERT(std::all_of(slice->get_new_axis_mask().begin(),
                                slice->get_new_axis_mask().end(),
                                [](int64_t v) { return v == 0; }),
                    "GFX Vulkan Slice: StridedSlice new_axis_mask is not supported");
    OPENVINO_ASSERT(std::all_of(slice->get_shrink_axis_mask().begin(),
                                slice->get_shrink_axis_mask().end(),
                                [](int64_t v) { return v == 0; }),
                    "GFX Vulkan Slice: StridedSlice shrink_axis_mask is not supported");
    OPENVINO_ASSERT(std::all_of(slice->get_ellipsis_mask().begin(),
                                slice->get_ellipsis_mask().end(),
                                [](int64_t v) { return v == 0; }),
                    "GFX Vulkan Slice: StridedSlice ellipsis_mask is not supported");

    auto begin = get_slice_const_i64(slice->input_value(1), "StridedSlice begin");
    auto end = get_slice_const_i64(slice->input_value(2), "StridedSlice end");
    std::vector<int64_t> strides(rank, 1);
    if (slice->get_input_size() > 3) {
        auto values = get_slice_const_i64(slice->input_value(3), "StridedSlice strides");
        OPENVINO_ASSERT(values.size() <= rank, "GFX Vulkan Slice: StridedSlice strides rank mismatch");
        std::copy(values.begin(), values.end(), strides.begin());
    }
    const auto& begin_mask = slice->get_begin_mask();
    const auto& end_mask = slice->get_end_mask();
    for (size_t axis = 0; axis < rank; ++axis) {
        const auto dim = static_cast<int64_t>(in_shape[axis]);
        const bool masked_begin = axis < begin_mask.size() && begin_mask[axis] != 0;
        const bool masked_end = axis < end_mask.size() && end_mask[axis] != 0;
        const int64_t step = strides[axis];
        OPENVINO_ASSERT(step != 0, "GFX Vulkan Slice: StridedSlice zero step is not supported");
        int64_t start = axis < begin.size() ? begin[axis] : 0;
        int64_t finish = axis < end.size() ? end[axis] : dim;
        start = masked_begin ? (step < 0 ? dim - 1 : 0) : normalize_runtime_slice_index(start, dim, true);
        finish = masked_end ? (step < 0 ? -1 : dim) : normalize_runtime_slice_index(finish, dim, false);
        (void)finish;
        meta.starts[axis] = static_cast<int32_t>(start);
        meta.steps[axis] = static_cast<int32_t>(step);
    }
    return meta;
}

}  // namespace

VulkanStage::VulkanStage(const std::shared_ptr<const ov::Node>& node)
    : MlirStage(node) {}

VulkanStage::~VulkanStage() = default;

void VulkanStage::init(GpuBufferManager* buffer_manager) {
    MlirStage::init(buffer_manager);
}

void VulkanStage::compile(GpuBufferManager* buffer_manager) {
    if (should_use_matmul_linear()) {
        MlirStage::init(buffer_manager);
        prepare_matmul_linear_kernel();
        return;
    }
    MlirStage::compile(buffer_manager);
    prepare_specialized_kernels();
}

std::shared_ptr<ICompiledKernel> VulkanStage::compile_specialized_kernel_from_mlir(mlir::ModuleOp module,
                                                                                   const std::string& entry_name,
                                                                                   uint32_t arg_count,
                                                                                   const char* error_prefix) {
    KernelSource src = make_kernel_source_from_mlir(module, entry_name, arg_count);
    src.signature.output_arg_count = 1;
    VulkanCodegenBackend backend;
    std::string log;
    auto kernel = backend.compile(src, &log);
    OPENVINO_ASSERT(kernel, error_prefix, log);
    kernel->prepare_runtime_artifacts();
    return kernel;
}

void VulkanStage::prepare_specialized_kernels() {
    const auto conv_plan = conv_route_plan();
    if ((m_type == "Split" || m_type == "VariadicSplit") && !has_absorbed_input_transpose()) {
        prepare_split_kernel();
        return;
    }
    if (should_use_concat_chunked()) {
        prepare_concat_kernel();
        return;
    }
    if (should_use_slice_chunked()) {
        prepare_slice_kernel();
        return;
    }
    if (should_use_interpolate_chunked()) {
        prepare_interpolate_kernel();
        return;
    }
    if (should_use_transpose_chunked()) {
        prepare_transpose_kernel();
        return;
    }
    if (should_use_convert_chunked()) {
        prepare_convert_kernel();
        return;
    }
    if (should_use_gather_linear()) {
        prepare_gather_linear_kernel();
        return;
    }
    if (should_use_gather_embedding()) {
        prepare_gather_embedding_kernel();
        return;
    }
    if (should_use_matmul_linear()) {
        prepare_matmul_linear_kernel();
        return;
    }
    if (should_use_broadcast_chunked()) {
        prepare_broadcast_kernel();
        return;
    }
    if (should_use_select_chunked()) {
        prepare_select_kernel();
        return;
    }
    if (should_use_reduce_last_axis()) {
        prepare_reduce_last_axis_kernel();
        return;
    }
    if (should_use_rms_chunked()) {
        prepare_rms_kernel();
        return;
    }
    if (should_use_softmax_chunked()) {
        prepare_softmax_kernel();
        return;
    }
    if (should_use_group_conv2d_chunked()) {
        prepare_group_conv2d_kernel();
        return;
    }
    if (conv_plan.kind == GfxConvRouteKind::Direct3x3) {
        prepare_conv2d_3x3_direct_kernel();
        return;
    }
    if (conv_plan.kind == GfxConvRouteKind::Chunked &&
        conv_plan.algorithm.kind != GfxConvAlgorithmKind::Im2ColMatMul &&
        conv_plan.algorithm.kind != GfxConvAlgorithmKind::Indirect) {
        prepare_conv2d_chunk_kernel();
        return;
    }
    if (should_use_binary_same_shape()) {
        prepare_binary_same_shape_kernel();
        return;
    }
    if (should_use_binary_bias_add()) {
        prepare_binary_bias_add_kernel();
        return;
    }
    if (should_use_binary_chunked()) {
        prepare_binary_kernel();
        return;
    }
    if (should_use_unary_chunked()) {
        prepare_unary_kernel();
    }
}

void VulkanStage::prepare_unary_kernel() {
    const auto op_key = unary_chunk_key(m_type);
    OPENVINO_ASSERT(!op_key.empty(), "GFX Vulkan unary chunked: unsupported op ", m_type);
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan unary chunked: unsupported element type ",
                    elem_type);
    if (m_linear_unary_kernel && m_linear_unary_elem_type == elem_type && m_linear_unary_key == op_key) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_linear_unary_module(ctx, elem_type, op_key);
    m_linear_unary_kernel =
        compile_specialized_kernel_from_mlir(module, "linear_unary", 3, "GFX Vulkan unary chunked: kernel compile failed: ");
    m_linear_unary_elem_type = elem_type;
    m_linear_unary_key = op_key;
    m_linear_unary_launch_abi = extract_launch_operand_abi(module);
    m_linear_unary_scalar_args = extract_kernel_scalar_values(module);
}

void VulkanStage::prepare_binary_kernel() {
    const auto op_key = binary_chunk_key(m_type);
    OPENVINO_ASSERT(!op_key.empty(), "GFX Vulkan binary chunked: unsupported op ", m_type);
    const ov::element::Type src0_et = resolve_stage_input_element_type(m_node, 0, !m_inputs.empty() ? m_inputs[0] : nullptr);
    const ov::element::Type src1_et = resolve_stage_input_element_type(m_node, 1, m_inputs.size() > 1 ? m_inputs[1] : nullptr);
    const ov::element::Type dst_et = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_binary_io_types(src0_et, src1_et, dst_et, op_key),
                    "GFX Vulkan binary chunked: unsupported element types ",
                    src0_et,
                    " / ",
                    src1_et,
                    " -> ",
                    dst_et,
                    " for op ",
                    op_key);
    size_t meta_rank = 0;
    if (!m_kernel_extra_inputs.empty() && !m_kernel_extra_inputs[0].shape.empty()) {
        meta_rank = m_kernel_extra_inputs[0].shape[0];
    } else if (m_node && m_node->get_output_partial_shape(0).rank().is_static()) {
        meta_rank = static_cast<size_t>(m_node->get_output_partial_shape(0).rank().get_length());
    }
    OPENVINO_ASSERT(meta_rank != 0, "GFX Vulkan binary chunked: invalid metadata rank");
    if (m_linear_binary_kernel &&
        m_linear_binary_src0_elem_type == src0_et &&
        m_linear_binary_src1_elem_type == src1_et &&
        m_linear_binary_dst_elem_type == dst_et &&
        m_linear_binary_key == op_key &&
        m_linear_binary_rank == meta_rank) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_linear_binary_module(ctx, src0_et, src1_et, dst_et, op_key, meta_rank);
    m_linear_binary_kernel =
        compile_specialized_kernel_from_mlir(module, "linear_binary", 4, "GFX Vulkan binary chunked: kernel compile failed: ");
    m_linear_binary_src0_elem_type = src0_et;
    m_linear_binary_src1_elem_type = src1_et;
    m_linear_binary_dst_elem_type = dst_et;
    m_linear_binary_key = op_key;
    m_linear_binary_rank = meta_rank;
    m_linear_binary_launch_abi = extract_launch_operand_abi(module);
}

void VulkanStage::prepare_binary_same_shape_kernel() {
    const auto op_key = binary_chunk_key(m_type);
    OPENVINO_ASSERT(!op_key.empty(), "GFX Vulkan same-shape binary: unsupported op ", m_type);
    const ov::element::Type src0_et = resolve_stage_input_element_type(m_node, 0, !m_inputs.empty() ? m_inputs[0] : nullptr);
    const ov::element::Type src1_et = resolve_stage_input_element_type(m_node, 1, m_inputs.size() > 1 ? m_inputs[1] : nullptr);
    const ov::element::Type dst_et = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_binary_io_types(src0_et, src1_et, dst_et, op_key),
                    "GFX Vulkan same-shape binary: unsupported element types ",
                    src0_et,
                    " / ",
                    src1_et,
                    " -> ",
                    dst_et,
                    " for op ",
                    op_key);
    if (m_linear_binary_same_shape_kernel &&
        m_linear_binary_same_shape_src0_elem_type == src0_et &&
        m_linear_binary_same_shape_src1_elem_type == src1_et &&
        m_linear_binary_same_shape_dst_elem_type == dst_et &&
        m_linear_binary_same_shape_key == op_key) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_linear_binary_same_shape_module(ctx, src0_et, src1_et, dst_et, op_key);
    m_linear_binary_same_shape_kernel = compile_specialized_kernel_from_mlir(
        module,
        "linear_binary_same_shape",
        3,
        "GFX Vulkan same-shape binary: kernel compile failed: ");
    m_linear_binary_same_shape_src0_elem_type = src0_et;
    m_linear_binary_same_shape_src1_elem_type = src1_et;
    m_linear_binary_same_shape_dst_elem_type = dst_et;
    m_linear_binary_same_shape_key = op_key;
}

void VulkanStage::prepare_binary_bias_add_kernel() {
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan bias add: unsupported element type ",
                    elem_type);
    if (m_binary_bias_add_kernel && m_binary_bias_add_elem_type == elem_type) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_binary_bias_add_module(ctx, elem_type);
    m_binary_bias_add_kernel = compile_specialized_kernel_from_mlir(
        module,
        "binary_bias_add",
        3,
        "GFX Vulkan bias add: kernel compile failed: ");
    m_binary_bias_add_elem_type = elem_type;
}

void VulkanStage::prepare_conv2d_1x1_kernel() {
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    OPENVINO_ASSERT(conv, "GFX Vulkan conv2d 1x1: node cast failed");
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan conv2d 1x1: unsupported element type ",
                    elem_type);
    const auto caps = query_parallelism_caps(m_buffer_manager);
    const auto total = static_cast<uint64_t>(tensor_elements(conv->get_output_shape(0)));
    const auto launch_plan =
        select_chunk_dispatch_plan(caps, "conv2d_1x1", total, static_cast<uint64_t>(conv->get_input_shape(0).at(1)));
    if (m_conv2d_1x1_kernel && m_conv2d_1x1_elem_type == elem_type &&
        m_conv2d_1x1_threads_per_group == launch_plan.threads_per_group) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_conv2d_1x1_module(ctx, elem_type, launch_plan.threads_per_group);
    m_conv2d_1x1_kernel =
        compile_specialized_kernel_from_mlir(module,
                                             "conv2d_1x1",
                                             m_has_bias ? 5 : 4,
                                             "GFX Vulkan conv2d 1x1: kernel compile failed: ");
    m_conv2d_1x1_elem_type = elem_type;
    m_conv2d_1x1_threads_per_group = launch_plan.threads_per_group;
}

void VulkanStage::prepare_conv2d_3x3_direct_kernel() {
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    OPENVINO_ASSERT(conv, "GFX Vulkan conv2d 3x3 direct: node cast failed");
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan conv2d 3x3 direct: unsupported element type ",
                    elem_type);
    const auto caps = query_parallelism_caps(m_buffer_manager);
    auto plan = select_conv2d_direct_plan(caps,
                                          conv->get_output_shape(0),
                                          conv->get_input_shape(0).at(1),
                                          conv->get_output_shape(0).at(1),
                                          conv->get_input_shape(0).at(1) *
                                              conv->get_input_shape(1).at(2) *
                                              conv->get_input_shape(1).at(3),
                                          conv->get_strides().at(0) == 2 &&
                                              conv->get_strides().at(1) == 2);
    if (m_conv2d_3x3_force_safe_variant) {
        plan = make_safe_conv2d_direct_plan(caps);
    }
    if (m_conv2d_3x3_direct_kernel && m_conv2d_3x3_direct_elem_type == elem_type &&
        m_conv2d_3x3_direct_oc_block == plan.output_channel_block &&
        m_conv2d_3x3_direct_threads_per_group == plan.threads_per_group &&
        m_conv2d_3x3_direct_variant == plan.variant) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module =
        build_conv2d_3x3_direct_module(ctx, elem_type, plan.output_channel_block, plan.threads_per_group, plan.variant);
    const char* entry_name = plan.variant.rfind("conv2d_direct_xy", 0) == 0
                                 ? "conv2d_3x3_direct_xy"
                                 : (plan.output_channel_block == 1 ? "conv2d_3x3_direct" : "conv2d_3x3_direct_oc2");
    m_conv2d_3x3_direct_kernel = compile_specialized_kernel_from_mlir(
        module,
        entry_name,
        m_has_bias ? 4 : 3,
        "GFX Vulkan conv2d 3x3 direct: kernel compile failed: ");
    m_conv2d_3x3_direct_elem_type = elem_type;
    m_conv2d_3x3_direct_oc_block = plan.output_channel_block;
    m_conv2d_3x3_direct_threads_per_group = plan.threads_per_group;
    m_conv2d_3x3_direct_variant = plan.variant;
}

void VulkanStage::prepare_conv2d_chunk_kernel() {
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan conv2d chunked: unsupported element type ",
                    elem_type);
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    OPENVINO_ASSERT(conv, "GFX Vulkan conv2d chunked: node cast failed");
    const auto caps = query_parallelism_caps(m_buffer_manager);
    const auto& out_shape = conv->get_output_shape(0);
    const auto& in_shape = conv->get_input_shape(0);
    const auto& w_shape = conv->get_input_shape(1);
    const auto work_per_elem =
        static_cast<uint64_t>(in_shape.at(1)) * static_cast<uint64_t>(w_shape.at(2)) *
        static_cast<uint64_t>(w_shape.at(3));
    const bool stride2 = conv->get_strides().at(0) > 1 || conv->get_strides().at(1) > 1;
    const auto spatial_plan = select_conv_parallelism(caps,
                                                      out_shape,
                                                      static_cast<uint64_t>(in_shape.at(1)),
                                                      static_cast<uint64_t>(w_shape.at(0)),
                                                      work_per_elem,
                                                      stride2,
                                                      /*depthwise=*/false);
    const uint32_t threads_h = std::max<uint32_t>(1u, spatial_plan.dispatch.threads_h);
    const uint32_t threads_w = std::max<uint32_t>(1u, spatial_plan.dispatch.threads_w);
    if (m_conv2d_chunk_kernel && m_conv2d_chunk_elem_type == elem_type &&
        m_conv2d_chunk_threads_h == threads_h &&
        m_conv2d_chunk_threads_w == threads_w) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_conv2d_chunk_module(ctx, elem_type, threads_h, threads_w);
    m_conv2d_chunk_kernel =
        compile_specialized_kernel_from_mlir(module, "conv2d_chunk", 3, "GFX Vulkan conv2d chunked: kernel compile failed: ");
    m_conv2d_chunk_elem_type = elem_type;
    m_conv2d_chunk_threads_h = threads_h;
    m_conv2d_chunk_threads_w = threads_w;
}

void VulkanStage::prepare_group_conv2d_kernel() {
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan group_conv2d chunked: unsupported element type ",
                    elem_type);
    auto gconv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(m_node);
    OPENVINO_ASSERT(gconv, "GFX Vulkan group_conv2d chunked: node cast failed");
    const auto caps = query_parallelism_caps(m_buffer_manager);
    const auto total = static_cast<uint64_t>(tensor_elements(gconv->get_output_shape(0)));
    const auto work_per_elem =
        static_cast<uint64_t>(gconv->get_input_shape(1).at(3)) * static_cast<uint64_t>(gconv->get_input_shape(1).at(4));
    const auto launch_plan = select_chunk_dispatch_plan(caps, "group_conv2d", total, work_per_elem);
    if (m_group_conv2d_kernel && m_group_conv2d_elem_type == elem_type &&
        m_group_conv2d_threads_per_group == launch_plan.threads_per_group) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_group_conv2d_chunk_module(ctx, elem_type, launch_plan.threads_per_group);
    m_group_conv2d_kernel = compile_specialized_kernel_from_mlir(
        module,
        "group_conv2d_direct",
        3,
        "GFX Vulkan group_conv2d chunked: kernel compile failed: ");
    m_group_conv2d_elem_type = elem_type;
    m_group_conv2d_threads_per_group = launch_plan.threads_per_group;
}

void VulkanStage::prepare_softmax_kernel() {
    bool log_softmax = false;
    if (ov::as_type_ptr<const ov::op::v5::LogSoftmax>(m_node)) {
        log_softmax = true;
    }
    const auto* input = !m_inputs.empty() ? m_inputs[0] : nullptr;
    const ov::element::Type elem_type =
        input ? (input->expected_type == ov::element::dynamic ? input->buf.type : input->expected_type)
              : (m_node ? m_node->get_input_element_type(0) : ov::element::dynamic);
    OPENVINO_ASSERT(elem_type == ov::element::f16 || elem_type == ov::element::f32,
                    "GFX Vulkan Softmax: unsupported element type ",
                    elem_type);
    if (m_softmax_row_kernel && m_softmax_elem_type == elem_type && m_softmax_log_kernel == log_softmax) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_softmax_row_module(ctx, elem_type, log_softmax);
    m_softmax_row_kernel = compile_specialized_kernel_from_mlir(
        module,
        log_softmax ? "logsoftmax_row" : "softmax_row",
        3,
        "GFX Vulkan Softmax: kernel compile failed: ");
    m_softmax_elem_type = elem_type;
    m_softmax_log_kernel = log_softmax;
}

void VulkanStage::prepare_concat_kernel() {
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    if (m_concat_single_kernel && m_concat_elem_type == elem_type) {
        if (!m_node || m_node->get_input_size() != 2 ||
            (m_concat_binary_kernel && m_concat_binary_elem_type == elem_type)) {
            return;
        }
    }
    auto& ctx = gfx_mlir_context();
    if (!m_concat_single_kernel || m_concat_elem_type != elem_type) {
        auto module = build_concat_single_module(ctx, elem_type);
        m_concat_single_kernel =
            compile_specialized_kernel_from_mlir(module, "concat_single", 3, "GFX Vulkan Concat: kernel compile failed: ");
        m_concat_elem_type = elem_type;
    }
    if (m_node && m_node->get_input_size() == 2 &&
        (!m_concat_binary_kernel || m_concat_binary_elem_type != elem_type)) {
        auto module = build_concat_binary_module(ctx, elem_type);
        m_concat_binary_kernel = compile_specialized_kernel_from_mlir(
            module,
            "concat_binary",
            4,
            "GFX Vulkan Concat binary: kernel compile failed: ");
        m_concat_binary_elem_type = elem_type;
    }
}

void VulkanStage::prepare_slice_kernel() {
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    if (m_slice_linear_kernel && m_slice_elem_type == elem_type) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_slice_linear_module(ctx, elem_type);
    m_slice_linear_kernel =
        compile_specialized_kernel_from_mlir(module, "slice_linear", 8, "GFX Vulkan Slice: kernel compile failed: ");
    m_slice_elem_type = elem_type;
}

void VulkanStage::prepare_split_kernel() {
    const auto* input = !m_inputs.empty() ? m_inputs[0] : nullptr;
    const ov::element::Type elem_type =
        input ? (input->expected_type == ov::element::dynamic ? input->buf.type : input->expected_type)
              : (m_node ? m_node->get_input_element_type(0) : ov::element::dynamic);
    if (m_split_single_kernel && m_split_elem_type == elem_type) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_split_single_module(ctx, elem_type);
    m_split_single_kernel =
        compile_specialized_kernel_from_mlir(module, "split_single", 3, "GFX Vulkan Split: kernel compile failed: ");
    m_split_elem_type = elem_type;
}

void VulkanStage::prepare_transpose_kernel() {
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    if (m_transpose_kernel && m_transpose_elem_type == elem_type) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_transpose_module(ctx, elem_type);
    m_transpose_kernel = compile_specialized_kernel_from_mlir(
        module,
        "transpose_direct",
        2,
        "GFX Vulkan Transpose: kernel compile failed: ");
    m_transpose_elem_type = elem_type;
}

void VulkanStage::prepare_interpolate_kernel() {
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan Interpolate: unsupported element type ",
                    elem_type);
    if (m_interpolate_kernel && m_interpolate_elem_type == elem_type) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_interpolate_module(ctx, elem_type);
    m_interpolate_kernel = compile_specialized_kernel_from_mlir(
        module,
        "interpolate_direct",
        3,
        "GFX Vulkan Interpolate: kernel compile failed: ");
    m_interpolate_elem_type = elem_type;
}

void VulkanStage::prepare_convert_kernel() {
    const auto src_et = m_node->get_input_element_type(0);
    const auto dst_et = m_node->get_output_element_type(0);
    OPENVINO_ASSERT(is_supported_linear_convert_type(src_et, dst_et),
                    "GFX Vulkan Convert: unsupported conversion ",
                    src_et,
                    " -> ",
                    dst_et);
    if (m_convert_linear_kernel && m_convert_src_elem_type == src_et && m_convert_dst_elem_type == dst_et) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_convert_linear_module(ctx, src_et, dst_et);
    m_convert_linear_kernel = compile_specialized_kernel_from_mlir(
        module,
        "convert_linear",
        3,
        "GFX Vulkan Convert: kernel compile failed: ");
    m_convert_src_elem_type = src_et;
    m_convert_dst_elem_type = dst_et;
}

void VulkanStage::prepare_gather_linear_kernel() {
    OPENVINO_ASSERT(m_node, "GFX Vulkan Gather: missing node");
    const auto data_et = m_node->get_input_element_type(0);
    const auto idx_et = m_node->get_input_element_type(1);
    OPENVINO_ASSERT(is_supported_gather_linear_type(data_et, idx_et),
                    "GFX Vulkan Gather: unsupported types ",
                    data_et,
                    " / ",
                    idx_et);
    if (m_gather_linear_kernel &&
        m_gather_linear_data_elem_type == data_et &&
        m_gather_linear_index_elem_type == idx_et) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_gather_linear_module(ctx, data_et, idx_et);
    m_gather_linear_kernel = compile_specialized_kernel_from_mlir(
        module,
        "gather_linear",
        4,
        "GFX Vulkan Gather: kernel compile failed: ");
    m_gather_linear_data_elem_type = data_et;
    m_gather_linear_index_elem_type = idx_et;
}

void VulkanStage::prepare_gather_embedding_kernel() {
    OPENVINO_ASSERT(m_node, "GFX Vulkan Gather: missing node");
    const auto data_shape = m_node->get_input_shape(0);
    const auto vocab = data_shape[0];
    const auto hidden = data_shape[1];
    const auto data_et = m_node->get_input_element_type(0);
    const auto idx_et = m_node->get_input_element_type(1);
    OPENVINO_ASSERT(is_supported_gather_embedding_type(data_et, idx_et),
                    "GFX Vulkan Gather: unsupported embedding types ",
                    data_et,
                    " / ",
                    idx_et);
    if (m_gather_embedding_kernel &&
        m_gather_data_elem_type == data_et &&
        m_gather_index_elem_type == idx_et &&
        m_gather_vocab == vocab &&
        m_gather_hidden == hidden) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_gather_embedding_module(ctx, data_et, idx_et, vocab, hidden);
    m_gather_embedding_kernel = compile_specialized_kernel_from_mlir(
        module,
        "gather_embedding",
        4,
        "GFX Vulkan Gather: kernel compile failed: ");
    m_gather_data_elem_type = data_et;
    m_gather_index_elem_type = idx_et;
    m_gather_vocab = vocab;
    m_gather_hidden = hidden;
}

void VulkanStage::prepare_reduce_last_axis_kernel() {
    const auto op_key = reduce_last_axis_key(m_type);
    OPENVINO_ASSERT(!op_key.empty(), "GFX Vulkan reduce last-axis: unsupported op ", m_type);
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan reduce last-axis: unsupported element type ",
                    elem_type);
    if (m_reduce_last_axis_kernel &&
        m_reduce_last_axis_elem_type == elem_type &&
        m_reduce_last_axis_key == op_key) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_reduce_last_axis_module(ctx, elem_type, op_key);
    m_reduce_last_axis_kernel = compile_specialized_kernel_from_mlir(
        module,
        "reduce_last_axis",
        3,
        "GFX Vulkan reduce last-axis: kernel compile failed: ");
    m_reduce_last_axis_elem_type = elem_type;
    m_reduce_last_axis_key = op_key;
}

void VulkanStage::prepare_rms_kernel() {
    auto rms = ov::as_type_ptr<const ov::op::internal::RMS>(m_node);
    OPENVINO_ASSERT(rms, "GFX Vulkan RMS: node cast failed");
    const auto in_pshape = rms->get_input_partial_shape(0);
    const auto gamma_pshape = rms->get_input_partial_shape(1);
    OPENVINO_ASSERT(in_pshape.rank().is_static() && in_pshape.rank().get_length() > 0,
                    "GFX Vulkan RMS: input rank must be static");
    const auto rank = static_cast<size_t>(in_pshape.rank().get_length());
    OPENVINO_ASSERT(in_pshape[rank - 1].is_static(),
                    "GFX Vulkan RMS: hidden dimension must be static");
    OPENVINO_ASSERT(gamma_pshape.is_static(), "GFX Vulkan RMS: gamma shape must be static");
    const size_t hidden = static_cast<size_t>(in_pshape[rank - 1].get_length());
    const size_t gamma_size = ov::shape_size(gamma_pshape.to_shape());
    const uint32_t reduction_threads = gfx_rms_parallel_reduction_threads(static_cast<uint32_t>(hidden));
    const auto input_et = rms->get_input_element_type(0);
    const auto gamma_et = rms->get_input_element_type(1);
    const auto output_et = rms->get_output_element_type(0);
    OPENVINO_ASSERT(is_supported_linear_elem_type(input_et) &&
                        is_supported_linear_elem_type(gamma_et) &&
                        is_supported_linear_elem_type(output_et),
                    "GFX Vulkan RMS: unsupported element types ",
                    input_et,
                    " / ",
                    gamma_et,
                    " -> ",
                    output_et);
    if (m_rms_kernel &&
        m_rms_input_elem_type == input_et &&
        m_rms_gamma_elem_type == gamma_et &&
        m_rms_output_elem_type == output_et &&
        m_rms_hidden == hidden &&
        m_rms_gamma_size == gamma_size &&
        m_rms_reduction_threads == reduction_threads &&
        m_rms_epsilon == static_cast<float>(rms->get_epsilon())) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_rms_module(ctx,
                                   input_et,
                                   gamma_et,
                                   output_et,
                                   hidden,
                                   gamma_size,
                                   reduction_threads,
                                   static_cast<float>(rms->get_epsilon()));
    m_rms_kernel =
        compile_specialized_kernel_from_mlir(module, "rms_linear", 3, "GFX Vulkan RMS: kernel compile failed: ");
    m_rms_input_elem_type = input_et;
    m_rms_gamma_elem_type = gamma_et;
    m_rms_output_elem_type = output_et;
    m_rms_hidden = static_cast<uint32_t>(hidden);
    m_rms_gamma_size = gamma_size;
    m_rms_reduction_threads = reduction_threads;
    m_rms_epsilon = static_cast<float>(rms->get_epsilon());
}

void VulkanStage::prepare_matmul_linear_kernel() {
    auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(m_node);
    OPENVINO_ASSERT(mm, "GFX Vulkan MatMul: node cast failed");
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan MatMul: unsupported element type ",
                    elem_type);
    const bool ta = mm->get_transpose_a();
    const bool tb = mm->get_transpose_b();
    if (m_matmul_linear_kernel &&
        m_matmul_linear_elem_type == elem_type &&
        m_matmul_linear_transpose_a == ta &&
        m_matmul_linear_transpose_b == tb) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_matmul_linear_module(ctx, elem_type, ta, tb);
    m_matmul_linear_kernel = compile_specialized_kernel_from_mlir(
        module,
        "matmul_linear",
        4,
        "GFX Vulkan MatMul: kernel compile failed: ");
    m_matmul_linear_elem_type = elem_type;
    m_matmul_linear_transpose_a = ta;
    m_matmul_linear_transpose_b = tb;
}

void VulkanStage::prepare_broadcast_kernel() {
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_broadcast_elem_type(elem_type),
                    "GFX Vulkan Broadcast: unsupported element type ",
                    elem_type);
    const auto rank = static_cast<size_t>(m_node->get_output_partial_shape(0).rank().get_length());
    OPENVINO_ASSERT(rank != 0, "GFX Vulkan Broadcast: scalar output is not supported");
    if (m_broadcast_kernel && m_broadcast_elem_type == elem_type && m_broadcast_rank == rank) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_broadcast_module(ctx, elem_type, rank);
    m_broadcast_kernel = compile_specialized_kernel_from_mlir(
        module,
        "broadcast_linear",
        3,
        "GFX Vulkan Broadcast: kernel compile failed: ");
    m_broadcast_elem_type = elem_type;
    m_broadcast_rank = rank;
}

void VulkanStage::prepare_select_kernel() {
    const ov::element::Type cond_type =
        resolve_stage_input_element_type(m_node, 0, m_inputs.size() > 0 ? m_inputs[0] : nullptr);
    const ov::element::Type data_type =
        resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(cond_type == ov::element::boolean,
                    "GFX Vulkan Select: unsupported condition element type ",
                    cond_type);
    OPENVINO_ASSERT(is_supported_broadcast_elem_type(data_type),
                    "GFX Vulkan Select: unsupported data element type ",
                    data_type);
    const auto rank = static_cast<size_t>(m_node->get_output_partial_shape(0).rank().get_length());
    OPENVINO_ASSERT(rank != 0, "GFX Vulkan Select: scalar output is not supported");
    if (m_select_kernel && m_select_cond_elem_type == cond_type &&
        m_select_data_elem_type == data_type && m_select_rank == rank) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_select_module(ctx, cond_type, data_type, rank);
    m_select_kernel = compile_specialized_kernel_from_mlir(
        module,
        "select_linear",
        5,
        "GFX Vulkan Select: kernel compile failed: ");
    m_select_cond_elem_type = cond_type;
    m_select_data_elem_type = data_type;
    m_select_rank = rank;
}

GpuStageSubmitPolicy VulkanStage::submit_policy() const {
    const auto opt_plan = optimization_plan();
    const auto& conv_plan = opt_plan.conv;
    GfxStageRuntimeTraits traits{};
    traits.binary_chunked = should_use_binary_chunked();
    traits.binary_same_shape = should_use_binary_same_shape();
    traits.binary_bias_add = should_use_binary_bias_add();
    traits.unary_chunked = should_use_unary_chunked();
    traits.softmax_chunked = should_use_softmax_chunked();
    traits.conv2d_3x3_direct = conv_plan.kind == GfxConvRouteKind::Direct3x3;
    traits.conv2d_chunked = conv_plan.kind == GfxConvRouteKind::Chunked;
    traits.group_conv2d_chunked = should_use_group_conv2d_chunked();
    traits.transpose_chunked = should_use_transpose_chunked();
    traits.split_concat_chunked = should_use_concat_chunked() ||
                                  ((m_type == "Split" || m_type == "VariadicSplit") && !has_absorbed_input_transpose());
    traits.convert_chunked = should_use_convert_chunked();
    auto plan = select_stage_optimization_plan(m_buffer_manager,
                                               GpuBackend::Vulkan,
                                               m_type,
                                               m_node,
                                               resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output),
                                               m_has_bias,
                                               m_has_activation,
                                               m_has_bn,
                                               traits);
    return plan.execution.submit;
}

void VulkanStage::execute(GpuCommandBufferHandle command_buffer) {
    const auto conv_plan = conv_route_plan();
    // Keep Split/Concat on compact Vulkan kernels. The generic tensor-result
    // lowering is correct on Metal but still too expensive on current mobile
    // Vulkan stacks for these layout-heavy stages.
    if ((m_type == "Split" || m_type == "VariadicSplit") && !has_absorbed_input_transpose()) {
        execute_split_chunked(command_buffer);
        return;
    }
    if (should_use_concat_chunked()) {
        execute_concat_chunked(command_buffer);
        return;
    }
    if (should_use_slice_chunked()) {
        execute_slice_chunked(command_buffer);
        return;
    }
    if (should_use_interpolate_chunked()) {
        execute_interpolate_chunked(command_buffer);
        return;
    }
    if (should_use_transpose_chunked()) {
        execute_transpose_chunked(command_buffer);
        return;
    }
    if (should_use_convert_chunked()) {
        execute_convert_chunked(command_buffer);
        return;
    }
    if (should_use_gather_linear()) {
        execute_gather_linear(command_buffer);
        return;
    }
    if (should_use_gather_embedding()) {
        execute_gather_embedding(command_buffer);
        return;
    }
    if (should_use_matmul_linear()) {
        execute_matmul_linear(command_buffer);
        return;
    }
    if (should_use_broadcast_chunked()) {
        execute_broadcast_chunked(command_buffer);
        return;
    }
    if (should_use_select_chunked()) {
        execute_select_chunked(command_buffer);
        return;
    }
    if (should_use_reduce_last_axis()) {
        execute_reduce_last_axis(command_buffer);
        return;
    }
    if (should_use_rms_chunked()) {
        execute_rms_chunked(command_buffer);
        return;
    }
    if (should_use_softmax_chunked()) {
        execute_softmax_chunked(command_buffer);
        return;
    }
    if (should_use_group_conv2d_chunked()) {
        execute_group_conv2d_chunked(command_buffer);
        return;
    }
    if (conv_plan.kind == GfxConvRouteKind::Direct3x3) {
        execute_conv2d_3x3_direct(command_buffer);
        return;
    }
    if (conv_plan.kind == GfxConvRouteKind::Chunked) {
        if (conv_plan.algorithm.kind == GfxConvAlgorithmKind::Im2ColMatMul ||
            conv_plan.algorithm.kind == GfxConvAlgorithmKind::Indirect) {
            MlirStage::execute(command_buffer);
            return;
        }
        execute_conv2d_chunked(command_buffer);
        return;
    }
    if (gfx_log_debug_enabled() && (m_type == "Add" || m_type == "Swish" || m_type == "Multiply" || m_type == "Sigmoid")) {
        GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
        const ov::Shape& dispatch_shape = !m_output_shape.empty() ? m_output_shape : (output ? output->shape : ov::Shape{});
        const auto elem_type = resolve_stage_element_type(m_node, output);
        gfx_log_debug("VulkanExec") << "Chunked candidates: type=" << m_type
                                    << " unary=" << (should_use_unary_chunked() ? "yes" : "no")
                                    << " binary=" << (should_use_binary_chunked() ? "yes" : "no")
                                    << " bias_broadcast_add="
                                    << ((m_type == "Add" && is_bias_broadcast_add(m_node)) ? "yes" : "no")
                                    << " inputs=" << m_inputs.size()
                                    << " extras=" << m_kernel_extra_inputs.size()
                                    << " out_rank=" << dispatch_shape.size()
                                    << " out_elems=" << (dispatch_shape.empty() ? 0 : tensor_elements(dispatch_shape))
                                    << " elem_type=" << elem_type;
    }
    if (should_use_binary_same_shape()) {
        execute_binary_same_shape(command_buffer);
        return;
    }
    if (should_use_binary_bias_add()) {
        execute_binary_bias_add(command_buffer);
        return;
    }
    if (should_use_binary_chunked()) {
        execute_binary_chunked(command_buffer);
        return;
    }
    if (should_use_unary_chunked()) {
        execute_unary_chunked(command_buffer);
        return;
    }
    MlirStage::execute(command_buffer);
}

void VulkanStage::enable_profiling(bool enable) {
    MlirStage::enable_profiling(enable);
}

void VulkanStage::set_profiler(void* profiler,
                               uint32_t node_id,
                               const std::string& node_name,
                               const std::string& node_type) {
    MlirStage::set_profiler(profiler, node_id, node_name, node_type);
}

void VulkanStage::set_inputs(const std::vector<GpuTensor*>& inputs) {
    MlirStage::set_inputs(inputs);
}

void VulkanStage::set_output(GpuTensor* output) {
    MlirStage::set_output(output);
}

void VulkanStage::set_outputs(const std::vector<std::unique_ptr<GpuTensor>>& outputs) {
    MlirStage::set_outputs(outputs);
}

std::unique_ptr<GpuStage> VulkanStage::clone() const {
    auto stage = std::make_unique<VulkanStage>(m_node);
    clone_into(*stage);
    return stage;
}

bool VulkanStage::fuse_activation(ActivationKind kind, float alpha) {
    return MlirStage::fuse_activation(kind, alpha);
}

bool VulkanStage::fuse_batchnorm(const BatchNormParams& params) {
    return MlirStage::fuse_batchnorm(params);
}

bool VulkanStage::fuse_bias(const BiasParams& params) {
    return MlirStage::fuse_bias(params);
}

bool VulkanStage::should_use_unary_chunked() const {
    if (unary_chunk_key(m_type).empty()) {
        return false;
    }
    auto resolve_input = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
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
    GpuTensor* input0 = resolve_input(0);
    if (!m_node && !input0) {
        return false;
    }
    if (m_node && (!m_node->get_input_partial_shape(0).rank().is_static() ||
                   !m_node->get_output_partial_shape(0).rank().is_static() ||
                   m_node->get_input_partial_shape(0).rank().get_length() !=
                       m_node->get_output_partial_shape(0).rank().get_length())) {
        return false;
    }
    ov::Shape dispatch_shape = !m_output_shape.empty() ? m_output_shape : (input0 ? input0->shape : ov::Shape{});
    if (dispatch_shape.empty() && m_node && m_node->get_output_partial_shape(0).is_static()) {
        dispatch_shape = m_node->get_output_shape(0);
    }
    const ov::element::Type et = resolve_stage_element_type(m_node, nullptr);
    if (!is_supported_linear_elem_type(et)) {
        return false;
    }
    return dispatch_shape.empty() || tensor_elements(dispatch_shape) >= kLargeLinearChunkElems;
}

bool VulkanStage::should_use_concat_chunked() const {
    if (m_type != "Concat" || has_absorbed_input_transpose() || !m_node) {
        return false;
    }
    auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(m_node);
    if (!concat || concat->get_input_size() == 0 || concat->get_output_size() != 1) {
        return false;
    }
    const auto out_pshape = m_node->get_output_partial_shape(0);
    if (!out_pshape.rank().is_static() || out_pshape.rank().get_length() == 0) {
        return false;
    }
    for (size_t input_idx = 0; input_idx < concat->get_input_size(); ++input_idx) {
        const auto in_pshape = m_node->get_input_partial_shape(input_idx);
        if (!in_pshape.rank().is_static() ||
            in_pshape.rank().get_length() != out_pshape.rank().get_length()) {
            return false;
        }
    }
    return is_supported_linear_elem_type(
        resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output));
}

bool VulkanStage::should_use_slice_chunked() const {
    if ((m_type != "Slice" && m_type != "StridedSlice") || !m_node || m_node->get_output_size() != 1) {
        return false;
    }
    const auto in_pshape = m_node->get_input_partial_shape(0);
    const auto out_pshape = m_node->get_output_partial_shape(0);
    if (!in_pshape.rank().is_static() || !out_pshape.rank().is_static()) {
        return false;
    }
    const auto rank = out_pshape.rank().get_length();
    if (rank <= 0 || rank > 6) {
        return false;
    }
    return resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output) != ov::element::dynamic;
}

bool VulkanStage::should_use_softmax_chunked() const {
    if (m_type != "Softmax" && m_type != "LogSoftmax") {
        return false;
    }
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    const ov::Shape& dispatch_shape =
        !m_output_shape.empty() ? m_output_shape : (output ? output->shape : ov::Shape{});
    if (dispatch_shape.empty()) {
        return false;
    }
    const ov::element::Type et = resolve_stage_element_type(m_node, nullptr);
    return is_supported_linear_elem_type(et) &&
           tensor_elements(dispatch_shape) >= kLargeLinearChunkElems;
}

bool VulkanStage::should_use_transpose_chunked() const {
    if (m_type != "Transpose" || !m_node || m_node->get_input_size() != 2 || m_node->get_output_size() != 1) {
        return false;
    }
    auto tr = ov::as_type_ptr<const ov::op::v1::Transpose>(m_node);
    if (!tr) {
        return false;
    }
    if (!m_node->get_input_partial_shape(0).is_static() || !m_node->get_output_partial_shape(0).is_static()) {
        return false;
    }
    auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(tr->input_value(1).get_node_shared_ptr());
    if (!perm_const) {
        return false;
    }
    const auto& out_shape = m_node->get_output_shape(0);
    if (out_shape.empty() || tensor_elements(out_shape) < kLargeLinearChunkElems) {
        return false;
    }
    return is_supported_linear_elem_type(resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output));
}

bool VulkanStage::should_skip_generic_kernel_compile(const GfxStageOptimizationPlan& plan) const {
    if ((m_type == "Split" || m_type == "VariadicSplit") && !has_absorbed_input_transpose()) {
        return true;
    }
    if (should_use_concat_chunked()) {
        return true;
    }
    if (should_use_slice_chunked()) {
        return true;
    }
    if (should_use_interpolate_chunked() || should_use_convert_chunked() || should_use_softmax_chunked() || should_use_binary_same_shape() ||
        should_use_gather_linear() || should_use_gather_embedding() || should_use_matmul_linear() || should_use_broadcast_chunked() || should_use_select_chunked() || should_use_reduce_last_axis() || should_use_rms_chunked() || should_use_binary_bias_add() || should_use_binary_chunked() || should_use_unary_chunked() ||
        should_use_transpose_chunked()) {
        return true;
    }
    if (should_use_group_conv2d_chunked() ||
        plan.conv.kind == GfxConvRouteKind::Direct3x3) {
        return true;
    }
    return plan.conv.kind == GfxConvRouteKind::Chunked &&
           plan.conv.algorithm.kind != GfxConvAlgorithmKind::Im2ColMatMul &&
           plan.conv.algorithm.kind != GfxConvAlgorithmKind::Indirect;
}

GfxStageOptimizationPlan VulkanStage::optimization_plan() const {
    GfxStageRuntimeTraits traits{};
    traits.binary_chunked = should_use_binary_chunked();
    traits.binary_same_shape = should_use_binary_same_shape();
    traits.binary_bias_add = should_use_binary_bias_add();
    traits.unary_chunked = should_use_unary_chunked();
    traits.softmax_chunked = should_use_softmax_chunked();
    traits.transpose_chunked = should_use_transpose_chunked();
    traits.split_concat_chunked = should_use_concat_chunked() ||
                                  ((m_type == "Split" || m_type == "VariadicSplit") && !has_absorbed_input_transpose());
    traits.convert_chunked = should_use_convert_chunked();
    return select_stage_optimization_plan(m_buffer_manager,
                                          GpuBackend::Vulkan,
                                          m_type,
                                          m_node,
                                          resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output),
                                          m_has_bias,
                                          m_has_activation,
                                          m_has_bn,
                                          traits);
}

GfxConvRoutePlan VulkanStage::conv_route_plan() const {
    return optimization_plan().conv;
}

bool VulkanStage::should_use_conv2d_chunked() const {
    return conv_route_plan().kind == GfxConvRouteKind::Chunked;
}

bool VulkanStage::should_use_interpolate_chunked() const {
    if (m_type != "Interpolate" || !m_node || m_node->get_input_size() == 0 || m_node->get_output_size() == 0) {
        return false;
    }
    if (!m_node->get_input_partial_shape(0).is_static() || !m_node->get_output_partial_shape(0).is_static()) {
        return false;
    }
    const auto in_shape = m_node->get_input_shape(0);
    const auto out_shape = m_node->get_output_shape(0);
    if (in_shape.size() != 4 || out_shape.size() != 4) {
        return false;
    }
    if (!ov::as_type_ptr<const ov::op::v0::Interpolate>(m_node) &&
        !ov::as_type_ptr<const ov::op::v4::Interpolate>(m_node) &&
        !ov::as_type_ptr<const ov::op::v11::Interpolate>(m_node)) {
        return false;
    }
    return is_supported_linear_elem_type(resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output));
}

bool VulkanStage::should_use_conv2d_1x1_chunked() const {
    return conv_route_plan().kind == GfxConvRouteKind::Direct1x1;
}

bool VulkanStage::should_use_conv2d_3x3_direct() const {
    return conv_route_plan().kind == GfxConvRouteKind::Direct3x3;
}

bool VulkanStage::should_use_group_conv2d_chunked() const {
    return conv_route_plan().kind == GfxConvRouteKind::GroupChunked && !has_absorbed_input_transpose();
}

bool VulkanStage::should_use_binary_chunked() const {
    const auto op_key = binary_chunk_key(m_type);
    if (op_key.empty()) {
        return false;
    }
    if (has_absorbed_input_transpose()) {
        return false;
    }
    if (!m_node || m_node->get_input_size() != 2 || m_node->get_output_size() != 1) {
        return false;
    }
    const auto in0_pshape = m_node->get_input_partial_shape(0);
    const auto in1_pshape = m_node->get_input_partial_shape(1);
    const auto out_pshape = m_node->get_output_partial_shape(0);
    if (!in0_pshape.rank().is_static() || !in1_pshape.rank().is_static() || !out_pshape.rank().is_static()) {
        return false;
    }
    if (out_pshape.rank().get_length() == 0) {
        return false;
    }
    ov::Shape dispatch_shape;
    if (!m_output_shape.empty()) {
        dispatch_shape = m_output_shape;
    } else if (out_pshape.is_static()) {
        dispatch_shape = m_node->get_output_shape(0);
    }
    if (!is_supported_binary_io_types(m_node->get_input_element_type(0),
                                      m_node->get_input_element_type(1),
                                      m_node->get_output_element_type(0),
                                      op_key)) {
        return false;
    }
    return dispatch_shape.empty() || tensor_elements(dispatch_shape) >= kLargeLinearChunkElems;
}

bool VulkanStage::should_use_binary_same_shape() const {
    const auto op_key = binary_chunk_key(m_type);
    if (op_key.empty() || !m_node || m_node->get_input_size() != 2 || m_node->get_output_size() != 1) {
        return false;
    }
    if (has_absorbed_input_transpose()) {
        return false;
    }
    if (!m_node->get_input_partial_shape(0).is_static() ||
        !m_node->get_input_partial_shape(1).is_static() ||
        !m_node->get_output_partial_shape(0).is_static()) {
        return false;
    }
    const auto& out_shape = m_node->get_output_shape(0);
    if (out_shape.empty()) {
        return false;
    }
    return m_node->get_input_shape(0) == out_shape &&
           m_node->get_input_shape(1) == out_shape &&
           is_supported_binary_io_types(m_node->get_input_element_type(0),
                                        m_node->get_input_element_type(1),
                                        m_node->get_output_element_type(0),
                                        op_key) &&
           tensor_elements(out_shape) >= kLargeLinearChunkElems;
}

bool VulkanStage::should_use_binary_bias_add() const {
    // Keep bias-shaped Add on the shared MLIR/SPIR-V path for Vulkan.
    // The dedicated fast-path has shown correctness issues on mobile-class
    // drivers even when fed with clean externalized inputs.
    return false;
}

bool VulkanStage::should_use_convert_chunked() const {
    if (m_type != "Convert" || !m_node || m_node->get_input_size() != 1 || m_node->get_output_size() != 1) {
        return false;
    }
    auto cvt = ov::as_type_ptr<const ov::op::v0::Convert>(m_node);
    if (!cvt) {
        return false;
    }
    const auto in_pshape = m_node->get_input_partial_shape(0);
    const auto out_pshape = m_node->get_output_partial_shape(0);
    if (!in_pshape.rank().is_static() || !out_pshape.rank().is_static() ||
        in_pshape.rank().get_length() != out_pshape.rank().get_length()) {
        return false;
    }
    if (!is_supported_linear_convert_type(m_node->get_input_element_type(0), m_node->get_output_element_type(0))) {
        return false;
    }
    if (in_pshape.is_static() && out_pshape.is_static()) {
        const auto& in_shape = m_node->get_input_shape(0);
        const auto& out_shape = m_node->get_output_shape(0);
        return !in_shape.empty() && !out_shape.empty() &&
               tensor_elements(in_shape) == tensor_elements(out_shape) &&
               tensor_elements(out_shape) >= kLargeLinearChunkElems;
    }
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    const ov::Shape runtime_shape =
        output && !output->shape.empty()
            ? output->shape
            : (!m_inputs.empty() && m_inputs[0] && !m_inputs[0]->shape.empty() ? m_inputs[0]->shape : ov::Shape{});
    return runtime_shape.empty() || tensor_elements(runtime_shape) >= kLargeLinearChunkElems;
}

bool VulkanStage::should_use_gather_embedding() const {
    if (m_type != "Gather" || !m_node || m_node->get_input_size() != 3 || m_node->get_output_size() != 1) {
        return false;
    }
    if (!(ov::as_type_ptr<const ov::op::v1::Gather>(m_node) ||
          ov::as_type_ptr<const ov::op::v7::Gather>(m_node) ||
          ov::as_type_ptr<const ov::op::v8::Gather>(m_node))) {
        return false;
    }
    int64_t batch_dims = 0;
    if (auto gather_v7 = ov::as_type_ptr<const ov::op::v7::Gather>(m_node)) {
        batch_dims = gather_v7->get_batch_dims();
    } else if (auto gather_v8 = ov::as_type_ptr<const ov::op::v8::Gather>(m_node)) {
        batch_dims = gather_v8->get_batch_dims();
    }
    if (batch_dims != 0) {
        return false;
    }
    const auto data_pshape = m_node->get_input_partial_shape(0);
    const auto idx_pshape = m_node->get_input_partial_shape(1);
    const auto out_pshape = m_node->get_output_partial_shape(0);
    if (!data_pshape.is_static() ||
        !idx_pshape.rank().is_static() ||
        !out_pshape.rank().is_static() ||
        data_pshape.rank().get_length() != 2 ||
        out_pshape.rank().get_length() != idx_pshape.rank().get_length() + 1) {
        return false;
    }
    auto axis_c = ov::as_type_ptr<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(2));
    if (!axis_c) {
        return false;
    }
    const auto axis_v = axis_c->cast_vector<int64_t>();
    if (axis_v.size() != 1) {
        return false;
    }
    int64_t axis = axis_v[0];
    if (axis < 0) {
        axis += 2;
    }
    return axis == 0 &&
           data_pshape[1].is_static() &&
           is_supported_gather_embedding_type(m_node->get_input_element_type(0),
                                              m_node->get_input_element_type(1));
}

bool VulkanStage::should_use_gather_linear() const {
    if (m_type != "Gather" || !m_node || m_node->get_input_size() != 3 || m_node->get_output_size() != 1) {
        return false;
    }
    if (!(ov::as_type_ptr<const ov::op::v1::Gather>(m_node) ||
          ov::as_type_ptr<const ov::op::v7::Gather>(m_node) ||
          ov::as_type_ptr<const ov::op::v8::Gather>(m_node))) {
        return false;
    }
    int64_t batch_dims = 0;
    if (auto gather_v7 = ov::as_type_ptr<const ov::op::v7::Gather>(m_node)) {
        batch_dims = gather_v7->get_batch_dims();
    } else if (auto gather_v8 = ov::as_type_ptr<const ov::op::v8::Gather>(m_node)) {
        batch_dims = gather_v8->get_batch_dims();
    }
    if (batch_dims != 0) {
        return false;
    }
    const auto data_pshape = m_node->get_input_partial_shape(0);
    const auto idx_pshape = m_node->get_input_partial_shape(1);
    const auto out_pshape = m_node->get_output_partial_shape(0);
    if (!data_pshape.rank().is_static() || !idx_pshape.rank().is_static() || !out_pshape.rank().is_static()) {
        return false;
    }
    auto axis_c = ov::as_type_ptr<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(2));
    if (!axis_c) {
        return false;
    }
    const auto axis_v = axis_c->cast_vector<int64_t>();
    if (axis_v.size() != 1) {
        return false;
    }
    return is_supported_gather_linear_type(m_node->get_input_element_type(0), m_node->get_input_element_type(1));
}

bool VulkanStage::should_use_reduce_last_axis() const {
    if (reduce_last_axis_key(m_type).empty() || !m_node || m_node->get_input_size() < 2 || m_node->get_output_size() != 1) {
        return false;
    }
    ov::AxisSet axes;
    bool keep_dims = false;
    if (auto mean = ov::as_type_ptr<const ov::op::v1::ReduceMean>(m_node)) {
        if (!mean->reduction_axes_constant()) {
            return false;
        }
        axes = mean->get_reduction_axes();
        keep_dims = mean->get_keep_dims();
    } else if (auto sum = ov::as_type_ptr<const ov::op::v1::ReduceSum>(m_node)) {
        if (!sum->reduction_axes_constant()) {
            return false;
        }
        axes = sum->get_reduction_axes();
        keep_dims = sum->get_keep_dims();
    } else {
        return false;
    }
    (void)keep_dims;
    const auto in_pshape = m_node->get_input_partial_shape(0);
    const auto out_pshape = m_node->get_output_partial_shape(0);
    if (!in_pshape.rank().is_static() || !out_pshape.rank().is_static() || axes.size() != 1) {
        return false;
    }
    const auto in_rank = static_cast<size_t>(in_pshape.rank().get_length());
    if (in_rank == 0 || *axes.begin() != in_rank - 1) {
        return false;
    }
    if (!in_pshape[in_rank - 1].is_static()) {
        return false;
    }
    const ov::element::Type et = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    return is_supported_broadcast_elem_type(et);
}

bool VulkanStage::should_use_rms_chunked() const {
    if (m_type != "RMS" || !m_node || m_node->get_input_size() != 2 || m_node->get_output_size() != 1) {
        return false;
    }
    auto rms = ov::as_type_ptr<const ov::op::internal::RMS>(m_node);
    if (!rms) {
        return false;
    }
    const auto in_pshape = m_node->get_input_partial_shape(0);
    const auto gamma_pshape = m_node->get_input_partial_shape(1);
    if (!in_pshape.rank().is_static() || in_pshape.rank().get_length() < 1 || !gamma_pshape.is_static()) {
        return false;
    }
    const auto rank = static_cast<size_t>(in_pshape.rank().get_length());
    if (!in_pshape[rank - 1].is_static()) {
        return false;
    }
    const auto hidden = static_cast<size_t>(in_pshape[rank - 1].get_length());
    if (hidden == 0) {
        return false;
    }
    const auto gamma_size = ov::shape_size(gamma_pshape.to_shape());
    if (gamma_size != 1 && gamma_size != hidden) {
        return false;
    }
    return is_supported_linear_elem_type(m_node->get_input_element_type(0)) &&
           is_supported_linear_elem_type(m_node->get_input_element_type(1)) &&
           is_supported_linear_elem_type(m_node->get_output_element_type(0));
}

bool VulkanStage::should_use_matmul_linear() const {
    if (m_type != "MatMul" || !m_node || m_node->get_input_size() != 2 || m_node->get_output_size() != 1) {
        return false;
    }
    auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(m_node);
    if (!mm) {
        return false;
    }
    const auto a_pshape = m_node->get_input_partial_shape(0);
    const auto b_pshape = m_node->get_input_partial_shape(1);
    const auto out_pshape = m_node->get_output_partial_shape(0);
    if (!a_pshape.rank().is_static() || !b_pshape.rank().is_static()) {
        return false;
    }
    const bool ta = mm->get_transpose_a();
    const bool tb = mm->get_transpose_b();
    const auto a_rank = a_pshape.rank().get_length();
    const auto b_rank = b_pshape.rank().get_length();
    if (a_rank < 2 || a_rank > 4 || b_rank < 2 || b_rank > 4) {
        return false;
    }
    if (out_pshape.rank().is_static()) {
        const auto out_rank = out_pshape.rank().get_length();
        if (out_rank < 2 || out_rank > 4) {
            return false;
        }
    }
    const auto a_k = a_pshape[static_cast<size_t>(a_rank - (ta ? 2 : 1))];
    if (a_k.is_static()) {
        const auto b_k = b_pshape[static_cast<size_t>(b_rank - (tb ? 1 : 2))];
        if (b_k.is_static() && a_k.get_length() != b_k.get_length()) {
            if (tb) {
                return false;
            }
            const auto b_last = b_pshape[static_cast<size_t>(b_rank - 1)];
            if (!b_last.is_static() || a_k.get_length() != b_last.get_length()) {
                return false;
            }
        }
    }
    const ov::element::Type et = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    return is_supported_linear_elem_type(et);
}

bool VulkanStage::should_use_broadcast_chunked() const {
    if (m_type != "Broadcast" || !m_node || m_node->get_input_size() < 2 || m_node->get_output_size() != 1) {
        return false;
    }
    if (!ov::as_type_ptr<const ov::op::v1::Broadcast>(m_node) &&
        !ov::as_type_ptr<const ov::op::v3::Broadcast>(m_node)) {
        return false;
    }
    const auto in_pshape = m_node->get_input_partial_shape(0);
    const auto out_pshape = m_node->get_output_partial_shape(0);
    if (!in_pshape.rank().is_static() || !out_pshape.rank().is_static()) {
        return false;
    }
    const auto rank = out_pshape.rank().get_length();
    if (rank <= 0 || rank > 6) {
        return false;
    }
    const ov::element::Type et = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    return is_supported_broadcast_elem_type(et);
}

bool VulkanStage::should_use_select_chunked() const {
    if (m_type != "Select" || !m_node || m_node->get_input_size() != 3 || m_node->get_output_size() != 1) {
        return false;
    }
    if (has_absorbed_input_transpose()) {
        return false;
    }
    if (!ov::as_type_ptr<const ov::op::v1::Select>(m_node)) {
        return false;
    }
    for (size_t input_idx = 0; input_idx < 3; ++input_idx) {
        if (!m_node->get_input_partial_shape(input_idx).rank().is_static()) {
            return false;
        }
    }
    const auto out_pshape = m_node->get_output_partial_shape(0);
    if (!out_pshape.rank().is_static()) {
        return false;
    }
    const auto rank = out_pshape.rank().get_length();
    if (rank <= 0 || rank > 6) {
        return false;
    }
    const ov::element::Type cond_type =
        resolve_stage_input_element_type(m_node, 0, m_inputs.size() > 0 ? m_inputs[0] : nullptr);
    const ov::element::Type true_type =
        resolve_stage_input_element_type(m_node, 1, m_inputs.size() > 1 ? m_inputs[1] : nullptr);
    const ov::element::Type false_type =
        resolve_stage_input_element_type(m_node, 2, m_inputs.size() > 2 ? m_inputs[2] : nullptr);
    const ov::element::Type data_type =
        resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    return cond_type == ov::element::boolean &&
           true_type == data_type &&
           false_type == data_type &&
           is_supported_broadcast_elem_type(data_type);
}

std::shared_ptr<ICompiledKernel> VulkanStage::compile_kernel(const KernelSource& source,
                                                             std::string* log) {
    VulkanCodegenBackend backend;
    return backend.compile(source, log);
}

KernelExecutionHooks* VulkanStage::prepare_profiling(ProfileState& state,
                                                     KernelExecutionHooks& hooks) {
    auto* vk_profiler = static_cast<VulkanProfiler*>(profiler_handle());
    if (!vk_profiler) {
        return nullptr;
    }
    vk_profiler->begin_node(profile_node_id(),
                            profile_node_name().c_str(),
                            profile_node_type().c_str(),
                            "GFX");
    const auto sample = vk_profiler->reserve_samples();
    hooks.on_begin = [vk_profiler, sample](GpuCommandEncoderHandle enc) {
        vk_profiler->write_timestamp(reinterpret_cast<VkCommandBuffer>(enc), sample.begin);
    };
    hooks.on_end = [vk_profiler, sample](GpuCommandEncoderHandle enc) {
        vk_profiler->write_timestamp(reinterpret_cast<VkCommandBuffer>(enc), sample.end);
    };
    hooks.on_complete = [vk_profiler, sample, node_id = profile_node_id()]() {
        vk_profiler->end_node(node_id, sample);
    };
    hooks.on_event = [vk_profiler](std::string_view event) {
        vk_profiler->increment_counter(event);
        if (event == "vulkan_owns_command_buffer") {
            vk_profiler->record_segment("hazard", "owns_command_buffer", std::chrono::microseconds{0});
        } else if (event == "vulkan_internal_submit_wait") {
            vk_profiler->record_segment("hazard", "internal_submit_wait", std::chrono::microseconds{0});
        }
    };
    hooks.on_counter = [vk_profiler](std::string_view name, uint64_t delta) {
        vk_profiler->increment_counter(name, delta);
    };
    hooks.on_segment = [vk_profiler](std::string_view phase,
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
        vk_profiler->record_segment(phase,
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
    (void)state;
    return &hooks;
}

void VulkanStage::finalize_profiling(const ProfileState& state) {
    (void)state;
}

mlir::ModuleOp VulkanStage::build_linear_unary_module(mlir::MLIRContext& ctx,
                                                      const ov::element::Type& et,
                                                      const std::string& op_key) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    math::MathDialect>();

    Type elem_ty;
    switch (et) {
        case ov::element::f16: elem_ty = Float16Type::get(&ctx); break;
        case ov::element::f32: elem_ty = Float32Type::get(&ctx); break;
        default: OPENVINO_THROW("GFX Vulkan unary chunked: unsupported element type ", et);
    }
    Type compute_ty = elem_ty;
    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), m_has_bias ? 4 : 3));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds",
                 m_has_bias ? make_i32_array_attr({1, 1, 1, 1}) : make_i32_array_attr({1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices",
                 m_has_bias ? make_i32_array_attr({0, 1, 2, 3}) : make_i32_array_attr({0, 1, 2}));
    auto in_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto out_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto param_ty = MemRefType::get({2}, IntegerType::get(&ctx, 32));
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "linear_unary",
                                                 b.getFunctionType(TypeRange{in_ty, param_ty, out_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(mlir::DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));
    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto c0 = body.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = body.create<arith::ConstantIndexOp>(loc, 1);
    auto load_param = [&](int idx) -> Value {
        auto idx_val = body.create<arith::ConstantIndexOp>(loc, idx);
        auto v_i32 = body.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{idx_val});
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), v_i32);
    };
    auto cast_to_compute = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == elem_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, elem_ty, value);
    };
    auto emit_unary = [&](OpBuilder& builder, Value x) -> Value {
        auto xc = cast_to_compute(builder, x);
        if (op_key == "swish") {
            auto one = builder.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 1.0));
            auto zero = builder.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 0.0));
            auto cond = builder.create<arith::CmpFOp>(loc,
                                                      arith::CmpFPredicate::OGE,
                                                      xc,
                                                      zero);
            auto neg = builder.create<arith::NegFOp>(loc, xc);
            auto exp_neg = builder.create<math::ExpOp>(loc, neg);
            auto pos_denom = builder.create<arith::AddFOp>(loc, one, exp_neg);
            auto pos = builder.create<arith::DivFOp>(loc, xc, pos_denom);
            auto exp_pos = builder.create<math::ExpOp>(loc, xc);
            auto neg_denom = builder.create<arith::AddFOp>(loc, one, exp_pos);
            auto neg_num = builder.create<arith::MulFOp>(loc, xc, exp_pos);
            auto neg_res = builder.create<arith::DivFOp>(loc, neg_num, neg_denom);
            return cast_to_output(builder, builder.create<arith::SelectOp>(loc, cond, pos, neg_res));
        }
        if (op_key == "sigmoid") {
            auto neg = builder.create<arith::NegFOp>(loc, xc);
            auto exp = builder.create<math::ExpOp>(loc, neg);
            auto one = builder.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 1.0));
            auto denom = builder.create<arith::AddFOp>(loc, one, exp);
            return cast_to_output(builder, builder.create<arith::DivFOp>(loc, one, denom));
        }
        if (op_key == "relu") {
            auto zero = builder.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 0.0));
            return cast_to_output(builder, builder.create<arith::MaximumFOp>(loc, xc, zero));
        }
        if (op_key == "tanh") {
            return cast_to_output(builder, builder.create<math::TanhOp>(loc, xc));
        }
        if (op_key == "sqrt") {
            return cast_to_output(builder, builder.create<math::SqrtOp>(loc, xc));
        }
        if (op_key == "exp") {
            return cast_to_output(builder, builder.create<math::ExpOp>(loc, xc));
        }
        if (op_key == "log") {
            return cast_to_output(builder, builder.create<math::LogOp>(loc, xc));
        }
        if (op_key == "floor") {
            return cast_to_output(builder, builder.create<math::FloorOp>(loc, xc));
        }
        if (op_key == "ceil") {
            return cast_to_output(builder, builder.create<math::CeilOp>(loc, xc));
        }
        if (op_key == "neg") {
            return cast_to_output(builder, builder.create<arith::NegFOp>(loc, xc));
        }
        if (op_key == "sin") {
            return cast_to_output(builder, builder.create<math::SinOp>(loc, xc));
        }
        if (op_key == "cos") {
            return cast_to_output(builder, builder.create<math::CosOp>(loc, xc));
        }
        OPENVINO_THROW("GFX Vulkan unary chunked: unsupported op ", op_key);
    };

    Value offset = load_param(0);
    Value count = load_param(1);
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value local_idx = body.create<arith::AddIOp>(loc,
                                                 body.create<arith::MulIOp>(loc, bid, bdim),
                                                 tid);
    auto active =
        body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local_idx, count);
    auto active_if = body.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value idx = then_builder.create<arith::AddIOp>(loc, offset, local_idx);
        Value val = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{idx});
        Value out = emit_unary(then_builder, val);
        then_builder.create<memref::StoreOp>(loc, out, fn.getArgument(2), ValueRange{idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_conv2d_1x1_module(mlir::MLIRContext& ctx,
                                                    const ov::element::Type& et,
                                                    uint32_t threads_per_group) {
    using namespace mlir;
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    OPENVINO_ASSERT(conv, "GFX Vulkan conv2d 1x1: node cast failed");

    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    math::MathDialect>();

    Type elem_ty;
    switch (et) {
        case ov::element::f16: elem_ty = Float16Type::get(&ctx); break;
        case ov::element::f32: elem_ty = Float32Type::get(&ctx); break;
        default: OPENVINO_THROW("GFX Vulkan conv2d 1x1: unsupported element type ", et);
    }
    Type compute_ty = elem_ty;
    OPENVINO_ASSERT(!m_has_activation || m_activation == ActivationKind::Relu,
                    "GFX Vulkan conv2d 1x1: only Relu activation fusion is supported");

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), m_has_bias ? 5 : 4));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds",
                 m_has_bias ? make_i32_array_attr({1, 1, 1, 1, 1}) : make_i32_array_attr({1, 1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices",
                 m_has_bias ? make_i32_array_attr({0, 1, 2, 3, 4}) : make_i32_array_attr({0, 1, 2, 3}));

    auto input_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto weight_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto bias_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto param_ty = MemRefType::get({7}, IntegerType::get(&ctx, 32));
    auto output_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);

    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "conv2d_1x1",
                                                 b.getFunctionType(m_has_bias ? TypeRange{input_ty, weight_ty, bias_ty, param_ty, output_ty}
                                                                              : TypeRange{input_ty, weight_ty, param_ty, output_ty},
                                                                   {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(mlir::DenseI32ArrayAttr::get(&ctx, {static_cast<int32_t>(std::max<uint32_t>(1u, threads_per_group)), 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto cast_to_compute = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == elem_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, elem_ty, value);
    };
    auto load_param = [&](OpBuilder& builder, int idx) -> Value {
        auto idx_val = builder.create<arith::ConstantIndexOp>(loc, idx);
        auto arg_idx = m_has_bias ? 3 : 2;
        auto v_i32 = builder.create<memref::LoadOp>(loc, fn.getArgument(arg_idx), ValueRange{idx_val});
        return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), v_i32);
    };

    Value c_in = load_param(body, 1);
    Value out_h = load_param(body, 5);
    Value out_w = load_param(body, 6);
    Value c_out = load_param(body, 4);
    Value total = body.create<arith::MulIOp>(loc,
                                             load_param(body, 0),
                                             body.create<arith::MulIOp>(loc,
                                                                        c_out,
                                                                        body.create<arith::MulIOp>(loc, out_h, out_w)));
    auto zero = body.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 0.0f));
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value global_idx = body.create<arith::AddIOp>(loc,
                                                  body.create<arith::MulIOp>(loc, bid, bdim),
                                                  tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, global_idx, total);
    auto active_if = body.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value rem = global_idx;
        Value ow = then_builder.create<arith::RemUIOp>(loc, rem, out_w);
        rem = then_builder.create<arith::DivUIOp>(loc, rem, out_w);
        Value oh = then_builder.create<arith::RemUIOp>(loc, rem, out_h);
        rem = then_builder.create<arith::DivUIOp>(loc, rem, out_h);
        Value oc = then_builder.create<arith::RemUIOp>(loc, rem, c_out);
        Value n = then_builder.create<arith::DivUIOp>(loc, rem, c_out);

        Value acc = zero.getResult();
        auto for_ic = then_builder.create<scf::ForOp>(loc,
                                                      then_builder.create<arith::ConstantIndexOp>(loc, 0),
                                                      c_in,
                                                      then_builder.create<arith::ConstantIndexOp>(loc, 1),
                                                      ValueRange{acc});
        {
            auto loop_builder = OpBuilder::atBlockBegin(for_ic.getBody());
            Value ic = for_ic.getInductionVar();
            Value input_offset = loop_builder.create<arith::AddIOp>(
                loc,
                loop_builder.create<arith::MulIOp>(
                    loc,
                    loop_builder.create<arith::AddIOp>(
                        loc,
                        loop_builder.create<arith::MulIOp>(loc, n, c_in),
                        ic),
                    loop_builder.create<arith::MulIOp>(loc, out_h, out_w)),
                loop_builder.create<arith::AddIOp>(
                    loc,
                    loop_builder.create<arith::MulIOp>(loc, oh, out_w),
                    ow));
            Value weight_offset = loop_builder.create<arith::AddIOp>(
                loc,
                loop_builder.create<arith::MulIOp>(loc, oc, c_in),
                ic);
            Value input_val = loop_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{input_offset});
            Value weight_val = loop_builder.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{weight_offset});
            Value prod = loop_builder.create<arith::MulFOp>(
                loc,
                cast_to_compute(loop_builder, input_val),
                cast_to_compute(loop_builder, weight_val));
            Value next = loop_builder.create<arith::AddFOp>(loc, for_ic.getRegionIterArgs()[0], prod);
            loop_builder.create<scf::YieldOp>(loc, next);
        }
        acc = for_ic.getResult(0);
        if (m_has_bias) {
            Value bias_val = then_builder.create<memref::LoadOp>(loc, fn.getArgument(2), ValueRange{oc});
            acc = then_builder.create<arith::AddFOp>(loc, acc, cast_to_compute(then_builder, bias_val));
        }
        if (m_has_activation) {
            acc = then_builder.create<arith::MaximumFOp>(loc, acc, zero);
        }
        Value output_offset = then_builder.create<arith::AddIOp>(
            loc,
            then_builder.create<arith::MulIOp>(
                loc,
                then_builder.create<arith::AddIOp>(
                    loc,
                    then_builder.create<arith::MulIOp>(loc, n, c_out),
                    oc),
                then_builder.create<arith::MulIOp>(loc, out_h, out_w)),
            then_builder.create<arith::AddIOp>(
                loc,
                then_builder.create<arith::MulIOp>(loc, oh, out_w),
                ow));
        then_builder.create<memref::StoreOp>(loc,
                                             cast_to_output(then_builder, acc),
                                             fn.getArgument(m_has_bias ? 4 : 3),
                                             ValueRange{output_offset});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_conv2d_3x3_direct_module(mlir::MLIRContext& ctx,
                                                           const ov::element::Type& et,
                                                           uint32_t output_channel_block,
                                                           uint32_t threads_per_group,
                                                           const std::string& variant) {
    using namespace mlir;
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    OPENVINO_ASSERT(conv, "GFX Vulkan conv2d 3x3 direct: node cast failed");
    const auto& in_shape = conv->get_input_shape(0);
    const auto& w_shape = conv->get_input_shape(1);
    const auto& out_shape = conv->get_output_shape(0);

    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    math::MathDialect>();

    Type elem_ty;
    switch (et) {
        case ov::element::f16: elem_ty = Float16Type::get(&ctx); break;
        case ov::element::f32: elem_ty = Float32Type::get(&ctx); break;
        default: OPENVINO_THROW("GFX Vulkan conv2d 3x3 direct: unsupported element type ", et);
    }
    Type compute_ty = elem_ty;
    OPENVINO_ASSERT(!m_has_activation || m_activation == ActivationKind::Relu,
                    "GFX Vulkan conv2d 3x3 direct: only Relu activation fusion is supported");
    const uint32_t oc_block = std::max<uint32_t>(1u, output_channel_block);
    const bool spatial_xy_variant = variant.rfind("conv2d_direct_xy", 0) == 0;
    const bool xy32x2_variant = variant == "conv2d_direct_xy32x2" || variant == "conv2d_direct_xy32x2_dense_s2";
    const bool xy16x4_variant = variant == "conv2d_direct_xy16x4" || variant == "conv2d_direct_xy16x4_dense_s2";
    const int32_t tile_x = xy32x2_variant ? 32 : (xy16x4_variant ? 16 : 8);
    const int32_t tile_y = xy32x2_variant ? 2 : (xy16x4_variant ? 4 : 8);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), m_has_bias ? 4 : 3));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds",
                 m_has_bias ? make_i32_array_attr({1, 1, 1, 1}) : make_i32_array_attr({1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices",
                 m_has_bias ? make_i32_array_attr({0, 1, 2, 3}) : make_i32_array_attr({0, 1, 2}));

    auto input_ty = MemRefType::get({static_cast<int64_t>(in_shape.at(0)),
                                     static_cast<int64_t>(in_shape.at(1)),
                                     static_cast<int64_t>(in_shape.at(2)),
                                     static_cast<int64_t>(in_shape.at(3))},
                                    elem_ty);
    auto weight_ty = MemRefType::get({static_cast<int64_t>(w_shape.at(0)),
                                      static_cast<int64_t>(w_shape.at(1)),
                                      static_cast<int64_t>(w_shape.at(2)),
                                      static_cast<int64_t>(w_shape.at(3))},
                                     elem_ty);
    auto bias_ty = MemRefType::get({static_cast<int64_t>(out_shape.at(1))}, elem_ty);
    auto output_ty = MemRefType::get({static_cast<int64_t>(out_shape.at(0)),
                                      static_cast<int64_t>(out_shape.at(1)),
                                      static_cast<int64_t>(out_shape.at(2)),
                                      static_cast<int64_t>(out_shape.at(3))},
                                     elem_ty);

    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    SmallVector<Type, 4> arg_types{input_ty, weight_ty};
    if (m_has_bias) {
        arg_types.push_back(bias_ty);
    }
    arg_types.push_back(output_ty);
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 spatial_xy_variant ? "conv2d_3x3_direct_xy"
                                                                    : (oc_block == 1 ? "conv2d_3x3_direct"
                                                                                     : "conv2d_3x3_direct_oc2"),
                                                 b.getFunctionType(arg_types, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(
        &ctx,
        spatial_xy_variant ? SmallVector<int32_t, 3>{tile_x, tile_y, 1}
                           : SmallVector<int32_t, 3>{static_cast<int32_t>(threads_per_group), 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto cast_to_compute = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == elem_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, elem_ty, value);
    };

    Value c0 = body.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = body.create<arith::ConstantIndexOp>(loc, 1);
    Value c_out = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape.at(1)));
    Value oc_block_val = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(oc_block));
    Value c_out_groups = body.create<arith::ConstantIndexOp>(
        loc, static_cast<int64_t>((out_shape.at(1) + oc_block - 1) / oc_block));
    Value in_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape.at(2)));
    Value in_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape.at(3)));
    Value out_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape.at(2)));
    Value out_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape.at(3)));
    Value stride_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_strides().at(0)));
    Value stride_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_strides().at(1)));
    Value pad_top = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_pads_begin().at(0)));
    Value pad_left = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_pads_begin().at(1)));
    auto zero = body.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 0.0f));
    Value ow;
    Value oh;
    Value n;
    Value oc_base;
    scf::IfOp active_if;
    if (spatial_xy_variant) {
        Value bid_x = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
        Value bid_y = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
        Value bid_z = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::z);
        Value bdim_x = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
        Value bdim_y = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::y);
        Value tid_x = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
        Value tid_y = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
        ow = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid_x, bdim_x), tid_x);
        oh = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid_y, bdim_y), tid_y);
        Value oc_group = body.create<arith::RemUIOp>(loc, bid_z, c_out_groups);
        n = body.create<arith::DivUIOp>(loc, bid_z, c_out_groups);
        oc_base = body.create<arith::MulIOp>(loc, oc_group, oc_block_val);
        auto ow_valid = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, ow, out_w);
        auto oh_valid = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, oh, out_h);
        auto active = body.create<arith::AndIOp>(loc, ow_valid, oh_valid);
        active_if = body.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
    } else {
        Value total = body.create<arith::ConstantIndexOp>(
            loc,
            static_cast<int64_t>(out_shape.at(0) * ((out_shape.at(1) + oc_block - 1) / oc_block) * out_shape.at(2) *
                                 out_shape.at(3)));
        Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
        Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
        Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
        Value global_idx = body.create<arith::AddIOp>(loc,
                                                      body.create<arith::MulIOp>(loc, bid, bdim),
                                                      tid);
        auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, global_idx, total);
        active_if = body.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
        OpBuilder then_builder = active_if.getThenBodyBuilder();
        Value rem = global_idx;
        ow = then_builder.create<arith::RemUIOp>(loc, rem, out_w);
        rem = then_builder.create<arith::DivUIOp>(loc, rem, out_w);
        oh = then_builder.create<arith::RemUIOp>(loc, rem, out_h);
        rem = then_builder.create<arith::DivUIOp>(loc, rem, out_h);
        Value oc_group = then_builder.create<arith::RemUIOp>(loc, rem, c_out_groups);
        n = then_builder.create<arith::DivUIOp>(loc, rem, c_out_groups);
        oc_base = then_builder.create<arith::MulIOp>(loc, oc_group, oc_block_val);
    }
    {
        auto then_builder = active_if.getThenBodyBuilder();
        SmallVector<Value, 4> accs;
        SmallVector<Value, 4> oc_values;
        SmallVector<Value, 4> oc_valids;
        accs.reserve(oc_block);
        oc_values.reserve(oc_block);
        oc_valids.reserve(oc_block);
        for (uint32_t lane = 0; lane < oc_block; ++lane) {
            Value lane_val = then_builder.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(lane));
            Value oc = then_builder.create<arith::AddIOp>(loc, oc_base, lane_val);
            oc_values.push_back(oc);
            oc_valids.push_back(
                then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, oc, c_out));
            accs.push_back(zero.getResult());
        }
        Value base_ih = then_builder.create<arith::SubIOp>(
            loc, then_builder.create<arith::MulIOp>(loc, oh, stride_h), pad_top);
        Value base_iw = then_builder.create<arith::SubIOp>(
            loc, then_builder.create<arith::MulIOp>(loc, ow, stride_w), pad_left);
        auto one = then_builder.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 1.0f));
        const auto in_channels = static_cast<int64_t>(in_shape.at(1));
        for (int64_t ic_idx = 0; ic_idx < in_channels; ++ic_idx) {
            Value ic = then_builder.create<arith::ConstantIndexOp>(loc, ic_idx);
            for (int64_t kh_idx = 0; kh_idx < 3; ++kh_idx) {
                Value kh = then_builder.create<arith::ConstantIndexOp>(loc, kh_idx);
                for (int64_t kw_idx = 0; kw_idx < 3; ++kw_idx) {
                    Value kw = then_builder.create<arith::ConstantIndexOp>(loc, kw_idx);
                    Value ih = then_builder.create<arith::AddIOp>(loc, base_ih, kh);
                    Value iw = then_builder.create<arith::AddIOp>(loc, base_iw, kw);

                    auto ih_ge0 = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, ih, c0);
                    auto ih_lt = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ih, in_h);
                    auto iw_ge0 = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, iw, c0);
                    auto iw_lt = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iw, in_w);
                    auto in_bounds = then_builder.create<arith::AndIOp>(loc, ih_ge0, ih_lt);
                    in_bounds = then_builder.create<arith::AndIOp>(loc, in_bounds, iw_ge0);
                    in_bounds = then_builder.create<arith::AndIOp>(loc, in_bounds, iw_lt);

                    Value ih_nonneg = then_builder.create<arith::SelectOp>(loc, ih_ge0, ih, c0);
                    Value iw_nonneg = then_builder.create<arith::SelectOp>(loc, iw_ge0, iw, c0);
                    Value in_h_last = then_builder.create<arith::SubIOp>(loc, in_h, c1);
                    Value in_w_last = then_builder.create<arith::SubIOp>(loc, in_w, c1);
                    auto ih_bounded =
                        then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ih_nonneg, in_h);
                    auto iw_bounded =
                        then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iw_nonneg, in_w);
                    Value ih_safe = then_builder.create<arith::SelectOp>(loc, ih_bounded, ih_nonneg, in_h_last);
                    Value iw_safe = then_builder.create<arith::SelectOp>(loc, iw_bounded, iw_nonneg, in_w_last);
                    Value input_val = then_builder.create<memref::LoadOp>(
                        loc, fn.getArgument(0), ValueRange{n, ic, ih_safe, iw_safe});
                    Value input_comp = cast_to_compute(then_builder, input_val);
                    Value mask = then_builder.create<arith::SelectOp>(loc, in_bounds, one, zero);
                    Value masked_input = then_builder.create<arith::MulFOp>(loc, input_comp, mask);

                    for (uint32_t lane = 0; lane < oc_block; ++lane) {
                        Value safe_oc =
                            then_builder.create<arith::SelectOp>(loc, oc_valids[lane], oc_values[lane], c0);
                        Value weight_val = then_builder.create<memref::LoadOp>(
                            loc, fn.getArgument(1), ValueRange{safe_oc, ic, kh, kw});
                        Value prod = then_builder.create<arith::MulFOp>(
                            loc, masked_input, cast_to_compute(then_builder, weight_val));
                        Value updated = then_builder.create<arith::AddFOp>(loc, accs[lane], prod);
                        accs[lane] = then_builder.create<arith::SelectOp>(loc, oc_valids[lane], updated, accs[lane]);
                    }
                }
            }
        }
        for (uint32_t lane = 0; lane < oc_block; ++lane) {
            if (m_has_bias) {
                Value safe_oc =
                    then_builder.create<arith::SelectOp>(loc, oc_valids[lane], oc_values[lane], c0);
                Value bias_val = then_builder.create<memref::LoadOp>(loc, fn.getArgument(2), ValueRange{safe_oc});
                accs[lane] = then_builder.create<arith::AddFOp>(loc, accs[lane], cast_to_compute(then_builder, bias_val));
            }
            if (m_has_activation) {
                accs[lane] = then_builder.create<arith::MaximumFOp>(loc, accs[lane], zero);
            }
            auto store_if = then_builder.create<scf::IfOp>(loc, oc_valids[lane], /*withElseRegion=*/false);
            auto store_builder = store_if.getThenBodyBuilder();
            store_builder.create<memref::StoreOp>(loc,
                                                  cast_to_output(store_builder, accs[lane]),
                                                  fn.getArgument(m_has_bias ? 3 : 2),
                                                  ValueRange{n, oc_values[lane], oh, ow});
        }
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_conv2d_chunk_module(mlir::MLIRContext& ctx,
                                                      const ov::element::Type& et,
                                                      uint32_t threads_h,
                                                      uint32_t threads_w) {
    using namespace mlir;
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    OPENVINO_ASSERT(conv, "GFX Vulkan conv2d chunked: node cast failed");
    const auto& in_shape = conv->get_input_shape(0);
    const auto& w_shape = conv->get_input_shape(1);
    const auto& out_shape = conv->get_output_shape(0);

    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    math::MathDialect>();

    Type elem_ty;
    switch (et) {
        case ov::element::f16: elem_ty = Float16Type::get(&ctx); break;
        case ov::element::f32: elem_ty = Float32Type::get(&ctx); break;
        default: OPENVINO_THROW("GFX Vulkan conv2d chunked: unsupported element type ", et);
    }
    Type compute_ty = elem_ty;

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 3));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2}));
    auto input_ty = MemRefType::get({static_cast<int64_t>(in_shape.at(0)),
                                     static_cast<int64_t>(in_shape.at(1)),
                                     static_cast<int64_t>(in_shape.at(2)),
                                     static_cast<int64_t>(in_shape.at(3))},
                                    elem_ty);
    auto weight_ty = MemRefType::get({static_cast<int64_t>(w_shape.at(0)),
                                      static_cast<int64_t>(w_shape.at(1)),
                                      static_cast<int64_t>(w_shape.at(2)),
                                      static_cast<int64_t>(w_shape.at(3))},
                                     elem_ty);
    auto output_ty = MemRefType::get({static_cast<int64_t>(out_shape.at(0)),
                                      static_cast<int64_t>(out_shape.at(1)),
                                      static_cast<int64_t>(out_shape.at(2)),
                                      static_cast<int64_t>(out_shape.at(3))},
                                     elem_ty);
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "conv2d_chunk",
                                                 b.getFunctionType(TypeRange{input_ty, weight_ty, output_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(
        mlir::DenseI32ArrayAttr::get(&ctx,
                                     {static_cast<int32_t>(std::max<uint32_t>(1u, threads_w)),
                                      static_cast<int32_t>(std::max<uint32_t>(1u, threads_h)),
                                      1}));
    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto c0 = body.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = body.create<arith::ConstantIndexOp>(loc, 1);
    auto cast_to_compute = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == elem_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, elem_ty, value);
    };

    Value batch = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape.at(0)));
    Value c_in = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape.at(1)));
    Value in_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape.at(2)));
    Value in_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape.at(3)));
    Value c_out = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(w_shape.at(0)));
    Value k_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(w_shape.at(2)));
    Value k_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(w_shape.at(3)));
    Value stride_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_strides().at(0)));
    Value stride_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_strides().at(1)));
    Value dil_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_dilations().at(0)));
    Value dil_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_dilations().at(1)));
    Value pad_top = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_pads_begin().at(0)));
    Value pad_left = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(conv->get_pads_begin().at(1)));
    Value out_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape.at(2)));
    Value out_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape.at(3)));
    auto zero = body.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 0.0f));
    const auto in_channels = static_cast<int64_t>(in_shape.at(1));
    const auto kernel_h = static_cast<int64_t>(w_shape.at(2));
    const auto kernel_w = static_cast<int64_t>(w_shape.at(3));
    Value bid_x = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bid_y = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
    Value bid_z = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::z);
    Value bdim_x = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value bdim_y = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::y);
    Value tid_x = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value tid_y = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
    Value ow = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid_x, bdim_x), tid_x);
    Value oh = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid_y, bdim_y), tid_y);
    Value oc = body.create<arith::RemUIOp>(loc, bid_z, c_out);
    Value n = body.create<arith::DivUIOp>(loc, bid_z, c_out);
    auto ow_valid = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, ow, out_w);
    auto oh_valid = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, oh, out_h);
    auto n_valid = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, n, batch);
    auto active = body.create<arith::AndIOp>(loc, ow_valid, oh_valid);
    active = body.create<arith::AndIOp>(loc, active, n_valid);
    auto active_if = body.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value acc = zero.getResult();
        Value base_ih = then_builder.create<arith::SubIOp>(
            loc, then_builder.create<arith::MulIOp>(loc, oh, stride_h), pad_top);
        Value base_iw = then_builder.create<arith::SubIOp>(
            loc, then_builder.create<arith::MulIOp>(loc, ow, stride_w), pad_left);
        auto one = then_builder.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 1.0f));
        Value in_h_last = then_builder.create<arith::SubIOp>(loc, in_h, c1);
        Value in_w_last = then_builder.create<arith::SubIOp>(loc, in_w, c1);
        for (int64_t ic_idx = 0; ic_idx < in_channels; ++ic_idx) {
            Value ic = then_builder.create<arith::ConstantIndexOp>(loc, ic_idx);
            for (int64_t kh_idx = 0; kh_idx < kernel_h; ++kh_idx) {
                Value kh = then_builder.create<arith::ConstantIndexOp>(loc, kh_idx);
                for (int64_t kw_idx = 0; kw_idx < kernel_w; ++kw_idx) {
                    Value kw = then_builder.create<arith::ConstantIndexOp>(loc, kw_idx);
                    Value ih = then_builder.create<arith::AddIOp>(
                        loc, base_ih, then_builder.create<arith::MulIOp>(loc, kh, dil_h));
                    Value iw = then_builder.create<arith::AddIOp>(
                        loc, base_iw, then_builder.create<arith::MulIOp>(loc, kw, dil_w));

                    auto ih_ge0 = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, ih, c0);
                    auto ih_lt = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ih, in_h);
                    auto iw_ge0 = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, iw, c0);
                    auto iw_lt = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iw, in_w);
                    auto in_bounds = then_builder.create<arith::AndIOp>(loc, ih_ge0, ih_lt);
                    in_bounds = then_builder.create<arith::AndIOp>(loc, in_bounds, iw_ge0);
                    in_bounds = then_builder.create<arith::AndIOp>(loc, in_bounds, iw_lt);

                    Value ih_nonneg = then_builder.create<arith::SelectOp>(loc, ih_ge0, ih, c0);
                    Value iw_nonneg = then_builder.create<arith::SelectOp>(loc, iw_ge0, iw, c0);
                    auto ih_bounded =
                        then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ih_nonneg, in_h);
                    auto iw_bounded =
                        then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iw_nonneg, in_w);
                    Value ih_safe = then_builder.create<arith::SelectOp>(loc, ih_bounded, ih_nonneg, in_h_last);
                    Value iw_safe = then_builder.create<arith::SelectOp>(loc, iw_bounded, iw_nonneg, in_w_last);
                    Value input_val = then_builder.create<memref::LoadOp>(
                        loc, fn.getArgument(0), ValueRange{n, ic, ih_safe, iw_safe});
                    Value weight_val = then_builder.create<memref::LoadOp>(
                        loc, fn.getArgument(1), ValueRange{oc, ic, kh, kw});
                    Value input_comp = cast_to_compute(then_builder, input_val);
                    Value mask = then_builder.create<arith::SelectOp>(loc, in_bounds, one, zero);
                    Value masked_input = then_builder.create<arith::MulFOp>(loc, input_comp, mask);
                    Value prod = then_builder.create<arith::MulFOp>(
                        loc, masked_input, cast_to_compute(then_builder, weight_val));
                    acc = then_builder.create<arith::AddFOp>(loc, acc, prod);
                }
            }
        }
        then_builder.create<memref::StoreOp>(loc,
                                             cast_to_output(then_builder, acc),
                                             fn.getArgument(2),
                                             ValueRange{n, oc, oh, ow});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_group_conv2d_chunk_module(mlir::MLIRContext& ctx,
                                                            const ov::element::Type& et,
                                                            uint32_t threads_per_group) {
    using namespace mlir;
    auto gconv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(m_node);
    OPENVINO_ASSERT(gconv, "GFX Vulkan group_conv2d chunked: node cast failed");
    const auto& in_shape = gconv->get_input_shape(0);
    const auto& w_shape = gconv->get_input_shape(1);
    const auto& out_shape = gconv->get_output_shape(0);
    OPENVINO_ASSERT(in_shape.size() == 4 && out_shape.size() == 4 && w_shape.size() == 5,
                    "GFX Vulkan group_conv2d chunked: unexpected ranks");
    OPENVINO_ASSERT(w_shape[0] == in_shape[1] && w_shape[0] == out_shape[1] && w_shape[1] == 1 && w_shape[2] == 1,
                    "GFX Vulkan group_conv2d chunked: only depthwise multiplier=1 is supported");

    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect>();

    Type elem_ty;
    switch (et) {
        case ov::element::f16: elem_ty = Float16Type::get(&ctx); break;
        case ov::element::f32: elem_ty = Float32Type::get(&ctx); break;
        default: OPENVINO_THROW("GFX Vulkan group_conv2d chunked: unsupported element type ", et);
    }
    Type compute_ty = elem_ty;

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 3));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2}));

    const int64_t input_elems = static_cast<int64_t>(tensor_elements(in_shape));
    const int64_t weight_elems = static_cast<int64_t>(tensor_elements(w_shape));
    const int64_t output_elems = static_cast<int64_t>(tensor_elements(out_shape));
    auto input_ty = MemRefType::get({input_elems}, elem_ty);
    auto weight_ty = MemRefType::get({weight_elems}, elem_ty);
    auto output_ty = MemRefType::get({output_elems}, elem_ty);

    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "group_conv2d_direct",
                                                 b.getFunctionType(TypeRange{input_ty, weight_ty, output_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(
        &ctx,
        {static_cast<int32_t>(std::max<uint32_t>(1u, threads_per_group)), 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto cast_to_compute = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == elem_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, elem_ty, value);
    };

    auto c0 = body.create<arith::ConstantIndexOp>(loc, 0);
    auto total = body.create<arith::ConstantIndexOp>(loc, output_elems);
    Value c_out = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape[1]));
    Value in_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape[2]));
    Value in_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(in_shape[3]));
    Value out_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape[2]));
    Value out_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(out_shape[3]));
    Value k_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(w_shape[3]));
    Value k_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(w_shape[4]));
    Value stride_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(gconv->get_strides()[0]));
    Value stride_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(gconv->get_strides()[1]));
    Value dil_h = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(gconv->get_dilations()[0]));
    Value dil_w = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(gconv->get_dilations()[1]));
    Value pad_top = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(gconv->get_pads_begin()[0]));
    Value pad_left = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(gconv->get_pads_begin()[1]));
    auto zero = body.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 0.0f));

    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value linear_idx = body.create<arith::AddIOp>(loc,
                                                  body.create<arith::MulIOp>(loc, bid, bdim),
                                                  tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, linear_idx, total);
    auto active_if = body.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value rem = linear_idx;
        Value ow = then_builder.create<arith::RemUIOp>(loc, rem, out_w);
        rem = then_builder.create<arith::DivUIOp>(loc, rem, out_w);
        Value oh = then_builder.create<arith::RemUIOp>(loc, rem, out_h);
        rem = then_builder.create<arith::DivUIOp>(loc, rem, out_h);
        Value oc = then_builder.create<arith::RemUIOp>(loc, rem, c_out);
        Value n = then_builder.create<arith::DivUIOp>(loc, rem, c_out);
        Value acc = zero.getResult();

        for (int64_t kh_idx = 0; kh_idx < static_cast<int64_t>(w_shape[3]); ++kh_idx) {
            Value kh = then_builder.create<arith::ConstantIndexOp>(loc, kh_idx);
            for (int64_t kw_idx = 0; kw_idx < static_cast<int64_t>(w_shape[4]); ++kw_idx) {
                Value kw = then_builder.create<arith::ConstantIndexOp>(loc, kw_idx);
                Value ih = then_builder.create<arith::AddIOp>(
                    loc,
                    then_builder.create<arith::MulIOp>(loc, oh, stride_h),
                    then_builder.create<arith::MulIOp>(loc, kh, dil_h));
                ih = then_builder.create<arith::SubIOp>(loc, ih, pad_top);
                Value iw = then_builder.create<arith::AddIOp>(
                    loc,
                    then_builder.create<arith::MulIOp>(loc, ow, stride_w),
                    then_builder.create<arith::MulIOp>(loc, kw, dil_w));
                iw = then_builder.create<arith::SubIOp>(loc, iw, pad_left);

                auto ih_ge0 = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, ih, c0);
                auto ih_lt = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, ih, in_h);
                auto iw_ge0 = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, iw, c0);
                auto iw_lt = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iw, in_w);
                auto in_bounds = then_builder.create<arith::AndIOp>(loc, ih_ge0, ih_lt);
                in_bounds = then_builder.create<arith::AndIOp>(loc, in_bounds, iw_ge0);
                in_bounds = then_builder.create<arith::AndIOp>(loc, in_bounds, iw_lt);

                auto if_in_bounds = then_builder.create<scf::IfOp>(loc, TypeRange{compute_ty}, in_bounds, true);
                {
                    auto then_inner = if_in_bounds.getThenBodyBuilder();
                    Value input_offset = then_inner.create<arith::AddIOp>(
                        loc,
                        then_inner.create<arith::MulIOp>(
                            loc,
                            then_inner.create<arith::AddIOp>(
                                loc,
                                then_inner.create<arith::MulIOp>(loc, n, c_out),
                                oc),
                            then_inner.create<arith::MulIOp>(loc, in_h, in_w)),
                        then_inner.create<arith::AddIOp>(
                            loc,
                            then_inner.create<arith::MulIOp>(loc, ih, in_w),
                            iw));
                    Value weight_offset = then_inner.create<arith::AddIOp>(
                        loc,
                        then_inner.create<arith::MulIOp>(
                            loc,
                            oc,
                            then_inner.create<arith::MulIOp>(loc, k_h, k_w)),
                        then_inner.create<arith::AddIOp>(
                            loc,
                            then_inner.create<arith::MulIOp>(loc, kh, k_w),
                            kw));
                    Value input_val = then_inner.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{input_offset});
                    Value weight_val = then_inner.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{weight_offset});
                    Value prod = then_inner.create<arith::MulFOp>(
                        loc,
                        cast_to_compute(then_inner, input_val),
                        cast_to_compute(then_inner, weight_val));
                    Value sum = then_inner.create<arith::AddFOp>(loc, acc, prod);
                    then_inner.create<scf::YieldOp>(loc, sum);
                }
                {
                    auto else_inner = if_in_bounds.getElseBodyBuilder();
                    else_inner.create<scf::YieldOp>(loc, acc);
                }
                acc = if_in_bounds.getResult(0);
            }
        }

        Value out_val = cast_to_output(then_builder, acc);
        then_builder.create<memref::StoreOp>(loc, out_val, fn.getArgument(2), ValueRange{linear_idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_linear_binary_module(mlir::MLIRContext& ctx,
                                                       const ov::element::Type& src0_et,
                                                       const ov::element::Type& src1_et,
                                                       const ov::element::Type& dst_et,
                                                       const std::string& op_key,
                                                       size_t meta_rank) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    math::MathDialect>();

    OPENVINO_ASSERT(is_supported_binary_io_types(src0_et, src1_et, dst_et, op_key),
                    "GFX Vulkan binary chunked: unsupported element types for op ",
                    op_key);
    Type src0_ty = to_binary_storage_type(ctx, src0_et);
    Type src1_ty = to_binary_storage_type(ctx, src1_et);
    Type dst_ty = to_binary_storage_type(ctx, dst_et);
    Type compute_ty = (src0_et == ov::element::f16) ? static_cast<Type>(Float16Type::get(&ctx))
                                                    : static_cast<Type>(Float32Type::get(&ctx));

    OPENVINO_ASSERT(meta_rank != 0, "GFX Vulkan binary chunked: metadata rank must be non-zero");

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 4));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2, 3}));
    auto in0_ty = MemRefType::get({ShapedType::kDynamic}, src0_ty);
    auto in1_ty = MemRefType::get({ShapedType::kDynamic}, src1_ty);
    auto out_ty = MemRefType::get({ShapedType::kDynamic}, dst_ty);
    auto param_ty = MemRefType::get({ShapedType::kDynamic}, IntegerType::get(&ctx, 32));
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "linear_binary",
                                                 b.getFunctionType(TypeRange{in0_ty, in1_ty, param_ty, out_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(mlir::DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));
    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto c0 = body.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = body.create<arith::ConstantIndexOp>(loc, 1);
    auto load_param = [&](int idx) -> Value {
        auto idx_val = body.create<arith::ConstantIndexOp>(loc, idx);
        auto v_i32 = body.create<memref::LoadOp>(loc, fn.getArgument(2), ValueRange{idx_val});
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), v_i32);
    };
    auto load_param_at = [&](OpBuilder& builder, Value idx) -> Value {
        auto v_i32 = builder.create<memref::LoadOp>(loc, fn.getArgument(2), ValueRange{idx});
        return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), v_i32);
    };
    auto cast_to_compute = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == dst_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, dst_ty, value);
    };
    auto emit_binary = [&](OpBuilder& builder, Value lhs, Value rhs) -> Value {
        if (is_arithmetic_binary_key(op_key)) {
            if (llvm::isa<mlir::FloatType>(lhs.getType())) {
                auto a = cast_to_compute(builder, lhs);
                auto bval = cast_to_compute(builder, rhs);
                if (op_key == "add") {
                    return cast_to_output(builder, builder.create<arith::AddFOp>(loc, a, bval));
                }
                if (op_key == "sub") {
                    return cast_to_output(builder, builder.create<arith::SubFOp>(loc, a, bval));
                }
                if (op_key == "mul") {
                    return cast_to_output(builder, builder.create<arith::MulFOp>(loc, a, bval));
                }
                if (op_key == "div") {
                    return cast_to_output(builder, builder.create<arith::DivFOp>(loc, a, bval));
                }
                if (op_key == "pow") {
                    return cast_to_output(builder, builder.create<math::PowFOp>(loc, a, bval));
                }
            } else {
                if (op_key == "add") {
                    return builder.create<arith::AddIOp>(loc, lhs, rhs);
                }
                if (op_key == "sub") {
                    return builder.create<arith::SubIOp>(loc, lhs, rhs);
                }
                if (op_key == "mul") {
                    return builder.create<arith::MulIOp>(loc, lhs, rhs);
                }
                if (op_key == "div") {
                    return builder.create<arith::DivSIOp>(loc, lhs, rhs);
                }
            }
        }
        if (is_compare_binary_key(op_key)) {
            if (llvm::isa<mlir::FloatType>(lhs.getType())) {
                if (op_key == "eq") {
                    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, lhs, rhs);
                }
                if (op_key == "ne") {
                    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, lhs, rhs);
                }
                if (op_key == "lt") {
                    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, lhs, rhs);
                }
                if (op_key == "gt") {
                    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, lhs, rhs);
                }
                if (op_key == "le") {
                    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE, lhs, rhs);
                }
                if (op_key == "ge") {
                    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, lhs, rhs);
                }
            } else {
                if (op_key == "eq") {
                    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, rhs);
                }
                if (op_key == "ne") {
                    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lhs, rhs);
                }
                if (op_key == "lt") {
                    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs);
                }
                if (op_key == "gt") {
                    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs, rhs);
                }
                if (op_key == "le") {
                    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, lhs, rhs);
                }
                if (op_key == "ge") {
                    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, lhs, rhs);
                }
            }
        }
        if (is_logical_binary_key(op_key)) {
            if (op_key == "land") {
                return builder.create<arith::AndIOp>(loc, lhs, rhs);
            }
            if (op_key == "lor") {
                return builder.create<arith::OrIOp>(loc, lhs, rhs);
            }
            if (op_key == "lxor") {
                return builder.create<arith::XOrIOp>(loc, lhs, rhs);
            }
        }
        OPENVINO_THROW("GFX Vulkan binary chunked: unsupported op ", op_key);
    };

    Value offset = load_param(0);
    Value count = load_param(1);
    Value rank = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(meta_rank));
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value local_idx = body.create<arith::AddIOp>(loc,
                                                 body.create<arith::MulIOp>(loc, bid, bdim),
                                                 tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local_idx, count);
    auto active_if = body.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value linear_idx = then_builder.create<arith::AddIOp>(loc, offset, local_idx);
        auto for_dims = then_builder.create<scf::ForOp>(loc, c0, rank, c1, ValueRange{linear_idx, c0, c0});
        auto dim_builder = OpBuilder::atBlockBegin(for_dims.getBody());
        Value rev_dim = dim_builder.create<arith::SubIOp>(loc, rank, c1);
        rev_dim = dim_builder.create<arith::SubIOp>(loc, rev_dim, for_dims.getInductionVar());
        auto meta_base = [&](int64_t base) -> Value {
            return dim_builder.create<arith::ConstantIndexOp>(loc, base);
        };
        Value dim_size = load_param_at(dim_builder, dim_builder.create<arith::AddIOp>(loc, meta_base(2), rev_dim));
        Value coord = dim_builder.create<arith::RemUIOp>(loc, for_dims.getRegionIterArgs()[0], dim_size);
        Value next_rem = dim_builder.create<arith::DivUIOp>(loc, for_dims.getRegionIterArgs()[0], dim_size);
        Value stride0_base = meta_base(2 + static_cast<int64_t>(meta_rank));
        Value stride1_base = meta_base(2 + static_cast<int64_t>(2 * meta_rank));
        Value stride0 = load_param_at(dim_builder, dim_builder.create<arith::AddIOp>(loc, stride0_base, rev_dim));
        Value stride1 = load_param_at(dim_builder, dim_builder.create<arith::AddIOp>(loc, stride1_base, rev_dim));
        Value off0 = dim_builder.create<arith::AddIOp>(
            loc,
            for_dims.getRegionIterArgs()[1],
            dim_builder.create<arith::MulIOp>(loc, coord, stride0));
        Value off1 = dim_builder.create<arith::AddIOp>(
            loc,
            for_dims.getRegionIterArgs()[2],
            dim_builder.create<arith::MulIOp>(loc, coord, stride1));
        if (!for_dims.getBody()->empty() && for_dims.getBody()->back().hasTrait<OpTrait::IsTerminator>()) {
            for_dims.getBody()->back().erase();
        }
        auto dim_end = OpBuilder::atBlockEnd(for_dims.getBody());
        dim_end.create<scf::YieldOp>(loc, ValueRange{next_rem, off0, off1});

        Value lhs = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{for_dims.getResult(1)});
        Value rhs = then_builder.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{for_dims.getResult(2)});
        Value out = emit_binary(then_builder, lhs, rhs);
        then_builder.create<memref::StoreOp>(loc, out, fn.getArgument(3), ValueRange{linear_idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_linear_binary_same_shape_module(mlir::MLIRContext& ctx,
                                                                  const ov::element::Type& src0_et,
                                                                  const ov::element::Type& src1_et,
                                                                  const ov::element::Type& dst_et,
                                                                  const std::string& op_key) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    math::MathDialect>();

    OPENVINO_ASSERT(is_supported_binary_io_types(src0_et, src1_et, dst_et, op_key),
                    "GFX Vulkan same-shape binary: unsupported element types for op ",
                    op_key);
    Type src0_ty = to_binary_storage_type(ctx, src0_et);
    Type src1_ty = to_binary_storage_type(ctx, src1_et);
    Type dst_ty = to_binary_storage_type(ctx, dst_et);
    Type compute_ty = (src0_et == ov::element::f16) ? static_cast<Type>(Float16Type::get(&ctx))
                                                    : static_cast<Type>(Float32Type::get(&ctx));

    OPENVINO_ASSERT(m_node && m_node->get_output_partial_shape(0).is_static() &&
                        m_node->get_input_partial_shape(0).is_static() &&
                        m_node->get_input_partial_shape(1).is_static(),
                    "GFX Vulkan same-shape binary: static shapes required");
    const ov::Shape out_shape = m_node->get_output_shape(0);
    OPENVINO_ASSERT(m_node->get_input_shape(0) == out_shape && m_node->get_input_shape(1) == out_shape,
                    "GFX Vulkan same-shape binary: shapes must match");
    const int64_t total = static_cast<int64_t>(tensor_elements(out_shape));

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 3));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2}));

    auto in0_ty = MemRefType::get({total}, src0_ty);
    auto in1_ty = MemRefType::get({total}, src1_ty);
    auto out_ty = MemRefType::get({total}, dst_ty);
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "linear_binary_same_shape",
                                                 b.getFunctionType(TypeRange{in0_ty, in1_ty, out_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto total_idx = body.create<arith::ConstantIndexOp>(loc, total);
    auto cast_to_compute = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == dst_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, dst_ty, value);
    };
    auto emit_binary = [&](OpBuilder& builder, Value lhs, Value rhs) -> Value {
        if (is_arithmetic_binary_key(op_key)) {
            if (llvm::isa<mlir::FloatType>(lhs.getType())) {
                auto a = cast_to_compute(builder, lhs);
                auto bval = cast_to_compute(builder, rhs);
                if (op_key == "add") {
                    return cast_to_output(builder, builder.create<arith::AddFOp>(loc, a, bval));
                }
                if (op_key == "sub") {
                    return cast_to_output(builder, builder.create<arith::SubFOp>(loc, a, bval));
                }
                if (op_key == "mul") {
                    return cast_to_output(builder, builder.create<arith::MulFOp>(loc, a, bval));
                }
                if (op_key == "div") {
                    return cast_to_output(builder, builder.create<arith::DivFOp>(loc, a, bval));
                }
                if (op_key == "pow") {
                    return cast_to_output(builder, builder.create<math::PowFOp>(loc, a, bval));
                }
            } else {
                if (op_key == "add") {
                    return builder.create<arith::AddIOp>(loc, lhs, rhs);
                }
                if (op_key == "sub") {
                    return builder.create<arith::SubIOp>(loc, lhs, rhs);
                }
                if (op_key == "mul") {
                    return builder.create<arith::MulIOp>(loc, lhs, rhs);
                }
                if (op_key == "div") {
                    return builder.create<arith::DivSIOp>(loc, lhs, rhs);
                }
            }
        }
        if (is_compare_binary_key(op_key)) {
            if (llvm::isa<mlir::FloatType>(lhs.getType())) {
                if (op_key == "eq") {
                    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, lhs, rhs);
                }
                if (op_key == "ne") {
                    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, lhs, rhs);
                }
                if (op_key == "lt") {
                    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, lhs, rhs);
                }
                if (op_key == "gt") {
                    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, lhs, rhs);
                }
                if (op_key == "le") {
                    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE, lhs, rhs);
                }
                if (op_key == "ge") {
                    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE, lhs, rhs);
                }
            } else {
                if (op_key == "eq") {
                    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, rhs);
                }
                if (op_key == "ne") {
                    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, lhs, rhs);
                }
                if (op_key == "lt") {
                    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs);
                }
                if (op_key == "gt") {
                    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs, rhs);
                }
                if (op_key == "le") {
                    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, lhs, rhs);
                }
                if (op_key == "ge") {
                    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, lhs, rhs);
                }
            }
        }
        if (is_logical_binary_key(op_key)) {
            if (op_key == "land") {
                return builder.create<arith::AndIOp>(loc, lhs, rhs);
            }
            if (op_key == "lor") {
                return builder.create<arith::OrIOp>(loc, lhs, rhs);
            }
            if (op_key == "lxor") {
                return builder.create<arith::XOrIOp>(loc, lhs, rhs);
            }
        }
        OPENVINO_THROW("GFX Vulkan same-shape binary: unsupported op ", op_key);
    };

    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value idx = body.create<arith::AddIOp>(loc,
                                           body.create<arith::MulIOp>(loc, bid, bdim),
                                           tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, idx, total_idx);
    auto active_if = body.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value lhs = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{idx});
        Value rhs = then_builder.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{idx});
        Value out = emit_binary(then_builder, lhs, rhs);
        then_builder.create<memref::StoreOp>(loc, out, fn.getArgument(2), ValueRange{idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_binary_bias_add_module(mlir::MLIRContext& ctx,
                                                         const ov::element::Type& et) {
    OPENVINO_ASSERT(m_node, "GFX Vulkan bias add: node is null");
    OPENVINO_ASSERT(m_node->get_input_partial_shape(0).is_static() &&
                        m_node->get_input_partial_shape(1).is_static() &&
                        m_node->get_output_partial_shape(0).is_static(),
                    "GFX Vulkan bias add: static shapes required");
    OPENVINO_ASSERT(et == ov::element::f16 || et == ov::element::f32,
                    "GFX Vulkan bias add: unsupported element type ",
                    et);

    auto add = std::dynamic_pointer_cast<const ov::op::v1::Add>(m_node);
    OPENVINO_ASSERT(add, "GFX Vulkan bias add: expected ov::op::v1::Add");
    auto module = build_mlir_for_node(add, ctx);
    OPENVINO_ASSERT(module, "GFX Vulkan bias add: failed to build shared MLIR module");
    return module;
}

void VulkanStage::execute_unary_chunked(GpuCommandBufferHandle command_buffer) {
    auto resolve_input = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
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
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    GpuTensor* input0 = resolve_input(0);
    OPENVINO_ASSERT(input0 && output,
                    "GFX Vulkan unary chunked: missing tensors");
    const auto op_key = unary_chunk_key(m_type);
    OPENVINO_ASSERT(!op_key.empty(), "GFX Vulkan unary chunked: unsupported op ", m_type);
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan unary chunked: unsupported element type ",
                    elem_type);
    if (!m_linear_unary_kernel || m_linear_unary_elem_type != elem_type || m_linear_unary_key != op_key) {
        auto& ctx = gfx_mlir_context();
        auto module = build_linear_unary_module(ctx, elem_type, op_key);
        KernelSource src = make_kernel_source_from_mlir(module, "linear_unary", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_linear_unary_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_linear_unary_kernel, "GFX Vulkan unary chunked: kernel compile failed: ", log);
        m_linear_unary_kernel->prepare_runtime_artifacts();
        m_linear_unary_elem_type = elem_type;
        m_linear_unary_key = op_key;
        m_linear_unary_launch_abi = extract_launch_operand_abi(module);
        m_linear_unary_scalar_args = extract_kernel_scalar_values(module);
        if (gfx_log_debug_enabled()) {
            std::ostringstream oss;
            oss << "Unary launch ABI kinds=";
            for (size_t i = 0; i < m_linear_unary_launch_abi.kinds.size(); ++i) {
                if (i) {
                    oss << ",";
                }
                oss << m_linear_unary_launch_abi.kinds[i];
            }
            oss << " arg_indices=";
            for (size_t i = 0; i < m_linear_unary_launch_abi.arg_indices.size(); ++i) {
                if (i) {
                    oss << ",";
                }
                oss << m_linear_unary_launch_abi.arg_indices[i];
            }
            oss << " scalar_values=";
            for (size_t i = 0; i < m_linear_unary_launch_abi.scalar_values.size(); ++i) {
                if (i) {
                    oss << ",";
                }
                oss << m_linear_unary_launch_abi.scalar_values[i];
            }
            oss << " scalar_known=";
            for (size_t i = 0; i < m_linear_unary_launch_abi.scalar_known.size(); ++i) {
                if (i) {
                    oss << ",";
                }
                oss << static_cast<int32_t>(m_linear_unary_launch_abi.scalar_known[i]);
            }
            gfx_log_debug("VulkanExec") << oss.str();
        }
    }

    ov::Shape dispatch_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
    if (dispatch_shape.empty() && !input0->shape.empty()) {
        dispatch_shape = input0->shape;
    }
    if (dispatch_shape.empty() && m_node && m_node->get_output_partial_shape(0).is_static()) {
        dispatch_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(!dispatch_shape.empty(), "GFX Vulkan unary chunked: output shape is unknown");
    output->shape = dispatch_shape;
    output->expected_type = elem_type;
    const uint32_t total = static_cast<uint32_t>(tensor_elements(dispatch_shape));
    for (uint32_t offset = 0; offset < total; offset += kLinearChunkElemsPerDispatch) {
        const uint32_t count = std::min<uint32_t>(kLinearChunkElemsPerDispatch, total - offset);
        struct LinearChunkParams {
            uint32_t offset;
            uint32_t count;
        } params{offset, count};
        std::vector<KernelArg> args;
        size_t tg = 1;
        if (!m_linear_unary_launch_abi.valid) {
            tg = std::min<size_t>(count,
                                  std::max<size_t>(1, m_linear_unary_kernel->clamp_threadgroup_size(64)));
            args = {
                make_buffer_arg(0, input0->buf),
                make_bytes_arg(1, &params, sizeof(params)),
                make_buffer_arg(2, output->buf),
            };
        } else {
            const int32_t dynamic_scalars[] = {static_cast<int32_t>(count), static_cast<int32_t>(offset)};
            args.reserve(m_linear_unary_launch_abi.kinds.size());
            size_t scalar_idx = 0;
            size_t dynamic_idx = 0;
            for (size_t i = 0; i < m_linear_unary_launch_abi.kinds.size(); ++i) {
                if (m_linear_unary_launch_abi.kinds[i] == 1) {
                    const int32_t arg_idx = m_linear_unary_launch_abi.arg_indices[i];
                    if (arg_idx == 0) {
                        args.push_back(make_buffer_arg(static_cast<uint32_t>(args.size()), input0->buf));
                    } else if (arg_idx == 1) {
                        args.push_back(make_bytes_arg(static_cast<uint32_t>(args.size()), &params, sizeof(params)));
                    } else if (arg_idx == 2) {
                        args.push_back(make_buffer_arg(static_cast<uint32_t>(args.size()), output->buf));
                    } else {
                        OPENVINO_THROW("GFX Vulkan unary chunked: unsupported memref arg index ", arg_idx);
                    }
                    continue;
                }
                OPENVINO_ASSERT(scalar_idx < m_linear_unary_launch_abi.scalar_values.size(),
                                "GFX Vulkan unary chunked: scalar ABI mismatch");
                int32_t scalar = m_linear_unary_launch_abi.scalar_values[scalar_idx];
                if (scalar_idx < m_linear_unary_launch_abi.scalar_known.size() &&
                    !m_linear_unary_launch_abi.scalar_known[scalar_idx]) {
                    OPENVINO_ASSERT(dynamic_idx < std::size(dynamic_scalars),
                                    "GFX Vulkan unary chunked: too many dynamic scalars");
                    scalar = dynamic_scalars[dynamic_idx++];
                }
                args.push_back(make_bytes_arg(static_cast<uint32_t>(args.size()), &scalar, sizeof(int32_t)));
                ++scalar_idx;
            }
        }
        KernelDispatch dispatch = make_1d_dispatch(count, tg);
        auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
        m_linear_unary_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
    }
}

void VulkanStage::execute_binary_chunked(GpuCommandBufferHandle command_buffer) {
    auto resolve_input = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
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
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    GpuTensor* input0 = resolve_input(0);
    GpuTensor* input1 = resolve_input(1);
    OPENVINO_ASSERT(input0 && input1 && output,
                    "GFX Vulkan binary chunked: missing tensors");
    const auto op_key = binary_chunk_key(m_type);
    OPENVINO_ASSERT(!op_key.empty(), "GFX Vulkan binary chunked: unsupported op ", m_type);
    const ov::element::Type src0_et = resolve_stage_input_element_type(m_node, 0, input0);
    const ov::element::Type src1_et = resolve_stage_input_element_type(m_node, 1, input1);
    const ov::element::Type dst_et = resolve_stage_element_type(m_node, output);
    OPENVINO_ASSERT(is_supported_binary_io_types(src0_et, src1_et, dst_et, op_key),
                    "GFX Vulkan binary chunked: unsupported element types ",
                    src0_et,
                    " / ",
                    src1_et,
                    " -> ",
                    dst_et,
                    " for op ",
                    op_key);
    ov::Shape in0_shape = input0->shape;
    ov::Shape in1_shape = input1->shape;
    if (in0_shape.empty() && m_node && m_node->get_input_partial_shape(0).is_static()) {
        in0_shape = m_node->get_input_shape(0);
    }
    if (in1_shape.empty() && m_node && m_node->get_input_partial_shape(1).is_static()) {
        in1_shape = m_node->get_input_shape(1);
    }
    OPENVINO_ASSERT(!in0_shape.empty() && !in1_shape.empty(),
                    "GFX Vulkan binary chunked: input shapes are unknown");
    ov::Shape dispatch_shape;
    if (!m_output_shape.empty()) {
        dispatch_shape = m_output_shape;
    } else if (!output->shape.empty()) {
        dispatch_shape = output->shape;
    } else if (m_node && m_node->get_output_partial_shape(0).is_static()) {
        dispatch_shape = m_node->get_output_shape(0);
    } else {
        dispatch_shape = broadcast_runtime_shape(in0_shape, in1_shape);
    }
    output->shape = dispatch_shape;
    output->expected_type = dst_et;

    size_t meta_rank = 0;
    if (!m_kernel_extra_inputs.empty() && !m_kernel_extra_inputs[0].shape.empty()) {
        meta_rank = m_kernel_extra_inputs[0].shape[0];
    } else if (m_node && m_node->get_output_partial_shape(0).rank().is_static()) {
        meta_rank = static_cast<size_t>(m_node->get_output_partial_shape(0).rank().get_length());
    } else {
        meta_rank = dispatch_shape.size();
    }
    OPENVINO_ASSERT(meta_rank != 0, "GFX Vulkan binary chunked: invalid metadata rank");
    OPENVINO_ASSERT(dispatch_shape.size() == meta_rank,
                    "GFX Vulkan binary chunked: runtime output rank mismatch");
    const auto stride0_vals = ov::gfx_plugin::compute_broadcast_element_strides(in0_shape, dispatch_shape);
    const auto stride1_vals = ov::gfx_plugin::compute_broadcast_element_strides(in1_shape, dispatch_shape);
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("VulkanExec") << "Binary chunked buffers in0=" << input0->buf.buffer
                                    << " in1=" << input1->buf.buffer
                                    << " out=" << output->buf.buffer
                                    << " in0_type=" << input0->buf.type << "/" << input0->expected_type
                                    << " in1_type=" << input1->buf.type << "/" << input1->expected_type
                                    << " out_type=" << output->buf.type << "/" << output->expected_type;
    }
    if (!m_linear_binary_kernel ||
        m_linear_binary_src0_elem_type != src0_et ||
        m_linear_binary_src1_elem_type != src1_et ||
        m_linear_binary_dst_elem_type != dst_et ||
        m_linear_binary_key != op_key ||
        m_linear_binary_rank != meta_rank) {
        auto& ctx = gfx_mlir_context();
        auto module = build_linear_binary_module(ctx, src0_et, src1_et, dst_et, op_key, meta_rank);
        KernelSource src = make_kernel_source_from_mlir(module, "linear_binary", /*arg_count=*/4);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_linear_binary_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_linear_binary_kernel, "GFX Vulkan binary chunked: kernel compile failed: ", log);
        m_linear_binary_kernel->prepare_runtime_artifacts();
        m_linear_binary_src0_elem_type = src0_et;
        m_linear_binary_src1_elem_type = src1_et;
        m_linear_binary_dst_elem_type = dst_et;
        m_linear_binary_key = op_key;
        m_linear_binary_rank = meta_rank;
        m_linear_binary_launch_abi = extract_launch_operand_abi(module);
        if (gfx_log_debug_enabled()) {
            std::ostringstream oss;
            oss << "Binary launch ABI kinds=";
            for (size_t i = 0; i < m_linear_binary_launch_abi.kinds.size(); ++i) {
                if (i) {
                    oss << ",";
                }
                oss << m_linear_binary_launch_abi.kinds[i];
            }
            oss << " arg_indices=";
            for (size_t i = 0; i < m_linear_binary_launch_abi.arg_indices.size(); ++i) {
                if (i) {
                    oss << ",";
                }
                oss << m_linear_binary_launch_abi.arg_indices[i];
            }
            oss << " scalar_values=";
            for (size_t i = 0; i < m_linear_binary_launch_abi.scalar_values.size(); ++i) {
                if (i) {
                    oss << ",";
                }
                oss << m_linear_binary_launch_abi.scalar_values[i];
            }
            oss << " scalar_known=";
            for (size_t i = 0; i < m_linear_binary_launch_abi.scalar_known.size(); ++i) {
                if (i) {
                    oss << ",";
                }
                oss << static_cast<int32_t>(m_linear_binary_launch_abi.scalar_known[i]);
            }
            gfx_log_debug("VulkanExec") << oss.str();
        }
    }

    const uint32_t total = static_cast<uint32_t>(tensor_elements(dispatch_shape));
    for (uint32_t offset = 0; offset < total; offset += kLinearChunkElemsPerDispatch) {
        const uint32_t count = std::min<uint32_t>(kLinearChunkElemsPerDispatch, total - offset);
        std::vector<int32_t> params(2 + 3 * meta_rank, 0);
        params[0] = static_cast<int32_t>(offset);
        params[1] = static_cast<int32_t>(count);
        for (size_t i = 0; i < meta_rank; ++i) {
            params[2 + i] = static_cast<int32_t>(dispatch_shape[i]);
            params[2 + meta_rank + i] = stride0_vals[i];
            params[2 + 2 * meta_rank + i] = stride1_vals[i];
        }
        const size_t params_bytes = params.size() * sizeof(int32_t);
        std::vector<KernelArg> args;
        size_t tg = 1;
        if (!m_linear_binary_launch_abi.valid) {
            tg = std::min<size_t>(count,
                                  std::max<size_t>(1, m_linear_binary_kernel->clamp_threadgroup_size(64)));
            args = {
                make_buffer_arg(0, input0->buf),
                make_buffer_arg(1, input1->buf),
                make_bytes_arg(2, params.data(), params_bytes),
                make_buffer_arg(3, output->buf),
            };
        } else {
            const int32_t dynamic_scalars[] = {static_cast<int32_t>(count), static_cast<int32_t>(offset)};
            args.reserve(m_linear_binary_launch_abi.kinds.size());
            size_t scalar_idx = 0;
            size_t dynamic_idx = 0;
            for (size_t i = 0; i < m_linear_binary_launch_abi.kinds.size(); ++i) {
                if (m_linear_binary_launch_abi.kinds[i] == 1) {
                    const int32_t arg_idx = m_linear_binary_launch_abi.arg_indices[i];
                    if (arg_idx == 0) {
                        args.push_back(make_buffer_arg(static_cast<uint32_t>(args.size()), input0->buf));
                    } else if (arg_idx == 1) {
                        args.push_back(make_buffer_arg(static_cast<uint32_t>(args.size()), input1->buf));
                    } else if (arg_idx == 2) {
                        args.push_back(make_bytes_arg(static_cast<uint32_t>(args.size()), params.data(), params_bytes));
                    } else if (arg_idx == 3) {
                        args.push_back(make_buffer_arg(static_cast<uint32_t>(args.size()), output->buf));
                    } else {
                        OPENVINO_THROW("GFX Vulkan binary chunked: unsupported memref arg index ", arg_idx);
                    }
                    continue;
                }
                OPENVINO_ASSERT(scalar_idx < m_linear_binary_launch_abi.scalar_values.size(),
                                "GFX Vulkan binary chunked: scalar ABI mismatch");
                int32_t scalar = m_linear_binary_launch_abi.scalar_values[scalar_idx];
                if (scalar_idx < m_linear_binary_launch_abi.scalar_known.size() &&
                    !m_linear_binary_launch_abi.scalar_known[scalar_idx]) {
                    OPENVINO_ASSERT(dynamic_idx < std::size(dynamic_scalars),
                                    "GFX Vulkan binary chunked: too many dynamic scalars");
                    scalar = dynamic_scalars[dynamic_idx++];
                }
                args.push_back(make_bytes_arg(static_cast<uint32_t>(args.size()), &scalar, sizeof(int32_t)));
                ++scalar_idx;
            }
        }
        KernelDispatch dispatch = make_1d_dispatch(count, tg);
        auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
        m_linear_binary_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
    }
}

void VulkanStage::execute_binary_same_shape(GpuCommandBufferHandle command_buffer) {
    auto resolve_input = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
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
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    GpuTensor* input0 = resolve_input(0);
    GpuTensor* input1 = resolve_input(1);
    OPENVINO_ASSERT(input0 && input1 && output, "GFX Vulkan same-shape binary: missing tensors");
    const auto op_key = binary_chunk_key(m_type);
    OPENVINO_ASSERT(!op_key.empty(), "GFX Vulkan same-shape binary: unsupported op ", m_type);
    const ov::element::Type src0_et = resolve_stage_input_element_type(m_node, 0, input0);
    const ov::element::Type src1_et = resolve_stage_input_element_type(m_node, 1, input1);
    const ov::element::Type dst_et = resolve_stage_element_type(m_node, output);
    OPENVINO_ASSERT(is_supported_binary_io_types(src0_et, src1_et, dst_et, op_key),
                    "GFX Vulkan same-shape binary: unsupported element types ",
                    src0_et,
                    " / ",
                    src1_et,
                    " -> ",
                    dst_et,
                    " for op ",
                    op_key);

    if (!m_linear_binary_same_shape_kernel ||
        m_linear_binary_same_shape_src0_elem_type != src0_et ||
        m_linear_binary_same_shape_src1_elem_type != src1_et ||
        m_linear_binary_same_shape_dst_elem_type != dst_et ||
        m_linear_binary_same_shape_key != op_key) {
        auto& ctx = gfx_mlir_context();
        auto module = build_linear_binary_same_shape_module(ctx, src0_et, src1_et, dst_et, op_key);
        KernelSource src = make_kernel_source_from_mlir(module, "linear_binary_same_shape", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_linear_binary_same_shape_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_linear_binary_same_shape_kernel,
                        "GFX Vulkan same-shape binary: kernel compile failed: ",
                        log);
        m_linear_binary_same_shape_kernel->prepare_runtime_artifacts();
        m_linear_binary_same_shape_src0_elem_type = src0_et;
        m_linear_binary_same_shape_src1_elem_type = src1_et;
        m_linear_binary_same_shape_dst_elem_type = dst_et;
        m_linear_binary_same_shape_key = op_key;
    }

    ov::Shape out_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
    if (out_shape.empty() && m_node && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    output->shape = out_shape;
    output->expected_type = dst_et;
    const uint32_t total = static_cast<uint32_t>(tensor_elements(out_shape));
    const size_t tg =
        std::min<size_t>(total, std::max<size_t>(1, m_linear_binary_same_shape_kernel->clamp_threadgroup_size(64)));
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    std::vector<KernelArg> args = {
        make_buffer_arg(0, input0->buf),
        make_buffer_arg(1, input1->buf),
        make_buffer_arg(2, output->buf),
    };
    m_linear_binary_same_shape_kernel->execute(command_buffer, dispatch, args, nullptr);
}

void VulkanStage::execute_binary_bias_add(GpuCommandBufferHandle command_buffer) {
    auto resolve_input = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
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
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    GpuTensor* input0 = resolve_input(0);
    GpuTensor* input1 = resolve_input(1);
    OPENVINO_ASSERT(input0 && input1 && output, "GFX Vulkan bias add: missing tensors");

    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan bias add: unsupported element type ",
                    elem_type);
    if (!m_binary_bias_add_kernel || m_binary_bias_add_elem_type != elem_type) {
        auto& ctx = gfx_mlir_context();
        auto module = build_binary_bias_add_module(ctx, elem_type);
        KernelSource src = make_kernel_source_from_mlir(module, "binary_bias_add", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_binary_bias_add_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_binary_bias_add_kernel, "GFX Vulkan bias add: kernel compile failed: ", log);
        m_binary_bias_add_kernel->prepare_runtime_artifacts();
        m_binary_bias_add_elem_type = elem_type;
    }

    const ov::Shape& dispatch_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
    const uint32_t total = static_cast<uint32_t>(tensor_elements(dispatch_shape));
    const size_t tg = std::min<size_t>(total,
                                       std::max<size_t>(1, m_binary_bias_add_kernel->clamp_threadgroup_size(64)));
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    std::vector<KernelArg> args = {
        make_buffer_arg(0, input0->buf),
        make_buffer_arg(1, input1->buf),
        make_buffer_arg(2, output->buf),
    };
    m_binary_bias_add_kernel->execute(command_buffer, dispatch, args, nullptr);
}

void VulkanStage::execute_conv2d_1x1_chunked(GpuCommandBufferHandle command_buffer) {
    if (m_conv2d_1x1_force_chunked_fallback) {
        execute_conv2d_chunked(command_buffer);
        return;
    }
    auto resolve_input = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
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
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    OPENVINO_ASSERT(conv, "GFX Vulkan conv2d 1x1: node cast failed");
    GpuTensor* input0 = resolve_input(0);
    GpuTensor* input1 = resolve_input(1);
    GpuTensor* bias = (m_has_bias && !m_kernel_extra_inputs.empty() && m_kernel_extra_inputs[0].buf.valid())
                          ? &m_kernel_extra_inputs[0]
                          : nullptr;
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(input0 && input1 && output, "GFX Vulkan conv2d 1x1: missing tensors");
    OPENVINO_ASSERT(!m_has_bias || bias, "GFX Vulkan conv2d 1x1: missing fused bias tensor");
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("VulkanExec") << "Using conv2d_1x1 path for " << m_name;
    }

    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan conv2d 1x1: unsupported element type ",
                    elem_type);
    const auto caps = query_parallelism_caps(m_buffer_manager);
    const auto total = static_cast<uint64_t>(tensor_elements(conv->get_output_shape(0)));
    const auto launch_plan =
        select_chunk_dispatch_plan(caps, "conv2d_1x1", total, static_cast<uint64_t>(conv->get_input_shape(0).at(1)));
    if (!m_conv2d_1x1_kernel || m_conv2d_1x1_elem_type != elem_type ||
        m_conv2d_1x1_threads_per_group != launch_plan.threads_per_group) {
        auto& ctx = gfx_mlir_context();
        auto module = build_conv2d_1x1_module(ctx, elem_type, launch_plan.threads_per_group);
        if (gfx_log_debug_enabled()) {
            std::string module_text;
            llvm::raw_string_ostream os(module_text);
            module.print(os);
            gfx_log_debug("VulkanExec") << "conv2d_1x1 module:\n" << module_text;
        }
        KernelSource src = make_kernel_source_from_mlir(module, "conv2d_1x1", /*arg_count=*/m_has_bias ? 5 : 4);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_conv2d_1x1_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_conv2d_1x1_kernel, "GFX Vulkan conv2d 1x1: kernel compile failed: ", log);
        m_conv2d_1x1_kernel->prepare_runtime_artifacts();
        m_conv2d_1x1_elem_type = elem_type;
        m_conv2d_1x1_threads_per_group = launch_plan.threads_per_group;
    }

    const size_t tg = std::min<size_t>(total,
                                       std::max<size_t>(1,
                                                        m_conv2d_1x1_kernel->clamp_threadgroup_size(
                                                            launch_plan.threads_per_group)));
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    const auto& in_shape = conv->get_input_shape(0);
    const auto& out_shape = conv->get_output_shape(0);
    const int32_t params[] = {
        static_cast<int32_t>(in_shape.at(0)),
        static_cast<int32_t>(in_shape.at(1)),
        static_cast<int32_t>(in_shape.at(2)),
        static_cast<int32_t>(in_shape.at(3)),
        static_cast<int32_t>(out_shape.at(1)),
        static_cast<int32_t>(out_shape.at(2)),
        static_cast<int32_t>(out_shape.at(3)),
    };
    std::vector<KernelArg> args;
    args.push_back(make_buffer_arg(0, input0->buf));
    args.push_back(make_buffer_arg(1, input1->buf));
    if (m_has_bias) {
        args.push_back(make_buffer_arg(2, bias->buf));
    }
    args.push_back(make_bytes_arg(static_cast<uint32_t>(args.size()), &params, sizeof(params)));
    args.push_back(make_buffer_arg(static_cast<uint32_t>(args.size()), output->buf));
    auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
    try {
        m_conv2d_1x1_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
    } catch (const std::exception& ex) {
        if (!is_vulkan_pipeline_creation_failure(ex)) {
            throw;
        }
        m_conv2d_1x1_force_chunked_fallback = true;
        gfx_log_info("VulkanExec") << "Falling back from conv2d_1x1 direct to chunked variant for "
                                   << m_name << ": " << ex.what();
        execute_conv2d_chunked(command_buffer);
    }
}

void VulkanStage::execute_conv2d_3x3_direct(GpuCommandBufferHandle command_buffer) {
    if (m_conv2d_3x3_force_chunked_fallback) {
        execute_conv2d_chunked(command_buffer);
        return;
    }
    auto resolve_input = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
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
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    OPENVINO_ASSERT(conv, "GFX Vulkan conv2d 3x3 direct: node cast failed");
    GpuTensor* input0 = resolve_input(0);
    GpuTensor* input1 = resolve_input(1);
    GpuTensor* bias = (m_has_bias && !m_kernel_extra_inputs.empty() && m_kernel_extra_inputs[0].buf.valid())
                          ? &m_kernel_extra_inputs[0]
                          : nullptr;
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(input0 && input1 && output, "GFX Vulkan conv2d 3x3 direct: missing tensors");
    OPENVINO_ASSERT(!m_has_bias || bias, "GFX Vulkan conv2d 3x3 direct: missing fused bias tensor");
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("VulkanExec") << "Using conv2d_3x3_direct path for " << m_name
                                    << " variant=" << conv_route_plan().algorithm.variant;
    }

    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan conv2d 3x3 direct: unsupported element type ",
                    elem_type);
    const auto caps = query_parallelism_caps(m_buffer_manager);
    auto plan = select_conv2d_direct_plan(caps,
                                          conv->get_output_shape(0),
                                          conv->get_input_shape(0).at(1),
                                          conv->get_output_shape(0).at(1),
                                          conv->get_input_shape(0).at(1) *
                                              conv->get_input_shape(1).at(2) *
                                              conv->get_input_shape(1).at(3),
                                          conv->get_strides().at(0) == 2 &&
                                              conv->get_strides().at(1) == 2);
    if (m_conv2d_3x3_force_safe_variant) {
        plan = make_safe_conv2d_direct_plan(caps);
    }
    if (!m_conv2d_3x3_direct_kernel || m_conv2d_3x3_direct_elem_type != elem_type ||
        m_conv2d_3x3_direct_oc_block != plan.output_channel_block ||
        m_conv2d_3x3_direct_threads_per_group != plan.threads_per_group ||
        m_conv2d_3x3_direct_variant != plan.variant) {
        auto& ctx = gfx_mlir_context();
        auto module =
            build_conv2d_3x3_direct_module(ctx, elem_type, plan.output_channel_block, plan.threads_per_group, plan.variant);
        if (gfx_log_debug_enabled()) {
            std::string module_text;
            llvm::raw_string_ostream os(module_text);
            module.print(os);
            gfx_log_debug("VulkanExec") << "conv2d_3x3_direct module:\n" << module_text;
        }
        const char* entry_name = plan.variant.rfind("conv2d_direct_xy", 0) == 0
                                     ? "conv2d_3x3_direct_xy"
                                     : (plan.output_channel_block == 1 ? "conv2d_3x3_direct" : "conv2d_3x3_direct_oc2");
        KernelSource src = make_kernel_source_from_mlir(module, entry_name, /*arg_count=*/m_has_bias ? 4 : 3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_conv2d_3x3_direct_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_conv2d_3x3_direct_kernel, "GFX Vulkan conv2d 3x3 direct: kernel compile failed: ", log);
        m_conv2d_3x3_direct_kernel->prepare_runtime_artifacts();
        m_conv2d_3x3_direct_elem_type = elem_type;
        m_conv2d_3x3_direct_oc_block = plan.output_channel_block;
        m_conv2d_3x3_direct_threads_per_group = plan.threads_per_group;
        m_conv2d_3x3_direct_variant = plan.variant;
    }

    const uint64_t out_groups =
        (static_cast<uint64_t>(conv->get_output_shape(0).at(1)) + plan.output_channel_block - 1) /
        plan.output_channel_block;
    KernelDispatch dispatch{};
    if (plan.variant.rfind("conv2d_direct_xy", 0) == 0) {
        const bool xy32x2_variant =
            plan.variant == "conv2d_direct_xy32x2" || plan.variant == "conv2d_direct_xy32x2_dense_s2";
        const bool xy16x4_variant =
            plan.variant == "conv2d_direct_xy16x4" || plan.variant == "conv2d_direct_xy16x4_dense_s2";
        const size_t tg_x = xy32x2_variant ? 32 : (xy16x4_variant ? 16 : 8);
        const size_t tg_y = xy32x2_variant ? 2 : (xy16x4_variant ? 4 : 8);
        dispatch = make_3d_dispatch(conv->get_output_shape(0).at(3),
                                    conv->get_output_shape(0).at(2),
                                    static_cast<size_t>(conv->get_output_shape(0).at(0) * out_groups),
                                    tg_x,
                                    tg_y,
                                    1);
    } else {
        const uint32_t total = static_cast<uint32_t>(conv->get_output_shape(0).at(0) * out_groups *
                                                     conv->get_output_shape(0).at(2) * conv->get_output_shape(0).at(3));
        const size_t tg = std::min<size_t>(total,
                                           std::max<size_t>(1,
                                                            m_conv2d_3x3_direct_kernel->clamp_threadgroup_size(
                                                                plan.threads_per_group)));
        dispatch = make_1d_dispatch(total, tg);
    }
    std::vector<KernelArg> args;
    args.push_back(make_buffer_arg(0, input0->buf));
    args.push_back(make_buffer_arg(1, input1->buf));
    if (m_has_bias) {
        args.push_back(make_buffer_arg(2, bias->buf));
    }
    args.push_back(make_buffer_arg(static_cast<uint32_t>(args.size()), output->buf));
    try {
        m_conv2d_3x3_direct_kernel->execute(command_buffer, dispatch, args, nullptr);
    } catch (const std::exception& ex) {
        if (!is_vulkan_pipeline_creation_failure(ex)) {
            throw;
        }
        if (!m_conv2d_3x3_force_safe_variant) {
            m_conv2d_3x3_force_safe_variant = true;
            gfx_log_info("VulkanExec") << "Retrying conv2d_3x3 direct with safe variant for "
                                       << m_name << ": " << ex.what();
            execute_conv2d_3x3_direct(command_buffer);
            return;
        }
        m_conv2d_3x3_force_chunked_fallback = true;
        gfx_log_info("VulkanExec") << "Falling back from conv2d_3x3 direct to chunked variant for "
                                   << m_name << ": " << ex.what();
        execute_conv2d_chunked(command_buffer);
    }
}

void VulkanStage::execute_conv2d_chunked(GpuCommandBufferHandle command_buffer) {
    auto resolve_input = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
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
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    OPENVINO_ASSERT(conv, "GFX Vulkan conv2d chunked: node cast failed");
    GpuTensor* input0 = resolve_input(0);
    GpuTensor* input1 = resolve_input(1);
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(input0 && input1 && output,
                    "GFX Vulkan conv2d chunked: missing tensors");
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("VulkanExec") << "Using conv2d_chunk path for " << m_name;
    }
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("VulkanExec") << "Conv2D chunked buffers in0=" << input0->buf.buffer
                                    << " in1=" << input1->buf.buffer
                                    << " out=" << output->buf.buffer
                                    << " in0_type=" << input0->buf.type << "/" << input0->expected_type
                                    << " in1_type=" << input1->buf.type << "/" << input1->expected_type
                                    << " out_type=" << output->buf.type << "/" << output->expected_type;
    }

    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan conv2d chunked: unsupported element type ",
                    elem_type);
    const auto& out_shape = conv->get_output_shape(0);
    const auto& in_shape = conv->get_input_shape(0);
    const auto& w_shape = conv->get_input_shape(1);
    const uint64_t work_per_elem = static_cast<uint64_t>(in_shape.at(1)) *
                                   static_cast<uint64_t>(w_shape.at(2)) *
                                   static_cast<uint64_t>(w_shape.at(3));
    const auto caps = query_parallelism_caps(m_buffer_manager);
    const bool stride2 = conv->get_strides().at(0) > 1 || conv->get_strides().at(1) > 1;
    const auto spatial_plan = select_conv_parallelism(caps,
                                                      out_shape,
                                                      static_cast<uint64_t>(in_shape.at(1)),
                                                      static_cast<uint64_t>(w_shape.at(0)),
                                                      work_per_elem,
                                                      stride2,
                                                      /*depthwise=*/false);
    const uint32_t threads_h = std::max<uint32_t>(1u, spatial_plan.dispatch.threads_h);
    const uint32_t threads_w = std::max<uint32_t>(1u, spatial_plan.dispatch.threads_w);
    if (!m_conv2d_chunk_kernel || m_conv2d_chunk_elem_type != elem_type ||
        m_conv2d_chunk_threads_h != threads_h ||
        m_conv2d_chunk_threads_w != threads_w) {
        auto& ctx = gfx_mlir_context();
        auto module = build_conv2d_chunk_module(ctx, elem_type, threads_h, threads_w);
        if (gfx_log_debug_enabled()) {
            std::string module_text;
            llvm::raw_string_ostream os(module_text);
            module.print(os);
            gfx_log_debug("VulkanExec") << "conv2d_chunk module:\n" << module_text;
        }
        KernelSource src = make_kernel_source_from_mlir(module, "conv2d_chunk", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_conv2d_chunk_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_conv2d_chunk_kernel, "GFX Vulkan conv2d chunked: kernel compile failed: ", log);
        m_conv2d_chunk_kernel->prepare_runtime_artifacts();
        m_conv2d_chunk_elem_type = elem_type;
        m_conv2d_chunk_threads_h = threads_h;
        m_conv2d_chunk_threads_w = threads_w;
    }
    std::vector<KernelArg> args{
        make_buffer_arg(0, input0->buf),
        make_buffer_arg(1, input1->buf),
        make_buffer_arg(2, output->buf),
    };
    auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
    KernelDispatch dispatch = make_3d_dispatch(out_shape.at(3),
                                               out_shape.at(2),
                                               out_shape.at(0) * out_shape.at(1),
                                               threads_w,
                                               threads_h,
                                               1);
    // Record chunked Conv2D into the caller-owned infer command buffer so the
    // infer path keeps a single queue submit while the kernel maps threads
    // directly to output [W, H, N*C] coordinates.
    m_conv2d_chunk_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
}

void VulkanStage::execute_group_conv2d_chunked(GpuCommandBufferHandle command_buffer) {
    auto resolve_input = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
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
    auto gconv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(m_node);
    OPENVINO_ASSERT(gconv, "GFX Vulkan group_conv2d chunked: node cast failed");
    GpuTensor* input0 = resolve_input(0);
    GpuTensor* input1 = resolve_input(1);
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(input0 && input1 && output,
                    "GFX Vulkan group_conv2d chunked: missing tensors");

    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan group_conv2d chunked: unsupported element type ",
                    elem_type);
    const auto& out_shape = gconv->get_output_shape(0);
    const auto& w_shape = gconv->get_input_shape(1);
    const auto caps = query_parallelism_caps(m_buffer_manager);
    const auto launch_plan = select_chunk_dispatch_plan(caps,
                                                        "group_conv2d",
                                                        static_cast<uint64_t>(tensor_elements(out_shape)),
                                                        static_cast<uint64_t>(w_shape[3]) *
                                                            static_cast<uint64_t>(w_shape[4]));
    if (!m_group_conv2d_kernel || m_group_conv2d_elem_type != elem_type ||
        m_group_conv2d_threads_per_group != launch_plan.threads_per_group) {
        auto& ctx = gfx_mlir_context();
        auto module = build_group_conv2d_chunk_module(ctx, elem_type, launch_plan.threads_per_group);
        KernelSource src = make_kernel_source_from_mlir(module, "group_conv2d_direct", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_group_conv2d_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_group_conv2d_kernel,
                        "GFX Vulkan group_conv2d chunked: kernel compile failed: ",
                        log);
        m_group_conv2d_kernel->prepare_runtime_artifacts();
        m_group_conv2d_elem_type = elem_type;
        m_group_conv2d_threads_per_group = launch_plan.threads_per_group;
    }

    output->shape = gconv->get_output_shape(0);
    const uint32_t total = static_cast<uint32_t>(tensor_elements(output->shape));
    std::vector<KernelArg> args{
        make_buffer_arg(0, input0->buf),
        make_buffer_arg(1, input1->buf),
        make_buffer_arg(2, output->buf),
    };
    auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
    const uint32_t tg = m_group_conv2d_kernel->clamp_threadgroup_size(launch_plan.threads_per_group);
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    m_group_conv2d_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
}

mlir::ModuleOp VulkanStage::build_split_single_module(mlir::MLIRContext& ctx,
                                                      const ov::element::Type& et) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect, gpu::GPUDialect, scf::SCFDialect, memref::MemRefDialect, arith::ArithDialect>();

    Type elem_ty = to_mlir_type(et, ctx, /*fallback_f32=*/true, /*allow_unsigned=*/true,
                                /*allow_small_ints=*/true, /*allow_bf16=*/false,
                                /*allow_boolean=*/false, /*signless_integers=*/true);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 3));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2}));

    auto in_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto param_ty = MemRefType::get({5}, IntegerType::get(&ctx, 32));
    SmallVector<Type, 3> arg_types{in_ty, param_ty, in_ty};
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "split_single",
                                                 b.getFunctionType(arg_types, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));
    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();

    // params: [0]=outer, [1]=axis_total, [2]=inner, [3]=axis_offset, [4]=slice_len
    auto c0 = body.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = body.create<arith::ConstantIndexOp>(loc, 1);
    auto load_param = [&](int idx) -> Value {
        auto idx_val = body.create<arith::ConstantIndexOp>(loc, idx);
        auto v_i32 = body.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{idx_val});
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), v_i32);
    };
    Value outer = load_param(0);
    Value axis_total = load_param(1);
    Value inner = load_param(2);
    Value axis_offset = load_param(3);
    Value slice_len = load_param(4);

    Value total = body.create<arith::MulIOp>(loc, outer,
                                             body.create<arith::MulIOp>(loc, slice_len, inner));
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, idx, total);
    auto active_if = body.create<scf::IfOp>(loc, active, false);
    {
        auto loop = active_if.getThenBodyBuilder();
        Value inner_idx = loop.create<arith::RemUIOp>(loc, idx, inner);
        Value tmp = loop.create<arith::DivUIOp>(loc, idx, inner);
        Value axis_idx = loop.create<arith::RemUIOp>(loc, tmp, slice_len);
        Value outer_idx = loop.create<arith::DivUIOp>(loc, tmp, slice_len);

        Value base0 = loop.create<arith::MulIOp>(loc, outer_idx, axis_total);
        Value base1 = loop.create<arith::AddIOp>(loc, base0, axis_offset);
        Value base2 = loop.create<arith::AddIOp>(loc, base1, axis_idx);
        Value src_index = loop.create<arith::MulIOp>(loc, base2, inner);
        src_index = loop.create<arith::AddIOp>(loc, src_index, inner_idx);

        Value val = loop.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{src_index});
        loop.create<memref::StoreOp>(loc, val, fn.getArgument(2), ValueRange{idx});
    }

    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_concat_single_module(mlir::MLIRContext& ctx,
                                                       const ov::element::Type& et) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect, gpu::GPUDialect, scf::SCFDialect, memref::MemRefDialect, arith::ArithDialect>();

    Type elem_ty = to_mlir_type(et, ctx, /*fallback_f32=*/true, /*allow_unsigned=*/true,
                                /*allow_small_ints=*/true, /*allow_bf16=*/false,
                                /*allow_boolean=*/false, /*signless_integers=*/true);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 3));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2}));

    auto io_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto param_ty = MemRefType::get({5}, IntegerType::get(&ctx, 32));
    SmallVector<Type, 3> arg_types{io_ty, param_ty, io_ty};
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "concat_single",
                                                 b.getFunctionType(arg_types, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));
    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();

    // params: [0]=outer, [1]=axis_total, [2]=inner, [3]=axis_offset, [4]=slice_len
    auto c0 = body.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = body.create<arith::ConstantIndexOp>(loc, 1);
    auto load_param = [&](int idx) -> Value {
        auto idx_val = body.create<arith::ConstantIndexOp>(loc, idx);
        auto v_i32 = body.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{idx_val});
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), v_i32);
    };
    Value outer = load_param(0);
    Value axis_total = load_param(1);
    Value inner = load_param(2);
    Value axis_offset = load_param(3);
    Value slice_len = load_param(4);

    Value total = body.create<arith::MulIOp>(loc, outer,
                                             body.create<arith::MulIOp>(loc, slice_len, inner));
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, idx, total);
    auto active_if = body.create<scf::IfOp>(loc, active, false);
    {
        auto loop = active_if.getThenBodyBuilder();
        Value inner_idx = loop.create<arith::RemUIOp>(loc, idx, inner);
        Value tmp = loop.create<arith::DivUIOp>(loc, idx, inner);
        Value axis_idx = loop.create<arith::RemUIOp>(loc, tmp, slice_len);
        Value outer_idx = loop.create<arith::DivUIOp>(loc, tmp, slice_len);

        Value base0 = loop.create<arith::MulIOp>(loc, outer_idx, axis_total);
        Value base1 = loop.create<arith::AddIOp>(loc, base0, axis_offset);
        Value base2 = loop.create<arith::AddIOp>(loc, base1, axis_idx);
        Value dst_index = loop.create<arith::MulIOp>(loc, base2, inner);
        dst_index = loop.create<arith::AddIOp>(loc, dst_index, inner_idx);

        Value val = loop.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{idx});
        loop.create<memref::StoreOp>(loc, val, fn.getArgument(2), ValueRange{dst_index});
    }

    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_concat_binary_module(mlir::MLIRContext& ctx,
                                                       const ov::element::Type& et) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect, gpu::GPUDialect, scf::SCFDialect, memref::MemRefDialect, arith::ArithDialect>();

    Type elem_ty = to_mlir_type(et, ctx, /*fallback_f32=*/true, /*allow_unsigned=*/true,
                                /*allow_small_ints=*/true, /*allow_bf16=*/false,
                                /*allow_boolean=*/false, /*signless_integers=*/true);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 4));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2, 3}));

    auto io_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto param_ty = MemRefType::get({5}, IntegerType::get(&ctx, 32));
    SmallVector<Type, 4> arg_types{io_ty, io_ty, param_ty, io_ty};
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "concat_binary",
                                                 b.getFunctionType(arg_types, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));
    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();

    // params: [0]=outer, [1]=axis_total, [2]=inner, [3]=slice0, [4]=slice1
    auto load_param = [&](int idx) -> Value {
        auto idx_val = body.create<arith::ConstantIndexOp>(loc, idx);
        auto v_i32 = body.create<memref::LoadOp>(loc, fn.getArgument(2), ValueRange{idx_val});
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), v_i32);
    };
    Value outer = load_param(0);
    Value axis_total = load_param(1);
    Value inner = load_param(2);
    Value slice0 = load_param(3);
    Value slice1 = load_param(4);
    Value active_axis = body.create<arith::AddIOp>(loc, slice0, slice1);
    Value total = body.create<arith::MulIOp>(loc, outer,
                                             body.create<arith::MulIOp>(loc, active_axis, inner));
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, idx, total);
    auto active_if = body.create<scf::IfOp>(loc, active, false);
    {
        auto loop = active_if.getThenBodyBuilder();
        Value inner_idx = loop.create<arith::RemUIOp>(loc, idx, inner);
        Value tmp = loop.create<arith::DivUIOp>(loc, idx, inner);
        Value axis_idx = loop.create<arith::RemUIOp>(loc, tmp, active_axis);
        Value outer_idx = loop.create<arith::DivUIOp>(loc, tmp, active_axis);

        Value dst_base0 = loop.create<arith::MulIOp>(loc, outer_idx, axis_total);
        Value dst_base1 = loop.create<arith::AddIOp>(loc, dst_base0, axis_idx);
        Value dst_index = loop.create<arith::MulIOp>(loc, dst_base1, inner);
        dst_index = loop.create<arith::AddIOp>(loc, dst_index, inner_idx);

        auto use_src0 = loop.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, axis_idx, slice0);
        auto select_input = loop.create<scf::IfOp>(loc, TypeRange{elem_ty}, use_src0, true);
        {
            auto then_builder = select_input.getThenBodyBuilder();
            Value src_base0 = then_builder.create<arith::MulIOp>(loc, outer_idx, slice0);
            Value src_base1 = then_builder.create<arith::AddIOp>(loc, src_base0, axis_idx);
            Value src_index = then_builder.create<arith::MulIOp>(loc, src_base1, inner);
            src_index = then_builder.create<arith::AddIOp>(loc, src_index, inner_idx);
            Value val = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{src_index});
            then_builder.create<scf::YieldOp>(loc, val);
        }
        {
            auto else_builder = select_input.getElseBodyBuilder();
            Value axis1_idx = else_builder.create<arith::SubIOp>(loc, axis_idx, slice0);
            Value src_base0 = else_builder.create<arith::MulIOp>(loc, outer_idx, slice1);
            Value src_base1 = else_builder.create<arith::AddIOp>(loc, src_base0, axis1_idx);
            Value src_index = else_builder.create<arith::MulIOp>(loc, src_base1, inner);
            src_index = else_builder.create<arith::AddIOp>(loc, src_index, inner_idx);
            Value val = else_builder.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{src_index});
            else_builder.create<scf::YieldOp>(loc, val);
        }
        loop.create<memref::StoreOp>(loc, select_input.getResult(0), fn.getArgument(3), ValueRange{dst_index});
    }

    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_slice_linear_module(mlir::MLIRContext& ctx,
                                                      const ov::element::Type& et) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect>();

    Type elem_ty = to_mlir_type(et, ctx, /*fallback_f32=*/true, /*allow_unsigned=*/true,
                                /*allow_small_ints=*/true, /*allow_bf16=*/false,
                                /*allow_boolean=*/false, /*signless_integers=*/true);
    Type i32_ty = IntegerType::get(&ctx, 32);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 8));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1, 1, 1, 1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2, 3, 4, 5, 6, 7}));

    auto io_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto scalar_ty = MemRefType::get({1}, i32_ty);
    auto meta_ty = MemRefType::get({ShapedType::kDynamic}, i32_ty);
    SmallVector<Type, 8> arg_types{io_ty, scalar_ty, scalar_ty, meta_ty, meta_ty, meta_ty, meta_ty, io_ty};
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "slice_linear",
                                                 b.getFunctionType(arg_types, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto c0 = body.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = body.create<arith::ConstantIndexOp>(loc, 1);
    auto load_scalar = [&](BlockArgument arg) -> Value {
        auto raw = body.create<memref::LoadOp>(loc, arg, ValueRange{c0});
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), raw);
    };
    auto load_meta = [&](OpBuilder& builder, BlockArgument arg, Value idx) -> Value {
        auto raw = builder.create<memref::LoadOp>(loc, arg, ValueRange{idx});
        return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), raw);
    };

    Value total = load_scalar(fn.getArgument(1));
    Value rank = load_scalar(fn.getArgument(2));
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, idx, total);
    auto active_if = body.create<scf::IfOp>(loc, active, false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        auto for_dims = then_builder.create<scf::ForOp>(loc, c0, rank, c1, ValueRange{idx, c0});
        auto dim_builder = OpBuilder::atBlockBegin(for_dims.getBody());
        Value rev_dim = dim_builder.create<arith::SubIOp>(loc, rank, c1);
        rev_dim = dim_builder.create<arith::SubIOp>(loc, rev_dim, for_dims.getInductionVar());
        Value dim_size = load_meta(dim_builder, fn.getArgument(3), rev_dim);
        Value coord = dim_builder.create<arith::RemUIOp>(loc, for_dims.getRegionIterArgs()[0], dim_size);
        Value next_rem = dim_builder.create<arith::DivUIOp>(loc, for_dims.getRegionIterArgs()[0], dim_size);
        Value start = load_meta(dim_builder, fn.getArgument(5), rev_dim);
        Value step = load_meta(dim_builder, fn.getArgument(6), rev_dim);
        Value stride = load_meta(dim_builder, fn.getArgument(4), rev_dim);
        Value src_dim = dim_builder.create<arith::AddIOp>(
            loc,
            start,
            dim_builder.create<arith::MulIOp>(loc, coord, step));
        Value next_src = dim_builder.create<arith::AddIOp>(
            loc,
            for_dims.getRegionIterArgs()[1],
            dim_builder.create<arith::MulIOp>(loc, src_dim, stride));
        if (!for_dims.getBody()->empty() && for_dims.getBody()->back().hasTrait<OpTrait::IsTerminator>()) {
            for_dims.getBody()->back().erase();
        }
        auto dim_end = OpBuilder::atBlockEnd(for_dims.getBody());
        dim_end.create<scf::YieldOp>(loc, ValueRange{next_rem, next_src});

        Value value = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{for_dims.getResult(1)});
        then_builder.create<memref::StoreOp>(loc, value, fn.getArgument(7), ValueRange{idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_softmax_row_module(mlir::MLIRContext& ctx,
                                                     const ov::element::Type& et,
                                                     bool log_softmax) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    mlir::math::MathDialect>();

    Type elem_ty;
    switch (et) {
        case ov::element::f16: elem_ty = Float16Type::get(&ctx); break;
        case ov::element::f32: elem_ty = Float32Type::get(&ctx); break;
        default: OPENVINO_THROW("GFX Vulkan Softmax: unsupported element type ", et);
    }
    Type compute_ty = elem_ty;

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());

    auto io_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto param_ty = MemRefType::get({4}, IntegerType::get(&ctx, 32));
    SmallVector<Type, 3> arg_types{io_ty, param_ty, io_ty};
    auto fn = b.create<func::FuncOp>(UnknownLoc::get(&ctx),
                                     log_softmax ? "logsoftmax_row" : "softmax_row",
                                     b.getFunctionType(arg_types, {}));
    auto* entry = fn.addEntryBlock();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();

    auto cast_to_compute = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == elem_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, elem_ty, value);
    };
    auto load_param = [&](int idx) -> Value {
        auto idx_val = body.create<arith::ConstantIndexOp>(loc, idx);
        auto v_i32 = body.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{idx_val});
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), v_i32);
    };

    Value row_begin = load_param(0);
    Value row_count = load_param(1);
    Value cols = load_param(2);
    Value inner = load_param(3);
    auto c0 = body.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = body.create<arith::ConstantIndexOp>(loc, 1);
    auto neg_inf = body.create<arith::ConstantOp>(loc,
                                                  FloatAttr::get(compute_ty, -std::numeric_limits<float>::infinity()));
    auto zero = body.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 0.0f));
    auto for_rows = body.create<scf::ForOp>(loc, c0, row_count, c1);
    auto brows = OpBuilder::atBlockBegin(for_rows.getBody());
    auto flat_row = brows.create<arith::AddIOp>(loc, row_begin, for_rows.getInductionVar());
    auto outer = brows.create<arith::DivUIOp>(loc, flat_row, inner);
    auto inner_idx = brows.create<arith::RemUIOp>(loc, flat_row, inner);
    Value row_base = brows.create<arith::MulIOp>(loc, outer, cols);
    row_base = brows.create<arith::MulIOp>(loc, row_base, inner);
    row_base = brows.create<arith::AddIOp>(loc, row_base, inner_idx);

    auto for_max = brows.create<scf::ForOp>(loc, c0, cols, c1, ValueRange{neg_inf});
    auto bmax = OpBuilder::atBlockBegin(for_max.getBody());
    auto max_idx = bmax.create<arith::AddIOp>(
        loc,
        row_base,
        bmax.create<arith::MulIOp>(loc, for_max.getInductionVar(), inner));
    auto max_raw = bmax.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{max_idx});
    auto max_val = cast_to_compute(bmax, max_raw);
    auto max_cur = for_max.getRegionIterArgs()[0];
    auto max_cmp = bmax.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, max_val, max_cur);
    auto max_sel = bmax.create<arith::SelectOp>(loc, max_cmp, max_val, max_cur);
    if (!for_max.getBody()->empty() && for_max.getBody()->back().hasTrait<OpTrait::IsTerminator>()) {
        for_max.getBody()->back().erase();
    }
    auto bmax_end = OpBuilder::atBlockEnd(for_max.getBody());
    bmax_end.create<scf::YieldOp>(loc, max_sel.getResult());

    auto for_sum = brows.create<scf::ForOp>(loc, c0, cols, c1, ValueRange{zero});
    auto bsum = OpBuilder::atBlockBegin(for_sum.getBody());
    auto sum_idx = bsum.create<arith::AddIOp>(
        loc,
        row_base,
        bsum.create<arith::MulIOp>(loc, for_sum.getInductionVar(), inner));
    auto sum_raw = bsum.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{sum_idx});
    auto sum_val = cast_to_compute(bsum, sum_raw);
    auto sum_diff = bsum.create<arith::SubFOp>(loc, sum_val, for_max.getResult(0));
    auto sum_exp = bsum.create<mlir::math::ExpOp>(loc, sum_diff);
    auto sum_cur = for_sum.getRegionIterArgs()[0];
    auto sum_next = bsum.create<arith::AddFOp>(loc, sum_cur, sum_exp);
    if (!for_sum.getBody()->empty() && for_sum.getBody()->back().hasTrait<OpTrait::IsTerminator>()) {
        for_sum.getBody()->back().erase();
    }
    auto bsum_end = OpBuilder::atBlockEnd(for_sum.getBody());
    bsum_end.create<scf::YieldOp>(loc, sum_next.getResult());

    auto for_write = brows.create<scf::ForOp>(loc, c0, cols, c1);
    auto bwrite = OpBuilder::atBlockBegin(for_write.getBody());
    auto out_idx = bwrite.create<arith::AddIOp>(
        loc,
        row_base,
        bwrite.create<arith::MulIOp>(loc, for_write.getInductionVar(), inner));
    auto out_raw = bwrite.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{out_idx});
    auto out_compute = cast_to_compute(bwrite, out_raw);
    auto out_diff = bwrite.create<arith::SubFOp>(loc, out_compute, for_max.getResult(0));
    Value out_value;
    if (log_softmax) {
        auto log_sum = bwrite.create<mlir::math::LogOp>(loc, for_sum.getResult(0));
        out_value = bwrite.create<arith::SubFOp>(loc, out_diff, log_sum);
    } else {
        auto out_exp = bwrite.create<mlir::math::ExpOp>(loc, out_diff);
        out_value = bwrite.create<arith::DivFOp>(loc, out_exp, for_sum.getResult(0));
    }
    auto out_cast = cast_to_output(bwrite, out_value);
    OpBuilder bwrite_end(for_write.getBody(), for_write.getBody()->getTerminator()->getIterator());
    bwrite_end.create<memref::StoreOp>(loc, out_cast, fn.getArgument(2), ValueRange{out_idx});
    body.setInsertionPointAfter(for_rows);
    body.create<func::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_interpolate_module(mlir::MLIRContext& ctx,
                                                     const ov::element::Type& et) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    math::MathDialect>();

    Type elem_ty;
    switch (et) {
        case ov::element::f16: elem_ty = Float16Type::get(&ctx); break;
        case ov::element::f32: elem_ty = Float32Type::get(&ctx); break;
        default: OPENVINO_THROW("GFX Vulkan Interpolate: unsupported element type ", et);
    }
    const Type compute_ty = Float32Type::get(&ctx);
    const auto i32_ty = IntegerType::get(&ctx, 32);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(i32_ty, 3));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2}));

    auto io_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto param_ty = MemRefType::get({9}, i32_ty);
    SmallVector<Type, 3> arg_types{io_ty, param_ty, io_ty};
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "interpolate_direct",
                                                 b.getFunctionType(arg_types, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {8, 8, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    const auto loc = fn.getLoc();

    auto c0 = body.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = body.create<arith::ConstantIndexOp>(loc, 1);
    auto c0_i32 = body.create<arith::ConstantIntOp>(loc, 0, 32);
    auto c1_i32 = body.create<arith::ConstantIntOp>(loc, 1, 32);
    auto c_half = body.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 0.5f));
    auto c_one = body.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 1.0f));

    auto load_param_i32 = [&](int idx) -> Value {
        auto idx_val = body.create<arith::ConstantIndexOp>(loc, idx);
        return body.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{idx_val});
    };
    auto load_param_index = [&](int idx) -> Value {
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), load_param_i32(idx));
    };
    auto cast_to_compute = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == elem_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, elem_ty, value);
    };

    Value n_dim = load_param_index(0);
    Value c_dim = load_param_index(1);
    Value h_in = load_param_index(2);
    Value w_in = load_param_index(3);
    Value h_out = load_param_index(4);
    Value w_out = load_param_index(5);
    Value coord_mode = load_param_i32(6);
    Value nearest_flag = load_param_i32(7);
    Value nc_dim = body.create<arith::MulIOp>(loc, n_dim, c_dim);

    Value bid_x = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bid_y = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::y);
    Value bid_z = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::z);
    Value bdim_x = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value bdim_y = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::y);
    Value bdim_z = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::z);
    Value tid_x = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value tid_y = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::y);
    Value tid_z = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::z);

    Value w = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid_x, bdim_x), tid_x);
    Value h = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid_y, bdim_y), tid_y);
    Value nc = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid_z, bdim_z), tid_z);

    auto w_valid = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, w, w_out);
    auto h_valid = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, h, h_out);
    auto nc_valid = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, nc, nc_dim);
    auto active = body.create<arith::AndIOp>(loc, body.create<arith::AndIOp>(loc, w_valid, h_valid), nc_valid);
    auto active_if = body.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        auto make_f32 = [&](float value) -> Value {
            return then_builder.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, value));
        };

        Value n_dim_t = n_dim;
        Value c_dim_t = c_dim;
        Value h_in_t = h_in;
        Value w_in_t = w_in;
        Value h_out_t = h_out;
        Value w_out_t = w_out;
        Value coord_mode_t = coord_mode;
        Value nearest_flag_t = nearest_flag;

        Value c = then_builder.create<arith::RemUIOp>(loc, nc, c_dim_t);
        Value n = then_builder.create<arith::DivUIOp>(loc, nc, c_dim_t);

        Value h_i32 = then_builder.create<arith::IndexCastOp>(loc, i32_ty, h);
        Value w_i32 = then_builder.create<arith::IndexCastOp>(loc, i32_ty, w);
        Value fh = then_builder.create<arith::SIToFPOp>(loc, compute_ty, h_i32);
        Value fw = then_builder.create<arith::SIToFPOp>(loc, compute_ty, w_i32);

        auto scale_coord = [&](Value coord,
                               Value in_dim,
                               Value out_dim,
                               Value coord_mode_val) -> Value {
            Value in_i32 = then_builder.create<arith::IndexCastOp>(loc, i32_ty, in_dim);
            Value out_i32 = then_builder.create<arith::IndexCastOp>(loc, i32_ty, out_dim);
            Value in_f = then_builder.create<arith::SIToFPOp>(loc, compute_ty, in_i32);
            Value out_f = then_builder.create<arith::SIToFPOp>(loc, compute_ty, out_i32);

            Value asym = then_builder.create<arith::MulFOp>(loc,
                                                            coord,
                                                            then_builder.create<arith::DivFOp>(loc, in_f, out_f));

            Value half_scaled = then_builder.create<arith::SubFOp>(
                loc,
                then_builder.create<arith::MulFOp>(
                    loc,
                    then_builder.create<arith::AddFOp>(loc, coord, c_half),
                    then_builder.create<arith::DivFOp>(loc, in_f, out_f)),
                c_half);

            Value align = asym;
            auto out_gt_one = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, out_dim, c1);
            auto align_if = then_builder.create<scf::IfOp>(loc, TypeRange{compute_ty}, out_gt_one, /*withElseRegion=*/true);
            {
                auto align_then = align_if.getThenBodyBuilder();
                Value in_minus_one = align_then.create<arith::SubIOp>(loc, in_dim, c1);
                Value out_minus_one = align_then.create<arith::SubIOp>(loc, out_dim, c1);
                Value in_m1_i32 = align_then.create<arith::IndexCastOp>(loc, i32_ty, in_minus_one);
                Value out_m1_i32 = align_then.create<arith::IndexCastOp>(loc, i32_ty, out_minus_one);
                Value in_m1_f = align_then.create<arith::SIToFPOp>(loc, compute_ty, in_m1_i32);
                Value out_m1_f = align_then.create<arith::SIToFPOp>(loc, compute_ty, out_m1_i32);
                Value scale = align_then.create<arith::DivFOp>(loc, in_m1_f, out_m1_f);
                align_then.create<scf::YieldOp>(loc,
                                                ValueRange{align_then.create<arith::MulFOp>(loc, coord, scale).getResult()});
            }
            {
                auto align_else = align_if.getElseBodyBuilder();
                align_else.create<scf::YieldOp>(loc, ValueRange{asym});
            }
            align = align_if.getResult(0);

            auto is_align = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, coord_mode_val, c1_i32);
            auto is_half = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, coord_mode_val, c0_i32);
            Value selected = then_builder.create<arith::SelectOp>(loc, is_half, half_scaled, asym);
            return then_builder.create<arith::SelectOp>(loc, is_align, align, selected);
        };

        fh = scale_coord(fh, h_in_t, h_out_t, coord_mode_t);
        fw = scale_coord(fw, w_in_t, w_out_t, coord_mode_t);

        auto floor_to_i32 = [&](Value coord_f) -> Value {
            Value i32 = then_builder.create<arith::FPToSIOp>(loc, i32_ty, coord_f);
            Value i32f = then_builder.create<arith::SIToFPOp>(loc, compute_ty, i32);
            auto lt = then_builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, coord_f, i32f);
            auto adj = then_builder.create<arith::SubIOp>(loc, i32, c1_i32);
            return then_builder.create<arith::SelectOp>(loc, lt, adj, i32);
        };
        auto clamp_idx = [&](OpBuilder& builder, Value idx, Value maxv) -> Value {
            auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
            auto lt0 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, idx, zero);
            auto gt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, idx, maxv);
            auto clamped_low = builder.create<arith::SelectOp>(loc, lt0, zero, idx);
            return builder.create<arith::SelectOp>(loc, gt, maxv, clamped_low);
        };
        auto make_flat_index = [&](OpBuilder& builder, Value n_idx, Value c_idx, Value h_idx, Value w_idx) -> Value {
            Value idx = builder.create<arith::AddIOp>(
                loc,
                builder.create<arith::MulIOp>(loc, n_idx, c_dim_t),
                c_idx);
            idx = builder.create<arith::AddIOp>(
                loc,
                builder.create<arith::MulIOp>(loc, idx, h_in_t),
                h_idx);
            idx = builder.create<arith::AddIOp>(
                loc,
                builder.create<arith::MulIOp>(loc, idx, w_in_t),
                w_idx);
            return idx;
        };
        auto make_output_flat_index = [&](OpBuilder& builder, Value n_idx, Value c_idx, Value h_idx, Value w_idx) -> Value {
            Value idx = builder.create<arith::AddIOp>(
                loc,
                builder.create<arith::MulIOp>(loc, n_idx, c_dim_t),
                c_idx);
            idx = builder.create<arith::AddIOp>(
                loc,
                builder.create<arith::MulIOp>(loc, idx, h_out_t),
                h_idx);
            idx = builder.create<arith::AddIOp>(
                loc,
                builder.create<arith::MulIOp>(loc, idx, w_out_t),
                w_idx);
            return idx;
        };

        Value h0_i32 = floor_to_i32(fh);
        Value w0_i32 = floor_to_i32(fw);
        Value h0f = then_builder.create<arith::SIToFPOp>(loc, compute_ty, h0_i32);
        Value w0f = then_builder.create<arith::SIToFPOp>(loc, compute_ty, w0_i32);
        Value h0 = then_builder.create<arith::IndexCastOp>(loc, then_builder.getIndexType(), h0_i32);
        Value w0 = then_builder.create<arith::IndexCastOp>(loc, then_builder.getIndexType(), w0_i32);
        Value max_h = then_builder.create<arith::SubIOp>(loc, h_in_t, c1);
        Value max_w = then_builder.create<arith::SubIOp>(loc, w_in_t, c1);
        h0 = clamp_idx(then_builder, h0, max_h);
        w0 = clamp_idx(then_builder, w0, max_w);

        Value value = {};
        auto is_nearest = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, nearest_flag_t, c0_i32);
        auto sample_if = then_builder.create<scf::IfOp>(loc, TypeRange{elem_ty}, is_nearest, /*withElseRegion=*/true);
        {
            auto nearest_builder = sample_if.getThenBodyBuilder();
            Value src_idx = make_flat_index(nearest_builder, n, c, h0, w0);
            Value src = nearest_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{src_idx});
            nearest_builder.create<scf::YieldOp>(loc, ValueRange{src});
        }
        {
            auto linear_builder = sample_if.getElseBodyBuilder();
            auto c1_index = linear_builder.create<arith::ConstantIndexOp>(loc, 1);
            Value h1 = linear_builder.create<arith::AddIOp>(loc, h0, c1_index);
            Value w1 = linear_builder.create<arith::AddIOp>(loc, w0, c1_index);
            h1 = clamp_idx(linear_builder, h1, max_h);
            w1 = clamp_idx(linear_builder, w1, max_w);

            Value idx00 = make_flat_index(linear_builder, n, c, h0, w0);
            Value idx01 = make_flat_index(linear_builder, n, c, h0, w1);
            Value idx10 = make_flat_index(linear_builder, n, c, h1, w0);
            Value idx11 = make_flat_index(linear_builder, n, c, h1, w1);

            Value v00 = linear_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{idx00});
            Value v01 = linear_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{idx01});
            Value v10 = linear_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{idx10});
            Value v11 = linear_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{idx11});

            Value v00f = cast_to_compute(linear_builder, v00);
            Value v01f = cast_to_compute(linear_builder, v01);
            Value v10f = cast_to_compute(linear_builder, v10);
            Value v11f = cast_to_compute(linear_builder, v11);

            Value dh = linear_builder.create<arith::SubFOp>(loc, fh, h0f);
            Value dw = linear_builder.create<arith::SubFOp>(loc, fw, w0f);
            Value one_minus_dw = linear_builder.create<arith::SubFOp>(loc, c_one, dw);
            Value one_minus_dh = linear_builder.create<arith::SubFOp>(loc, c_one, dh);
            Value v0 = linear_builder.create<arith::AddFOp>(
                loc,
                linear_builder.create<arith::MulFOp>(loc, v00f, one_minus_dw),
                linear_builder.create<arith::MulFOp>(loc, v01f, dw));
            Value v1 = linear_builder.create<arith::AddFOp>(
                loc,
                linear_builder.create<arith::MulFOp>(loc, v10f, one_minus_dw),
                linear_builder.create<arith::MulFOp>(loc, v11f, dw));
            Value vf = linear_builder.create<arith::AddFOp>(
                loc,
                linear_builder.create<arith::MulFOp>(loc, v0, one_minus_dh),
                linear_builder.create<arith::MulFOp>(loc, v1, dh));
            linear_builder.create<scf::YieldOp>(loc, ValueRange{cast_to_output(linear_builder, vf)});
        }
        value = sample_if.getResult(0);

        Value out_idx = make_output_flat_index(then_builder, n, c, h, w);
        then_builder.create<memref::StoreOp>(loc, value, fn.getArgument(2), ValueRange{out_idx});
    }

    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_transpose_module(mlir::MLIRContext& ctx,
                                                   const ov::element::Type& et) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect>();

    auto tr = ov::as_type_ptr<const ov::op::v1::Transpose>(m_node);
    OPENVINO_ASSERT(tr, "GFX Vulkan transpose: node cast failed");
    OPENVINO_ASSERT(tr->get_input_partial_shape(0).is_static() && tr->get_output_partial_shape(0).is_static(),
                    "GFX Vulkan transpose: static shapes required");
    auto perm_const = ov::as_type_ptr<const ov::op::v0::Constant>(tr->input_value(1).get_node_shared_ptr());
    OPENVINO_ASSERT(perm_const, "GFX Vulkan transpose: perm must be constant");

    Type elem_ty;
    switch (et) {
        case ov::element::f16: elem_ty = Float16Type::get(&ctx); break;
        case ov::element::f32: elem_ty = Float32Type::get(&ctx); break;
        default: OPENVINO_THROW("GFX Vulkan transpose: unsupported element type ", et);
    }

    const ov::Shape in_shape = tr->get_input_shape(0);
    const ov::Shape out_shape = tr->get_output_shape(0);
    const auto perm = perm_const->cast_vector<int64_t>();
    const size_t rank = perm.size();
    OPENVINO_ASSERT(rank == in_shape.size() && rank == out_shape.size(),
                    "GFX Vulkan transpose: rank mismatch");

    std::vector<int64_t> in_strides(rank, 1);
    std::vector<int64_t> out_strides(rank, 1);
    for (int64_t i = static_cast<int64_t>(rank) - 2; i >= 0; --i) {
        in_strides[static_cast<size_t>(i)] =
            in_strides[static_cast<size_t>(i) + 1] * static_cast<int64_t>(in_shape[static_cast<size_t>(i) + 1]);
        out_strides[static_cast<size_t>(i)] =
            out_strides[static_cast<size_t>(i) + 1] * static_cast<int64_t>(out_shape[static_cast<size_t>(i) + 1]);
    }
    const int64_t total_elems = static_cast<int64_t>(tensor_elements(out_shape));

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 2));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1}));

    auto flat_ty = MemRefType::get({total_elems}, elem_ty);
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "transpose_direct",
                                                 b.getFunctionType(TypeRange{flat_ty, flat_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto total = body.create<arith::ConstantIndexOp>(loc, total_elems);
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value out_idx = body.create<arith::AddIOp>(loc,
                                               body.create<arith::MulIOp>(loc, bid, bdim),
                                               tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, out_idx, total);
    auto active_if = body.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value remaining = out_idx;
        Value in_idx = then_builder.create<arith::ConstantIndexOp>(loc, 0);
        for (size_t d = 0; d < rank; ++d) {
            auto out_stride = then_builder.create<arith::ConstantIndexOp>(loc, out_strides[d]);
            Value coord = remaining;
            if (out_strides[d] != 1) {
                coord = then_builder.create<arith::DivUIOp>(loc, remaining, out_stride);
                remaining = then_builder.create<arith::RemUIOp>(loc, remaining, out_stride);
            } else if (d + 1 != rank) {
                remaining = then_builder.create<arith::ConstantIndexOp>(loc, 0);
            }
            auto in_stride = then_builder.create<arith::ConstantIndexOp>(loc, in_strides[static_cast<size_t>(perm[d])]);
            auto contrib = then_builder.create<arith::MulIOp>(loc, coord, in_stride);
            in_idx = then_builder.create<arith::AddIOp>(loc, in_idx, contrib);
        }
        Value val = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{in_idx});
        then_builder.create<memref::StoreOp>(loc, val, fn.getArgument(1), ValueRange{out_idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_convert_linear_module(mlir::MLIRContext& ctx,
                                                        const ov::element::Type& src_et,
                                                        const ov::element::Type& dst_et) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect>();

    Type src_ty = ov::gfx_plugin::to_mlir_type(src_et,
                                               ctx,
                                               /*fallback_f32=*/false,
                                               /*allow_unsigned=*/true,
                                               /*allow_small_ints=*/true,
                                               /*allow_bf16=*/false,
                                               /*allow_boolean=*/true,
                                               /*signless_integers=*/true);
    Type dst_ty = ov::gfx_plugin::to_mlir_type(dst_et,
                                               ctx,
                                               /*fallback_f32=*/false,
                                               /*allow_unsigned=*/true,
                                               /*allow_small_ints=*/true,
                                               /*allow_bf16=*/false,
                                               /*allow_boolean=*/true,
                                               /*signless_integers=*/true);

    auto in_ty = MemRefType::get({ShapedType::kDynamic}, src_ty);
    auto param_ty = MemRefType::get({2}, IntegerType::get(&ctx, 32));
    auto out_ty = MemRefType::get({ShapedType::kDynamic}, dst_ty);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 3));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2}));

    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "convert_linear",
                                                 b.getFunctionType(TypeRange{in_ty, param_ty, out_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto load_param = [&](int idx) -> Value {
        auto idx_val = body.create<arith::ConstantIndexOp>(loc, idx);
        auto v_i32 = body.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{idx_val});
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), v_i32);
    };
    Value offset = load_param(0);
    Value count = load_param(1);
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value local_idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local_idx, count);
    auto active_if = body.create<scf::IfOp>(loc, active, false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value idx = then_builder.create<arith::AddIOp>(loc, offset, local_idx);
        Value src = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{idx});
        Value dst = src;
        if (src_ty != dst_ty) {
            auto src_int = mlir::dyn_cast<IntegerType>(src_ty);
            auto dst_int = mlir::dyn_cast<IntegerType>(dst_ty);
            auto src_float = mlir::dyn_cast<FloatType>(src_ty);
            auto dst_float = mlir::dyn_cast<FloatType>(dst_ty);
            if (src_int && dst_int) {
                if (dst_int.getWidth() > src_int.getWidth()) {
                    dst = is_unsigned_convert_elem_type(src_et)
                              ? then_builder.create<arith::ExtUIOp>(loc, dst_ty, src).getResult()
                              : then_builder.create<arith::ExtSIOp>(loc, dst_ty, src).getResult();
                } else {
                    dst = then_builder.create<arith::TruncIOp>(loc, dst_ty, src);
                }
            } else if (src_float && dst_float) {
                if (dst_float.getWidth() > src_float.getWidth()) {
                    dst = then_builder.create<arith::ExtFOp>(loc, dst_ty, src);
                } else {
                    dst = then_builder.create<arith::TruncFOp>(loc, dst_ty, src);
                }
            } else if (src_int && dst_float) {
                dst = is_unsigned_convert_elem_type(src_et)
                          ? then_builder.create<arith::UIToFPOp>(loc, dst_ty, src).getResult()
                          : then_builder.create<arith::SIToFPOp>(loc, dst_ty, src).getResult();
            } else if (src_float && dst_int) {
                dst = is_unsigned_convert_elem_type(dst_et)
                          ? then_builder.create<arith::FPToUIOp>(loc, dst_ty, src).getResult()
                          : then_builder.create<arith::FPToSIOp>(loc, dst_ty, src).getResult();
            } else {
                OPENVINO_THROW("GFX Vulkan convert: unsupported conversion ", src_et, " -> ", dst_et);
            }
        }
        then_builder.create<memref::StoreOp>(loc, dst, fn.getArgument(2), ValueRange{idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_gather_linear_module(mlir::MLIRContext& ctx,
                                                       const ov::element::Type& data_et,
                                                       const ov::element::Type& idx_et) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect>();

    Type data_ty = to_mlir_type(data_et, ctx, /*fallback_f32=*/true, /*allow_unsigned=*/true,
                                /*allow_small_ints=*/true, /*allow_bf16=*/false,
                                /*allow_boolean=*/true, /*signless_integers=*/true);
    Type idx_ty = to_mlir_type(idx_et, ctx, /*fallback_f32=*/true, /*allow_unsigned=*/true,
                               /*allow_small_ints=*/true, /*allow_bf16=*/false,
                               /*allow_boolean=*/false, /*signless_integers=*/true);
    auto data_memref_ty = MemRefType::get({ShapedType::kDynamic}, data_ty);
    auto idx_memref_ty = MemRefType::get({ShapedType::kDynamic}, idx_ty);
    auto params_ty = MemRefType::get({6}, IntegerType::get(&ctx, 32));
    auto out_memref_ty = MemRefType::get({ShapedType::kDynamic}, data_ty);
    auto i64_ty = IntegerType::get(&ctx, 64);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 4));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2, 3}));

    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "gather_linear",
                                                 b.getFunctionType(TypeRange{data_memref_ty, idx_memref_ty, params_ty, out_memref_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto zero_i64 = body.create<arith::ConstantIntOp>(loc, 0, 64);
    auto one_i64 = body.create<arith::ConstantIntOp>(loc, 1, 64);
    auto load_param = [&](int idx) -> Value {
        auto idx_val = body.create<arith::ConstantIndexOp>(loc, idx);
        auto v_i32 = body.create<memref::LoadOp>(loc, fn.getArgument(2), ValueRange{idx_val});
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), v_i32);
    };
    Value outer = load_param(0);
    Value inner = load_param(1);
    Value axis_dim = load_param(2);
    Value indices_count = load_param(3);
    Value offset = load_param(4);
    Value count = load_param(5);
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value local_idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local_idx, count);
    auto active_if = body.create<scf::IfOp>(loc, active, false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value out_idx = then_builder.create<arith::AddIOp>(loc, offset, local_idx);
        Value inner_idx = then_builder.create<arith::RemUIOp>(loc, out_idx, inner);
        Value rem = then_builder.create<arith::DivUIOp>(loc, out_idx, inner);
        Value gather_idx_pos = then_builder.create<arith::RemUIOp>(loc, rem, indices_count);
        Value outer_idx = then_builder.create<arith::DivUIOp>(loc, rem, indices_count);
        Value gather_raw = then_builder.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{gather_idx_pos});
        Value gather_i64 = gather_raw;
        if (idx_ty.isInteger(32)) {
            gather_i64 = then_builder.create<arith::ExtSIOp>(loc, i64_ty, gather_raw).getResult();
        } else if (!idx_ty.isInteger(64)) {
            OPENVINO_THROW("GFX Vulkan Gather: only i32/i64 indices are supported");
        }
        Value axis_dim_i64 = then_builder.create<arith::IndexCastOp>(loc, i64_ty, axis_dim);
        Value max_i64 = then_builder.create<arith::SubIOp>(loc, axis_dim_i64, one_i64);
        Value neg_pred = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gather_i64, zero_i64);
        Value gather_plus = then_builder.create<arith::AddIOp>(loc, gather_i64, axis_dim_i64);
        Value gather_fixed = then_builder.create<arith::SelectOp>(loc, neg_pred, gather_plus, gather_i64);
        Value lt0 = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, gather_fixed, zero_i64);
        Value gtmax = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, gather_fixed, max_i64);
        Value gather_clamped = then_builder.create<arith::SelectOp>(loc, lt0, zero_i64, gather_fixed);
        gather_clamped = then_builder.create<arith::SelectOp>(loc, gtmax, max_i64, gather_clamped);
        Value gather_idx = then_builder.create<arith::IndexCastOp>(loc, then_builder.getIndexType(), gather_clamped);
        Value data_idx = then_builder.create<arith::AddIOp>(
            loc,
            then_builder.create<arith::MulIOp>(
                loc,
                then_builder.create<arith::AddIOp>(
                    loc,
                    then_builder.create<arith::MulIOp>(loc, outer_idx, axis_dim),
                    gather_idx),
                inner),
            inner_idx);
        Value value = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{data_idx});
        then_builder.create<memref::StoreOp>(loc, value, fn.getArgument(3), ValueRange{out_idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_gather_embedding_module(mlir::MLIRContext& ctx,
                                                          const ov::element::Type& data_et,
                                                          const ov::element::Type& idx_et,
                                                          size_t vocab,
                                                          size_t hidden) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect>();

    Type data_ty = data_et == ov::element::f16
                       ? Type(Float16Type::get(&ctx))
                       : Type(Float32Type::get(&ctx));
    Type idx_ty = idx_et == ov::element::i64 ? IntegerType::get(&ctx, 64) : IntegerType::get(&ctx, 32);
    auto weights_ty = MemRefType::get({static_cast<int64_t>(vocab * hidden)}, data_ty);
    auto indices_ty = MemRefType::get({ShapedType::kDynamic}, idx_ty);
    auto params_ty = MemRefType::get({5}, IntegerType::get(&ctx, 32));
    auto output_ty = MemRefType::get({ShapedType::kDynamic}, data_ty);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 4));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2, 3}));

    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "gather_embedding",
                                                 b.getFunctionType(TypeRange{weights_ty, indices_ty, params_ty, output_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto load_param = [&](int idx) -> Value {
        auto idx_val = body.create<arith::ConstantIndexOp>(loc, idx);
        auto v_i32 = body.create<memref::LoadOp>(loc, fn.getArgument(2), ValueRange{idx_val});
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), v_i32);
    };
    Value vocab_v = load_param(0);
    Value hidden_v = load_param(1);
    Value offset = load_param(3);
    Value count = load_param(4);
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value local_idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local_idx, count);
    auto active_if = body.create<scf::IfOp>(loc, active, false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value out_idx = then_builder.create<arith::AddIOp>(loc, offset, local_idx);
        Value token = then_builder.create<arith::DivUIOp>(loc, out_idx, hidden_v);
        Value channel = then_builder.create<arith::RemUIOp>(loc, out_idx, hidden_v);
        Value raw = then_builder.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{token});

        auto i64_ty = IntegerType::get(&ctx, 64);
        Value raw_i64 = raw;
        if (idx_et == ov::element::i32) {
            raw_i64 = then_builder.create<arith::ExtSIOp>(loc, i64_ty, raw).getResult();
        }
        Value vocab_i64 = then_builder.create<arith::IndexCastOp>(loc, i64_ty, vocab_v);
        Value zero_i64 = then_builder.create<arith::ConstantIntOp>(loc, 0, 64);
        Value one_i64 = then_builder.create<arith::ConstantIntOp>(loc, 1, 64);
        Value max_i64 = then_builder.create<arith::SubIOp>(loc, vocab_i64, one_i64);
        Value neg = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, raw_i64, zero_i64);
        Value wrapped = then_builder.create<arith::AddIOp>(loc, raw_i64, vocab_i64);
        Value normalized = then_builder.create<arith::SelectOp>(loc, neg, wrapped, raw_i64);
        Value lt0 = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, normalized, zero_i64);
        Value gtmax = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, normalized, max_i64);
        Value clamped = then_builder.create<arith::SelectOp>(loc, lt0, zero_i64, normalized);
        clamped = then_builder.create<arith::SelectOp>(loc, gtmax, max_i64, clamped);
        Value gather_idx = then_builder.create<arith::IndexCastOp>(loc, then_builder.getIndexType(), clamped);
        Value weight_offset =
            then_builder.create<arith::AddIOp>(loc,
                                               then_builder.create<arith::MulIOp>(loc, gather_idx, hidden_v),
                                               channel);
        Value value = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{weight_offset});
        then_builder.create<memref::StoreOp>(loc, value, fn.getArgument(3), ValueRange{out_idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_reduce_last_axis_module(mlir::MLIRContext& ctx,
                                                          const ov::element::Type& et,
                                                          const std::string& op_key) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect>();

    Type elem_ty;
    if (et == ov::element::f16) {
        elem_ty = Float16Type::get(&ctx);
    } else if (et == ov::element::f32) {
        elem_ty = Float32Type::get(&ctx);
    } else {
        OPENVINO_THROW("GFX Vulkan reduce last-axis: unsupported element type ", et);
    }
    Type compute_ty = Float32Type::get(&ctx);
    auto input_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto params_ty = MemRefType::get({4}, IntegerType::get(&ctx, 32));
    auto output_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 3));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2}));

    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "reduce_last_axis",
                                                 b.getFunctionType(TypeRange{input_ty, params_ty, output_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto load_param = [&](int idx) -> Value {
        auto idx_val = body.create<arith::ConstantIndexOp>(loc, idx);
        auto v_i32 = body.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{idx_val});
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), v_i32);
    };
    auto cast_to_compute = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](OpBuilder& builder, Value value) -> Value {
        if (elem_ty == compute_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, elem_ty, value);
    };

    Value reduce = load_param(1);
    Value offset = load_param(2);
    Value count = load_param(3);
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value local_idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local_idx, count);
    auto active_if = body.create<scf::IfOp>(loc, active, false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value out_idx = then_builder.create<arith::AddIOp>(loc, offset, local_idx);
        Value base = then_builder.create<arith::MulIOp>(loc, out_idx, reduce);
        Value c0 = then_builder.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = then_builder.create<arith::ConstantIndexOp>(loc, 1);
        Value zero = then_builder.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 0.0f));
        auto loop = then_builder.create<scf::ForOp>(loc, c0, reduce, c1, ValueRange{zero});
        {
            auto loop_builder = OpBuilder::atBlockBegin(loop.getBody());
            Value in_idx = loop_builder.create<arith::AddIOp>(loc, base, loop.getInductionVar());
            Value val = loop_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{in_idx});
            Value sum = loop_builder.create<arith::AddFOp>(loc, loop.getRegionIterArgs()[0], cast_to_compute(loop_builder, val));
            if (!loop.getBody()->empty() && loop.getBody()->back().hasTrait<OpTrait::IsTerminator>()) {
                loop.getBody()->back().erase();
            }
            auto loop_end = OpBuilder::atBlockEnd(loop.getBody());
            loop_end.create<scf::YieldOp>(loc, ValueRange{sum});
        }
        Value result = loop.getResult(0);
        if (op_key == "mean") {
            Value reduce_f = then_builder.create<arith::IndexCastOp>(loc, IntegerType::get(&ctx, 32), reduce);
            reduce_f = then_builder.create<arith::SIToFPOp>(loc, compute_ty, reduce_f);
            result = then_builder.create<arith::DivFOp>(loc, result, reduce_f);
        } else if (op_key != "sum") {
            OPENVINO_THROW("GFX Vulkan reduce last-axis: unsupported op ", op_key);
        }
        then_builder.create<memref::StoreOp>(loc, cast_to_output(then_builder, result), fn.getArgument(2), ValueRange{out_idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_rms_module(mlir::MLIRContext& ctx,
                                             const ov::element::Type& input_et,
                                             const ov::element::Type& gamma_et,
                                             const ov::element::Type& output_et,
                                             size_t hidden,
                                             size_t gamma_size,
                                             uint32_t reduction_threads,
                                             float epsilon) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    math::MathDialect>();

    auto to_float_ty = [&](const ov::element::Type& et, const char* what) -> Type {
        if (et == ov::element::f16) {
            return Float16Type::get(&ctx);
        }
        if (et == ov::element::f32) {
            return Float32Type::get(&ctx);
        }
        OPENVINO_THROW("GFX Vulkan RMS: unsupported ", what, " element type ", et);
    };
    Type input_ty = to_float_ty(input_et, "input");
    Type gamma_ty = to_float_ty(gamma_et, "gamma");
    Type output_ty = to_float_ty(output_et, "output");
    Type compute_ty = Float32Type::get(&ctx);
    const uint32_t threads = reduction_threads > 1 ? reduction_threads : 1u;
    OPENVINO_ASSERT(hidden > 0, "GFX Vulkan RMS: hidden dimension must be positive");
    OPENVINO_ASSERT(gamma_size == 1 || gamma_size == hidden,
                    "GFX Vulkan RMS: unsupported gamma size ",
                    gamma_size,
                    " for hidden ",
                    hidden);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 3));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2}));

    auto input_mem_ty = MemRefType::get({ShapedType::kDynamic}, input_ty);
    auto gamma_mem_ty = MemRefType::get({ShapedType::kDynamic}, gamma_ty);
    auto output_mem_ty = MemRefType::get({ShapedType::kDynamic}, output_ty);
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "rms_linear",
                                                 b.getFunctionType(TypeRange{input_mem_ty, gamma_mem_ty, output_mem_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {static_cast<int32_t>(threads), 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto c0 = body.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = body.create<arith::ConstantIndexOp>(loc, 1);
    auto hidden_v = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(hidden));
    auto threads_v = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(threads));
    auto eps_v = body.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, epsilon));
    auto hidden_f_i32 =
        body.create<arith::ConstantOp>(loc,
                                       IntegerAttr::get(IntegerType::get(&ctx, 32), static_cast<int64_t>(hidden)));
    auto hidden_f = body.create<arith::SIToFPOp>(loc, compute_ty, hidden_f_i32);

    auto cast_to_compute = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == output_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, output_ty, value);
    };
    auto emit_scale = [&](OpBuilder& builder, Value linear_idx, Value col, Value inv) {
        Value x = builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{linear_idx});
        Value gamma_idx = gamma_size == 1 ? c0.getResult() : col;
        Value gamma = builder.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{gamma_idx});
        Value scaled = builder.create<arith::MulFOp>(
            loc,
            builder.create<arith::MulFOp>(loc, cast_to_compute(builder, x), inv),
            cast_to_compute(builder, gamma));
        builder.create<memref::StoreOp>(loc, cast_to_output(builder, scaled), fn.getArgument(2), ValueRange{linear_idx});
    };

    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value global_idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    if (threads == 1) {
        Value row = body.create<arith::DivUIOp>(loc, global_idx, hidden_v);
        Value col = body.create<arith::RemUIOp>(loc, global_idx, hidden_v);
        Value base = body.create<arith::MulIOp>(loc, row, hidden_v);
        Value zero = body.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 0.0f));
        auto loop = body.create<scf::ForOp>(loc, c0, hidden_v, c1, ValueRange{zero});
        {
            auto loop_builder = OpBuilder::atBlockBegin(loop.getBody());
            Value in_idx = loop_builder.create<arith::AddIOp>(loc, base, loop.getInductionVar());
            Value x = loop_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{in_idx});
            Value xf = cast_to_compute(loop_builder, x);
            Value sq = loop_builder.create<arith::MulFOp>(loc, xf, xf);
            Value sum = loop_builder.create<arith::AddFOp>(loc, loop.getRegionIterArgs()[0], sq);
            if (!loop.getBody()->empty() && loop.getBody()->back().hasTrait<OpTrait::IsTerminator>()) {
                loop.getBody()->back().erase();
            }
            auto loop_end = OpBuilder::atBlockEnd(loop.getBody());
            loop_end.create<scf::YieldOp>(loc, ValueRange{sum});
        }
        Value mean = body.create<arith::DivFOp>(loc, loop.getResult(0), hidden_f);
        Value inv = body.create<math::RsqrtOp>(loc, body.create<arith::AddFOp>(loc, mean, eps_v));
        emit_scale(body, global_idx, col, inv);
    } else {
        auto partial_ty = MemRefType::get({static_cast<int64_t>(threads)},
                                          compute_ty,
                                          MemRefLayoutAttrInterface{},
                                          b.getI64IntegerAttr(3));
        Value partial = body.create<memref::AllocOp>(loc, partial_ty);
        Value lane = tid;
        Value row = body.create<arith::DivUIOp>(loc, global_idx, threads_v);
        Value base = body.create<arith::MulIOp>(loc, row, hidden_v);
        Value zero = body.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 0.0f));
        auto loop = body.create<scf::ForOp>(loc, lane, hidden_v, threads_v, ValueRange{zero});
        {
            auto loop_builder = OpBuilder::atBlockBegin(loop.getBody());
            Value in_idx = loop_builder.create<arith::AddIOp>(loc, base, loop.getInductionVar());
            Value x = loop_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{in_idx});
            Value xf = cast_to_compute(loop_builder, x);
            Value sq = loop_builder.create<arith::MulFOp>(loc, xf, xf);
            Value sum = loop_builder.create<arith::AddFOp>(loc, loop.getRegionIterArgs()[0], sq);
            if (!loop.getBody()->empty() && loop.getBody()->back().hasTrait<OpTrait::IsTerminator>()) {
                loop.getBody()->back().erase();
            }
            auto loop_end = OpBuilder::atBlockEnd(loop.getBody());
            loop_end.create<scf::YieldOp>(loc, ValueRange{sum});
        }
        body.create<memref::StoreOp>(loc, loop.getResult(0), partial, ValueRange{lane});
        body.create<gpu::BarrierOp>(loc);
        for (uint32_t stride = threads / 2; stride > 0; stride >>= 1) {
            auto stride_v = body.create<arith::ConstantIndexOp>(loc, stride);
            auto lane_active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, lane, stride_v);
            auto reduce_if = body.create<scf::IfOp>(loc, lane_active, false);
            {
                auto then_builder = reduce_if.getThenBodyBuilder();
                Value other_lane = then_builder.create<arith::AddIOp>(loc, lane, stride_v);
                Value lhs = then_builder.create<memref::LoadOp>(loc, partial, ValueRange{lane});
                Value rhs = then_builder.create<memref::LoadOp>(loc, partial, ValueRange{other_lane});
                Value sum = then_builder.create<arith::AddFOp>(loc, lhs, rhs);
                then_builder.create<memref::StoreOp>(loc, sum, partial, ValueRange{lane});
            }
            body.setInsertionPointAfter(reduce_if);
            body.create<gpu::BarrierOp>(loc);
        }
        Value total_sum = body.create<memref::LoadOp>(loc, partial, ValueRange{c0});
        Value mean = body.create<arith::DivFOp>(loc, total_sum, hidden_f);
        Value inv = body.create<math::RsqrtOp>(loc, body.create<arith::AddFOp>(loc, mean, eps_v));
        auto out_loop = body.create<scf::ForOp>(loc, lane, hidden_v, threads_v);
        {
            auto loop_builder = OpBuilder::atBlockBegin(out_loop.getBody());
            Value col = out_loop.getInductionVar();
            Value out_idx = loop_builder.create<arith::AddIOp>(loc, base, col);
            emit_scale(loop_builder, out_idx, col, inv);
        }
    }
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_matmul_linear_module(mlir::MLIRContext& ctx,
                                                       const ov::element::Type& et,
                                                       bool transpose_a,
                                                       bool transpose_b) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect>();

    Type elem_ty;
    if (et == ov::element::f16) {
        elem_ty = Float16Type::get(&ctx);
    } else if (et == ov::element::f32) {
        elem_ty = Float32Type::get(&ctx);
    } else {
        OPENVINO_THROW("GFX Vulkan MatMul: unsupported element type ", et);
    }
    Type compute_ty = Float32Type::get(&ctx);
    auto input_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto weight_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto params_ty = MemRefType::get({8}, IntegerType::get(&ctx, 32));
    auto output_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 4));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        attrs.reserve(values.size());
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2, 3}));

    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "matmul_linear",
                                                 b.getFunctionType(TypeRange{input_ty, weight_ty, params_ty, output_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto load_param = [&](int idx) -> Value {
        auto idx_val = body.create<arith::ConstantIndexOp>(loc, idx);
        auto v_i32 = body.create<memref::LoadOp>(loc, fn.getArgument(2), ValueRange{idx_val});
        return body.create<arith::IndexCastOp>(loc, body.getIndexType(), v_i32);
    };
    auto cast_to_compute = [&](OpBuilder& builder, Value value) -> Value {
        if (value.getType() == compute_ty) {
            return value;
        }
        return builder.create<arith::ExtFOp>(loc, compute_ty, value);
    };
    auto cast_to_output = [&](OpBuilder& builder, Value value) -> Value {
        if (elem_ty == compute_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, elem_ty, value);
    };

    Value batch = load_param(0);
    Value m_dim = load_param(1);
    Value n_dim = load_param(2);
    Value k_dim = load_param(3);
    Value weight_batch = load_param(4);
    Value offset = load_param(5);
    Value count = load_param(6);
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value local_idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local_idx, count);
    auto active_if = body.create<scf::IfOp>(loc, active, false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value out_idx = then_builder.create<arith::AddIOp>(loc, offset, local_idx);
        Value n = then_builder.create<arith::RemUIOp>(loc, out_idx, n_dim);
        Value rem = then_builder.create<arith::DivUIOp>(loc, out_idx, n_dim);
        Value m = then_builder.create<arith::RemUIOp>(loc, rem, m_dim);
        Value b_idx = then_builder.create<arith::DivUIOp>(loc, rem, m_dim);
        Value c0 = then_builder.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = then_builder.create<arith::ConstantIndexOp>(loc, 1);
        Value one_batch = then_builder.create<arith::ConstantIndexOp>(loc, 1);
        Value wb_is_one = then_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, weight_batch, one_batch);
        Value wb = then_builder.create<arith::SelectOp>(loc, wb_is_one, c0, b_idx);
        Value zero = then_builder.create<arith::ConstantOp>(loc, FloatAttr::get(compute_ty, 0.0f));
        auto loop = then_builder.create<scf::ForOp>(loc, c0, k_dim, c1, ValueRange{zero});
        {
            auto loop_builder = OpBuilder::atBlockBegin(loop.getBody());
            Value k = loop.getInductionVar();
            Value a_off;
            if (transpose_a) {
                a_off = loop_builder.create<arith::AddIOp>(
                    loc,
                    loop_builder.create<arith::MulIOp>(loc,
                                                       loop_builder.create<arith::AddIOp>(
                                                           loc,
                                                           loop_builder.create<arith::MulIOp>(loc, b_idx, k_dim),
                                                           k),
                                                       m_dim),
                    m);
            } else {
                a_off = loop_builder.create<arith::AddIOp>(
                    loc,
                    loop_builder.create<arith::MulIOp>(loc,
                                                       loop_builder.create<arith::AddIOp>(
                                                           loc,
                                                           loop_builder.create<arith::MulIOp>(loc, b_idx, m_dim),
                                                           m),
                                                       k_dim),
                    k);
            }
            Value b_off;
            if (transpose_b) {
                b_off = loop_builder.create<arith::AddIOp>(
                    loc,
                    loop_builder.create<arith::MulIOp>(loc,
                                                       loop_builder.create<arith::AddIOp>(
                                                           loc,
                                                           loop_builder.create<arith::MulIOp>(loc, wb, n_dim),
                                                           n),
                                                       k_dim),
                    k);
            } else {
                b_off = loop_builder.create<arith::AddIOp>(
                    loc,
                    loop_builder.create<arith::MulIOp>(loc,
                                                       loop_builder.create<arith::AddIOp>(
                                                           loc,
                                                           loop_builder.create<arith::MulIOp>(loc, wb, k_dim),
                                                           k),
                                                       n_dim),
                    n);
            }
            Value a_val = loop_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{a_off});
            Value b_val = loop_builder.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{b_off});
            Value prod = loop_builder.create<arith::MulFOp>(loc, cast_to_compute(loop_builder, a_val), cast_to_compute(loop_builder, b_val));
            Value sum = loop_builder.create<arith::AddFOp>(loc, loop.getRegionIterArgs()[0], prod);
            if (!loop.getBody()->empty() && loop.getBody()->back().hasTrait<OpTrait::IsTerminator>()) {
                loop.getBody()->back().erase();
            }
            auto loop_end = OpBuilder::atBlockEnd(loop.getBody());
            loop_end.create<scf::YieldOp>(loc, ValueRange{sum});
        }
        then_builder.create<memref::StoreOp>(loc, cast_to_output(then_builder, loop.getResult(0)), fn.getArgument(3), ValueRange{out_idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    (void)batch;
    return mod;
}

mlir::ModuleOp VulkanStage::build_broadcast_module(mlir::MLIRContext& ctx,
                                                   const ov::element::Type& et,
                                                   size_t rank) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect>();
    auto elem_ty = to_mlir_type(et, ctx, /*fallback_f32=*/false,
                                /*allow_unsigned=*/true,
                                /*allow_small_ints=*/false,
                                /*allow_bf16=*/false,
                                /*allow_boolean=*/true,
                                /*signless_integers=*/true);
    auto input_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto params_ty = MemRefType::get({ShapedType::kDynamic}, IntegerType::get(&ctx, 32));
    auto output_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 3));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2}));
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "broadcast_linear",
                                                 b.getFunctionType(TypeRange{input_ty, params_ty, output_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));
    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto load_param_at = [&](OpBuilder& builder, Value idx) -> Value {
        auto v_i32 = builder.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{idx});
        return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), v_i32);
    };
    auto load_param = [&](int idx) -> Value {
        return load_param_at(body, body.create<arith::ConstantIndexOp>(loc, idx));
    };
    Value offset = load_param(0);
    Value count = load_param(1);
    Value rank_v = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(rank));
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value local_idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local_idx, count);
    auto active_if = body.create<scf::IfOp>(loc, active, false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value out_idx = then_builder.create<arith::AddIOp>(loc, offset, local_idx);
        Value c0 = then_builder.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = then_builder.create<arith::ConstantIndexOp>(loc, 1);
        auto loop = then_builder.create<scf::ForOp>(loc, c0, rank_v, c1, ValueRange{out_idx, c0});
        auto loop_builder = OpBuilder::atBlockBegin(loop.getBody());
        Value rev = loop_builder.create<arith::SubIOp>(loc, rank_v, c1);
        rev = loop_builder.create<arith::SubIOp>(loc, rev, loop.getInductionVar());
        Value dim_base = loop_builder.create<arith::ConstantIndexOp>(loc, 2);
        Value stride_base = loop_builder.create<arith::ConstantIndexOp>(loc, 2 + static_cast<int64_t>(rank));
        Value dim = load_param_at(loop_builder, loop_builder.create<arith::AddIOp>(loc, dim_base, rev));
        Value stride = load_param_at(loop_builder, loop_builder.create<arith::AddIOp>(loc, stride_base, rev));
        Value coord = loop_builder.create<arith::RemUIOp>(loc, loop.getRegionIterArgs()[0], dim);
        Value next = loop_builder.create<arith::DivUIOp>(loc, loop.getRegionIterArgs()[0], dim);
        Value in_off = loop_builder.create<arith::AddIOp>(
            loc,
            loop.getRegionIterArgs()[1],
            loop_builder.create<arith::MulIOp>(loc, coord, stride));
        if (!loop.getBody()->empty() && loop.getBody()->back().hasTrait<OpTrait::IsTerminator>()) {
            loop.getBody()->back().erase();
        }
        auto loop_end = OpBuilder::atBlockEnd(loop.getBody());
        loop_end.create<scf::YieldOp>(loc, ValueRange{next, in_off});
        Value value = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{loop.getResult(1)});
        then_builder.create<memref::StoreOp>(loc, value, fn.getArgument(2), ValueRange{out_idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_select_module(mlir::MLIRContext& ctx,
                                                const ov::element::Type& cond_et,
                                                const ov::element::Type& data_et,
                                                size_t rank) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
                    gpu::GPUDialect,
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect>();
    auto cond_ty = to_mlir_type(cond_et, ctx, /*fallback_f32=*/false,
                                /*allow_unsigned=*/false,
                                /*allow_small_ints=*/false,
                                /*allow_bf16=*/false,
                                /*allow_boolean=*/true,
                                /*signless_integers=*/true);
    auto data_ty = to_mlir_type(data_et, ctx, /*fallback_f32=*/false,
                                /*allow_unsigned=*/true,
                                /*allow_small_ints=*/false,
                                /*allow_bf16=*/false,
                                /*allow_boolean=*/true,
                                /*signless_integers=*/true);
    auto cond_memref_ty = MemRefType::get({ShapedType::kDynamic}, cond_ty);
    auto data_memref_ty = MemRefType::get({ShapedType::kDynamic}, data_ty);
    auto params_ty = MemRefType::get({ShapedType::kDynamic}, IntegerType::get(&ctx, 32));
    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 5));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1, 1, 1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1, 2, 3, 4}));
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "select_linear",
                                                 b.getFunctionType(TypeRange{cond_memref_ty,
                                                                             data_memref_ty,
                                                                             data_memref_ty,
                                                                             params_ty,
                                                                             data_memref_ty},
                                                                   {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));
    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    auto load_param_at = [&](OpBuilder& builder, Value idx) -> Value {
        auto v_i32 = builder.create<memref::LoadOp>(loc, fn.getArgument(3), ValueRange{idx});
        return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), v_i32);
    };
    auto load_param = [&](int idx) -> Value {
        return load_param_at(body, body.create<arith::ConstantIndexOp>(loc, idx));
    };
    Value offset = load_param(0);
    Value count = load_param(1);
    Value rank_v = body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(rank));
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value local_idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local_idx, count);
    auto active_if = body.create<scf::IfOp>(loc, active, false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value out_idx = then_builder.create<arith::AddIOp>(loc, offset, local_idx);
        Value c0 = then_builder.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = then_builder.create<arith::ConstantIndexOp>(loc, 1);
        auto loop = then_builder.create<scf::ForOp>(loc, c0, rank_v, c1, ValueRange{out_idx, c0, c0, c0});
        auto loop_builder = OpBuilder::atBlockBegin(loop.getBody());
        Value rev = loop_builder.create<arith::SubIOp>(loc, rank_v, c1);
        rev = loop_builder.create<arith::SubIOp>(loc, rev, loop.getInductionVar());
        Value dim_base = loop_builder.create<arith::ConstantIndexOp>(loc, 2);
        Value cond_stride_base = loop_builder.create<arith::ConstantIndexOp>(loc, 2 + static_cast<int64_t>(rank));
        Value true_stride_base = loop_builder.create<arith::ConstantIndexOp>(loc, 2 + static_cast<int64_t>(2 * rank));
        Value false_stride_base = loop_builder.create<arith::ConstantIndexOp>(loc, 2 + static_cast<int64_t>(3 * rank));
        Value dim = load_param_at(loop_builder, loop_builder.create<arith::AddIOp>(loc, dim_base, rev));
        Value cond_stride = load_param_at(loop_builder, loop_builder.create<arith::AddIOp>(loc, cond_stride_base, rev));
        Value true_stride = load_param_at(loop_builder, loop_builder.create<arith::AddIOp>(loc, true_stride_base, rev));
        Value false_stride = load_param_at(loop_builder, loop_builder.create<arith::AddIOp>(loc, false_stride_base, rev));
        Value coord = loop_builder.create<arith::RemUIOp>(loc, loop.getRegionIterArgs()[0], dim);
        Value next = loop_builder.create<arith::DivUIOp>(loc, loop.getRegionIterArgs()[0], dim);
        Value cond_off = loop_builder.create<arith::AddIOp>(
            loc,
            loop.getRegionIterArgs()[1],
            loop_builder.create<arith::MulIOp>(loc, coord, cond_stride));
        Value true_off = loop_builder.create<arith::AddIOp>(
            loc,
            loop.getRegionIterArgs()[2],
            loop_builder.create<arith::MulIOp>(loc, coord, true_stride));
        Value false_off = loop_builder.create<arith::AddIOp>(
            loc,
            loop.getRegionIterArgs()[3],
            loop_builder.create<arith::MulIOp>(loc, coord, false_stride));
        if (!loop.getBody()->empty() && loop.getBody()->back().hasTrait<OpTrait::IsTerminator>()) {
            loop.getBody()->back().erase();
        }
        auto loop_end = OpBuilder::atBlockEnd(loop.getBody());
        loop_end.create<scf::YieldOp>(loc, ValueRange{next, cond_off, true_off, false_off});
        Value cond = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{loop.getResult(1)});
        Value tval = then_builder.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{loop.getResult(2)});
        Value fval = then_builder.create<memref::LoadOp>(loc, fn.getArgument(2), ValueRange{loop.getResult(3)});
        Value selected = then_builder.create<arith::SelectOp>(loc, cond, tval, fval);
        then_builder.create<memref::StoreOp>(loc, selected, fn.getArgument(4), ValueRange{out_idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
}

void VulkanStage::execute_broadcast_chunked(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(!m_inputs.empty() && m_inputs[0] && m_inputs[0]->buf.valid(),
                    "GFX Vulkan Broadcast: missing input buffer");
    GpuTensor* input = m_inputs[0];
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(output && output->buf.valid(), "GFX Vulkan Broadcast: missing output buffer");
    ov::Shape in_shape = input->shape;
    if (in_shape.empty() && m_node && m_node->get_input_partial_shape(0).is_static()) {
        in_shape = m_node->get_input_shape(0);
    }
    ov::Shape out_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
    if (out_shape.empty() && m_node && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(!in_shape.empty() && !out_shape.empty(),
                    "GFX Vulkan Broadcast: runtime shapes are unknown");
    output->shape = out_shape;
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    output->expected_type = elem_type;
    const size_t rank = out_shape.size();
    if (!m_broadcast_kernel || m_broadcast_elem_type != elem_type || m_broadcast_rank != rank) {
        auto& ctx = gfx_mlir_context();
        auto module = build_broadcast_module(ctx, elem_type, rank);
        KernelSource src = make_kernel_source_from_mlir(module, "broadcast_linear", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_broadcast_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_broadcast_kernel, "GFX Vulkan Broadcast: kernel compile failed: ", log);
        m_broadcast_kernel->prepare_runtime_artifacts();
        m_broadcast_elem_type = elem_type;
        m_broadcast_rank = rank;
    }
    const auto strides = ov::gfx_plugin::compute_broadcast_element_strides(in_shape, out_shape);
    std::vector<int32_t> params(2 + 2 * rank, 0);
    const uint32_t total = static_cast<uint32_t>(tensor_elements(out_shape));
    params[0] = 0;
    params[1] = static_cast<int32_t>(total);
    for (size_t i = 0; i < rank; ++i) {
        params[2 + i] = static_cast<int32_t>(out_shape[i]);
        params[2 + rank + i] = strides[i];
    }
    const std::string key = m_name + "/broadcast_params/" + std::to_string(total);
    GpuBuffer params_buf = m_buffer_manager->wrap_const(key, params.data(), params.size() * sizeof(int32_t), ov::element::i32);
    OPENVINO_ASSERT(params_buf.valid(), "GFX Vulkan Broadcast: failed to wrap params");
    params_buf.owned = false;
    std::vector<KernelArg> args{
        make_buffer_arg(0, input->buf),
        make_buffer_arg(1, params_buf),
        make_buffer_arg(2, output->buf),
    };
    const uint32_t tg = m_broadcast_kernel->clamp_threadgroup_size(64);
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    m_broadcast_kernel->execute(command_buffer, dispatch, args, nullptr);
}

void VulkanStage::execute_select_chunked(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(m_inputs.size() >= 3, "GFX Vulkan Select: requires 3 inputs");
    GpuTensor* cond = m_inputs[0];
    GpuTensor* tval = m_inputs[1];
    GpuTensor* fval = m_inputs[2];
    OPENVINO_ASSERT(cond && cond->buf.valid(), "GFX Vulkan Select: missing cond buffer");
    OPENVINO_ASSERT(tval && tval->buf.valid(), "GFX Vulkan Select: missing true buffer");
    OPENVINO_ASSERT(fval && fval->buf.valid(), "GFX Vulkan Select: missing false buffer");
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(output && output->buf.valid(), "GFX Vulkan Select: missing output buffer");
    auto resolve_input_shape = [&](size_t input_idx, GpuTensor* tensor) {
        ov::Shape shape = tensor ? tensor->shape : ov::Shape{};
        if (shape.empty() && m_node && m_node->get_input_partial_shape(input_idx).is_static()) {
            shape = m_node->get_input_shape(input_idx);
        }
        return shape;
    };
    const ov::Shape cond_shape = resolve_input_shape(0, cond);
    const ov::Shape true_shape = resolve_input_shape(1, tval);
    const ov::Shape false_shape = resolve_input_shape(2, fval);
    ov::Shape out_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
    if (out_shape.empty() && m_node && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(!out_shape.empty(), "GFX Vulkan Select: runtime output shape is unknown");
    output->shape = out_shape;
    const ov::element::Type cond_type = resolve_stage_input_element_type(m_node, 0, cond);
    const ov::element::Type data_type = resolve_stage_element_type(m_node, output);
    output->expected_type = data_type;
    const size_t rank = out_shape.size();
    if (!m_select_kernel || m_select_cond_elem_type != cond_type ||
        m_select_data_elem_type != data_type || m_select_rank != rank) {
        auto& ctx = gfx_mlir_context();
        auto module = build_select_module(ctx, cond_type, data_type, rank);
        KernelSource src = make_kernel_source_from_mlir(module, "select_linear", /*arg_count=*/5);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_select_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_select_kernel, "GFX Vulkan Select: kernel compile failed: ", log);
        m_select_kernel->prepare_runtime_artifacts();
        m_select_cond_elem_type = cond_type;
        m_select_data_elem_type = data_type;
        m_select_rank = rank;
    }
    const auto cond_strides = ov::gfx_plugin::compute_broadcast_element_strides(cond_shape, out_shape);
    const auto true_strides = ov::gfx_plugin::compute_broadcast_element_strides(true_shape, out_shape);
    const auto false_strides = ov::gfx_plugin::compute_broadcast_element_strides(false_shape, out_shape);
    std::vector<int32_t> params(2 + 4 * rank, 0);
    const uint32_t total = static_cast<uint32_t>(tensor_elements(out_shape));
    params[0] = 0;
    params[1] = static_cast<int32_t>(total);
    for (size_t i = 0; i < rank; ++i) {
        params[2 + i] = static_cast<int32_t>(out_shape[i]);
        params[2 + rank + i] = cond_strides[i];
        params[2 + 2 * rank + i] = true_strides[i];
        params[2 + 3 * rank + i] = false_strides[i];
    }
    std::string key = m_name + "/select_params/" + std::to_string(total);
    for (size_t i = 0; i < rank; ++i) {
        key += "_" + std::to_string(out_shape[i]) +
               "_" + std::to_string(cond_strides[i]) +
               "_" + std::to_string(true_strides[i]) +
               "_" + std::to_string(false_strides[i]);
    }
    GpuBuffer params_buf =
        m_buffer_manager->wrap_const(key, params.data(), params.size() * sizeof(int32_t), ov::element::i32);
    OPENVINO_ASSERT(params_buf.valid(), "GFX Vulkan Select: failed to wrap params");
    params_buf.owned = false;
    std::vector<KernelArg> args{
        make_buffer_arg(0, cond->buf),
        make_buffer_arg(1, tval->buf),
        make_buffer_arg(2, fval->buf),
        make_buffer_arg(3, params_buf),
        make_buffer_arg(4, output->buf),
    };
    const uint32_t tg = m_select_kernel->clamp_threadgroup_size(64);
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    m_select_kernel->execute(command_buffer, dispatch, args, nullptr);
}

void VulkanStage::execute_matmul_linear(GpuCommandBufferHandle command_buffer) {
    auto resolve_input = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
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
    auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(m_node);
    OPENVINO_ASSERT(mm, "GFX Vulkan MatMul: node cast failed");
    GpuTensor* input = resolve_input(0);
    GpuTensor* weights = resolve_input(1);
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(input && input->buf.valid() && weights && weights->buf.valid() && output && output->buf.valid(),
                    "GFX Vulkan MatMul: missing buffers");

    ov::Shape a_shape = input->shape;
    ov::Shape b_shape = weights->shape;
    if (a_shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        a_shape = m_node->get_input_shape(0);
    }
    if (b_shape.empty() && m_node->get_input_partial_shape(1).is_static()) {
        b_shape = m_node->get_input_shape(1);
    }
    OPENVINO_ASSERT(a_shape.size() >= 2 && a_shape.size() <= 4 && b_shape.size() >= 2 && b_shape.size() <= 4,
                    "GFX Vulkan MatMul: unsupported runtime ranks");
    const auto flattened = flatten_matmul_shapes_with_batch_broadcast(a_shape, b_shape, "GFX Vulkan MatMul");
    const auto& a3 = flattened.lhs;
    const auto& b3 = flattened.rhs;
    const bool ta = mm->get_transpose_a();
    const bool tb = mm->get_transpose_b();
    const size_t batch = static_cast<size_t>(flattened.batch);
    const size_t weight_batch = static_cast<size_t>(b3[0]);
    const size_t m_dim = static_cast<size_t>(ta ? a3[2] : a3[1]);
    const size_t k_a = static_cast<size_t>(ta ? a3[1] : a3[2]);
    size_t k_b = static_cast<size_t>(tb ? b3[2] : b3[1]);
    size_t n_dim = static_cast<size_t>(tb ? b3[1] : b3[2]);
    bool effective_tb = tb;
    if (!tb && k_b != k_a && static_cast<size_t>(b3[2]) == k_a) {
        k_b = static_cast<size_t>(b3[2]);
        n_dim = static_cast<size_t>(b3[1]);
        effective_tb = true;
    }
    OPENVINO_ASSERT(k_a == k_b, "GFX Vulkan MatMul: K mismatch ", k_a, " vs ", k_b);
    OPENVINO_ASSERT(weight_batch == 1 || weight_batch == batch,
                    "GFX Vulkan MatMul: unsupported batch broadcast ",
                    weight_batch,
                    " -> ",
                    batch);

    ov::Shape out_shape = flattened.batch_prefix;
    out_shape.push_back(m_dim);
    out_shape.push_back(n_dim);
    output->shape = out_shape;
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    output->expected_type = elem_type;

    if (!m_matmul_linear_kernel ||
        m_matmul_linear_elem_type != elem_type ||
        m_matmul_linear_transpose_a != ta ||
        m_matmul_linear_transpose_b != effective_tb) {
        auto& ctx = gfx_mlir_context();
        auto module = build_matmul_linear_module(ctx, elem_type, ta, effective_tb);
        KernelSource src = make_kernel_source_from_mlir(module, "matmul_linear", /*arg_count=*/4);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_matmul_linear_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_matmul_linear_kernel, "GFX Vulkan MatMul: kernel compile failed: ", log);
        m_matmul_linear_kernel->prepare_runtime_artifacts();
        m_matmul_linear_elem_type = elem_type;
        m_matmul_linear_transpose_a = ta;
        m_matmul_linear_transpose_b = effective_tb;
    }

    const uint32_t total = static_cast<uint32_t>(tensor_elements(out_shape));
    struct MatMulParams {
        uint32_t batch;
        uint32_t m;
        uint32_t n;
        uint32_t k;
        uint32_t weight_batch;
        uint32_t offset;
        uint32_t count;
        uint32_t reserved;
    } params{static_cast<uint32_t>(batch),
             static_cast<uint32_t>(m_dim),
             static_cast<uint32_t>(n_dim),
             static_cast<uint32_t>(k_a),
             static_cast<uint32_t>(weight_batch),
             0,
             total,
             0};
    const std::string key = m_name + "/matmul_params/" + std::to_string(batch) + "x" +
                            std::to_string(m_dim) + "x" + std::to_string(n_dim) + "x" + std::to_string(k_a);
    GpuBuffer params_buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
    OPENVINO_ASSERT(params_buf.valid(), "GFX Vulkan MatMul: failed to wrap params");
    params_buf.owned = false;
    std::vector<KernelArg> args{
        make_buffer_arg(0, input->buf),
        make_buffer_arg(1, weights->buf),
        make_buffer_arg(2, params_buf),
        make_buffer_arg(3, output->buf),
    };
    const uint32_t tg = m_matmul_linear_kernel->clamp_threadgroup_size(64);
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    m_matmul_linear_kernel->execute(command_buffer, dispatch, args, nullptr);
}

void VulkanStage::execute_reduce_last_axis(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(!m_inputs.empty() && m_inputs[0] && m_inputs[0]->buf.valid(),
                    "GFX Vulkan reduce last-axis: missing input buffer");
    OPENVINO_ASSERT(m_node, "GFX Vulkan reduce last-axis: node is null");
    GpuTensor* input = m_inputs[0];
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(output && output->buf.valid(), "GFX Vulkan reduce last-axis: missing output buffer");

    ov::Shape input_shape = input->shape;
    if (input_shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        input_shape = m_node->get_input_shape(0);
    }
    OPENVINO_ASSERT(!input_shape.empty(), "GFX Vulkan reduce last-axis: input shape is unknown");
    const size_t input_rank = input_shape.size();
    OPENVINO_ASSERT(input_rank != 0, "GFX Vulkan reduce last-axis: scalar input is not supported");
    const size_t reduce = input_shape.back();
    OPENVINO_ASSERT(reduce != 0, "GFX Vulkan reduce last-axis: invalid reduce dimension");

    bool keep_dims = false;
    if (auto mean = ov::as_type_ptr<const ov::op::v1::ReduceMean>(m_node)) {
        keep_dims = mean->get_keep_dims();
    } else if (auto sum = ov::as_type_ptr<const ov::op::v1::ReduceSum>(m_node)) {
        keep_dims = sum->get_keep_dims();
    }
    ov::Shape output_shape = input_shape;
    if (keep_dims) {
        output_shape.back() = 1;
    } else {
        output_shape.pop_back();
        if (output_shape.empty()) {
            output_shape = ov::Shape{1};
        }
    }
    output->shape = output_shape;
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    output->expected_type = elem_type;
    const auto op_key = reduce_last_axis_key(m_type);
    OPENVINO_ASSERT(!op_key.empty(), "GFX Vulkan reduce last-axis: unsupported op ", m_type);
    if (!m_reduce_last_axis_kernel ||
        m_reduce_last_axis_elem_type != elem_type ||
        m_reduce_last_axis_key != op_key) {
        auto& ctx = gfx_mlir_context();
        auto module = build_reduce_last_axis_module(ctx, elem_type, op_key);
        KernelSource src = make_kernel_source_from_mlir(module, "reduce_last_axis", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_reduce_last_axis_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_reduce_last_axis_kernel, "GFX Vulkan reduce last-axis: kernel compile failed: ", log);
        m_reduce_last_axis_kernel->prepare_runtime_artifacts();
        m_reduce_last_axis_elem_type = elem_type;
        m_reduce_last_axis_key = op_key;
    }

    const uint32_t total = static_cast<uint32_t>(tensor_elements(output_shape));
    struct ReduceParams {
        uint32_t outer;
        uint32_t reduce;
        uint32_t offset;
        uint32_t count;
    } params{total, static_cast<uint32_t>(reduce), 0, total};
    const std::string key = m_name + "/reduce_last_axis_params/" + std::to_string(total) + "x" + std::to_string(reduce);
    GpuBuffer params_buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
    OPENVINO_ASSERT(params_buf.valid(), "GFX Vulkan reduce last-axis: failed to wrap params");
    params_buf.owned = false;
    std::vector<KernelArg> args{
        make_buffer_arg(0, input->buf),
        make_buffer_arg(1, params_buf),
        make_buffer_arg(2, output->buf),
    };
    const uint32_t tg = m_reduce_last_axis_kernel->clamp_threadgroup_size(64);
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    m_reduce_last_axis_kernel->execute(command_buffer, dispatch, args, nullptr);
}

void VulkanStage::execute_rms_chunked(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(m_node, "GFX Vulkan RMS: node is null");
    auto rms = ov::as_type_ptr<const ov::op::internal::RMS>(m_node);
    OPENVINO_ASSERT(rms, "GFX Vulkan RMS: expected internal RMS node");
    OPENVINO_ASSERT(m_inputs.size() >= 2 && m_inputs[0] && m_inputs[0]->buf.valid(),
                    "GFX Vulkan RMS: missing input buffer");
    GpuTensor* input = m_inputs[0];
    GpuTensor* gamma = m_inputs[1];
    if ((!gamma || !gamma->buf.valid()) &&
        m_const_buffers &&
        m_const_buffers->buffers.size() > 1 &&
        m_const_buffers->present.size() > 1 &&
        m_const_buffers->present[1] &&
        m_const_buffers->buffers[1].buf.valid()) {
        gamma = &m_const_buffers->buffers[1];
    }
    OPENVINO_ASSERT(gamma && gamma->buf.valid(), "GFX Vulkan RMS: missing gamma buffer");
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(output && output->buf.valid(), "GFX Vulkan RMS: missing output buffer");

    ov::Shape out_shape = input->shape;
    if (out_shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        out_shape = m_node->get_input_shape(0);
    }
    OPENVINO_ASSERT(!out_shape.empty(), "GFX Vulkan RMS: input/output shape is unknown");
    output->shape = out_shape;
    output->expected_type = rms->get_output_element_type(0);

    const auto in_pshape = rms->get_input_partial_shape(0);
    OPENVINO_ASSERT(in_pshape.rank().is_static() && in_pshape.rank().get_length() > 0,
                    "GFX Vulkan RMS: input rank must be static");
    const size_t rank = static_cast<size_t>(in_pshape.rank().get_length());
    OPENVINO_ASSERT(in_pshape[rank - 1].is_static(),
                    "GFX Vulkan RMS: hidden dimension must be static");
    const size_t hidden = static_cast<size_t>(in_pshape[rank - 1].get_length());
    const auto gamma_shape = rms->get_input_partial_shape(1).to_shape();
    const size_t gamma_size = ov::shape_size(gamma_shape);
    const uint32_t reduction_threads = gfx_rms_parallel_reduction_threads(static_cast<uint32_t>(hidden));
    const auto input_et = rms->get_input_element_type(0);
    const auto gamma_et = rms->get_input_element_type(1);
    const auto output_et = rms->get_output_element_type(0);

    if (!m_rms_kernel ||
        m_rms_input_elem_type != input_et ||
        m_rms_gamma_elem_type != gamma_et ||
        m_rms_output_elem_type != output_et ||
        m_rms_hidden != hidden ||
        m_rms_gamma_size != gamma_size ||
        m_rms_reduction_threads != reduction_threads ||
        m_rms_epsilon != static_cast<float>(rms->get_epsilon())) {
        auto& ctx = gfx_mlir_context();
        auto module = build_rms_module(ctx,
                                       input_et,
                                       gamma_et,
                                       output_et,
                                       hidden,
                                       gamma_size,
                                       reduction_threads,
                                       static_cast<float>(rms->get_epsilon()));
        KernelSource src = make_kernel_source_from_mlir(module, "rms_linear", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_rms_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_rms_kernel, "GFX Vulkan RMS: kernel compile failed: ", log);
        m_rms_kernel->prepare_runtime_artifacts();
        m_rms_input_elem_type = input_et;
        m_rms_gamma_elem_type = gamma_et;
        m_rms_output_elem_type = output_et;
        m_rms_hidden = static_cast<uint32_t>(hidden);
        m_rms_gamma_size = gamma_size;
        m_rms_reduction_threads = reduction_threads;
        m_rms_epsilon = static_cast<float>(rms->get_epsilon());
    }

    const uint64_t total = static_cast<uint64_t>(tensor_elements(out_shape));
    OPENVINO_ASSERT(total % hidden == 0, "GFX Vulkan RMS: output elements are not divisible by hidden");
    const uint64_t rows = total / hidden;
    const uint32_t tg = m_rms_kernel->clamp_threadgroup_size(reduction_threads);
    KernelDispatch dispatch = reduction_threads > 1
                                  ? make_1d_dispatch(static_cast<size_t>(rows * reduction_threads), tg)
                                  : make_1d_dispatch(static_cast<size_t>(total), tg);
    std::vector<KernelArg> args{
        make_buffer_arg(0, input->buf),
        make_buffer_arg(1, gamma->buf),
        make_buffer_arg(2, output->buf),
    };
    m_rms_kernel->execute(command_buffer, dispatch, args, nullptr);
}

void VulkanStage::execute_softmax_chunked(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(!m_inputs.empty() && m_inputs[0] && m_inputs[0]->buf.valid(),
                    "GFX Vulkan Softmax: missing input buffer");
    OPENVINO_ASSERT(m_node, "GFX Vulkan Softmax: node is null");

    auto* input = m_inputs[0];
    GpuTensor* output = !m_outputs.empty() ? m_outputs[0] : m_output;
    OPENVINO_ASSERT(output && output->buf.valid(),
                    "GFX Vulkan Softmax: missing output buffer");
    OPENVINO_ASSERT(!input->shape.empty(), "GFX Vulkan Softmax: input shape is unknown");
    output->shape = input->shape;

    int64_t axis = -1;
    bool log_softmax = false;
    if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(m_node)) {
        axis = s1->get_axis();
    } else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(m_node)) {
        axis = s8->get_axis();
    } else if (auto ls = ov::as_type_ptr<const ov::op::v5::LogSoftmax>(m_node)) {
        axis = ls->get_axis();
        log_softmax = true;
    } else {
        OPENVINO_THROW("GFX Vulkan Softmax: unsupported node type");
    }

    const auto dims = compute_softmax_dims(input->shape, axis, "GFX Vulkan Softmax");
    const ov::element::Type elem_type =
        input->expected_type == ov::element::dynamic ? input->buf.type : input->expected_type;
    OPENVINO_ASSERT(elem_type == ov::element::f16 || elem_type == ov::element::f32,
                    "GFX Vulkan Softmax: unsupported element type ",
                    elem_type);

    if (!m_softmax_row_kernel || m_softmax_elem_type != elem_type || m_softmax_log_kernel != log_softmax) {
        m_softmax_elem_type = elem_type;
        m_softmax_log_kernel = log_softmax;
        auto& ctx = gfx_mlir_context();
        auto module = build_softmax_row_module(ctx, elem_type, log_softmax);
        KernelSource src = make_kernel_source_from_mlir(module,
                                                        log_softmax ? "logsoftmax_row" : "softmax_row",
                                                        /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_softmax_row_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_softmax_row_kernel, "GFX Vulkan Softmax: kernel compile failed: ", log);
        m_softmax_row_kernel->prepare_runtime_artifacts();
    }

    KernelDispatch dispatch = make_1d_dispatch(1, m_softmax_row_kernel->clamp_threadgroup_size(1));
    const uint32_t cols = static_cast<uint32_t>(dims.axis_len);
    const uint32_t inner = static_cast<uint32_t>(dims.inner == 0 ? 1 : dims.inner);
    const uint32_t total_rows = static_cast<uint32_t>(dims.rows);
    constexpr uint32_t kRowsPerDispatch = 256;
    for (uint32_t row_begin = 0; row_begin < total_rows; row_begin += kRowsPerDispatch) {
        const uint32_t row_count = std::min<uint32_t>(kRowsPerDispatch, total_rows - row_begin);
        struct SoftmaxRowParams {
            uint32_t row_begin;
            uint32_t row_count;
            uint32_t cols;
            uint32_t inner;
        } params{
            row_begin,
            row_count,
            cols,
            inner,
        };

        std::vector<KernelArg> args{
            make_buffer_arg(0, input->buf),
            make_bytes_arg(1, &params, sizeof(params)),
            make_buffer_arg(2, output->buf),
        };
        auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
        m_softmax_row_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
    }
}

void VulkanStage::execute_concat_chunked(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(m_node, "GFX Vulkan Concat: node is null");
    auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(m_node);
    OPENVINO_ASSERT(concat, "GFX Vulkan Concat: expected v0::Concat");

    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(output && output->buf.valid(), "GFX Vulkan Concat: missing output buffer");

    ov::Shape out_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
    if (out_shape.empty() && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(!out_shape.empty(), "GFX Vulkan Concat: output shape unknown");
    output->shape = out_shape;

    const size_t rank = out_shape.size();
    const int64_t axis_norm = normalize_axis(concat->get_axis(), rank, "GFX Vulkan Concat");
    size_t outer = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis_norm); ++i) outer *= out_shape[i];
    size_t inner = 1;
    for (size_t i = static_cast<size_t>(axis_norm) + 1; i < rank; ++i) inner *= out_shape[i];
    const size_t axis_total = out_shape[static_cast<size_t>(axis_norm)];

    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    if (!m_concat_single_kernel || m_concat_elem_type != elem_type) {
        auto& ctx = gfx_mlir_context();
        auto module = build_concat_single_module(ctx, elem_type);
        KernelSource src = make_kernel_source_from_mlir(module, "concat_single", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_concat_single_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_concat_single_kernel, "GFX Vulkan Concat: kernel compile failed: ", log);
        m_concat_single_kernel->prepare_runtime_artifacts();
        m_concat_elem_type = elem_type;
    }

    auto resolve_input = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
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

    auto input_shape = [&](GpuTensor* src, size_t input_idx) -> ov::Shape {
        OPENVINO_ASSERT(src && src->buf.valid(), "GFX Vulkan Concat: missing input buffer ", input_idx);
        ov::Shape src_shape = src->shape;
        if (src_shape.empty() && m_node->get_input_partial_shape(input_idx).is_static()) {
            src_shape = m_node->get_input_shape(input_idx);
        }
        OPENVINO_ASSERT(src_shape.size() == rank, "GFX Vulkan Concat: rank mismatch at input ", input_idx);
        return src_shape;
    };

    if (concat->get_input_size() == 2) {
        if (!m_concat_binary_kernel || m_concat_binary_elem_type != elem_type) {
            auto& ctx = gfx_mlir_context();
            auto module = build_concat_binary_module(ctx, elem_type);
            KernelSource src = make_kernel_source_from_mlir(module, "concat_binary", /*arg_count=*/4);
            src.signature.output_arg_count = 1;
            VulkanCodegenBackend backend;
            std::string log;
            m_concat_binary_kernel = backend.compile(src, &log);
            OPENVINO_ASSERT(m_concat_binary_kernel, "GFX Vulkan Concat binary: kernel compile failed: ", log);
            m_concat_binary_kernel->prepare_runtime_artifacts();
            m_concat_binary_elem_type = elem_type;
        }

        GpuTensor* src0 = resolve_input(0);
        GpuTensor* src1 = resolve_input(1);
        const ov::Shape src0_shape = input_shape(src0, 0);
        const ov::Shape src1_shape = input_shape(src1, 1);
        const size_t slice0 = src0_shape[static_cast<size_t>(axis_norm)];
        const size_t slice1 = src1_shape[static_cast<size_t>(axis_norm)];
        OPENVINO_ASSERT(slice0 + slice1 == axis_total,
                        "GFX Vulkan Concat: binary axis total mismatch");

        struct ConcatBinaryParams {
            uint32_t outer;
            uint32_t axis_total;
            uint32_t inner;
            uint32_t slice0;
            uint32_t slice1;
        } params{static_cast<uint32_t>(outer),
                 static_cast<uint32_t>(axis_total),
                 static_cast<uint32_t>(inner == 0 ? 1 : inner),
                 static_cast<uint32_t>(slice0),
                 static_cast<uint32_t>(slice1)};

        const std::string key = m_name + "/concat_binary_params";
        GpuBuffer params_buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
        OPENVINO_ASSERT(params_buf.valid(), "GFX Vulkan Concat binary: failed to wrap params");
        params_buf.owned = false;

        std::vector<KernelArg> args{
            make_buffer_arg(0, src0->buf),
            make_buffer_arg(1, src1->buf),
            make_buffer_arg(2, params_buf),
            make_buffer_arg(3, output->buf),
        };
        auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
        const uint32_t total =
            static_cast<uint32_t>(outer * (slice0 + slice1) * (inner == 0 ? 1 : inner));
        if (total == 0) {
            return;
        }
        const uint32_t tg = m_concat_binary_kernel->clamp_threadgroup_size(64);
        KernelDispatch dispatch = make_1d_dispatch(total, tg);
        m_concat_binary_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
        return;
    }

    size_t axis_offset = 0;
    for (size_t i = 0; i < concat->get_input_size(); ++i) {
        GpuTensor* src = resolve_input(i);
        ov::Shape src_shape = input_shape(src, i);
        const size_t slice = src_shape[static_cast<size_t>(axis_norm)];

        struct ConcatParams {
            uint32_t outer;
            uint32_t axis_total;
            uint32_t inner;
            uint32_t axis_offset;
            uint32_t slice_len;
        } params{static_cast<uint32_t>(outer),
                 static_cast<uint32_t>(axis_total),
                 static_cast<uint32_t>(inner == 0 ? 1 : inner),
                 static_cast<uint32_t>(axis_offset),
                 static_cast<uint32_t>(slice)};

        const std::string key = m_name + "/concat_params/" + std::to_string(i);
        GpuBuffer params_buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
        OPENVINO_ASSERT(params_buf.valid(), "GFX Vulkan Concat: failed to wrap params");
        params_buf.owned = false;

        std::vector<KernelArg> args{
            make_buffer_arg(0, src->buf),
            make_buffer_arg(1, params_buf),
            make_buffer_arg(2, output->buf),
        };
        auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
        const uint32_t total = static_cast<uint32_t>(outer * slice * (inner == 0 ? 1 : inner));
        const uint32_t tg = m_concat_single_kernel->clamp_threadgroup_size(64);
        KernelDispatch dispatch = make_1d_dispatch(total, tg);
        m_concat_single_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
        axis_offset += slice;
    }
}

void VulkanStage::execute_slice_chunked(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(!m_inputs.empty() && m_inputs[0] && m_inputs[0]->buf.valid(),
                    "GFX Vulkan Slice: missing input buffer");
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(output && output->buf.valid(), "GFX Vulkan Slice: missing output buffer");

    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    if (!m_slice_linear_kernel || m_slice_elem_type != elem_type) {
        auto& ctx = gfx_mlir_context();
        auto module = build_slice_linear_module(ctx, elem_type);
        KernelSource src = make_kernel_source_from_mlir(module, "slice_linear", /*arg_count=*/8);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_slice_linear_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_slice_linear_kernel, "GFX Vulkan Slice: kernel compile failed: ", log);
        m_slice_linear_kernel->prepare_runtime_artifacts();
        m_slice_elem_type = elem_type;
    }

    ov::Shape out_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
    if (out_shape.empty() && m_node && m_node->get_output_partial_shape(0).is_static()) {
        out_shape = m_node->get_output_shape(0);
    }
    OPENVINO_ASSERT(!out_shape.empty(), "GFX Vulkan Slice: output shape unknown");
    ov::Shape in_shape = m_inputs[0]->shape;
    if (in_shape.empty() && m_node && m_node->get_input_partial_shape(0).is_static()) {
        in_shape = m_node->get_input_shape(0);
    }
    OPENVINO_ASSERT(!in_shape.empty(), "GFX Vulkan Slice: input shape unknown");
    output->shape = out_shape;
    const auto meta = build_runtime_slice_meta(m_node, in_shape, out_shape);

    std::ostringstream key_suffix;
    key_suffix << "/" << meta.total << "r" << meta.rank << "_";
    for (size_t i = 0; i < out_shape.size(); ++i) {
        if (i) {
            key_suffix << 'x';
        }
        key_suffix << out_shape[i];
    }

    auto wrap_meta = [&](const std::string& name,
                         const void* data,
                         size_t bytes,
                         const ov::element::Type& type) {
        GpuBuffer buf = m_buffer_manager->wrap_const(m_name + "/" + name + key_suffix.str(), data, bytes, type);
        OPENVINO_ASSERT(buf.valid(), "GFX Vulkan Slice: failed to wrap ", name, " metadata");
        buf.owned = false;
        return buf;
    };
    GpuBuffer total_buf = wrap_meta("slice_total", &meta.total, sizeof(meta.total), ov::element::u32);
    GpuBuffer rank_buf = wrap_meta("slice_rank", &meta.rank, sizeof(meta.rank), ov::element::u32);
    GpuBuffer out_shape_buf = wrap_meta("slice_out_shape",
                                        meta.out_shape.data(),
                                        meta.out_shape.size() * sizeof(uint32_t),
                                        ov::element::u32);
    GpuBuffer in_stride_buf = wrap_meta("slice_in_stride",
                                        meta.in_stride.data(),
                                        meta.in_stride.size() * sizeof(uint32_t),
                                        ov::element::u32);
    GpuBuffer starts_buf = wrap_meta("slice_starts",
                                     meta.starts.data(),
                                     meta.starts.size() * sizeof(int32_t),
                                     ov::element::i32);
    GpuBuffer steps_buf = wrap_meta("slice_steps",
                                    meta.steps.data(),
                                    meta.steps.size() * sizeof(int32_t),
                                    ov::element::i32);

    std::vector<KernelArg> args{
        make_buffer_arg(0, m_inputs[0]->buf),
        make_buffer_arg(1, total_buf),
        make_buffer_arg(2, rank_buf),
        make_buffer_arg(3, out_shape_buf),
        make_buffer_arg(4, in_stride_buf),
        make_buffer_arg(5, starts_buf),
        make_buffer_arg(6, steps_buf),
        make_buffer_arg(7, output->buf),
    };
    auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
    const uint32_t total = static_cast<uint32_t>(tensor_elements(out_shape));
    const uint32_t tg = m_slice_linear_kernel->clamp_threadgroup_size(64);
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    m_slice_linear_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
}

void VulkanStage::execute_transpose_chunked(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(m_node, "GFX Vulkan Transpose: node is null");
    auto tr = ov::as_type_ptr<const ov::op::v1::Transpose>(m_node);
    OPENVINO_ASSERT(tr, "GFX Vulkan Transpose: expected v1::Transpose");
    OPENVINO_ASSERT(!m_inputs.empty() && m_inputs[0] && m_inputs[0]->buf.valid(),
                    "GFX Vulkan Transpose: missing input buffer");
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(output && output->buf.valid(), "GFX Vulkan Transpose: missing output buffer");

    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    if (!m_transpose_kernel || m_transpose_elem_type != elem_type) {
        auto& ctx = gfx_mlir_context();
        auto module = build_transpose_module(ctx, elem_type);
        KernelSource src = make_kernel_source_from_mlir(module, "transpose_direct", /*arg_count=*/2);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_transpose_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_transpose_kernel, "GFX Vulkan Transpose: kernel compile failed: ", log);
        m_transpose_kernel->prepare_runtime_artifacts();
        m_transpose_elem_type = elem_type;
    }

    output->shape = tr->get_output_shape(0);
    const uint32_t total = static_cast<uint32_t>(tensor_elements(output->shape));
    std::vector<KernelArg> args{
        make_buffer_arg(0, m_inputs[0]->buf),
        make_buffer_arg(1, output->buf),
    };
    auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
    const uint32_t tg = m_transpose_kernel->clamp_threadgroup_size(64);
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    m_transpose_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
}

void VulkanStage::execute_interpolate_chunked(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(m_node, "GFX Vulkan Interpolate: node is null");
    OPENVINO_ASSERT(!m_inputs.empty() && m_inputs[0] && m_inputs[0]->buf.valid(),
                    "GFX Vulkan Interpolate: missing input buffer");
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(output && output->buf.valid(), "GFX Vulkan Interpolate: missing output buffer");

    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan Interpolate: unsupported element type ",
                    elem_type);
    if (!m_interpolate_kernel || m_interpolate_elem_type != elem_type) {
        auto& ctx = gfx_mlir_context();
        auto module = build_interpolate_module(ctx, elem_type);
        KernelSource src = make_kernel_source_from_mlir(module, "interpolate_direct", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_interpolate_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_interpolate_kernel, "GFX Vulkan Interpolate: kernel compile failed: ", log);
        m_interpolate_kernel->prepare_runtime_artifacts();
        m_interpolate_elem_type = elem_type;
    }

    const ov::Shape in_shape = m_node->get_input_shape(0);
    const ov::Shape out_shape = m_node->get_output_shape(0);
    OPENVINO_ASSERT(in_shape.size() == 4 && out_shape.size() == 4,
                    "GFX Vulkan Interpolate: expects NCHW rank4");

    int32_t coord_mode = 0;
    int32_t nearest = 1;
    if (auto v0 = ov::as_type_ptr<const ov::op::v0::Interpolate>(m_node)) {
        nearest = ov::util::to_lower(v0->get_attrs().mode) == "nearest" ? 1 : 0;
        coord_mode = v0->get_attrs().align_corners ? 1 : 0;
    } else if (auto v4 = ov::as_type_ptr<const ov::op::v4::Interpolate>(m_node)) {
        using Base = ov::op::util::InterpolateBase;
        nearest = v4->get_attrs().mode == Base::InterpolateMode::NEAREST ? 1 : 0;
        switch (v4->get_attrs().coordinate_transformation_mode) {
            case Base::CoordinateTransformMode::ALIGN_CORNERS:
                coord_mode = 1;
                break;
            case Base::CoordinateTransformMode::ASYMMETRIC:
                coord_mode = 2;
                break;
            case Base::CoordinateTransformMode::HALF_PIXEL:
            default:
                coord_mode = 0;
                break;
        }
    } else if (auto v11 = ov::as_type_ptr<const ov::op::v11::Interpolate>(m_node)) {
        using Base = ov::op::util::InterpolateBase;
        nearest = v11->get_attrs().mode == Base::InterpolateMode::NEAREST ? 1 : 0;
        switch (v11->get_attrs().coordinate_transformation_mode) {
            case Base::CoordinateTransformMode::ALIGN_CORNERS:
                coord_mode = 1;
                break;
            case Base::CoordinateTransformMode::ASYMMETRIC:
                coord_mode = 2;
                break;
            case Base::CoordinateTransformMode::HALF_PIXEL:
            default:
                coord_mode = 0;
                break;
        }
    } else {
        OPENVINO_THROW("GFX Vulkan Interpolate: unsupported op kind");
    }

    struct InterpolateParams {
        int32_t n;
        int32_t c;
        int32_t h_in;
        int32_t w_in;
        int32_t h_out;
        int32_t w_out;
        int32_t coord_mode;
        int32_t nearest;
        int32_t nearest_mode;
    } params{};
    params.n = static_cast<int32_t>(in_shape[0]);
    params.c = static_cast<int32_t>(in_shape[1]);
    params.h_in = static_cast<int32_t>(in_shape[2]);
    params.w_in = static_cast<int32_t>(in_shape[3]);
    params.h_out = static_cast<int32_t>(out_shape[2]);
    params.w_out = static_cast<int32_t>(out_shape[3]);
    params.coord_mode = coord_mode;
    params.nearest = nearest;
    params.nearest_mode = 0;

    output->shape = out_shape;
    const std::string key = m_name + "/interpolate_params";
    GpuBuffer params_buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
    OPENVINO_ASSERT(params_buf.valid(), "GFX Vulkan Interpolate: failed to wrap params");
    params_buf.owned = false;

    std::vector<KernelArg> args{
        make_buffer_arg(0, m_inputs[0]->buf),
        make_buffer_arg(1, params_buf),
        make_buffer_arg(2, output->buf),
    };
    auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
    const size_t tg_total = std::max<size_t>(1, m_interpolate_kernel->clamp_threadgroup_size(64));
    OPENVINO_ASSERT(tg_total >= 64, "GFX Vulkan Interpolate: expected at least 64 threads per group");
    KernelDispatch dispatch =
        make_3d_dispatch(out_shape[3], out_shape[2], out_shape[0] * out_shape[1], 8, 8, 1);
    m_interpolate_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
}

void VulkanStage::execute_convert_chunked(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(!m_inputs.empty() && m_inputs[0] && m_inputs[0]->buf.valid(),
                    "GFX Vulkan Convert: missing input buffer");
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(output && output->buf.valid(), "GFX Vulkan Convert: missing output buffer");

    const auto src_et = m_node->get_input_element_type(0);
    const auto dst_et = m_node->get_output_element_type(0);
    OPENVINO_ASSERT(is_supported_linear_convert_type(src_et, dst_et),
                    "GFX Vulkan Convert: unsupported conversion ",
                    src_et,
                    " -> ",
                    dst_et);

    if (!m_convert_linear_kernel || m_convert_src_elem_type != src_et || m_convert_dst_elem_type != dst_et) {
        auto& ctx = gfx_mlir_context();
        auto module = build_convert_linear_module(ctx, src_et, dst_et);
        KernelSource src = make_kernel_source_from_mlir(module, "convert_linear", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_convert_linear_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_convert_linear_kernel, "GFX Vulkan Convert: kernel compile failed: ", log);
        m_convert_linear_kernel->prepare_runtime_artifacts();
        m_convert_src_elem_type = src_et;
        m_convert_dst_elem_type = dst_et;
    }

    if (output->shape.empty()) {
        if (!m_output_shape.empty()) {
            output->shape = m_output_shape;
        } else if (!m_inputs.empty() && m_inputs[0] && !m_inputs[0]->shape.empty()) {
            output->shape = m_inputs[0]->shape;
        } else if (m_node->get_output_partial_shape(0).is_static()) {
            output->shape = m_node->get_output_shape(0);
        }
    }
    OPENVINO_ASSERT(!output->shape.empty(), "GFX Vulkan Convert: output shape is unknown");
    output->expected_type = dst_et;
    const uint32_t total = static_cast<uint32_t>(tensor_elements(output->shape));
    struct ConvertParams {
        uint32_t offset;
        uint32_t count;
    } params{0, total};
    const std::string key = m_name + "/convert_params/" + std::to_string(total);
    GpuBuffer params_buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
    OPENVINO_ASSERT(params_buf.valid(), "GFX Vulkan Convert: failed to wrap params");
    params_buf.owned = false;
    std::vector<KernelArg> args{
        make_buffer_arg(0, m_inputs[0]->buf),
        make_buffer_arg(1, params_buf),
        make_buffer_arg(2, output->buf),
    };
    const uint32_t tg = m_convert_linear_kernel->clamp_threadgroup_size(64);
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    m_convert_linear_kernel->execute(command_buffer, dispatch, args, nullptr);
}

void VulkanStage::execute_gather_linear(GpuCommandBufferHandle command_buffer) {
    auto resolve_input = [&](size_t input_idx) -> GpuTensor* {
        GpuTensor* tensor = input_idx < m_inputs.size() ? m_inputs[input_idx] : nullptr;
        if (tensor && tensor->buf.valid()) {
            return tensor;
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
    OPENVINO_ASSERT(m_node, "GFX Vulkan Gather: missing node");
    GpuTensor* data = resolve_input(0);
    GpuTensor* indices = resolve_input(1);
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(data && data->buf.valid() && indices && indices->buf.valid(),
                    "GFX Vulkan Gather: missing input buffers");
    OPENVINO_ASSERT(output && output->buf.valid(), "GFX Vulkan Gather: missing output buffer");

    ov::Shape data_shape = data->shape;
    if (data_shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        data_shape = m_node->get_input_shape(0);
    }
    ov::Shape indices_shape = indices->shape;
    if (indices_shape.empty() && m_node->get_input_partial_shape(1).is_static()) {
        indices_shape = m_node->get_input_shape(1);
    }
    OPENVINO_ASSERT(!data_shape.empty() && !indices_shape.empty(), "GFX Vulkan Gather: runtime shapes are unknown");

    auto axis_c = ov::as_type_ptr<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(2));
    OPENVINO_ASSERT(axis_c, "GFX Vulkan Gather: axis must be constant");
    const auto axis_v = axis_c->cast_vector<int64_t>();
    OPENVINO_ASSERT(axis_v.size() == 1, "GFX Vulkan Gather: axis must be scalar");
    const auto dims = compute_gather_linear_dims(data_shape, indices_shape, axis_v[0], "GFX Vulkan Gather");
    ov::Shape out_shape = compute_gather_output_shape(data_shape, indices_shape, dims.axis, "GFX Vulkan Gather");
    output->shape = out_shape;
    output->expected_type = m_node->get_output_element_type(0);

    const auto data_et = m_node->get_input_element_type(0);
    const auto idx_et = m_node->get_input_element_type(1);
    if (!m_gather_linear_kernel ||
        m_gather_linear_data_elem_type != data_et ||
        m_gather_linear_index_elem_type != idx_et) {
        auto& ctx = gfx_mlir_context();
        auto module = build_gather_linear_module(ctx, data_et, idx_et);
        KernelSource src = make_kernel_source_from_mlir(module, "gather_linear", /*arg_count=*/4);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_gather_linear_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_gather_linear_kernel, "GFX Vulkan Gather: kernel compile failed: ", log);
        m_gather_linear_kernel->prepare_runtime_artifacts();
        m_gather_linear_data_elem_type = data_et;
        m_gather_linear_index_elem_type = idx_et;
    }

    const uint32_t total = static_cast<uint32_t>(tensor_elements(out_shape));
    if (total == 0) {
        return;
    }
    struct GatherParams {
        uint32_t outer;
        uint32_t inner;
        uint32_t axis_dim;
        uint32_t indices_count;
        uint32_t offset;
        uint32_t count;
    } params{static_cast<uint32_t>(dims.outer),
             static_cast<uint32_t>(dims.inner),
             static_cast<uint32_t>(dims.axis_dim),
             static_cast<uint32_t>(dims.indices_count),
             0,
             total};
    const std::string key = m_name + "/gather_linear_params/" + std::to_string(total) + "/" +
                            std::to_string(dims.outer) + "x" + std::to_string(dims.indices_count) + "x" +
                            std::to_string(dims.inner);
    GpuBuffer params_buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
    OPENVINO_ASSERT(params_buf.valid(), "GFX Vulkan Gather: failed to wrap params");
    params_buf.owned = false;
    std::vector<KernelArg> args{
        make_buffer_arg(0, data->buf),
        make_buffer_arg(1, indices->buf),
        make_buffer_arg(2, params_buf),
        make_buffer_arg(3, output->buf),
    };
    const uint32_t tg = m_gather_linear_kernel->clamp_threadgroup_size(64);
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    m_gather_linear_kernel->execute(command_buffer, dispatch, args, nullptr);
}

void VulkanStage::execute_gather_embedding(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(m_inputs.size() >= 2 &&
                        m_inputs[0] && m_inputs[0]->buf.valid() &&
                        m_inputs[1] && m_inputs[1]->buf.valid(),
                    "GFX Vulkan Gather: missing input buffers");
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    OPENVINO_ASSERT(output && output->buf.valid(), "GFX Vulkan Gather: missing output buffer");
    const auto data_shape = m_node->get_input_shape(0);
    OPENVINO_ASSERT(data_shape.size() == 2, "GFX Vulkan Gather: expected embedding table rank 2");
    const uint32_t vocab = static_cast<uint32_t>(data_shape[0]);
    const uint32_t hidden = static_cast<uint32_t>(data_shape[1]);

    ov::Shape indices_shape = m_inputs[1]->shape;
    if (indices_shape.empty() && m_node->get_input_partial_shape(1).is_static()) {
        indices_shape = m_node->get_input_shape(1);
    }
    OPENVINO_ASSERT(!indices_shape.empty(), "GFX Vulkan Gather: indices shape is unknown");
    ov::Shape out_shape = indices_shape;
    out_shape.push_back(hidden);
    output->shape = out_shape;
    output->expected_type = m_node->get_output_element_type(0);

    const auto data_et = m_node->get_input_element_type(0);
    const auto idx_et = m_node->get_input_element_type(1);
    if (!m_gather_embedding_kernel ||
        m_gather_data_elem_type != data_et ||
        m_gather_index_elem_type != idx_et ||
        m_gather_vocab != vocab ||
        m_gather_hidden != hidden) {
        auto& ctx = gfx_mlir_context();
        auto module = build_gather_embedding_module(ctx, data_et, idx_et, vocab, hidden);
        KernelSource src = make_kernel_source_from_mlir(module, "gather_embedding", /*arg_count=*/4);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_gather_embedding_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_gather_embedding_kernel, "GFX Vulkan Gather: kernel compile failed: ", log);
        m_gather_embedding_kernel->prepare_runtime_artifacts();
        m_gather_data_elem_type = data_et;
        m_gather_index_elem_type = idx_et;
        m_gather_vocab = vocab;
        m_gather_hidden = hidden;
    }

    const uint32_t total = static_cast<uint32_t>(tensor_elements(output->shape));
    struct GatherParams {
        uint32_t vocab;
        uint32_t hidden;
        uint32_t tokens;
        uint32_t offset;
        uint32_t count;
    } params{vocab,
             hidden,
             static_cast<uint32_t>(tensor_elements(indices_shape)),
             0,
             total};
    const std::string key = m_name + "/gather_embedding_params/" + std::to_string(total);
    GpuBuffer params_buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
    OPENVINO_ASSERT(params_buf.valid(), "GFX Vulkan Gather: failed to wrap params");
    params_buf.owned = false;
    std::vector<KernelArg> args{
        make_buffer_arg(0, m_inputs[0]->buf),
        make_buffer_arg(1, m_inputs[1]->buf),
        make_buffer_arg(2, params_buf),
        make_buffer_arg(3, output->buf),
    };
    const uint32_t tg = m_gather_embedding_kernel->clamp_threadgroup_size(64);
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    m_gather_embedding_kernel->execute(command_buffer, dispatch, args, nullptr);
}

void VulkanStage::execute_split_chunked(GpuCommandBufferHandle command_buffer) {
    OPENVINO_ASSERT(!m_inputs.empty() && m_inputs[0] && m_inputs[0]->buf.valid(),
                    "GFX Vulkan Split: missing input buffer");
    std::vector<GpuTensor*> outputs = m_outputs;
    if (outputs.empty() && m_output) {
        outputs.push_back(m_output);
    }
    OPENVINO_ASSERT(!outputs.empty(), "GFX Vulkan Split: missing output buffers");
    const auto& in = *m_inputs[0];
    OPENVINO_ASSERT(!in.shape.empty(), "GFX Vulkan Split: input shape unknown");

    int64_t axis = 0;
    std::vector<size_t> split_sizes;
    if (auto s = ov::as_type_ptr<const ov::op::v1::Split>(m_node)) {
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(s->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "Split axis must be constant");
        axis = axis_const->cast_vector<int64_t>()[0];
        const auto parts = s->get_num_splits();
        OPENVINO_ASSERT(parts > 0, "Split parts zero");
        const size_t axis_len = in.shape.at(static_cast<size_t>(normalize_axis(axis, in.shape.size(), "GFX Vulkan Split")));
        OPENVINO_ASSERT(axis_len % parts == 0, "Split axis not divisible");
        split_sizes.assign(parts, axis_len / parts);
    } else if (auto vs = ov::as_type_ptr<const ov::op::v1::VariadicSplit>(m_node)) {
        auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(1).get_node_shared_ptr());
        OPENVINO_ASSERT(axis_const, "VariadicSplit axis must be constant");
        axis = axis_const->cast_vector<int64_t>()[0];
        auto lengths_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(2).get_node_shared_ptr());
        OPENVINO_ASSERT(lengths_const, "VariadicSplit lengths must be constant");
        auto lengths = lengths_const->cast_vector<int64_t>();
        split_sizes.reserve(lengths.size());
        size_t axis_len = in.shape.at(static_cast<size_t>(normalize_axis(axis, in.shape.size(), "GFX Vulkan Split")));
        size_t sum = 0;
        for (auto v : lengths) {
            OPENVINO_ASSERT(v >= 0, "VariadicSplit negative length");
            split_sizes.push_back(static_cast<size_t>(v));
            sum += static_cast<size_t>(v);
        }
        OPENVINO_ASSERT(sum == axis_len, "VariadicSplit sizes mismatch");
    } else {
        OPENVINO_THROW("GFX Vulkan Split: unsupported node type");
    }

    const size_t rank = in.shape.size();
    const size_t axis_norm = static_cast<size_t>(normalize_axis(axis, rank, "GFX Vulkan Split"));
    const size_t axis_len_total = in.shape[axis_norm];
    size_t inner = 1;
    for (size_t d = axis_norm + 1; d < rank; ++d) inner *= in.shape[d];
    size_t outer = 1;
    for (size_t d = 0; d < axis_norm; ++d) outer *= in.shape[d];

    for (size_t oi = 0; oi < outputs.size(); ++oi) {
        auto* out = outputs[oi];
        if (!out) {
            continue;
        }
        ov::Shape out_shape = in.shape;
        out_shape[axis_norm] = split_sizes[oi];
        out->shape = std::move(out_shape);
        if (gfx_log_debug_enabled() && out->buf.valid()) {
            std::ostringstream oss;
            oss << "Split output[" << oi << "]"
                << " buf=" << out->buf.buffer
                << " uid=" << out->buf.allocation_uid
                << " off=" << out->buf.offset
                << " bytes=" << out->buf.size
                << " shape=" << out->shape
                << " type=" << out->expected_type;
            gfx_log_debug("VulkanExec") << oss.str();
        }
    }

    // Compile compact split kernel once per element type.
    if (!m_split_single_kernel || m_split_elem_type != in.expected_type) {
        m_split_elem_type = in.expected_type;
        auto& ctx = gfx_mlir_context();
        auto module = build_split_single_module(ctx, m_split_elem_type);
        KernelSource src = make_kernel_source_from_mlir(module, "split_single", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_split_single_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_split_single_kernel, "GFX Vulkan Split: kernel compile failed: ", log);
        m_split_single_kernel->prepare_runtime_artifacts();
    }

    size_t offset_along_axis = 0;
    for (size_t oi = 0; oi < outputs.size(); ++oi) {
        auto* out = outputs[oi];
        OPENVINO_ASSERT(out && out->buf.valid(), "GFX Vulkan Split: missing output buffer");
        OPENVINO_ASSERT(oi < split_sizes.size(), "GFX Vulkan Split: split size missing");
        const size_t slice = split_sizes[oi];

        struct SplitParams {
            uint32_t outer;
            uint32_t axis_total;
            uint32_t inner;
            uint32_t axis_offset;
            uint32_t slice_len;
        } params{static_cast<uint32_t>(outer),
                 static_cast<uint32_t>(axis_len_total),
                 static_cast<uint32_t>(inner == 0 ? 1 : inner),
                 static_cast<uint32_t>(offset_along_axis),
                 static_cast<uint32_t>(slice)};

        const std::string key = m_name + "/split_params/" + std::to_string(oi);
        GpuBuffer params_buf = m_buffer_manager->wrap_const(key, &params, sizeof(params), ov::element::u8);
        OPENVINO_ASSERT(params_buf.valid(), "GFX Vulkan Split: failed to wrap params");
        params_buf.owned = false;
        std::vector<KernelArg> args{
            make_buffer_arg(0, in.buf),
            make_buffer_arg(1, params_buf),
            make_buffer_arg(2, out->buf),
        };
        auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
        const uint32_t total = static_cast<uint32_t>(outer * slice * (inner == 0 ? 1 : inner));
        const uint32_t tg = m_split_single_kernel->clamp_threadgroup_size(64);
        KernelDispatch dispatch = make_1d_dispatch(total, tg);
        m_split_single_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
        offset_along_axis += slice;
    }
}

}  // namespace gfx_plugin
}  // namespace ov
