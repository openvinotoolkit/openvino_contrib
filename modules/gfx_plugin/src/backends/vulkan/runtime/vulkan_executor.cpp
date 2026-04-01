// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/vulkan_executor.hpp"

#include <chrono>

#include "backends/vulkan/codegen/vulkan_codegen_backend.hpp"
#include "backends/vulkan/runtime/profiling/profiler.hpp"
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
#include "openvino/op/interpolate.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
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
// Keep chunked Conv2D bounded for mobile Vulkan drivers, but large enough to
// avoid excessive per-dispatch overhead once chunks are recorded into the
// shared infer command buffer.
constexpr uint32_t kConv2DChunkElemsPerDispatch = 1024;

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

bool is_supported_linear_elem_type(const ov::element::Type& et) {
    return et == ov::element::f16 || et == ov::element::f32;
}

bool is_supported_linear_convert_type(const ov::element::Type& src_et, const ov::element::Type& dst_et) {
    return src_et == ov::element::u8 && (dst_et == ov::element::f16 || dst_et == ov::element::f32);
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
    return {};
}

}  // namespace

VulkanStage::VulkanStage(const std::shared_ptr<const ov::Node>& node)
    : MlirStage(node) {}

VulkanStage::~VulkanStage() = default;

void VulkanStage::init(GpuBufferManager* buffer_manager) {
    MlirStage::init(buffer_manager);
}

void VulkanStage::compile(GpuBufferManager* buffer_manager) {
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
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan binary chunked: unsupported element type ",
                    elem_type);
    OPENVINO_ASSERT(!m_kernel_extra_inputs.empty() && !m_kernel_extra_inputs[0].shape.empty(),
                    "GFX Vulkan binary chunked: missing metadata shape");
    const size_t meta_rank = m_kernel_extra_inputs[0].shape[0];
    OPENVINO_ASSERT(meta_rank != 0, "GFX Vulkan binary chunked: invalid metadata rank");
    if (m_linear_binary_kernel && m_linear_binary_elem_type == elem_type && m_linear_binary_key == op_key &&
        m_linear_binary_rank == meta_rank) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_linear_binary_module(ctx, elem_type, op_key, meta_rank);
    m_linear_binary_kernel =
        compile_specialized_kernel_from_mlir(module, "linear_binary", 4, "GFX Vulkan binary chunked: kernel compile failed: ");
    m_linear_binary_elem_type = elem_type;
    m_linear_binary_key = op_key;
    m_linear_binary_rank = meta_rank;
    m_linear_binary_launch_abi = extract_launch_operand_abi(module);
}

void VulkanStage::prepare_binary_same_shape_kernel() {
    const auto op_key = binary_chunk_key(m_type);
    OPENVINO_ASSERT(!op_key.empty(), "GFX Vulkan same-shape binary: unsupported op ", m_type);
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan same-shape binary: unsupported element type ",
                    elem_type);
    if (m_linear_binary_same_shape_kernel && m_linear_binary_same_shape_elem_type == elem_type &&
        m_linear_binary_same_shape_key == op_key) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_linear_binary_same_shape_module(ctx, elem_type, op_key);
    m_linear_binary_same_shape_kernel = compile_specialized_kernel_from_mlir(
        module,
        "linear_binary_same_shape",
        3,
        "GFX Vulkan same-shape binary: kernel compile failed: ");
    m_linear_binary_same_shape_elem_type = elem_type;
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
    const auto total = static_cast<uint64_t>(tensor_elements(conv->get_output_shape(0)));
    const auto work_per_elem =
        static_cast<uint64_t>(conv->get_input_shape(0).at(1)) * static_cast<uint64_t>(conv->get_input_shape(1).at(2)) *
        static_cast<uint64_t>(conv->get_input_shape(1).at(3));
    const auto launch_plan = select_chunk_dispatch_plan(caps, "conv2d", total, work_per_elem);
    if (m_conv2d_chunk_kernel && m_conv2d_chunk_elem_type == elem_type &&
        m_conv2d_chunk_threads_per_group == launch_plan.threads_per_group) {
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_conv2d_chunk_module(ctx, elem_type, launch_plan.threads_per_group);
    m_conv2d_chunk_kernel =
        compile_specialized_kernel_from_mlir(module, "conv2d_chunk", 4, "GFX Vulkan conv2d chunked: kernel compile failed: ");
    m_conv2d_chunk_elem_type = elem_type;
    m_conv2d_chunk_threads_per_group = launch_plan.threads_per_group;
    m_conv2d_chunk_launch_abi = extract_launch_operand_abi(module);
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
        return;
    }
    auto& ctx = gfx_mlir_context();
    auto module = build_concat_single_module(ctx, elem_type);
    m_concat_single_kernel =
        compile_specialized_kernel_from_mlir(module, "concat_single", 3, "GFX Vulkan Concat: kernel compile failed: ");
    m_concat_elem_type = elem_type;
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
        2,
        "GFX Vulkan Convert: kernel compile failed: ");
    m_convert_src_elem_type = src_et;
    m_convert_dst_elem_type = dst_et;
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
    GpuTensor* output = !m_outputs.empty() ? m_outputs.front() : m_output;
    GpuTensor* input0 = resolve_input(0);
    if (!input0 || !output) {
        return false;
    }
    const ov::Shape& dispatch_shape = !m_output_shape.empty() ? m_output_shape : input0->shape;
    if (dispatch_shape.empty()) {
        return false;
    }
    const ov::element::Type et = resolve_stage_element_type(m_node, nullptr);
    return is_supported_linear_elem_type(et) &&
           tensor_elements(dispatch_shape) >= kLargeLinearChunkElems;
}

bool VulkanStage::should_use_concat_chunked() const {
    if (m_type != "Concat" || has_absorbed_input_transpose() || !m_node) {
        return false;
    }
    auto concat = ov::as_type_ptr<const ov::op::v0::Concat>(m_node);
    if (!concat || concat->get_input_size() == 0 || concat->get_output_size() != 1) {
        return false;
    }
    if (!m_node->get_output_partial_shape(0).is_static()) {
        return false;
    }
    const auto& out_shape = m_node->get_output_shape(0);
    if (out_shape.empty()) {
        return false;
    }
    return is_supported_linear_elem_type(
        resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output));
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
    if (should_use_interpolate_chunked() || should_use_convert_chunked() || should_use_softmax_chunked() || should_use_binary_same_shape() ||
        should_use_binary_bias_add() || should_use_binary_chunked() || should_use_unary_chunked() ||
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
    if (binary_chunk_key(m_type).empty()) {
        return false;
    }
    if (has_absorbed_input_transpose()) {
        return false;
    }
    if (!m_node || m_node->get_input_size() != 2 || m_node->get_output_size() != 1) {
        return false;
    }
    if (!m_node->get_input_partial_shape(0).is_static() ||
        !m_node->get_input_partial_shape(1).is_static() ||
        !m_node->get_output_partial_shape(0).is_static()) {
        return false;
    }
    const ov::Shape& dispatch_shape = !m_output_shape.empty() ? m_output_shape : m_node->get_output_shape(0);
    if (dispatch_shape.empty()) {
        return false;
    }
    if (m_kernel_extra_inputs.size() < 3) {
        return false;
    }
    const ov::element::Type et = resolve_stage_element_type(m_node, nullptr);
    return is_supported_linear_elem_type(et) &&
           tensor_elements(dispatch_shape) >= kLargeLinearChunkElems;
}

bool VulkanStage::should_use_binary_same_shape() const {
    if (binary_chunk_key(m_type).empty() || !m_node || m_node->get_input_size() != 2 || m_node->get_output_size() != 1) {
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
    const ov::element::Type elem_type =
        resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    return m_node->get_input_shape(0) == out_shape &&
           m_node->get_input_shape(1) == out_shape &&
           is_supported_linear_elem_type(elem_type) &&
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
    if (!m_node->get_input_partial_shape(0).is_static() || !m_node->get_output_partial_shape(0).is_static()) {
        return false;
    }
    const auto& in_shape = m_node->get_input_shape(0);
    const auto& out_shape = m_node->get_output_shape(0);
    if (in_shape.empty() || out_shape.empty() || tensor_elements(in_shape) != tensor_elements(out_shape)) {
        return false;
    }
    return is_supported_linear_convert_type(m_node->get_input_element_type(0), m_node->get_output_element_type(0)) &&
           tensor_elements(out_shape) >= kLargeLinearChunkElems;
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
    OPENVINO_ASSERT(m_node && m_node->get_output_partial_shape(0).is_static() &&
                        m_node->get_input_partial_shape(0).is_static(),
                    "GFX Vulkan unary chunked: static shapes required");
    const ov::Shape out_shape = m_node->get_output_shape(0);
    const ov::Shape in_shape = m_node->get_input_shape(0);

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
    auto in_ty = MemRefType::get({static_cast<int64_t>(tensor_elements(in_shape))}, elem_ty);
    auto out_ty = MemRefType::get({static_cast<int64_t>(tensor_elements(out_shape))}, elem_ty);
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
                                                      uint32_t threads_per_group) {
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
    const int64_t input_elems = static_cast<int64_t>(tensor_elements(in_shape));
    const int64_t weight_elems = static_cast<int64_t>(tensor_elements(w_shape));
    const int64_t output_elems = static_cast<int64_t>(tensor_elements(out_shape));
    auto input_ty = MemRefType::get({input_elems}, elem_ty);
    auto weight_ty = MemRefType::get({weight_elems}, elem_ty);
    auto output_ty = MemRefType::get({output_elems}, elem_ty);
    auto param_ty = MemRefType::get({2}, IntegerType::get(&ctx, 32));
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "conv2d_chunk",
                                                 b.getFunctionType(TypeRange{input_ty, weight_ty, param_ty, output_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(mlir::DenseI32ArrayAttr::get(&ctx, {static_cast<int32_t>(std::max<uint32_t>(1u, threads_per_group)), 1, 1}));
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

    Value offset = load_param(0);
    Value count = load_param(1);
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
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value global_idx = body.create<arith::AddIOp>(loc,
                                                  body.create<arith::MulIOp>(loc, bid, bdim),
                                                  tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, global_idx, count);
    auto active_if = body.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value local_idx = global_idx;
        Value linear_idx = then_builder.create<arith::AddIOp>(loc, offset, local_idx);
        Value rem = linear_idx;
        Value ow = then_builder.create<arith::RemUIOp>(loc, rem, out_w);
        rem = then_builder.create<arith::DivUIOp>(loc, rem, out_w);
        Value oh = then_builder.create<arith::RemUIOp>(loc, rem, out_h);
        rem = then_builder.create<arith::DivUIOp>(loc, rem, out_h);
        Value oc = then_builder.create<arith::RemUIOp>(loc, rem, c_out);
        Value n = then_builder.create<arith::DivUIOp>(loc, rem, c_out);
        Value acc = zero.getResult();
        for (int64_t ic_idx = 0; ic_idx < in_channels; ++ic_idx) {
            Value ic = then_builder.create<arith::ConstantIndexOp>(loc, ic_idx);
            for (int64_t kh_idx = 0; kh_idx < kernel_h; ++kh_idx) {
                Value kh = then_builder.create<arith::ConstantIndexOp>(loc, kh_idx);
                for (int64_t kw_idx = 0; kw_idx < kernel_w; ++kw_idx) {
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

                    auto if_in_bounds = then_builder.create<scf::IfOp>(
                        loc, TypeRange{compute_ty}, in_bounds, /*withElseRegion=*/true);
                    {
                        auto then_inner = if_in_bounds.getThenBodyBuilder();
                        Value input_offset = then_inner.create<arith::AddIOp>(
                            loc,
                            then_inner.create<arith::MulIOp>(
                                loc,
                                then_inner.create<arith::AddIOp>(
                                    loc,
                                    then_inner.create<arith::MulIOp>(loc, n, c_in),
                                    ic),
                                then_inner.create<arith::MulIOp>(loc, in_h, in_w)),
                            then_inner.create<arith::AddIOp>(
                                loc,
                                then_inner.create<arith::MulIOp>(loc, ih, in_w),
                                iw));
                        Value weight_offset = then_inner.create<arith::AddIOp>(
                            loc,
                            then_inner.create<arith::MulIOp>(
                                loc,
                                then_inner.create<arith::AddIOp>(
                                    loc,
                                    then_inner.create<arith::MulIOp>(loc, oc, c_in),
                                    ic),
                                then_inner.create<arith::MulIOp>(loc, k_h, k_w)),
                            then_inner.create<arith::AddIOp>(
                                loc,
                                then_inner.create<arith::MulIOp>(loc, kh, k_w),
                                kw));
                        Value input_val = then_inner.create<memref::LoadOp>(
                            loc,
                            fn.getArgument(0),
                            ValueRange{input_offset});
                        Value weight_val = then_inner.create<memref::LoadOp>(
                            loc,
                            fn.getArgument(1),
                            ValueRange{weight_offset});
                        Value prod = then_inner.create<arith::MulFOp>(
                            loc,
                            cast_to_compute(then_inner, input_val),
                            cast_to_compute(then_inner, weight_val));
                        Value sum = then_inner.create<arith::AddFOp>(loc, prod, acc);
                        then_inner.create<scf::YieldOp>(loc, sum);
                    }
                    {
                        auto else_inner = if_in_bounds.getElseBodyBuilder();
                        else_inner.create<scf::YieldOp>(loc, acc);
                    }
                    acc = if_in_bounds.getResult(0);
                }
            }
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
                                             fn.getArgument(3),
                                             ValueRange{output_offset});
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
                                                       const ov::element::Type& et,
                                                       const std::string& op_key,
                                                       size_t meta_rank) {
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
        default: OPENVINO_THROW("GFX Vulkan binary chunked: unsupported element type ", et);
    }
    Type compute_ty = elem_ty;

    OPENVINO_ASSERT(m_node && m_node->get_output_partial_shape(0).is_static() &&
                        m_node->get_input_partial_shape(0).is_static() &&
                        m_node->get_input_partial_shape(1).is_static(),
                    "GFX Vulkan binary chunked: static shapes required");
    const ov::Shape out_shape = m_node->get_output_shape(0);
    const ov::Shape in0_shape = m_node->get_input_shape(0);
    const ov::Shape in1_shape = m_node->get_input_shape(1);
    OPENVINO_ASSERT(meta_rank == out_shape.size() && meta_rank != 0,
                    "GFX Vulkan binary chunked: metadata rank must match output rank");
    std::vector<int32_t> dims_i32(out_shape.size(), 0);
    for (size_t i = 0; i < out_shape.size(); ++i) {
        dims_i32[i] = static_cast<int32_t>(out_shape[i]);
    }
    const auto stride0_vals = ov::gfx_plugin::compute_broadcast_element_strides(in0_shape, out_shape);
    const auto stride1_vals = ov::gfx_plugin::compute_broadcast_element_strides(in1_shape, out_shape);

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
    auto in0_ty = MemRefType::get({static_cast<int64_t>(tensor_elements(in0_shape))}, elem_ty);
    auto in1_ty = MemRefType::get({static_cast<int64_t>(tensor_elements(in1_shape))}, elem_ty);
    auto out_ty = MemRefType::get({static_cast<int64_t>(tensor_elements(out_shape))}, elem_ty);
    auto param_ty = MemRefType::get({2}, IntegerType::get(&ctx, 32));
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
    auto emit_binary = [&](OpBuilder& builder, Value lhs, Value rhs) -> Value {
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
        auto for_dims = then_builder.create<scf::ForOp>(loc,
                                                        c0,
                                                        rank,
                                                        c1,
                                                        ValueRange{linear_idx, c0, c0});
        auto dim_builder = OpBuilder::atBlockBegin(for_dims.getBody());
        Value rev_dim = dim_builder.create<arith::SubIOp>(loc, rank, c1);
        rev_dim = dim_builder.create<arith::SubIOp>(loc, rev_dim, for_dims.getInductionVar());
        auto load_const_meta = [&](const std::vector<int32_t>& values, Value index) -> Value {
            OPENVINO_ASSERT(!values.empty(), "GFX Vulkan binary chunked: empty metadata constants");
            Value result = dim_builder.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(values.front()));
            for (int64_t i = static_cast<int64_t>(values.size()) - 1; i >= 0; --i) {
                auto idx_c = dim_builder.create<arith::ConstantIndexOp>(loc, i);
                auto cond = dim_builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, index, idx_c);
                auto val = dim_builder.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(values[static_cast<size_t>(i)]));
                result = dim_builder.create<arith::SelectOp>(loc, cond, val, result);
            }
            return result;
        };
        Value dim_size = load_const_meta(dims_i32, rev_dim);
        Value coord = dim_builder.create<arith::RemUIOp>(loc, for_dims.getRegionIterArgs()[0], dim_size);
        Value next_rem = dim_builder.create<arith::DivUIOp>(loc, for_dims.getRegionIterArgs()[0], dim_size);
        Value stride0 = load_const_meta(stride0_vals, rev_dim);
        Value stride1 = load_const_meta(stride1_vals, rev_dim);
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
        default: OPENVINO_THROW("GFX Vulkan same-shape binary: unsupported element type ", et);
    }
    Type compute_ty = elem_ty;

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

    auto in_ty = MemRefType::get({total}, elem_ty);
    auto out_ty = MemRefType::get({total}, elem_ty);
    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "linear_binary_same_shape",
                                                 b.getFunctionType(TypeRange{in_ty, in_ty, out_ty}, {}),
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
        if (value.getType() == elem_ty) {
            return value;
        }
        return builder.create<arith::TruncFOp>(loc, elem_ty, value);
    };
    auto emit_binary = [&](OpBuilder& builder, Value lhs, Value rhs) -> Value {
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

    const ov::Shape& dispatch_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
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
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan binary chunked: unsupported element type ",
                    elem_type);
    OPENVINO_ASSERT(!m_kernel_extra_inputs[0].shape.empty(),
                    "GFX Vulkan binary chunked: missing metadata shape");
    const size_t meta_rank = m_kernel_extra_inputs[0].shape[0];
    OPENVINO_ASSERT(meta_rank != 0, "GFX Vulkan binary chunked: invalid metadata rank");
    if (gfx_log_debug_enabled()) {
        gfx_log_debug("VulkanExec") << "Binary chunked buffers in0=" << input0->buf.buffer
                                    << " in1=" << input1->buf.buffer
                                    << " out=" << output->buf.buffer
                                    << " in0_type=" << input0->buf.type << "/" << input0->expected_type
                                    << " in1_type=" << input1->buf.type << "/" << input1->expected_type
                                    << " out_type=" << output->buf.type << "/" << output->expected_type;
    }
    if (!m_linear_binary_kernel || m_linear_binary_elem_type != elem_type ||
        m_linear_binary_key != op_key || m_linear_binary_rank != meta_rank) {
        auto& ctx = gfx_mlir_context();
        auto module = build_linear_binary_module(ctx, elem_type, op_key, meta_rank);
        KernelSource src = make_kernel_source_from_mlir(module, "linear_binary", /*arg_count=*/4);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_linear_binary_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_linear_binary_kernel, "GFX Vulkan binary chunked: kernel compile failed: ", log);
        m_linear_binary_kernel->prepare_runtime_artifacts();
        m_linear_binary_elem_type = elem_type;
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

    const ov::Shape& dispatch_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
    const uint32_t total = static_cast<uint32_t>(tensor_elements(dispatch_shape));
    for (uint32_t offset = 0; offset < total; offset += kLinearChunkElemsPerDispatch) {
        const uint32_t count = std::min<uint32_t>(kLinearChunkElemsPerDispatch, total - offset);
        struct LinearChunkParams {
            uint32_t offset;
            uint32_t count;
        } params{offset, count};
        std::vector<KernelArg> args;
        size_t tg = 1;
        if (!m_linear_binary_launch_abi.valid) {
            tg = std::min<size_t>(count,
                                  std::max<size_t>(1, m_linear_binary_kernel->clamp_threadgroup_size(64)));
            args = {
                make_buffer_arg(0, input0->buf),
                make_buffer_arg(1, input1->buf),
                make_bytes_arg(2, &params, sizeof(params)),
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
                        args.push_back(make_bytes_arg(static_cast<uint32_t>(args.size()), &params, sizeof(params)));
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
    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan same-shape binary: unsupported element type ",
                    elem_type);

    if (!m_linear_binary_same_shape_kernel || m_linear_binary_same_shape_elem_type != elem_type ||
        m_linear_binary_same_shape_key != op_key) {
        auto& ctx = gfx_mlir_context();
        auto module = build_linear_binary_same_shape_module(ctx, elem_type, op_key);
        KernelSource src = make_kernel_source_from_mlir(module, "linear_binary_same_shape", /*arg_count=*/3);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_linear_binary_same_shape_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_linear_binary_same_shape_kernel,
                        "GFX Vulkan same-shape binary: kernel compile failed: ",
                        log);
        m_linear_binary_same_shape_kernel->prepare_runtime_artifacts();
        m_linear_binary_same_shape_elem_type = elem_type;
        m_linear_binary_same_shape_key = op_key;
    }

    const ov::Shape& out_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
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
    const uint32_t total = static_cast<uint32_t>(tensor_elements(out_shape));
    const uint64_t work_per_elem = static_cast<uint64_t>(in_shape.at(1)) *
                                   static_cast<uint64_t>(w_shape.at(2)) *
                                   static_cast<uint64_t>(w_shape.at(3));
    const auto caps = query_parallelism_caps(m_buffer_manager);
    const auto chunk_plan = select_chunk_dispatch_plan(caps, "conv2d", total, work_per_elem);
    if (!m_conv2d_chunk_kernel || m_conv2d_chunk_elem_type != elem_type ||
        m_conv2d_chunk_threads_per_group != chunk_plan.threads_per_group) {
        auto& ctx = gfx_mlir_context();
        auto module = build_conv2d_chunk_module(ctx, elem_type, chunk_plan.threads_per_group);
        if (gfx_log_debug_enabled()) {
            std::string module_text;
            llvm::raw_string_ostream os(module_text);
            module.print(os);
            gfx_log_debug("VulkanExec") << "conv2d_chunk module:\n" << module_text;
        }
        KernelSource src = make_kernel_source_from_mlir(module, "conv2d_chunk", /*arg_count=*/4);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_conv2d_chunk_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_conv2d_chunk_kernel, "GFX Vulkan conv2d chunked: kernel compile failed: ", log);
        m_conv2d_chunk_kernel->prepare_runtime_artifacts();
        m_conv2d_chunk_elem_type = elem_type;
        m_conv2d_chunk_threads_per_group = chunk_plan.threads_per_group;
        m_conv2d_chunk_launch_abi = extract_launch_operand_abi(module);
    }

    struct Conv2DChunkParams {
        uint32_t offset;
        uint32_t count;
    };

    const uint32_t elems_per_dispatch =
        std::max<uint32_t>(kConv2DChunkElemsPerDispatch, chunk_plan.elems_per_dispatch);
    for (uint32_t offset = 0; offset < total; offset += elems_per_dispatch) {
        Conv2DChunkParams params{};
        params.offset = offset;
        params.count = std::min<uint32_t>(elems_per_dispatch, total - offset);
        const size_t tg = std::min<size_t>(params.count,
                                           std::max<size_t>(1,
                                                            m_conv2d_chunk_kernel->clamp_threadgroup_size(
                                                                chunk_plan.threads_per_group)));
        KernelDispatch dispatch = make_1d_dispatch(params.count, tg);
        std::vector<KernelArg> args;
        if (!m_conv2d_chunk_launch_abi.valid) {
            args = {
                make_buffer_arg(0, input0->buf),
                make_buffer_arg(1, input1->buf),
                make_bytes_arg(2, &params, sizeof(params)),
                make_buffer_arg(3, output->buf),
            };
        } else {
            const int32_t dynamic_scalars[] = {static_cast<int32_t>(params.count), static_cast<int32_t>(params.offset)};
            args.reserve(m_conv2d_chunk_launch_abi.kinds.size());
            size_t scalar_idx = 0;
            size_t dynamic_idx = 0;
            for (size_t i = 0; i < m_conv2d_chunk_launch_abi.kinds.size(); ++i) {
                if (m_conv2d_chunk_launch_abi.kinds[i] == 1) {
                    const int32_t arg_idx = m_conv2d_chunk_launch_abi.arg_indices[i];
                    if (arg_idx == 0) {
                        args.push_back(make_buffer_arg(static_cast<uint32_t>(args.size()), input0->buf));
                    } else if (arg_idx == 1) {
                        args.push_back(make_buffer_arg(static_cast<uint32_t>(args.size()), input1->buf));
                    } else if (arg_idx == 2) {
                        args.push_back(make_bytes_arg(static_cast<uint32_t>(args.size()), &params, sizeof(params)));
                    } else if (arg_idx == 3) {
                        args.push_back(make_buffer_arg(static_cast<uint32_t>(args.size()), output->buf));
                    } else {
                        OPENVINO_THROW("GFX Vulkan conv2d chunked: unsupported memref arg index ", arg_idx);
                    }
                    continue;
                }
                OPENVINO_ASSERT(scalar_idx < m_conv2d_chunk_launch_abi.scalar_values.size(),
                                "GFX Vulkan conv2d chunked: scalar ABI mismatch");
                int32_t scalar = m_conv2d_chunk_launch_abi.scalar_values[scalar_idx];
                if (scalar_idx < m_conv2d_chunk_launch_abi.scalar_known.size() &&
                    !m_conv2d_chunk_launch_abi.scalar_known[scalar_idx]) {
                    OPENVINO_ASSERT(dynamic_idx < std::size(dynamic_scalars),
                                    "GFX Vulkan conv2d chunked: too many dynamic scalars");
                    scalar = dynamic_scalars[dynamic_idx++];
                }
                args.push_back(make_bytes_arg(static_cast<uint32_t>(args.size()), &scalar, sizeof(int32_t)));
                ++scalar_idx;
            }
        }
        auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
        // Record chunked Conv2D into the caller-owned infer command buffer so
        // the Vulkan infer path can batch submits across stages/chunks instead
        // of forcing a queue submit/wait for every dispatch.
        m_conv2d_chunk_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
    }
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

    OPENVINO_ASSERT(m_node && m_node->get_input_partial_shape(0).is_static() &&
                        m_node->get_output_partial_shape(0).is_static(),
                    "GFX Vulkan convert: static shapes required");

    auto to_mlir_type = [&](const ov::element::Type& et) -> Type {
        if (et == ov::element::u8) {
            return IntegerType::get(&ctx, 8);
        }
        if (et == ov::element::f16) {
            return Float16Type::get(&ctx);
        }
        if (et == ov::element::f32) {
            return Float32Type::get(&ctx);
        }
        OPENVINO_THROW("GFX Vulkan convert: unsupported element type ", et);
    };
    Type src_ty = to_mlir_type(src_et);
    Type dst_ty = to_mlir_type(dst_et);

    const int64_t total = static_cast<int64_t>(tensor_elements(m_node->get_output_shape(0)));
    auto in_ty = MemRefType::get({total}, src_ty);
    auto out_ty = MemRefType::get({total}, dst_ty);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    mod->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), b.getUnitAttr());
    mod->setAttr("gfx.parallel_dispatch", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.prefer_parallel", BoolAttr::get(&ctx, true));
    mod->setAttr("gfx.fixed_arg_count", IntegerAttr::get(IntegerType::get(&ctx, 32), 2));
    auto make_i32_array_attr = [&](std::initializer_list<int32_t> values) {
        SmallVector<Attribute, 8> attrs;
        for (int32_t value : values) {
            attrs.push_back(b.getI32IntegerAttr(value));
        }
        return b.getArrayAttr(attrs);
    };
    mod->setAttr("gfx.kernel_operand_kinds", make_i32_array_attr({1, 1}));
    mod->setAttr("gfx.kernel_operand_arg_indices", make_i32_array_attr({0, 1}));

    auto gpu_mod = b.create<gpu::GPUModuleOp>(UnknownLoc::get(&ctx), "gfx_kernels");
    OpBuilder gpu_builder = OpBuilder::atBlockBegin(gpu_mod.getBody());
    auto fn = gpu_builder.create<gpu::GPUFuncOp>(UnknownLoc::get(&ctx),
                                                 "convert_linear",
                                                 b.getFunctionType(TypeRange{in_ty, out_ty}, {}),
                                                 TypeRange{},
                                                 TypeRange{});
    fn->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), b.getUnitAttr());
    fn.setKnownBlockSizeAttr(DenseI32ArrayAttr::get(&ctx, {64, 1, 1}));

    auto* entry = &fn.getBody().front();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();
    Value total_val = body.create<arith::ConstantIndexOp>(loc, total);
    Value bid = body.create<gpu::BlockIdOp>(loc, gpu::Dimension::x);
    Value bdim = body.create<gpu::BlockDimOp>(loc, gpu::Dimension::x);
    Value tid = body.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value idx = body.create<arith::AddIOp>(loc, body.create<arith::MulIOp>(loc, bid, bdim), tid);
    auto active = body.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, idx, total_val);
    auto active_if = body.create<scf::IfOp>(loc, active, false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value src = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{idx});
        Value dst = src;
        if (src_ty != dst_ty) {
            if (src_et == ov::element::u8 && (dst_et == ov::element::f16 || dst_et == ov::element::f32)) {
                dst = then_builder.create<arith::UIToFPOp>(loc, dst_ty, src);
            } else if (src_et == ov::element::f16 && dst_et == ov::element::f32) {
                dst = then_builder.create<arith::ExtFOp>(loc, dst_ty, src);
            } else if (src_et == ov::element::f32 && dst_et == ov::element::f16) {
                dst = then_builder.create<arith::TruncFOp>(loc, dst_ty, src);
            } else {
                OPENVINO_THROW("GFX Vulkan convert: unsupported conversion ", src_et, " -> ", dst_et);
            }
        }
        then_builder.create<memref::StoreOp>(loc, dst, fn.getArgument(1), ValueRange{idx});
    }
    body.setInsertionPointAfter(active_if);
    body.create<gpu::ReturnOp>(loc);
    return mod;
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

    size_t axis_offset = 0;
    for (size_t i = 0; i < concat->get_input_size(); ++i) {
        GpuTensor* src = resolve_input(i);
        OPENVINO_ASSERT(src && src->buf.valid(), "GFX Vulkan Concat: missing input buffer ", i);
        ov::Shape src_shape = src->shape;
        if (src_shape.empty() && m_node->get_input_partial_shape(i).is_static()) {
            src_shape = m_node->get_input_shape(i);
        }
        OPENVINO_ASSERT(src_shape.size() == rank, "GFX Vulkan Concat: rank mismatch at input ", i);
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
        KernelSource src = make_kernel_source_from_mlir(module, "convert_linear", /*arg_count=*/2);
        src.signature.output_arg_count = 1;
        VulkanCodegenBackend backend;
        std::string log;
        m_convert_linear_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_convert_linear_kernel, "GFX Vulkan Convert: kernel compile failed: ", log);
        m_convert_linear_kernel->prepare_runtime_artifacts();
        m_convert_src_elem_type = src_et;
        m_convert_dst_elem_type = dst_et;
    }

    output->shape = m_node->get_output_shape(0);
    output->expected_type = dst_et;
    const uint32_t total = static_cast<uint32_t>(tensor_elements(output->shape));
    std::vector<KernelArg> args{
        make_buffer_arg(0, m_inputs[0]->buf),
        make_buffer_arg(1, output->buf),
    };
    const uint32_t tg = m_convert_linear_kernel->clamp_threadgroup_size(64);
    KernelDispatch dispatch = make_1d_dispatch(total, tg);
    m_convert_linear_kernel->execute(command_buffer, dispatch, args, nullptr);
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
