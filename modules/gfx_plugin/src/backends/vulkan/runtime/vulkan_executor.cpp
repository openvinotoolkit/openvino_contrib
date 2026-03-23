// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backends/vulkan/runtime/vulkan_executor.hpp"

#include "backends/vulkan/codegen/vulkan_codegen_backend.hpp"
#include "backends/vulkan/runtime/profiling/profiler.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "kernel_ir/gfx_kernel_plan.hpp"
#include "mlir/gfx_mlir_kernel_metadata.hpp"
#include "mlir/gfx_mlir_type_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "runtime/gfx_logger.hpp"
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
}

void VulkanStage::execute(GpuCommandBufferHandle command_buffer) {
    // Keep Split on a compact per-output Vulkan kernel path. The generic
    // tensor-result lowering is correct on Metal but still unstable on
    // mobile Vulkan drivers.
    if (m_type == "Split" || m_type == "VariadicSplit") {
        execute_split_chunked(command_buffer);
        return;
    }
    if (m_type == "Concat") {
        execute_concat_chunked(command_buffer);
        return;
    }
    if (should_use_softmax_chunked()) {
        execute_softmax_chunked(command_buffer);
        return;
    }
    if (should_use_conv2d_chunked()) {
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
                                    << " inputs=" << m_inputs.size()
                                    << " extras=" << m_kernel_extra_inputs.size()
                                    << " out_rank=" << dispatch_shape.size()
                                    << " out_elems=" << (dispatch_shape.empty() ? 0 : tensor_elements(dispatch_shape))
                                    << " elem_type=" << elem_type;
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
    const ov::element::Type et = resolve_stage_element_type(m_node, output);
    return is_supported_linear_elem_type(et) &&
           tensor_elements(dispatch_shape) >= kLargeLinearChunkElems;
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
    const ov::element::Type et = resolve_stage_element_type(m_node, output);
    return is_supported_linear_elem_type(et) &&
           tensor_elements(dispatch_shape) >= kLargeLinearChunkElems;
}

bool VulkanStage::should_use_conv2d_chunked() const {
    if (m_type != "Convolution" || m_has_bias || m_has_activation || m_has_bn) {
        return false;
    }
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    if (!conv || conv->get_input_size() != 2 || conv->get_output_size() != 1) {
        return false;
    }
    const auto& out_shape = conv->get_output_shape(0);
    if (out_shape.size() != 4) {
        return false;
    }
    const auto elem_type = resolve_stage_element_type(m_node, !m_outputs.empty() ? m_outputs.front() : m_output);
    if (!is_supported_linear_elem_type(elem_type)) {
        return false;
    }
    const uint64_t out_elems = tensor_elements(out_shape);
    const uint64_t macs = out_elems *
                          static_cast<uint64_t>(conv->get_input_shape(0).at(1)) *
                          static_cast<uint64_t>(conv->get_input_shape(1).at(2)) *
                          static_cast<uint64_t>(conv->get_input_shape(1).at(3));
    return out_elems >= kLargeLinearChunkElems || macs >= (1ull << 20);
}

bool VulkanStage::should_use_binary_chunked() const {
    if (binary_chunk_key(m_type).empty()) {
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
    GpuTensor* input1 = resolve_input(1);
    if (!input0 || !input1 || !output) {
        return false;
    }
    const ov::Shape& dispatch_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
    if (dispatch_shape.empty()) {
        return false;
    }
    if (m_kernel_extra_inputs.size() < 3) {
        return false;
    }
    const ov::element::Type et = resolve_stage_element_type(m_node, output);
    return is_supported_linear_elem_type(et) &&
           tensor_elements(dispatch_shape) >= kLargeLinearChunkElems;
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
    mod->setAttr("gfx.dispatch_threads_h", IntegerAttr::get(IntegerType::get(&ctx, 32), 1));
    mod->setAttr("gfx.dispatch_threads_w", IntegerAttr::get(IntegerType::get(&ctx, 32), 1));
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
    auto io_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto param_ty = MemRefType::get({2}, IntegerType::get(&ctx, 32));
    auto fn = b.create<func::FuncOp>(UnknownLoc::get(&ctx),
                                     "linear_unary",
                                     b.getFunctionType(TypeRange{io_ty, param_ty, io_ty}, {}));
    auto* entry = fn.addEntryBlock();
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
    Value chunk_elems =
        body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(kLinearChunkElemsPerDispatch));
    auto par_out = body.create<scf::ParallelOp>(loc,
                                                ValueRange{c0},
                                                ValueRange{chunk_elems},
                                                ValueRange{c1});
    auto loop = OpBuilder::atBlockBegin(par_out.getBody());
    Value local_idx = par_out.getInductionVars()[0];
    auto active =
        loop.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local_idx, count);
    auto active_if = loop.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
        Value idx = then_builder.create<arith::AddIOp>(loc, offset, local_idx);
        Value val = then_builder.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{idx});
        Value out = emit_unary(then_builder, val);
        then_builder.create<memref::StoreOp>(loc, out, fn.getArgument(2), ValueRange{idx});
    }
    body.setInsertionPointAfter(par_out);
    body.create<func::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_conv2d_chunk_module(mlir::MLIRContext& ctx,
                                                      const ov::element::Type& et) {
    using namespace mlir;
    auto conv = ov::as_type_ptr<const ov::op::v1::Convolution>(m_node);
    OPENVINO_ASSERT(conv, "GFX Vulkan conv2d chunked: node cast failed");
    const auto& in_shape = conv->get_input_shape(0);
    const auto& w_shape = conv->get_input_shape(1);
    const auto& out_shape = conv->get_output_shape(0);

    ctx.loadDialect<func::FuncDialect,
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
    mod->setAttr("gfx.dispatch_threads_h", IntegerAttr::get(IntegerType::get(&ctx, 32), 1));
    mod->setAttr("gfx.dispatch_threads_w", IntegerAttr::get(IntegerType::get(&ctx, 32), 1));
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
    auto fn = b.create<func::FuncOp>(UnknownLoc::get(&ctx),
                                     "conv2d_chunk",
                                     b.getFunctionType(TypeRange{input_ty, weight_ty, param_ty, output_ty}, {}));
    auto* entry = fn.addEntryBlock();
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

    Value chunk_elems =
        body.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(kConv2DChunkElemsPerDispatch));
    auto par_out = body.create<scf::ParallelOp>(loc,
                                                ValueRange{c0},
                                                ValueRange{chunk_elems},
                                                ValueRange{c1});
    auto b_out = OpBuilder::atBlockBegin(par_out.getBody());
    Value local_idx = par_out.getInductionVars()[0];
    auto active =
        b_out.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, local_idx, count);
    auto active_if = b_out.create<scf::IfOp>(loc, active, /*withElseRegion=*/false);
    {
        auto then_builder = active_if.getThenBodyBuilder();
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
    body.setInsertionPointAfter(par_out);
    body.create<func::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_linear_binary_module(mlir::MLIRContext& ctx,
                                                       const ov::element::Type& et,
                                                       const std::string& op_key,
                                                       size_t meta_rank) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect,
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

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());
    auto io_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto param_ty = MemRefType::get({2}, IntegerType::get(&ctx, 32));
    OPENVINO_ASSERT(meta_rank != 0, "GFX Vulkan binary chunked: metadata rank must be positive");
    auto meta_ty = MemRefType::get({static_cast<int64_t>(meta_rank)}, IntegerType::get(&ctx, 32));
    SmallVector<Type, 7> arg_types{io_ty, io_ty, param_ty, meta_ty, meta_ty, meta_ty, io_ty};
    auto fn = b.create<func::FuncOp>(UnknownLoc::get(&ctx), "linear_binary", b.getFunctionType(arg_types, {}));
    auto* entry = fn.addEntryBlock();
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
    auto for_op = body.create<scf::ForOp>(loc, c0, count, c1);
    auto loop = OpBuilder::atBlockBegin(for_op.getBody());
    Value linear_idx = loop.create<arith::AddIOp>(loc, offset, for_op.getInductionVar());
    auto for_dims = loop.create<scf::ForOp>(loc,
                                            c0,
                                            rank,
                                            c1,
                                            ValueRange{linear_idx, c0, c0});
    auto dim_builder = OpBuilder::atBlockBegin(for_dims.getBody());
    Value rev_dim = dim_builder.create<arith::SubIOp>(loc, rank, c1);
    rev_dim = dim_builder.create<arith::SubIOp>(loc, rev_dim, for_dims.getInductionVar());
    auto load_meta = [&](Value meta, Value index) -> Value {
        auto loaded = dim_builder.create<memref::LoadOp>(loc, meta, ValueRange{index});
        return dim_builder.create<arith::IndexCastOp>(loc, dim_builder.getIndexType(), loaded);
    };
    Value dim_size = load_meta(fn.getArgument(3), rev_dim);
    Value coord = dim_builder.create<arith::RemUIOp>(loc, for_dims.getRegionIterArgs()[0], dim_size);
    Value next_rem = dim_builder.create<arith::DivUIOp>(loc, for_dims.getRegionIterArgs()[0], dim_size);
    Value stride0 = load_meta(fn.getArgument(4), rev_dim);
    Value stride1 = load_meta(fn.getArgument(5), rev_dim);
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

    Value lhs = loop.create<memref::LoadOp>(loc, fn.getArgument(0), ValueRange{for_dims.getResult(1)});
    Value rhs = loop.create<memref::LoadOp>(loc, fn.getArgument(1), ValueRange{for_dims.getResult(2)});
    Value out = emit_binary(loop, lhs, rhs);
    OpBuilder loop_end(for_op.getBody(), for_op.getBody()->getTerminator()->getIterator());
    loop_end.create<memref::StoreOp>(loc, out, fn.getArgument(6), ValueRange{linear_idx});
    body.setInsertionPointAfter(for_op);
    body.create<func::ReturnOp>(loc);
    return mod;
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
        VulkanCodegenBackend backend;
        std::string log;
        m_linear_unary_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_linear_unary_kernel, "GFX Vulkan unary chunked: kernel compile failed: ", log);
        m_linear_unary_elem_type = elem_type;
        m_linear_unary_key = op_key;
        m_linear_unary_launch_abi = extract_launch_operand_abi(module);
        m_linear_unary_scalar_args = extract_kernel_scalar_values(module);
    }

    const ov::Shape& dispatch_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
    const uint32_t total = static_cast<uint32_t>(tensor_elements(dispatch_shape));
    for (uint32_t offset = 0; offset < total; offset += kLinearChunkElemsPerDispatch) {
        const uint32_t count = std::min<uint32_t>(kLinearChunkElemsPerDispatch, total - offset);
        struct LinearChunkParams {
            uint32_t offset;
            uint32_t count;
        } params{offset, count};
        KernelDispatch dispatch = make_1d_dispatch(count, 1);
        OPENVINO_ASSERT(m_linear_unary_launch_abi.valid,
                        "GFX Vulkan unary chunked: missing launch ABI");
        const int32_t dynamic_scalars[] = {static_cast<int32_t>(count), static_cast<int32_t>(offset)};
        std::vector<KernelArg> args;
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
    OPENVINO_ASSERT(m_kernel_extra_inputs.size() >= 3,
                    "GFX Vulkan binary chunked: missing broadcast metadata");
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
    if (!m_linear_binary_kernel || m_linear_binary_elem_type != elem_type ||
        m_linear_binary_key != op_key || m_linear_binary_rank != meta_rank) {
        auto& ctx = gfx_mlir_context();
        auto module = build_linear_binary_module(ctx, elem_type, op_key, meta_rank);
        KernelSource src = make_kernel_source_from_mlir(module, "linear_binary", /*arg_count=*/7);
        VulkanCodegenBackend backend;
        std::string log;
        m_linear_binary_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_linear_binary_kernel, "GFX Vulkan binary chunked: kernel compile failed: ", log);
        m_linear_binary_elem_type = elem_type;
        m_linear_binary_key = op_key;
        m_linear_binary_rank = meta_rank;
    }

    const ov::Shape& dispatch_shape = !m_output_shape.empty() ? m_output_shape : output->shape;
    const uint32_t total = static_cast<uint32_t>(tensor_elements(dispatch_shape));
    KernelDispatch dispatch = make_1d_dispatch(1, m_linear_binary_kernel->clamp_threadgroup_size(1));
    for (uint32_t offset = 0; offset < total; offset += kLinearChunkElemsPerDispatch) {
        const uint32_t count = std::min<uint32_t>(kLinearChunkElemsPerDispatch, total - offset);
        struct LinearChunkParams {
            uint32_t offset;
            uint32_t count;
        } params{offset, count};
        std::vector<KernelArg> args{
            make_buffer_arg(0, input0->buf),
            make_buffer_arg(1, input1->buf),
            make_bytes_arg(2, &params, sizeof(params)),
            make_buffer_arg(3, m_kernel_extra_inputs[0].buf),
            make_buffer_arg(4, m_kernel_extra_inputs[1].buf),
            make_buffer_arg(5, m_kernel_extra_inputs[2].buf),
            make_buffer_arg(6, output->buf),
        };
        auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
        m_linear_binary_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
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

    const ov::element::Type elem_type = resolve_stage_element_type(m_node, output);
    OPENVINO_ASSERT(is_supported_linear_elem_type(elem_type),
                    "GFX Vulkan conv2d chunked: unsupported element type ",
                    elem_type);
    if (!m_conv2d_chunk_kernel || m_conv2d_chunk_elem_type != elem_type) {
        auto& ctx = gfx_mlir_context();
        auto module = build_conv2d_chunk_module(ctx, elem_type);
        KernelSource src = make_kernel_source_from_mlir(module, "conv2d_chunk", /*arg_count=*/4);
        VulkanCodegenBackend backend;
        std::string log;
        m_conv2d_chunk_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_conv2d_chunk_kernel, "GFX Vulkan conv2d chunked: kernel compile failed: ", log);
        m_conv2d_chunk_elem_type = elem_type;
        m_conv2d_chunk_launch_abi = extract_launch_operand_abi(module);
    }

    const auto& out_shape = conv->get_output_shape(0);
    struct Conv2DChunkParams {
        uint32_t offset;
        uint32_t count;
    };

    const uint32_t total = static_cast<uint32_t>(tensor_elements(out_shape));
    for (uint32_t offset = 0; offset < total; offset += kConv2DChunkElemsPerDispatch) {
        Conv2DChunkParams params{};
        params.offset = offset;
        params.count = std::min<uint32_t>(kConv2DChunkElemsPerDispatch, total - offset);
        KernelDispatch dispatch = make_1d_dispatch(kConv2DChunkElemsPerDispatch, 1);

        OPENVINO_ASSERT(m_conv2d_chunk_launch_abi.valid,
                        "GFX Vulkan conv2d chunked: missing launch ABI");
        const int32_t dynamic_scalars[] = {static_cast<int32_t>(params.count), static_cast<int32_t>(params.offset)};
        std::vector<KernelArg> args;
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
        auto bound_args = materialize_kernel_bytes_args(args, *m_buffer_manager, m_name.c_str());
        // Record chunked Conv2D into the caller-owned infer command buffer so
        // the Vulkan infer path can batch submits across stages/chunks instead
        // of forcing a queue submit/wait for every dispatch.
        m_conv2d_chunk_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
    }
}

mlir::ModuleOp VulkanStage::build_split_single_module(mlir::MLIRContext& ctx,
                                                      const ov::element::Type& et) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect, scf::SCFDialect, memref::MemRefDialect, arith::ArithDialect>();

    Type elem_ty = to_mlir_type(et, ctx, /*fallback_f32=*/true, /*allow_unsigned=*/true,
                                /*allow_small_ints=*/true, /*allow_bf16=*/false,
                                /*allow_boolean=*/false, /*signless_integers=*/true);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());

    auto in_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto param_ty = MemRefType::get({5}, IntegerType::get(&ctx, 32));
    SmallVector<Type, 3> arg_types{in_ty, param_ty, in_ty};
    auto fn = b.create<func::FuncOp>(UnknownLoc::get(&ctx), "split_single",
                                     b.getFunctionType(arg_types, {}));
    auto* entry = fn.addEntryBlock();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();

    // params: [0]=outer, [1]=axis_total, [2]=inner, [3]=axis_offset, [4]=slice_len
    auto c0 = body.create<arith::ConstantIndexOp>(loc, 0);
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
    auto for_op = body.create<scf::ForOp>(loc, c0, total, body.create<arith::ConstantIndexOp>(loc, 1));
    OpBuilder loop(&for_op.getBody()->front());
    Value idx = for_op.getInductionVar();

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
    OpBuilder loop_end(for_op.getBody(), for_op.getBody()->getTerminator()->getIterator());
    loop_end.create<memref::StoreOp>(loc, val, fn.getArgument(2), ValueRange{idx});

    body.setInsertionPointAfter(for_op);
    body.create<func::ReturnOp>(loc);
    return mod;
}

mlir::ModuleOp VulkanStage::build_concat_single_module(mlir::MLIRContext& ctx,
                                                       const ov::element::Type& et) {
    using namespace mlir;
    ctx.loadDialect<func::FuncDialect, scf::SCFDialect, memref::MemRefDialect, arith::ArithDialect>();

    Type elem_ty = to_mlir_type(et, ctx, /*fallback_f32=*/true, /*allow_unsigned=*/true,
                                /*allow_small_ints=*/true, /*allow_bf16=*/false,
                                /*allow_boolean=*/false, /*signless_integers=*/true);

    auto mod = ModuleOp::create(UnknownLoc::get(&ctx));
    OpBuilder b(mod.getBody(), mod.getBody()->begin());

    auto io_ty = MemRefType::get({ShapedType::kDynamic}, elem_ty);
    auto param_ty = MemRefType::get({5}, IntegerType::get(&ctx, 32));
    SmallVector<Type, 3> arg_types{io_ty, param_ty, io_ty};
    auto fn = b.create<func::FuncOp>(UnknownLoc::get(&ctx), "concat_single",
                                     b.getFunctionType(arg_types, {}));
    auto* entry = fn.addEntryBlock();
    OpBuilder body(entry, entry->begin());
    auto loc = fn.getLoc();

    // params: [0]=outer, [1]=axis_total, [2]=inner, [3]=axis_offset, [4]=slice_len
    auto c0 = body.create<arith::ConstantIndexOp>(loc, 0);
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
    auto for_op = body.create<scf::ForOp>(loc, c0, total, body.create<arith::ConstantIndexOp>(loc, 1));
    OpBuilder loop(&for_op.getBody()->front());
    Value idx = for_op.getInductionVar();

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
    OpBuilder loop_end(for_op.getBody(), for_op.getBody()->getTerminator()->getIterator());
    loop_end.create<memref::StoreOp>(loc, val, fn.getArgument(2), ValueRange{dst_index});

    body.setInsertionPointAfter(for_op);
    body.create<func::ReturnOp>(loc);
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
        VulkanCodegenBackend backend;
        std::string log;
        m_softmax_row_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_softmax_row_kernel, "GFX Vulkan Softmax: kernel compile failed: ", log);
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
        VulkanCodegenBackend backend;
        std::string log;
        m_concat_single_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_concat_single_kernel, "GFX Vulkan Concat: kernel compile failed: ", log);
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
        KernelDispatch dispatch = make_1d_dispatch(1, m_concat_single_kernel->clamp_threadgroup_size(1));
        m_concat_single_kernel->execute(command_buffer, dispatch, bound_args, nullptr);
        axis_offset += slice;
    }
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
    }

    // Compile compact split kernel once per element type.
    if (!m_split_single_kernel || m_split_elem_type != in.expected_type) {
        m_split_elem_type = in.expected_type;
        auto& ctx = gfx_mlir_context();
        auto module = build_split_single_module(ctx, m_split_elem_type);
        KernelSource src = make_kernel_source_from_mlir(module, "split_single", /*arg_count=*/3);
        VulkanCodegenBackend backend;
        std::string log;
        m_split_single_kernel = backend.compile(src, &log);
        OPENVINO_ASSERT(m_split_single_kernel, "GFX Vulkan Split: kernel compile failed: ", log);
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
        GpuTensor params_tensor;
        params_tensor.buf = params_buf;
        params_tensor.shape = {5};
        params_tensor.expected_type = ov::element::u8;

        std::vector<size_t> kernel_inputs{0};  // input tensor index in m_inputs
        std::vector<GpuTensor> extra{params_tensor};
        std::vector<GpuTensor*> outs{out};

        KernelArgsBundle bundle = build_kernel_args_from_metadata(
            /*operand_kinds=*/{},
            /*operand_arg_indices=*/{},
            /*scalar_args=*/{},
            kernel_inputs,
            /*kernel_input_arg_count=*/0,
            extra,
            outs,
            [&](size_t idx) { return m_inputs[idx]; },
            m_name.c_str());

        KernelDispatch dispatch = make_1d_dispatch(1, m_split_single_kernel->clamp_threadgroup_size(1));
        m_split_single_kernel->execute(command_buffer, dispatch, bundle.args, nullptr);
        offset_along_axis += slice;
    }
}

}  // namespace gfx_plugin
}  // namespace ov
