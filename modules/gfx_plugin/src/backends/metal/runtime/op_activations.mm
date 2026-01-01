// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_activations.hpp"

#include <cmath>
#include <limits>

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/parameter.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "runtime/gfx_logger.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir_codegen/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
}  // namespace

MetalActivationOp::MetalActivationOp(const std::shared_ptr<const ov::Node>& node,
                                     ActivationKind kind,
                                     float alpha,
                                     double clamp_min,
                                     double clamp_max,
                                     void* device,
                                     void* queue)
    : MetalOp(node->get_friendly_name(),
              "Activation",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_kind(kind),
      m_alpha(alpha),
      m_clamp_min(clamp_min),
      m_clamp_max(clamp_max),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    OPENVINO_ASSERT(node->get_input_size() == 1, "Activation expects single input");
    m_element_type = node->get_output_element_type(0);
    const bool is_float = (m_element_type == ov::element::f32 || m_element_type == ov::element::f16);
    const bool is_int = m_element_type.is_integral_number();
    const bool is_bool = (m_element_type == ov::element::boolean);
    if (m_kind == ActivationKind::Sign || m_kind == ActivationKind::Abs) {
        OPENVINO_ASSERT(is_float || is_int, "Activation supports only f16/f32 or integer for Sign/Abs");
    } else if (m_kind == ActivationKind::LogicalNot) {
        OPENVINO_ASSERT(is_int || is_bool, "LogicalNot supports only integer/bool types");
    } else {
        OPENVINO_ASSERT(is_float, "Activation supports only f16/f32");
    }
}

void MetalActivationOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalActivationOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }

    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    std::optional<std::pair<double, double>> clamp_range;
    if (m_kind == ActivationKind::Clamp) {
        clamp_range = std::make_pair(m_clamp_min, m_clamp_max);
    }
    auto module = build_mlir_unary_from_node(m_node, ctx, m_kind, m_alpha, clamp_range);
    UnaryCodegenDesc desc;
    desc.activation = m_kind;
    desc.alpha = m_alpha;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    desc.clamp_min = m_clamp_min;
    desc.clamp_max = m_clamp_max;
    auto source = generate_msl_from_mlir(module, desc);

    KernelSpec spec(m_node, 3u);
    m_kernel = compile_msl_kernel(backend, spec, module, "unary_kernel", source, &log);
    OPENVINO_ASSERT(m_kernel, "MetalActivationOp: failed to compile unary kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalActivationOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    if (!is_compiled()) {
        compile(buffer_manager());
    }
    OPENVINO_ASSERT(!inputs().empty(), "Activation: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "Activation: input buffer is null");
    MetalTensor& dst = require_output();

    ov::Shape out_shape = !output_shape().empty() ? output_shape() : src->shape;
    OPENVINO_ASSERT(!out_shape.empty(), "Activation: output shape unknown");
    if (!src->shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(src->shape == m_node->get_input_shape(0),
                        "Activation: runtime input shape mismatch");
    }
    const size_t bytes = m_element_type.size() * ov::shape_size(out_shape);
    const size_t src_bytes = m_element_type.size() * ov::shape_size(src->shape.empty() ? out_shape : src->shape);
    OPENVINO_ASSERT(src->buf.size >= src_bytes, "Activation: input buffer too small");
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        // Keep output device-only; CPU-visible outputs are not supported.
        dst.buf = allocate_temp_buffer(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
        dst.expected_type = m_element_type;
    }

    const NSUInteger elems = static_cast<NSUInteger>(shape_size(out_shape));
    OPENVINO_ASSERT(elems <= std::numeric_limits<uint32_t>::max(),
                    "Activation: element count exceeds uint32 range");
    const uint32_t num_elems = static_cast<uint32_t>(elems);
    if (num_elems == 0) {
        return;
    }

    KernelDispatch dispatch = make_1d_dispatch(elems, m_kernel->clamp_threadgroup_size(64));
    std::vector<KernelArg> args;
    args.reserve(3);
    append_kernel_input_args(args, 1, [&](size_t) { return src; }, name().c_str());
    append_kernel_output_args(args, static_cast<uint32_t>(args.size()), &dst, name().c_str());
    args.push_back(make_bytes_arg(static_cast<uint32_t>(args.size()), &num_elems, sizeof(num_elems)));
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);

    dst.shape = out_shape;
    dst.expected_type = m_element_type;
}

}  // namespace gfx_plugin
}  // namespace ov
