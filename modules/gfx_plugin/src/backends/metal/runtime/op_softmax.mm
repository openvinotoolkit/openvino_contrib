// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_softmax.hpp"

#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "runtime/gfx_logger.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir_codegen/codegen_common.hpp"
#include "runtime/gfx_shape_utils.hpp"

namespace ov {
namespace gfx_plugin {

MetalSoftmaxOp::MetalSoftmaxOp(const std::shared_ptr<const ov::Node>& node,
                               void* device,
                               void* queue,
                               bool log_softmax)
    : MetalOp(node->get_friendly_name(),
              log_softmax ? "LogSoftmax" : "Softmax",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue),
      m_node(node),
      m_log_softmax(log_softmax) {
    OPENVINO_ASSERT(node->get_input_size() == 1, "Softmax expects single input");
    if (auto s1 = std::dynamic_pointer_cast<const ov::op::v1::Softmax>(node)) {
        m_axis = s1->get_axis();
    } else if (auto s8 = std::dynamic_pointer_cast<const ov::op::v8::Softmax>(node)) {
        m_axis = s8->get_axis();
    } else if (auto ls = std::dynamic_pointer_cast<const ov::op::v5::LogSoftmax>(node)) {
        m_axis = ls->get_axis();
    } else {
        OPENVINO_THROW("MetalSoftmaxOp: unsupported softmax version");
    }
    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f32 || m_element_type == ov::element::f16,
                    "Softmax supports only f16/f32");
    if (node->get_output_partial_shape(0).is_static()) {
        auto dims = compute_softmax_dims(node->get_output_shape(0), m_axis, "Softmax");
        m_desc.rows = static_cast<int64_t>(dims.rows);
        m_desc.cols = static_cast<int64_t>(dims.axis_len);
        m_desc.inner = static_cast<int64_t>(dims.inner);
    }
}

void MetalSoftmaxOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalSoftmaxOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    if (!m_device && buffer_manager) {
        m_device = (id<MTLDevice>)buffer_manager->device();
    }
    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto model = make_single_op_model(m_node);
    auto module = m_log_softmax ? build_mlir_logsoftmax_from_model(model, ctx)
                                : build_mlir_softmax_from_model(model, ctx);
    SoftmaxCodegenDesc desc{};
    desc.rows = m_desc.rows;
    desc.cols = m_desc.cols;
    desc.inner = m_desc.inner == 0 ? 1 : m_desc.inner;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    desc.log_softmax = m_log_softmax;
    auto source = generate_msl_from_mlir(module, desc);

    KernelSpec spec(m_node, 3u);
    m_kernel = compile_msl_kernel(backend, spec, module, "softmax_kernel", source, &log);
    OPENVINO_ASSERT(m_kernel, "MetalSoftmaxOp: failed to compile softmax kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalSoftmaxOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "Softmax: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "Softmax: input buffer is null");
    MetalTensor& dst = require_output();

    ov::Shape in_shape = src->shape.empty() ? output_shape() : src->shape;
    OPENVINO_ASSERT(!in_shape.empty(), "Softmax: input shape unknown");
    auto dims = compute_softmax_dims(in_shape, m_axis, "Softmax");
    OPENVINO_ASSERT(dims.rows > 0 && dims.axis_len > 0, "Softmax: invalid flattened dims");

    // Recompile kernel on-the-fly if runtime dims differ from the specialized pipeline.
    if (m_desc.rows != static_cast<int64_t>(dims.rows) ||
        m_desc.cols != static_cast<int64_t>(dims.axis_len) ||
        m_desc.inner != static_cast<int64_t>(dims.inner) ||
        !m_kernel) {
        m_desc.rows = static_cast<int64_t>(dims.rows);
        m_desc.cols = static_cast<int64_t>(dims.axis_len);
        m_desc.inner = static_cast<int64_t>(dims.inner);
        MetalCodegenBackend backend(m_device);
        std::string log;
        mlir::MLIRContext ctx;
        auto model = make_single_op_model(m_node);
        auto module = m_log_softmax ? build_mlir_logsoftmax_from_model(model, ctx)
                                    : build_mlir_softmax_from_model(model, ctx);
        SoftmaxCodegenDesc desc{};
        desc.rows = m_desc.rows;
        desc.cols = m_desc.cols;
        desc.inner = m_desc.inner == 0 ? 1 : m_desc.inner;
        desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
        desc.log_softmax = m_log_softmax;
        auto source = generate_msl_from_mlir(module, desc);
        KernelSpec spec(m_node, 3u);
        m_kernel = compile_msl_kernel(backend, spec, module, "softmax_kernel", source, &log);
        OPENVINO_ASSERT(m_kernel, "MetalSoftmaxOp: failed to recompile softmax kernel: ", log);
    }

    // Allocate output if needed.
    const size_t bytes = m_element_type.size() * ov::shape_size(in_shape);
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = allocate_temp_buffer(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.shape = in_shape;
    dst.expected_type = m_element_type;

    OPENVINO_ASSERT(src->buf.valid(), "Softmax: src buffer is null");
    OPENVINO_ASSERT(dst.buf.valid(), "Softmax: dst buffer is null");
    const size_t need_bytes = m_element_type.size() * ov::shape_size(in_shape);
    OPENVINO_ASSERT(src->buf.size >= need_bytes, "Softmax: src buffer too small");
    OPENVINO_ASSERT(dst.buf.size >= need_bytes, "Softmax: dst buffer too small");
    struct SoftmaxParams {
        uint32_t rows;
        uint32_t cols;
        uint32_t inner;
    } params{static_cast<uint32_t>(dims.rows),
             static_cast<uint32_t>(dims.axis_len),
             static_cast<uint32_t>(dims.inner == 0 ? 1u : dims.inner)};

    const NSUInteger total = static_cast<NSUInteger>(params.rows) * static_cast<NSUInteger>(params.cols);
    if (total == 0) {
        return;
    }
    const NSUInteger threads_per_tg = 64;
    KernelDispatch dispatch = make_1d_dispatch(total, m_kernel->clamp_threadgroup_size(threads_per_tg));

    std::vector<KernelArg> args;
    args.reserve(3);
    append_kernel_input_args(args,
                             std::vector<size_t>{0},
                             [&](size_t idx) { return inputs()[idx]; },
                             name().c_str());
    append_kernel_output_args(args,
                              static_cast<uint32_t>(args.size()),
                              std::vector<GpuTensor*>{&dst},
                              name().c_str());
    args.push_back(make_bytes_arg(static_cast<uint32_t>(args.size()), &params, sizeof(params)));
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
}

}  // namespace gfx_plugin
}  // namespace ov
