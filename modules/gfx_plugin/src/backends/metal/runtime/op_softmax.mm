// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_softmax.hpp"

#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "backends/metal/runtime/metal_backend.hpp"
#include "runtime/gfx_logger.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "mlir_builder.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/codegen/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
struct SoftmaxDims {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t inner = 1;
};

SoftmaxDims flatten_softmax_dims(const ov::Shape& shape, int64_t axis_in) {
    SoftmaxDims d{};
    if (shape.empty())
        return d;
    int64_t axis = axis_in;
    if (axis < 0)
        axis += static_cast<int64_t>(shape.size());
    OPENVINO_ASSERT(axis >= 0 && axis < static_cast<int64_t>(shape.size()), "Softmax: axis out of range");
    uint64_t outer = 1;
    for (int64_t i = 0; i < axis; ++i)
        outer *= shape[static_cast<size_t>(i)];
    uint64_t inner = 1;
    for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i)
        inner *= shape[i];
    uint64_t cols = shape[static_cast<size_t>(axis)];
    d.rows = static_cast<uint32_t>(outer * inner);
    d.cols = static_cast<uint32_t>(cols);
    d.inner = static_cast<uint32_t>(inner);
    return d;
}
}  // namespace

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
        auto d = flatten_softmax_dims(node->get_output_shape(0), m_axis);
        m_desc.rows = d.rows;
        m_desc.cols = d.cols;
        m_desc.inner = d.inner;
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
    auto dims = flatten_softmax_dims(in_shape, m_axis);
    OPENVINO_ASSERT(dims.rows > 0 && dims.cols > 0, "Softmax: invalid flattened dims");

    // Recompile kernel on-the-fly if runtime dims differ from the specialized pipeline.
    if (m_desc.rows != static_cast<int64_t>(dims.rows) ||
        m_desc.cols != static_cast<int64_t>(dims.cols) ||
        m_desc.inner != static_cast<int64_t>(dims.inner) ||
        !m_kernel) {
        m_desc.rows = dims.rows;
        m_desc.cols = dims.cols;
        m_desc.inner = dims.inner;
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
        dst.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
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
    } params{dims.rows, dims.cols, dims.inner == 0 ? 1u : dims.inner};

    const NSUInteger total = static_cast<NSUInteger>(params.rows) * static_cast<NSUInteger>(params.cols);
    if (total == 0) {
        return;
    }
    const NSUInteger threads_per_tg = 64;
    KernelDispatch dispatch = make_1d_dispatch(total, m_kernel->clamp_threadgroup_size(threads_per_tg));

    std::vector<KernelArg> args;
    args.reserve(3);
    args.push_back(make_buffer_arg(0, src->buf));
    args.push_back(make_buffer_arg(1, dst.buf));
    args.push_back(make_bytes_arg(2, &params, sizeof(params)));
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
}

}  // namespace gfx_plugin
}  // namespace ov
