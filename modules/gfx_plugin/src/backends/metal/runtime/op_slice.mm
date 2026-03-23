// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_slice.hpp"

#include <algorithm>
#include <numeric>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "runtime/gfx_logger.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
std::vector<int64_t> get_const_i64(const std::shared_ptr<const ov::Node>& n) {
    auto c = ov::as_type_ptr<const ov::op::v0::Constant>(n);
    OPENVINO_ASSERT(c, "Slice: inputs must be Constant");
    return c->cast_vector<int64_t>();
}

std::vector<int64_t> normalize_axes(const std::vector<int64_t>& axes, size_t rank) {
    std::vector<int64_t> out;
    out.reserve(axes.size());
    for (auto a : axes) {
        int64_t ax = a;
        if (ax < 0)
            ax += static_cast<int64_t>(rank);
        OPENVINO_ASSERT(ax >= 0 && ax < static_cast<int64_t>(rank), "Slice: axis out of range");
        out.push_back(ax);
    }
    return out;
}
}  // namespace

MetalSliceOp::MetalSliceOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "Slice",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    parse_slice(node);
}

void MetalSliceOp::parse_slice(const std::shared_ptr<const ov::Node>& node) {
    auto sl = ov::as_type_ptr<const ov::op::v8::Slice>(node);
    OPENVINO_ASSERT(sl, "MetalSliceOp expects v8::Slice");

    const auto in_shape = node->get_input_shape(0);
    OPENVINO_ASSERT(!in_shape.empty(), "Slice: input shape must be static");
    const size_t rank = in_shape.size();

    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f16 || m_element_type == ov::element::f32 ||
                        m_element_type == ov::element::i32 || m_element_type == ov::element::i64,
                    "Slice: element type not supported");

    auto starts = get_const_i64(node->get_input_node_shared_ptr(1));
    auto ends = get_const_i64(node->get_input_node_shared_ptr(2));
    auto steps = get_const_i64(node->get_input_node_shared_ptr(3));

    std::vector<int64_t> axes;
    if (node->get_input_size() > 4) {
        axes = get_const_i64(node->get_input_node_shared_ptr(4));
    } else {
        axes.resize(starts.size());
        std::iota(axes.begin(), axes.end(), 0);
    }
    OPENVINO_ASSERT(starts.size() == ends.size() && starts.size() == steps.size() &&
                        starts.size() == axes.size(),
                    "Slice: starts/ends/steps/axes size mismatch");

    axes = normalize_axes(axes, rank);

    m_starts.assign(rank, 0);
    m_steps.assign(rank, 1);
    m_out_shape.assign(rank, 0);

    for (size_t i = 0; i < rank; ++i)
        m_out_shape[i] = static_cast<uint32_t>(in_shape[i]);

    for (size_t i = 0; i < axes.size(); ++i) {
        const size_t axis = static_cast<size_t>(axes[i]);
        int64_t dim = static_cast<int64_t>(in_shape[axis]);
        int64_t step = steps[i];
        OPENVINO_ASSERT(step > 0, "Slice: only positive steps supported");
        int64_t start = starts[i];
        int64_t end = ends[i];
        if (start < 0) start += dim;
        if (end < 0) end += dim;
        start = std::max<int64_t>(0, std::min<int64_t>(start, dim));
        end = std::max<int64_t>(0, std::min<int64_t>(end, dim));
        if (end < start) end = start;
        int64_t len = (end - start + step - 1) / step;
        m_starts[axis] = static_cast<int32_t>(start);
        m_steps[axis] = static_cast<uint32_t>(step);
        m_out_shape[axis] = static_cast<uint32_t>(len);
    }

    // Precompute input strides (row-major).
    m_in_stride.assign(rank, 1);
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        m_in_stride[static_cast<size_t>(i)] =
            m_in_stride[static_cast<size_t>(i + 1)] * static_cast<uint32_t>(in_shape[static_cast<size_t>(i + 1)]);
    }
}

void MetalSliceOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

void MetalSliceOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    ConvertCodegenDesc desc{};
    desc.dst_type = m_element_type;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    mlir::MLIRContext ctx;
    auto module = build_mlir_for_node(m_node, ctx);
    auto msl_desc = desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) {
        return generate_msl_for_slice_generic(msl_desc, mod);
    };

    KernelSpec spec(m_node, 8u);
    m_kernel = compile_msl_kernel(backend, spec, module, "slice_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalSliceOp: failed to compile kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalSliceOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "Slice: missing input");
    MetalTensor* src = inputs()[0];
    OPENVINO_ASSERT(src && src->buf.valid(), "Slice: input buffer null");
    MetalTensor& dst = require_output();

    ov::Shape in_shape = !src->shape.empty() ? src->shape : ov::Shape{};
    if (in_shape.empty() && m_node->get_input_partial_shape(0).is_static()) {
        in_shape = m_node->get_input_shape(0);
    }
    OPENVINO_ASSERT(!in_shape.empty(), "Slice: input shape unknown");
    if (m_node->get_input_partial_shape(0).is_static()) {
        OPENVINO_ASSERT(in_shape == m_node->get_input_shape(0),
                        "Slice: runtime input shape mismatch");
    }

    ov::Shape out_shape;
    out_shape.reserve(m_out_shape.size());
    for (auto v : m_out_shape) out_shape.push_back(static_cast<size_t>(v));

    const size_t bytes = ov::shape_size(out_shape) * m_element_type.size();
    if (!dst.buf.valid() || dst.buf.size < bytes) {
        dst.buf = allocate_temp_buffer(bytes, m_element_type, /*persistent=*/false, dst.prefer_private);
    }
    dst.shape = out_shape;
    dst.expected_type = m_element_type;

    const uint32_t rank = static_cast<uint32_t>(m_out_shape.size());
    const uint32_t total = static_cast<uint32_t>(ov::shape_size(out_shape));
    const size_t in_bytes = ov::shape_size(in_shape) * m_element_type.size();
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "Slice: input buffer too small");
    if (total == 0) {
        return;
    }

    KernelDispatch dispatch = make_1d_dispatch(total, m_kernel->clamp_threadgroup_size(256));

    KernelArgsBuilder args_builder(name().c_str());
    append_kernel_input_args(args_builder, 1, [&](size_t) { return src; }, name().c_str());
    args_builder.add_output(&dst);
    args_builder.add_bytes(&total, sizeof(total));
    args_builder.add_bytes(&rank, sizeof(rank));
    args_builder.add_bytes(m_out_shape.data(), m_out_shape.size() * sizeof(uint32_t));
    args_builder.add_bytes(m_in_stride.data(), m_in_stride.size() * sizeof(uint32_t));
    args_builder.add_bytes(m_starts.data(), m_starts.size() * sizeof(int32_t));
    args_builder.add_bytes(m_steps.data(), m_steps.size() * sizeof(uint32_t));

    const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
}

}  // namespace gfx_plugin
}  // namespace ov
