// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_elementwise.hpp"

#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "runtime/metal_logger.hpp"

namespace ov {
namespace metal_plugin {

namespace {
struct BroadcastResult {
    bool success = false;
    std::vector<int64_t> out_shape;
    std::vector<int64_t> stride0;
    std::vector<int64_t> stride1;
};

BroadcastResult compute_broadcast(const ov::Shape& a_shape, const ov::Shape& b_shape) {
    BroadcastResult res;
    size_t rank = std::max(a_shape.size(), b_shape.size());
    if (rank == 0)
        rank = 1;
    ov::Shape a_norm(rank, 1), b_norm(rank, 1), out(rank, 1);
    auto copy_back = [&](const ov::Shape& src, ov::Shape& dst) {
        size_t off = rank - src.size();
        for (size_t i = 0; i < src.size(); ++i) dst[off + i] = src[i];
    };
    copy_back(a_shape, a_norm);
    copy_back(b_shape, b_norm);
    for (size_t k = 0; k < rank; ++k) {
        auto da = a_norm[k];
        auto db = b_norm[k];
        if (da != db && da != 1 && db != 1) {
            res.success = false;
            return res;
        }
        out[k] = std::max(da, db);
    }
    auto make_stride = [&](const ov::Shape& shp) {
        std::vector<int64_t> st(shp.size(), 1);
        for (int i = static_cast<int>(shp.size()) - 2; i >= 0; --i) {
            st[i] = st[i + 1] * static_cast<int64_t>(shp[i + 1]);
        }
        return st;
    };
    const auto a_stride = make_stride(a_norm);
    const auto b_stride = make_stride(b_norm);
    res.stride0.resize(rank);
    res.stride1.resize(rank);
    for (size_t k = 0; k < rank; ++k) {
        res.stride0[k] = (a_norm[k] == 1 ? 0 : a_stride[k]);
        res.stride1[k] = (b_norm[k] == 1 ? 0 : b_stride[k]);
    }
    res.out_shape.assign(out.begin(), out.end());
    res.success = true;
    return res;
}

inline id<MTLBuffer> to_mtl(const MetalBuffer& buf) {
    return (__bridge id<MTLBuffer>)buf.buffer;
}

size_t shape_elems(const std::vector<int64_t>& shp) {
    if (shp.empty())
        return 1;
    size_t r = 1;
    for (auto d : shp) r *= static_cast<size_t>(d);
    return r;
}

}  // namespace

MetalElementwiseOp::MetalElementwiseOp(const std::shared_ptr<const ov::Node>& node,
                                       KernelOpKind kind,
                                       void* device,
                                       void* queue)
    : MetalOp(node->get_friendly_name(),
              "Elementwise",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_kind(kind),
      m_node(node),
      m_element_type(node->get_output_element_type(0)),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    OPENVINO_ASSERT(node->get_input_size() == 2, "Elementwise expects two inputs");
}

void MetalElementwiseOp::refresh_shapes_from_inputs() {
    // Derive shapes from bound tensors if available.
    ov::Shape a_shape;
    ov::Shape b_shape;
    if (!inputs().empty() && inputs()[0]) {
        a_shape = inputs()[0]->shape;
    }
    if (inputs().size() > 1 && inputs()[1]) {
        b_shape = inputs()[1]->shape;
    }
    if (!a_shape.empty() && !b_shape.empty()) {
        auto br = compute_broadcast(a_shape, b_shape);
        if (br.success) {
            m_out_dims.assign(br.out_shape.begin(), br.out_shape.end());
            m_stride0.assign(br.stride0.begin(), br.stride0.end());
            m_stride1.assign(br.stride1.begin(), br.stride1.end());
            m_num_elems = shape_elems(br.out_shape);
            return;
        }
    }
    // Fallback to output_shape() if broadcast failed or inputs missing.
    const auto& base_shape = output_shape();
    m_out_dims.clear();
    for (auto d : base_shape) m_out_dims.push_back(static_cast<int>(d));
    if (m_out_dims.empty())
        m_out_dims.push_back(1);
    size_t rank = m_out_dims.size();
    m_stride0.assign(rank, 1);
    m_stride1.assign(rank, 1);
    m_num_elems = 1;
    for (auto d : m_out_dims) m_num_elems *= static_cast<size_t>(d);
}

void MetalElementwiseOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
    // Preload constant inputs if present.
    auto maybe_upload_const = [&](const std::shared_ptr<const ov::Node>& n,
                                 size_t input_idx,
                                 MetalTensor& tgt) {
        auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(n);
        if (!c)
            return;
        const auto cet = c->get_element_type();
        const size_t bytes = c->get_byte_size();
        tgt.buf = buffer_manager->allocate(bytes, cet, /*persistent=*/true, /*storageModePrivate=*/true);
        buffer_manager->upload(tgt.buf, c->get_data_ptr(), bytes);
        tgt.shape = c->get_shape();
        tgt.expected_type = cet;
        if (inputs().size() <= input_idx || inputs()[input_idx] == nullptr) {
            // Leave input pointer null; execute() will use tgt.
        }
    };

    if (m_node) {
        if (m_node->get_input_size() > 0)
            maybe_upload_const(m_node->get_input_node_shared_ptr(0), 0, m_const0);
        if (m_node->get_input_size() > 1)
            maybe_upload_const(m_node->get_input_node_shared_ptr(1), 1, m_const1);
    }

    // Compute broadcast info from static shapes (if available).
    ov::Shape a_shape;
    ov::Shape b_shape;
    if (!inputs().empty() && inputs()[0] && !inputs()[0]->shape.empty())
        a_shape = inputs()[0]->shape;
    if (inputs().size() > 1 && inputs()[1] && !inputs()[1]->shape.empty())
        b_shape = inputs()[1]->shape;
    if (a_shape.empty() && m_node && m_node->get_input_partial_shape(0).is_static())
        a_shape = m_node->get_input_shape(0);
    if (b_shape.empty() && m_node && m_node->get_input_partial_shape(1).is_static())
        b_shape = m_node->get_input_shape(1);

    bool is_broadcast = false;
    if (a_shape.empty() || b_shape.empty()) {
        m_out_dims.clear();
        for (auto d : output_shape()) m_out_dims.push_back(static_cast<int>(d));
        if (m_out_dims.empty()) m_out_dims.push_back(1);
        size_t rank = m_out_dims.size();
        m_stride0.assign(rank, 1);
        m_stride1.assign(rank, 1);
    } else {
        auto br = compute_broadcast(a_shape, b_shape);
        if (br.success) {
            m_out_dims.assign(br.out_shape.begin(), br.out_shape.end());
            m_stride0.assign(br.stride0.begin(), br.stride0.end());
            m_stride1.assign(br.stride1.begin(), br.stride1.end());
            auto to_vec64 = [](const ov::Shape& s) {
                return std::vector<int64_t>(s.begin(), s.end());
            };
            is_broadcast = (br.out_shape != to_vec64(a_shape) || br.out_shape != to_vec64(b_shape));
        }
    }
    m_num_elems = 1;
    for (auto d : m_out_dims) m_num_elems *= static_cast<size_t>(d);

    KernelOp op{};
    op.kind = m_kind;
    op.is_broadcast = is_broadcast;
    op.out_shape.assign(m_out_dims.begin(), m_out_dims.end());
    op.stride0.assign(m_stride0.begin(), m_stride0.end());
    op.stride1.assign(m_stride1.begin(), m_stride1.end());
    KernelTensor out_t{};
    out_t.shape.assign(op.out_shape.begin(), op.out_shape.end());
    out_t.dtype = resolve_metal_dtype(m_element_type);
    op.output = &out_t;
    op.input0 = nullptr;
    op.input1 = nullptr;
    op.dtype = out_t.dtype;
    op.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(m_element_type));

    MetalKernelCompiler compiler(m_device);
    std::string log;
    switch (m_kind) {
        case KernelOpKind::ElementwiseAdd: m_pipeline = compiler.compile_add_kernel(op, log); break;
        case KernelOpKind::ElementwiseSub: m_pipeline = compiler.compile_sub_kernel(op, log); break;
        case KernelOpKind::ElementwiseMul: m_pipeline = compiler.compile_mul_kernel(op, log); break;
        case KernelOpKind::ElementwiseDiv: m_pipeline = compiler.compile_div_kernel(op, log); break;
        default: OPENVINO_THROW("Unsupported elementwise kind in MetalElementwiseOp");
    }
    OPENVINO_ASSERT(m_pipeline, "Failed to compile elementwise pipeline: ", log);
}

void MetalElementwiseOp::execute() {
    OPENVINO_ASSERT(inputs().size() >= 2, "Eltwise: missing inputs");
    MetalTensor* in0 = inputs()[0] ? inputs()[0] : (m_const0.buf.valid() ? &m_const0 : nullptr);
    MetalTensor* in1 = inputs()[1] ? inputs()[1] : (m_const1.buf.valid() ? &m_const1 : nullptr);
    OPENVINO_ASSERT(in0 && in0->buf.valid(), "Eltwise: input0 is null");
    OPENVINO_ASSERT(in1 && in1->buf.valid(), "Eltwise: input1 is null");

    MetalTensor& out = require_output();
    if (out.shape.empty()) {
        out.shape.clear();
        if (!m_out_dims.empty()) {
            for (auto d : m_out_dims) out.shape.push_back(static_cast<size_t>(d));
        } else {
            out.shape = in0->shape.empty() ? in1->shape : in0->shape;
        }
    }
    if (!out.buf.valid()) {
        size_t bytes = ov::shape_size(out.shape) * m_element_type.size();
        out.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, /*storageModePrivate=*/true);
        out.expected_type = m_element_type;
    }

    if (m_out_dims.empty() || m_stride0.empty() || m_stride1.empty() || m_num_elems == 0) {
        refresh_shapes_from_inputs();
    }
    if (m_num_elems == 0) {
        m_num_elems = ov::shape_size(out.shape);
    }

    if (!m_queue) {
        m_queue = [m_device newCommandQueue];
    }
    id<MTLCommandBuffer> cb = [m_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m_pipeline];
    [enc setBuffer:to_mtl(in0->buf) offset:0 atIndex:0];
    [enc setBuffer:to_mtl(in1->buf) offset:0 atIndex:1];
    [enc setBuffer:to_mtl(out.buf) offset:0 atIndex:2];

    uint32_t num_elems = static_cast<uint32_t>(m_num_elems);
    uint32_t rank = static_cast<uint32_t>(m_out_dims.empty() ? 1 : m_out_dims.size());
    if (rank == 0) rank = 1;
    if (m_out_dims.empty()) {
        m_out_dims.push_back(static_cast<int>(m_num_elems));
    }
    if (m_stride0.empty()) m_stride0.assign(rank, 1);
    if (m_stride1.empty()) m_stride1.assign(rank, 1);

    [enc setBytes:&num_elems length:sizeof(num_elems) atIndex:3];
    [enc setBytes:&rank length:sizeof(rank) atIndex:4];
    [enc setBytes:m_out_dims.data() length:m_out_dims.size() * sizeof(int) atIndex:5];
    [enc setBytes:m_stride0.data() length:m_stride0.size() * sizeof(int) atIndex:6];
    [enc setBytes:m_stride1.data() length:m_stride1.size() * sizeof(int) atIndex:7];

    const NSUInteger threads_per_tg = 64;
    MTLSize grid = MTLSizeMake(num_elems, 1, 1);
    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);

    start_profiling();
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    stop_profiling_ms();

    out.expected_type = m_element_type;
}

}  // namespace metal_plugin
}  // namespace ov
