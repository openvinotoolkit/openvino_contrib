// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "runtime/metal_op_elementwise.hpp"

#include <string>

#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "runtime/metal_logger.hpp"
#include "runtime/metal_op_utils.hpp"
#include "mlir_builder.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir_codegen/codegen_common.hpp"

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
                                       EltwiseKind kind,
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
}

void MetalElementwiseOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    // Preload constant inputs if present.
    auto maybe_upload_const = [&](const std::shared_ptr<const ov::Node>& n,
                                 size_t input_idx,
                                 MetalTensor& tgt) {
        auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(n);
        if (!c)
            return;
        const auto cet = c->get_element_type();
        const size_t bytes = c->get_byte_size();
        const std::string key = m_node->get_friendly_name() + "/const_" + std::to_string(input_idx);
        tgt.buf = buffer_manager->wrap_const(key, c->get_data_ptr(), bytes, cet);
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
    bool a_known = false;
    bool b_known = false;
    if (!inputs().empty() && inputs()[0]) {
        a_shape = inputs()[0]->shape;
        if (!a_shape.empty())
            a_known = true;
    }
    if (inputs().size() > 1 && inputs()[1]) {
        b_shape = inputs()[1]->shape;
        if (!b_shape.empty())
            b_known = true;
    }
    if (!a_known && m_node && m_node->get_input_partial_shape(0).is_static()) {
        a_shape = m_node->get_input_shape(0);
        a_known = true;
    }
    if (!b_known && m_node && m_node->get_input_partial_shape(1).is_static()) {
        b_shape = m_node->get_input_shape(1);
        b_known = true;
    }

    bool is_broadcast = true;
    if (a_known && b_known) {
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
    } else if (!output_shape().empty()) {
        m_out_dims.clear();
        for (auto d : output_shape()) m_out_dims.push_back(static_cast<int>(d));
        size_t rank = m_out_dims.size();
        m_stride0.assign(rank, 1);
        m_stride1.assign(rank, 1);
    }
    m_num_elems = m_out_dims.empty() ? 0 : 1;
    for (auto d : m_out_dims) m_num_elems *= static_cast<size_t>(d);

    MetalKernelCompiler compiler(m_device);
    std::string log;
    EltwiseCodegenDesc desc{};
    desc.eltwise_kind = m_kind;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    desc.is_broadcast = is_broadcast;
    desc.out_shape.assign(m_out_dims.begin(), m_out_dims.end());
    desc.stride0.assign(m_stride0.begin(), m_stride0.end());
    desc.stride1.assign(m_stride1.begin(), m_stride1.end());
    mlir::MLIRContext ctx;
    auto model = make_single_op_model(m_node);
    mlir::ModuleOp module;
    switch (m_kind) {
        case EltwiseKind::Add:
            module = build_mlir_add_from_model(model, ctx);
            break;
        case EltwiseKind::Sub:
            module = build_mlir_sub_from_model(model, ctx);
            break;
        case EltwiseKind::Mul:
            module = build_mlir_mul_from_model(model, ctx);
            break;
        case EltwiseKind::Div:
            module = build_mlir_div_from_model(model, ctx);
            break;
        case EltwiseKind::Pow:
            module = build_mlir_pow_from_model(model, ctx);
            break;
        case EltwiseKind::Mod:
            module = build_mlir_mod_from_model(model, ctx);
            break;
        case EltwiseKind::FloorMod:
            module = build_mlir_floor_mod_from_model(model, ctx);
            break;
        case EltwiseKind::Prelu:
            module = build_mlir_prelu_from_model(model, ctx);
            break;
        case EltwiseKind::SquaredDiff:
            module = build_mlir_squared_difference_from_model(model, ctx);
            break;
        case EltwiseKind::Min:
            module = build_mlir_min_from_model(model, ctx);
            break;
        case EltwiseKind::Max:
            module = build_mlir_max_from_model(model, ctx);
            break;
        case EltwiseKind::LogicalAnd:
            module = build_mlir_logical_and_from_model(model, ctx);
            break;
        case EltwiseKind::LogicalOr:
            module = build_mlir_logical_or_from_model(model, ctx);
            break;
        case EltwiseKind::LogicalXor:
            module = build_mlir_logical_xor_from_model(model, ctx);
            break;
        case EltwiseKind::Equal:
            module = build_mlir_equal_from_model(model, ctx);
            break;
        case EltwiseKind::NotEqual:
            module = build_mlir_not_equal_from_model(model, ctx);
            break;
        case EltwiseKind::Less:
            module = build_mlir_less_from_model(model, ctx);
            break;
        case EltwiseKind::Greater:
            module = build_mlir_greater_from_model(model, ctx);
            break;
        case EltwiseKind::LessEqual:
            module = build_mlir_less_equal_from_model(model, ctx);
            break;
        case EltwiseKind::GreaterEqual:
            module = build_mlir_greater_equal_from_model(model, ctx);
            break;
        default:
            OPENVINO_THROW("MetalElementwiseOp: unsupported MLIR builder for eltwise kind");
    }
    auto source = generate_msl_from_mlir(module, desc);
    m_pipeline = compiler.compile_msl_from_source(source, "eltwise_kernel", log);
    OPENVINO_ASSERT(m_pipeline, "Failed to compile elementwise pipeline: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalElementwiseOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 2, "Eltwise: missing inputs");
    MetalTensor* in0 = inputs()[0] ? inputs()[0] : (m_const0.buf.valid() ? &m_const0 : nullptr);
    MetalTensor* in1 = inputs()[1] ? inputs()[1] : (m_const1.buf.valid() ? &m_const1 : nullptr);
    OPENVINO_ASSERT(in0 && in0->buf.valid(), "Eltwise: input0 is null");
    OPENVINO_ASSERT(in1 && in1->buf.valid(), "Eltwise: input1 is null");

    MetalTensor& out = require_output();

    // Always recompute broadcast metadata from runtime shapes to avoid stale dims for dynamic inputs.
    ov::Shape a_shape = !in0->shape.empty() ? in0->shape : ov::Shape{};
    ov::Shape b_shape = !in1->shape.empty() ? in1->shape : ov::Shape{};
    if (a_shape.empty() && m_node && m_node->get_input_partial_shape(0).is_static())
        a_shape = m_node->get_input_shape(0);
    if (b_shape.empty() && m_node && m_node->get_input_partial_shape(1).is_static())
        b_shape = m_node->get_input_shape(1);

    auto elem_size_for = [](const MetalTensor* t) -> size_t {
        if (!t)
            return 0;
        const auto et = t->expected_type == ov::element::dynamic ? t->buf.type : t->expected_type;
        return et.size();
    };
    if (!a_shape.empty()) {
        const size_t need = ov::shape_size(a_shape) * elem_size_for(in0);
        OPENVINO_ASSERT(in0->buf.size >= need, "Eltwise: input0 buffer too small");
    }
    if (!b_shape.empty()) {
        const size_t need = ov::shape_size(b_shape) * elem_size_for(in1);
        OPENVINO_ASSERT(in1->buf.size >= need, "Eltwise: input1 buffer too small");
    }

    ov::Shape out_shape;
    auto br = compute_broadcast(a_shape, b_shape);
    if (br.success) {
        out_shape.assign(br.out_shape.begin(), br.out_shape.end());
        m_out_dims.assign(br.out_shape.begin(), br.out_shape.end());
        m_stride0.assign(br.stride0.begin(), br.stride0.end());
        m_stride1.assign(br.stride1.begin(), br.stride1.end());
        m_num_elems = shape_elems(br.out_shape);
    } else {
        if (!a_shape.empty())
            out_shape = a_shape;
        else if (!b_shape.empty())
            out_shape = b_shape;
        else if (!output_shape().empty())
            out_shape = output_shape();
        if (out_shape.empty())
            out_shape = ov::Shape{1};
        m_out_dims.clear();
        for (auto d : out_shape) m_out_dims.push_back(static_cast<int>(d));
        size_t rank = m_out_dims.size();
        auto make_stride = [&](const ov::Shape& shp) {
            std::vector<int64_t> st(shp.size(), 1);
            for (int i = static_cast<int>(shp.size()) - 2; i >= 0; --i) {
                st[i] = st[i + 1] * static_cast<int64_t>(shp[i + 1]);
            }
            return st;
        };
        auto st = make_stride(out_shape);
        m_stride0.assign(st.begin(), st.end());
        m_stride1.assign(st.begin(), st.end());
        m_num_elems = ov::shape_size(out_shape);
    }

    out.shape = out_shape;
    out.expected_type = m_element_type;

    size_t bytes = m_num_elems * m_element_type.size();
    if (!out.buf.valid() || out.buf.size < bytes) {
        out.buf = buffer_manager()->allocate(bytes, m_element_type, /*persistent=*/false, out.prefer_private);
        out.expected_type = m_element_type;
    }

    id<MTLCommandBuffer> cb = static_cast<id<MTLCommandBuffer>>(cmd_buf_handle);
    OPENVINO_ASSERT(cb, "MetalElementwiseOp: command buffer is null");
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
    const NSUInteger tg_size = static_cast<NSUInteger>(metal_clamp_tg_size((void*)m_pipeline, threads_per_tg));
    MTLSize tg = MTLSizeMake(tg_size, 1, 1);

    start_profiling(enc);
    if (num_elems == 0) {
        [enc endEncoding];
        return;
    }
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    stop_profiling_ms(enc);
    [enc endEncoding];

    out.expected_type = m_element_type;
}

}  // namespace metal_plugin
}  // namespace ov
