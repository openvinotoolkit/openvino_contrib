// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_elementwise.hpp"

#include <string>

#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "runtime/gfx_logger.hpp"
#include "runtime/gfx_shape_utils.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/codegen_common.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
struct BroadcastResult {
    bool success = false;
    std::vector<int64_t> out_shape;
    std::vector<int64_t> stride0;
    std::vector<int64_t> stride1;
};

ov::element::Type resolve_tensor_type(const MetalTensor* t) {
    if (!t) {
        return ov::element::dynamic;
    }
    if (t->expected_type != ov::element::dynamic) {
        return t->expected_type;
    }
    return t->buf.type;
}

ov::element::Type resolve_runtime_type(const MetalTensor* in0,
                                       const MetalTensor* in1,
                                       ov::element::Type fallback) {
    const auto t0 = resolve_tensor_type(in0);
    const auto t1 = resolve_tensor_type(in1);
    if (t0 != ov::element::dynamic && t1 != ov::element::dynamic && t0 != t1) {
        OPENVINO_THROW("Eltwise: input element types mismatch (", t0.get_type_name(),
                       " vs ", t1.get_type_name(), ")");
    }
    if (t0 != ov::element::dynamic) {
        return t0;
    }
    if (t1 != ov::element::dynamic) {
        return t1;
    }
    if (fallback == ov::element::dynamic) {
        return ov::element::f32;
    }
    return fallback;
}

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

bool MetalElementwiseOp::fuse_activation(ActivationKind kind, float alpha) {
    if (m_has_activation) {
        return false;
    }
    if (kind != ActivationKind::Relu &&
        kind != ActivationKind::Sigmoid &&
        kind != ActivationKind::Tanh &&
        kind != ActivationKind::Gelu &&
        kind != ActivationKind::Swish &&
        kind != ActivationKind::HSwish &&
        kind != ActivationKind::HSigmoid) {
        return false;
    }
    if (m_element_type.is_integral_number() || m_element_type == ov::element::boolean) {
        return false;
    }
    m_has_activation = true;
    m_activation = kind;
    m_activation_alpha = alpha;
    return true;
}

bool MetalElementwiseOp::fuse_input_activation(size_t input_idx, ActivationKind kind, float alpha) {
    if (m_has_input_activation || input_idx >= 2) {
        return false;
    }
    if (kind != ActivationKind::Relu &&
        kind != ActivationKind::Sigmoid &&
        kind != ActivationKind::Tanh &&
        kind != ActivationKind::Gelu &&
        kind != ActivationKind::Swish &&
        kind != ActivationKind::HSwish &&
        kind != ActivationKind::HSigmoid) {
        return false;
    }
    if (m_element_type.is_integral_number() || m_element_type == ov::element::boolean) {
        return false;
    }
    m_has_input_activation = true;
    m_input_activation_index = input_idx;
    m_input_activation = kind;
    m_input_activation_alpha = alpha;
    return true;
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
            m_num_elems = static_cast<size_t>(shape_product(br.out_shape, 0, br.out_shape.size()));
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

    m_is_broadcast = is_broadcast;
    const ov::element::Type compile_type =
        m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    compile_kernel(buffer_manager, compile_type, m_is_broadcast);
    MetalOp::compile(buffer_manager);
}

void MetalElementwiseOp::compile_kernel(MetalBufferManager* buffer_manager,
                                        ov::element::Type elem_type,
                                        bool is_broadcast) {
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }

    MetalCodegenBackend backend(m_device);
    std::string log;
    EltwiseCodegenDesc desc{};
    desc.eltwise_kind = m_kind;
    desc.element_type = elem_type;
    desc.input0_type = m_node->get_input_element_type(0);
    desc.input1_type = m_node->get_input_element_type(1);
    desc.output_type = m_node->get_output_element_type(0);
    desc.is_broadcast = is_broadcast;
    desc.out_shape.assign(m_out_dims.begin(), m_out_dims.end());
    desc.stride0.assign(m_stride0.begin(), m_stride0.end());
    desc.stride1.assign(m_stride1.begin(), m_stride1.end());
    desc.has_activation = m_has_activation;
    desc.activation = m_activation;
    desc.alpha = m_activation_alpha;
    desc.has_input_activation = m_has_input_activation;
    desc.input_activation_index = static_cast<uint32_t>(m_input_activation_index);
    desc.input_activation = m_input_activation;
    desc.input_activation_alpha = m_input_activation_alpha;
    auto& ctx = gfx_mlir_context();
    auto module = build_mlir_for_node(m_node, ctx);
    auto msl_desc = desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };
    if (gfx_log_debug_enabled()) {
        const std::string preview = msl_generator(module);
        auto sig_pos = preview.find("kernel void");
        auto line_pos = preview.find("device const", sig_pos == std::string::npos ? 0 : sig_pos);
        if (line_pos != std::string::npos) {
            auto line_end = preview.find('\n', line_pos);
            const auto line = preview.substr(line_pos,
                                             (line_end == std::string::npos)
                                                 ? std::string::npos
                                                 : (line_end - line_pos));
            gfx_log_debug("Eltwise") << "msl_signature=" << line;
        }
        auto b_pos = preview.find("device const", line_pos == std::string::npos ? 0 : line_pos + 1);
        if (b_pos != std::string::npos) {
            auto line_end = preview.find('\n', b_pos);
            const auto line = preview.substr(b_pos,
                                             (line_end == std::string::npos)
                                                 ? std::string::npos
                                                 : (line_end - b_pos));
            gfx_log_debug("Eltwise") << "msl_signature2=" << line;
        }
        auto c_pos = preview.find("C[gid]");
        if (c_pos != std::string::npos) {
            auto line_start = preview.rfind('\n', c_pos);
            if (line_start == std::string::npos) {
                line_start = 0;
            } else {
                line_start += 1;
            }
            auto line_end = preview.find('\n', c_pos);
            const auto line = preview.substr(line_start,
                                             (line_end == std::string::npos)
                                                 ? std::string::npos
                                                 : (line_end - line_start));
            gfx_log_debug("Eltwise") << "msl_assign=" << line;
        }
    }

    KernelSpec spec(m_node, 8u);
    m_kernel = compile_msl_kernel(backend, spec, module, "eltwise_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "Failed to compile elementwise pipeline: ", log);
    m_compiled_type = elem_type;
}

void MetalElementwiseOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 2, "Eltwise: missing inputs");
    MetalTensor* in0 = inputs()[0] ? inputs()[0] : (m_const0.buf.valid() ? &m_const0 : nullptr);
    MetalTensor* in1 = inputs()[1] ? inputs()[1] : (m_const1.buf.valid() ? &m_const1 : nullptr);
    OPENVINO_ASSERT(in0 && in0->buf.valid(), "Eltwise: input0 is null");
    OPENVINO_ASSERT(in1 && in1->buf.valid(), "Eltwise: input1 is null");

    MetalTensor& out = require_output();
    const ov::element::Type runtime_type = (m_element_type == ov::element::dynamic)
                                               ? resolve_runtime_type(in0, in1, m_element_type)
                                               : m_element_type;
    const ov::element::Type out_type =
        (m_element_type == ov::element::dynamic) ? runtime_type : m_element_type;
    if (gfx_log_debug_enabled()) {
        const auto t0 = resolve_tensor_type(in0);
        const auto t1 = resolve_tensor_type(in1);
        gfx_log_debug("Eltwise") << "name=" << name()
                              << " node_type=" << (m_node ? m_node->get_type_name() : "null")
                              << " m_element_type=" << m_element_type.get_type_name()
                              << " in0_type=" << t0.get_type_name()
                              << " in1_type=" << t1.get_type_name()
                              << " runtime_type=" << runtime_type.get_type_name()
                              << " compiled_type=" << m_compiled_type.get_type_name()
                              << " in0_buf=" << in0->buf.buffer
                              << " in1_buf=" << in1->buf.buffer
                              << " out_buf=" << out.buf.buffer;
    }

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
        m_num_elems = static_cast<size_t>(shape_product(br.out_shape, 0, br.out_shape.size()));
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
    out.expected_type = out_type;

    if (!m_kernel || m_compiled_type != out_type) {
        compile_kernel(buffer_manager(), out_type, m_is_broadcast);
    }

    size_t bytes = m_num_elems * out_type.size();
    if (!out.buf.valid() || out.buf.size < bytes) {
        out.buf = allocate_temp_buffer(bytes, out_type, /*persistent=*/false, out.prefer_private);
        out.expected_type = out_type;
    }

    uint32_t num_elems = static_cast<uint32_t>(m_num_elems);
    uint32_t rank = static_cast<uint32_t>(m_out_dims.empty() ? 1 : m_out_dims.size());
    if (rank == 0) rank = 1;
    if (m_out_dims.empty()) {
        m_out_dims.push_back(static_cast<int>(m_num_elems));
    }
    if (m_stride0.empty()) m_stride0.assign(rank, 1);
    if (m_stride1.empty()) m_stride1.assign(rank, 1);
    if (gfx_log_debug_enabled()) {
        std::ostringstream dims;
        dims << "dims=[";
        for (size_t i = 0; i < m_out_dims.size(); ++i) {
            if (i) dims << ",";
            dims << m_out_dims[i];
        }
        dims << "] stride0=[";
        for (size_t i = 0; i < m_stride0.size(); ++i) {
            if (i) dims << ",";
            dims << m_stride0[i];
        }
        dims << "] stride1=[";
        for (size_t i = 0; i < m_stride1.size(); ++i) {
            if (i) dims << ",";
            dims << m_stride1[i];
        }
        dims << "] num=" << m_num_elems << " rank=" << rank;
        gfx_log_debug("Eltwise") << dims.str();
    }

    const NSUInteger threads_per_tg = 64;
    if (num_elems == 0) {
        return;
    }
    KernelDispatch dispatch = make_1d_dispatch(num_elems, m_kernel->clamp_threadgroup_size(threads_per_tg));

    KernelArgsBuilder args_builder(name().c_str());
    args_builder.add_inputs(2, [&](size_t idx) { return idx == 0 ? in0 : in1; });
    args_builder.add_output(&out);
    args_builder.add_bytes(&num_elems, sizeof(num_elems));
    args_builder.add_bytes(&rank, sizeof(rank));
    args_builder.add_bytes(m_out_dims.data(), m_out_dims.size() * sizeof(int));
    args_builder.add_bytes(m_stride0.data(), m_stride0.size() * sizeof(int));
    args_builder.add_bytes(m_stride1.data(), m_stride1.size() * sizeof(int));
    auto args = args_builder.finalize(nullptr, nullptr);
    if (gfx_log_debug_enabled()) {
        std::ostringstream oss;
        oss << "args=";
        for (const auto& arg : args) {
            oss << " [i=" << arg.index << " kind=" << (arg.kind == KernelArg::Kind::Buffer ? "buf" : "bytes");
            if (arg.kind == KernelArg::Kind::Buffer) {
                oss << " ptr=" << arg.buffer.buffer;
            } else {
                oss << " bytes=" << arg.byte_size;
            }
            oss << "]";
        }
        gfx_log_debug("Eltwise") << oss.str();
    }
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);

    out.expected_type = out_type;
}

}  // namespace gfx_plugin
}  // namespace ov
