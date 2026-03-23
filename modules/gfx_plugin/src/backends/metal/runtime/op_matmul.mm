// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_matmul.hpp"

#include <array>
#include <string>

#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/type/float16.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/gfx_mlir_kernel_builder.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/codegen_common.hpp"
#include "transforms/mlir_fused_ops.hpp"
#include "runtime/gfx_logger.hpp"

namespace ov {
namespace gfx_plugin {

namespace {
// Flatten arbitrary rank (2–4) shapes into [batch, M, K] or [batch, K, N]
std::vector<int64_t> flatten_to_3d(const ov::Shape& s) {
    OPENVINO_ASSERT(!s.empty(), "MatMul: empty shape");
    OPENVINO_ASSERT(s.size() >= 2 && s.size() <= 4, "MatMul supports ranks 2–4");
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < s.size(); ++i) batch *= static_cast<int64_t>(s[i]);
    int64_t m = static_cast<int64_t>(s[s.size() - 2]);
    int64_t k = static_cast<int64_t>(s[s.size() - 1]);
    return {batch, m, k};
}

bool compute_bias_dims(const BiasParams& params,
                       const MatMulCodegenDesc& desc,
                       std::array<int64_t, 3>& out_dims) {
    if (params.values.empty()) {
        return false;
    }
    if (params.shape.size() > 3) {
        return false;
    }
    const std::array<int64_t, 3> out_shape{{desc.batch, desc.M, desc.N}};
    std::array<int64_t, 3> aligned{{1, 1, 1}};
    const size_t offset = 3 - params.shape.size();
    for (size_t i = 0; i < params.shape.size(); ++i) {
        aligned[offset + i] = params.shape[i];
    }
    for (size_t i = 0; i < 3; ++i) {
        if (aligned[i] != 1 && aligned[i] != out_shape[i]) {
            return false;
        }
    }
    size_t expected = 1;
    for (auto d : params.shape) {
        if (d <= 0) {
            return false;
        }
        expected *= static_cast<size_t>(d);
    }
    if (expected != params.values.size()) {
        return false;
    }
    out_dims = aligned;
    return true;
}

}  // namespace

MetalMatMulOp::MetalMatMulOp(const std::shared_ptr<const ov::Node>& node, void* device, void* queue)
    : MetalOp(node->get_friendly_name(),
              "MatMul",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue),
      m_node(node) {
    fill_desc_from_node(node);
}

void MetalMatMulOp::fill_desc_from_node(const std::shared_ptr<const ov::Node>& node) {
    auto mm0 = ov::as_type_ptr<const ov::op::v0::MatMul>(node);
    OPENVINO_ASSERT(mm0, "MetalMatMulOp expects MatMul node");
    bool ta = mm0->get_transpose_a();
    bool tb = mm0->get_transpose_b();

    m_shape_a = node->get_input_shape(0);
    m_shape_b = node->get_input_shape(1);
    OPENVINO_ASSERT(!m_shape_a.empty() && !m_shape_b.empty(), "MatMul: static shapes required");

    auto a3 = flatten_to_3d(m_shape_a);
    auto b3 = flatten_to_3d(m_shape_b);

    const int64_t batch_a = a3[0];
    const int64_t batch_b = b3[0];
    const int64_t batch = std::max(batch_a, batch_b);
    OPENVINO_ASSERT(batch_a == batch_b || batch_a == 1 || batch_b == 1,
                    "MatMul: incompatible batch broadcast (", batch_a, " vs ", batch_b, ")");

    const int64_t M = ta ? a3[2] : a3[1];
    const int64_t K_a = ta ? a3[1] : a3[2];
    int64_t K_b = tb ? b3[2] : b3[1];
    int64_t N = tb ? b3[1] : b3[2];
    bool b_is_nk_layout = tb;
    if (!tb && K_b != K_a && b3[2] == K_a) {
        // Support pre-transposed weights [N, K].
        K_b = b3[2];
        N = b3[1];
        b_is_nk_layout = true;
    }
    OPENVINO_ASSERT(K_a == K_b, "MatMul: K mismatch (", K_a, " vs ", K_b, ")");

    m_element_type = node->get_output_element_type(0);
    OPENVINO_ASSERT(m_element_type == ov::element::f32 || m_element_type == ov::element::f16,
                    "MetalMatMulOp supports only f16/f32");

    m_desc.element_type = m_element_type;
    m_desc.input_a_type = node->get_input_element_type(0);
    m_desc.input_b_type = node->get_input_element_type(1);
    m_desc.output_type = m_element_type;
    m_desc.M = M;
    m_desc.N = N;
    m_desc.K = K_a;
    m_desc.batch = batch;
    m_desc.batch_a = batch_a;
    m_desc.batch_b = batch_b;
    m_desc.a_transpose = ta;
    m_desc.b_transpose = tb || b_is_nk_layout;
    m_desc.b_is_nk_layout = b_is_nk_layout;
}

void MetalMatMulOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

bool MetalMatMulOp::fuse_activation(ActivationKind kind, float alpha) {
    OPENVINO_ASSERT(!is_compiled(), "MetalMatMulOp: cannot fuse activation after compilation");
    m_has_activation = true;
    m_activation = kind;
    m_activation_alpha = alpha;
    m_desc.has_activation = true;
    m_desc.activation = kind;
    m_desc.alpha = alpha;
    return true;
}

bool MetalMatMulOp::fuse_bias(const BiasParams& params) {
    OPENVINO_ASSERT(!is_compiled(), "MetalMatMulOp: cannot fuse bias after compilation");
    if (params.empty()) {
        return false;
    }
    if (params.element_type != ov::element::f16 && params.element_type != ov::element::f32) {
        return false;
    }
    std::array<int64_t, 3> dims{{1, 1, 1}};
    if (!compute_bias_dims(params, m_desc, dims)) {
        return false;
    }
    m_has_bias = true;
    m_bias_params = params;
    m_desc.has_bias = true;
    m_desc.bias_type = params.element_type;
    m_desc.bias_dims = dims;
    return true;
}

void MetalMatMulOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    auto maybe_upload_const = [&](size_t idx, MetalTensor& tgt) {
        auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(idx));
        if (!c)
            return;
        const auto cet = c->get_element_type();
        OPENVINO_ASSERT(cet == m_element_type,
                        "MatMul const type mismatch: expected ",
                        m_element_type.get_type_name(),
                        " got ",
                        cet.get_type_name());
        const size_t bytes = c->get_byte_size();
        const std::string key = m_node->get_friendly_name() + "/const_" + std::to_string(idx);
        tgt.buf = buffer_manager->wrap_const(key, c->get_data_ptr(), bytes, cet);
        tgt.shape = c->get_shape();
        tgt.expected_type = cet;
    };
    if (m_node) {
        maybe_upload_const(0, m_constA);
        maybe_upload_const(1, m_constB);
    }
    if (m_has_bias) {
        const ov::element::Type et =
            (m_element_type == ov::element::dynamic) ? ov::element::f32 : m_element_type;
        const size_t count = m_bias_params.values.size();
        if (count) {
            const std::string key = m_node->get_friendly_name() + "/bias";
            if (et == ov::element::f16) {
                m_bias_f16.resize(count);
                for (size_t i = 0; i < count; ++i) {
                    m_bias_f16[i] = ov::float16(m_bias_params.values[i]);
                }
                m_bias = buffer_manager->wrap_const(key,
                                                    m_bias_f16.data(),
                                                    m_bias_f16.size() * sizeof(ov::float16),
                                                    et);
            } else {
                m_bias = buffer_manager->wrap_const(key,
                                                    m_bias_params.values.data(),
                                                    m_bias_params.values.size() * sizeof(float),
                                                    et);
            }
            OPENVINO_ASSERT(m_bias.valid(), "MetalMatMulOp: failed to wrap bias buffer");
        }
    }

    MetalCodegenBackend backend(m_device ? m_device : (id<MTLDevice>)buffer_manager->device());
    std::string log;
    mlir::MLIRContext ctx;
    auto module = build_mlir_for_node(m_node, ctx);
    if (m_has_activation) {
        const bool applied = apply_fused_activation(module, m_activation, m_activation_alpha);
        OPENVINO_ASSERT(applied, "MetalMatMulOp: failed to apply fused activation for ", name());
    }
    MatMulCodegenDesc desc = m_desc;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    desc.input_a_type = m_desc.input_a_type == ov::element::dynamic ? desc.element_type : m_desc.input_a_type;
    desc.input_b_type = m_desc.input_b_type == ov::element::dynamic ? desc.element_type : m_desc.input_b_type;
    desc.output_type = m_node->get_output_element_type(0);
    if (desc.bias_type == ov::element::dynamic && m_has_bias) {
        desc.bias_type = m_bias_params.element_type;
    }
    auto msl_desc = desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec(m_node, m_has_bias ? 4u : 3u);
    m_kernel = compile_msl_kernel(backend, spec, module, "matmul_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalMatMulOp: failed to compile matmul kernel: ", log);

    MetalOp::compile(buffer_manager);
}

void MetalMatMulOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(inputs().size() >= 2, "MatMul: missing inputs");
    MetalTensor* A = inputs()[0] ? inputs()[0] : (m_constA.buf.valid() ? &m_constA : nullptr);
    MetalTensor* B = inputs()[1] ? inputs()[1] : (m_constB.buf.valid() ? &m_constB : nullptr);
    OPENVINO_ASSERT(A && A->buf.valid(), "MatMul: input A is null");
    OPENVINO_ASSERT(B && B->buf.valid(), "MatMul: input B is null");

    MetalTensor& C = require_output();
    if (C.shape.empty()) {
        C.shape = output_shape();
        if (C.shape.empty()) {
            // Derive from static input shapes.
            const int64_t batch = m_desc.batch;
            if (batch == 1)
                C.shape = {static_cast<size_t>(m_desc.M), static_cast<size_t>(m_desc.N)};
            else
                C.shape = {static_cast<size_t>(batch),
                           static_cast<size_t>(m_desc.M),
                           static_cast<size_t>(m_desc.N)};
        }
    }
    const size_t c_bytes = ov::shape_size(C.shape) * m_element_type.size();
    if (!C.buf.valid() || C.buf.size < c_bytes) {
        C.buf = allocate_temp_buffer(c_bytes, m_element_type, /*persistent=*/false, C.prefer_private);
    }
    C.expected_type = m_element_type;

    const size_t elem_sz = m_element_type.size();
    const size_t need_a = ov::shape_size(A->shape.empty() ? m_shape_a : A->shape) * elem_sz;
    const size_t need_b = ov::shape_size(B->shape.empty() ? m_shape_b : B->shape) * elem_sz;
    const size_t need_c = ov::shape_size(C.shape) * elem_sz;
    OPENVINO_ASSERT(A->buf.size >= need_a, "MatMul: A buffer too small");
    OPENVINO_ASSERT(B->buf.size >= need_b, "MatMul: B buffer too small");
    OPENVINO_ASSERT(C.buf.size >= need_c, "MatMul: C buffer too small");

    const uint32_t total = static_cast<uint32_t>(m_desc.batch * m_desc.M * m_desc.N);
    if (total == 0) {
        return;
    }
    const NSUInteger threads_per_tg = 256;
    KernelDispatch dispatch = make_1d_dispatch(total, m_kernel->clamp_threadgroup_size(threads_per_tg));

    KernelArgsBuilder args_builder(name().c_str());
    args_builder.add_inputs(2, [&](size_t idx) { return idx == 0 ? A : B; });
    if (m_has_bias) {
        OPENVINO_ASSERT(m_bias.valid(), "MatMul: bias buffer is null");
        args_builder.add_input_buffer(m_bias, "bias");
    }
    args_builder.add_output(&C);
    auto args = args_builder.finalize(nullptr, nullptr);
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);
}

}  // namespace gfx_plugin
}  // namespace ov
