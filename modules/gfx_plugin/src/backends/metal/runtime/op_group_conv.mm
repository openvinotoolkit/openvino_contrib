// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#import "backends/metal/runtime/op_group_conv.hpp"

#include <numeric>
#include <cstdint>
#include <string>

#include "openvino/core/shape_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/type/float16.hpp"
#include "backends/metal/codegen/metal_codegen_backend.hpp"
#include "runtime/gfx_logger.hpp"
#include "backends/metal/runtime/metal_memory.hpp"
#include "backends/metal/runtime/op_utils.hpp"
#include "kernel_ir/gfx_kernel_args.hpp"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/codegen_common.hpp"

namespace ov {
namespace gfx_plugin {

namespace {

inline size_t element_size(const ov::element::Type& t) {
    return t.size();
}

inline uint32_t to_msl_activation(ActivationKind kind) {
    switch (kind) {
        case ActivationKind::Relu: return 1u;
        case ActivationKind::Sigmoid: return 2u;
        case ActivationKind::Tanh: return 3u;
        case ActivationKind::Elu: return 4u;
        case ActivationKind::Prelu: return 5u;
        case ActivationKind::Gelu: return 6u;
        case ActivationKind::Swish: return 7u;
        case ActivationKind::HSwish: return 8u;
        case ActivationKind::HSigmoid: return 9u;
        case ActivationKind::Abs: return 10u;
        case ActivationKind::Sign: return 11u;
        case ActivationKind::Clamp: return 12u;
        case ActivationKind::Identity:
        default:
            return 0u;
    }
}

}  // namespace

MetalGroupConvOp::MetalGroupConvOp(const std::shared_ptr<const ov::op::v1::GroupConvolution>& node,
                                   MetalDeviceHandle device,
                                   MetalCommandQueueHandle queue)
    : MetalOp(node->get_friendly_name(),
              "GroupConv2D",
              node->get_output_partial_shape(0).is_static() ? node->get_output_shape(0) : ov::Shape{},
              device,
              queue),
      m_node(node),
      m_device((id<MTLDevice>)device),
      m_queue((id<MTLCommandQueue>)queue) {
    OPENVINO_ASSERT(m_node, "MetalGroupConvOp: node is null");

    const auto strides = m_node->get_strides();
    const auto dilations = m_node->get_dilations();
    const auto pads_begin = m_node->get_pads_begin();
    const auto pads_end = m_node->get_pads_end();

    const auto& in_shape = m_node->get_input_shape(0);   // NCHW
    const auto& w_shape = m_node->get_input_shape(1);    // GOIHW

    OPENVINO_ASSERT(in_shape.size() == 4, "MetalGroupConvOp: expected NCHW input");
    OPENVINO_ASSERT(w_shape.size() == 5, "MetalGroupConvOp: expected GOIHW weights");

    m_element_type = m_node->get_output_element_type(0);
    m_desc.element_type = m_element_type;
    m_desc.N = static_cast<uint32_t>(in_shape.at(0));
    m_desc.C_in = static_cast<uint32_t>(in_shape.at(1));
    m_desc.H = static_cast<uint32_t>(in_shape.at(2));
    m_desc.W = static_cast<uint32_t>(in_shape.at(3));

    const uint32_t groups = static_cast<uint32_t>(w_shape.at(0));
    const uint32_t c_out_pg = static_cast<uint32_t>(w_shape.at(1));
    const uint32_t c_in_pg = static_cast<uint32_t>(w_shape.at(2));
    m_desc.groups = groups;
    m_desc.C_out_pg = c_out_pg;
    m_desc.C_in_pg = c_in_pg;
    m_desc.C_out = groups * c_out_pg;
    OPENVINO_ASSERT(m_desc.C_in == groups * c_in_pg,
                    "MetalGroupConvOp: input channels mismatch for groups");

    m_desc.kH = static_cast<uint32_t>(w_shape.at(3));
    m_desc.kW = static_cast<uint32_t>(w_shape.at(4));
    m_desc.strideH = static_cast<uint32_t>(strides.at(0));
    m_desc.strideW = static_cast<uint32_t>(strides.at(1));
    m_desc.dilationH = static_cast<uint32_t>(dilations.at(0));
    m_desc.dilationW = static_cast<uint32_t>(dilations.at(1));
    m_desc.padTop = static_cast<uint32_t>(pads_begin.at(0));
    m_desc.padLeft = static_cast<uint32_t>(pads_begin.at(1));
    m_desc.padBottom = static_cast<uint32_t>(pads_end.at(0));
    m_desc.padRight = static_cast<uint32_t>(pads_end.at(1));
    m_desc.outH = 0;
    m_desc.outW = 0;
    m_desc.has_bias = false;
    m_desc.has_bn = false;
    m_desc.has_activation = false;
}

void MetalGroupConvOp::init(MetalBufferManager* buffer_manager) {
    MetalOp::init(buffer_manager);
}

bool MetalGroupConvOp::fuse_activation(ActivationKind kind, float alpha) {
    OPENVINO_ASSERT(!is_compiled(), "MetalGroupConvOp: cannot fuse activation after compilation");
    m_has_activation = true;
    m_activation = kind;
    m_activation_alpha = alpha;
    m_desc.has_activation = true;
    m_desc.activation = kind;
    m_desc.alpha = alpha;
    return true;
}

bool MetalGroupConvOp::fuse_batchnorm(const BatchNormParams& params) {
    OPENVINO_ASSERT(!is_compiled(), "MetalGroupConvOp: cannot fuse batchnorm after compilation");
    if (params.empty()) {
        return false;
    }
    const size_t channels = params.gamma.size();
    if (channels == 0 ||
        params.beta.size() != channels ||
        params.mean.size() != channels ||
        params.var.size() != channels) {
        return false;
    }
    if (m_desc.C_out != 0 && channels != m_desc.C_out) {
        return false;
    }
    m_has_bn = true;
    m_bn_params = params;
    m_desc.has_bn = true;
    m_desc.epsilon = params.epsilon;
    return true;
}

bool MetalGroupConvOp::fuse_bias(const BiasParams& params) {
    OPENVINO_ASSERT(!is_compiled(), "MetalGroupConvOp: cannot fuse bias after compilation");
    if (params.empty()) {
        return false;
    }
    if (params.element_type != ov::element::f16 && params.element_type != ov::element::f32) {
        return false;
    }
    const size_t c_out = m_desc.C_out;
    if (params.shape.size() == 1) {
        if (static_cast<size_t>(params.shape[0]) != c_out) {
            return false;
        }
    } else if (params.shape.size() == 4) {
        if (params.shape[0] != 1 || params.shape[2] != 1 || params.shape[3] != 1) {
            return false;
        }
        if (static_cast<size_t>(params.shape[1]) != c_out) {
            return false;
        }
    } else {
        return false;
    }
    m_has_bias = true;
    m_bias_params = params;
    m_desc.has_bias = true;
    m_desc.bias_rank = static_cast<uint32_t>(params.shape.size());
    return true;
}

void MetalGroupConvOp::compile(MetalBufferManager* buffer_manager) {
    if (is_compiled()) {
        return;
    }
    if (!this->buffer_manager()) {
        MetalOp::init(buffer_manager);
    }
    prepare_weights();
    prepare_bias();
    prepare_batchnorm();
    OPENVINO_ASSERT(m_device, "MetalGroupConvOp: Metal device is null");
    MetalCodegenBackend backend(m_device);
    std::string log;
    Conv2DCodegenDesc desc = m_desc;
    desc.element_type = m_element_type == ov::element::dynamic ? ov::element::f32 : m_element_type;
    if (m_has_activation) {
        desc.has_activation = true;
        desc.activation = m_activation;
        desc.alpha = m_activation_alpha;
    }
    if (m_has_bn) {
        desc.has_bn = true;
        desc.epsilon = m_bn_params.epsilon;
    }
    mlir::ModuleOp module;
    auto msl_desc = desc;
    auto msl_generator = [msl_desc](mlir::ModuleOp mod) { return generate_msl_from_mlir(mod, msl_desc); };

    KernelSpec spec(m_node, 9u);
    m_kernel = compile_msl_kernel(backend, spec, module, "conv2d_kernel", msl_generator, &log);
    OPENVINO_ASSERT(m_kernel, "MetalGroupConvOp: failed to compile conv2d kernel: ", log);
    MetalOp::compile(buffer_manager);
}

void MetalGroupConvOp::prepare_weights() {
    OPENVINO_ASSERT(buffer_manager(), "MetalGroupConvOp: buffer manager is null");
    auto weights_const = std::dynamic_pointer_cast<const ov::op::v0::Constant>(m_node->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(weights_const, "MetalGroupConvOp requires constant weights");

    const auto& et = weights_const->get_element_type();
    const size_t bytes = element_size(et) * shape_size(weights_const->get_shape());
    const std::string key = m_node->get_friendly_name() + "/weights";
    m_weights = buffer_manager()->wrap_const(key, weights_const->get_data_ptr(), bytes, et);
    OPENVINO_ASSERT(m_weights.valid(), "MetalGroupConvOp: failed to wrap weights buffer");
}

void MetalGroupConvOp::prepare_batchnorm() {
    if (!m_has_bn) {
        return;
    }
    OPENVINO_ASSERT(buffer_manager(), "MetalGroupConvOp: buffer manager is null");
    const ov::element::Type et = (m_element_type == ov::element::dynamic) ? ov::element::f32 : m_element_type;
    const std::string base = m_node->get_friendly_name() + "/batchnorm/";
    const size_t channels = m_bn_params.gamma.size();
    if (channels == 0) {
        return;
    }

    auto wrap_const_vec = [&](const std::string& suffix,
                              const std::vector<float>& src,
                              std::vector<ov::float16>& storage_f16) -> MetalBuffer {
        OPENVINO_ASSERT(src.size() == channels, "MetalGroupConvOp: BN buffer size mismatch for ", suffix);
        if (et == ov::element::f16) {
            storage_f16.resize(src.size());
            for (size_t i = 0; i < src.size(); ++i) {
                storage_f16[i] = ov::float16(src[i]);
            }
            return buffer_manager()->wrap_const(base + suffix,
                                                storage_f16.data(),
                                                storage_f16.size() * sizeof(ov::float16),
                                                et);
        }
        return buffer_manager()->wrap_const(base + suffix,
                                            src.data(),
                                            src.size() * sizeof(float),
                                            et);
    };

    m_bn_gamma = wrap_const_vec("gamma", m_bn_params.gamma, m_bn_gamma_f16);
    m_bn_beta  = wrap_const_vec("beta",  m_bn_params.beta,  m_bn_beta_f16);
    m_bn_mean  = wrap_const_vec("mean",  m_bn_params.mean,  m_bn_mean_f16);
    m_bn_var   = wrap_const_vec("var",   m_bn_params.var,   m_bn_var_f16);

    OPENVINO_ASSERT(m_bn_gamma.valid() && m_bn_beta.valid() && m_bn_mean.valid() && m_bn_var.valid(),
                    "MetalGroupConvOp: failed to wrap batchnorm buffers");
}

void MetalGroupConvOp::prepare_bias() {
    if (!m_has_bias) {
        return;
    }
    OPENVINO_ASSERT(buffer_manager(), "MetalGroupConvOp: buffer manager is null");
    const ov::element::Type et =
        (m_element_type == ov::element::dynamic) ? ov::element::f32 : m_element_type;
    const size_t count = m_bias_params.values.size();
    if (count == 0) {
        return;
    }
    const std::string key = m_node->get_friendly_name() + "/bias";
    if (et == ov::element::f16) {
        m_bias_f16.resize(count);
        for (size_t i = 0; i < count; ++i) {
            m_bias_f16[i] = ov::float16(m_bias_params.values[i]);
        }
        m_bias = buffer_manager()->wrap_const(key,
                                              m_bias_f16.data(),
                                              m_bias_f16.size() * sizeof(ov::float16),
                                              et);
    } else {
        m_bias = buffer_manager()->wrap_const(key,
                                              m_bias_params.values.data(),
                                              m_bias_params.values.size() * sizeof(float),
                                              et);
    }
    OPENVINO_ASSERT(m_bias.valid(), "MetalGroupConvOp: failed to wrap bias buffer");
}

void MetalGroupConvOp::execute(MetalCommandBufferHandle cmd_buf_handle) {
    OPENVINO_ASSERT(!inputs().empty(), "MetalGroupConvOp: no inputs bound");
    MetalTensor* src = inputs().at(0);
    OPENVINO_ASSERT(src && src->buf.valid(), "MetalGroupConvOp: input buffer is null");
    MetalTensor& dst_tensor = require_output();

    const ov::Shape in_shape = !src->shape.empty() ? src->shape : m_node->get_input_shape(0);
    OPENVINO_ASSERT(in_shape.size() == 4, "MetalGroupConvOp: expected NCHW input");
    OPENVINO_ASSERT(in_shape[0] == m_desc.N &&
                        in_shape[1] == m_desc.C_in &&
                        in_shape[2] == m_desc.H &&
                        in_shape[3] == m_desc.W,
                    "MetalGroupConvOp: runtime input shape differs from compiled shape");
    const size_t in_bytes = ov::shape_size(in_shape) * element_size(m_element_type);
    OPENVINO_ASSERT(src->buf.size >= in_bytes, "MetalGroupConvOp: input buffer too small");

    const ov::Shape out_shape = !output_shape().empty() ? output_shape() : m_node->get_output_shape(0);
    OPENVINO_ASSERT(out_shape.size() == 4, "MetalGroupConvOp: expected NCHW output");
    const size_t out_bytes = ov::shape_size(out_shape) * element_size(m_element_type);
    if (!dst_tensor.buf.valid() || dst_tensor.buf.size < out_bytes) {
        const auto& et = m_node->get_output_element_type(0);
        size_t bytes = element_size(et);
        for (auto d : out_shape) bytes *= d;
        dst_tensor.buf = allocate_temp_buffer(bytes,
                                                    et,
                                                    /*persistent=*/false,
                                                    dst_tensor.prefer_private);
        dst_tensor.expected_type = et;
    }
    dst_tensor.shape = out_shape;
    dst_tensor.expected_type = m_node->get_output_element_type(0);

    const ov::Shape w_shape = m_node->get_input_shape(1);
    const size_t w_bytes = ov::shape_size(w_shape) * element_size(m_element_type);
    OPENVINO_ASSERT(m_weights.size >= w_bytes, "MetalGroupConvOp: weights buffer too small");

    MetalBuffer bias = m_bias;
    if (m_desc.has_bias) {
        OPENVINO_ASSERT(bias.valid(), "MetalGroupConvOp: bias buffer is null");
    }
    MetalBuffer gamma = m_bn_gamma;
    MetalBuffer beta = m_bn_beta;
    MetalBuffer mean = m_bn_mean;
    MetalBuffer var = m_bn_var;
    struct ConvParams {
        uint32_t N, C_in, H, W;
        uint32_t C_out;
        uint32_t groups;
        uint32_t C_in_pg;
        uint32_t C_out_pg;
        uint32_t kH, kW;
        uint32_t strideH, strideW;
        uint32_t dilationH, dilationW;
        uint32_t padTop, padLeft;
        uint32_t padBottom, padRight;
        uint32_t outH, outW;
        uint32_t has_bias;
        uint32_t has_bn;
        uint32_t activation;
        float alpha;
        float epsilon;
        float clamp_min;
        float clamp_max;
    } params{};
    params.N = m_desc.N;
    params.C_in = m_desc.C_in;
    params.H = m_desc.H;
    params.W = m_desc.W;
    params.C_out = m_desc.C_out;
    params.groups = m_desc.groups;
    params.C_in_pg = m_desc.C_in_pg;
    params.C_out_pg = m_desc.C_out_pg;
    params.kH = m_desc.kH;
    params.kW = m_desc.kW;
    params.strideH = m_desc.strideH;
    params.strideW = m_desc.strideW;
    params.dilationH = m_desc.dilationH;
    params.dilationW = m_desc.dilationW;
    params.padTop = m_desc.padTop;
    params.padLeft = m_desc.padLeft;
    params.padBottom = m_desc.padBottom;
    params.padRight = m_desc.padRight;
    params.outH = static_cast<uint32_t>(out_shape[2]);
    params.outW = static_cast<uint32_t>(out_shape[3]);
    OPENVINO_ASSERT(params.outH > 0 && params.outW > 0, "MetalGroupConvOp: output spatial dims must be positive");
    params.has_bias = 0;
    params.has_bn = m_desc.has_bn ? 1u : 0u;
    params.activation = m_desc.has_activation ? to_msl_activation(m_desc.activation) : 0u;
    params.alpha = m_activation_alpha;
    params.epsilon = m_desc.epsilon;
    params.clamp_min = 0.0f;
    params.clamp_max = 0.0f;

    const uint64_t total_u64 = static_cast<uint64_t>(params.N) *
                               static_cast<uint64_t>(params.outH) *
                               static_cast<uint64_t>(params.outW) *
                               static_cast<uint64_t>(params.C_out);
    OPENVINO_ASSERT(total_u64 > 0, "MetalGroupConvOp: computed zero elements");
    constexpr uint64_t kMaxGrid = 1ULL << 31;
    OPENVINO_ASSERT(total_u64 < kMaxGrid, "MetalGroupConvOp: computed grid too large: ", total_u64);
    const NSUInteger total = static_cast<NSUInteger>(total_u64);
    if (total == 0) {
        return;
    }
    KernelDispatch dispatch = make_1d_dispatch(total, m_kernel->clamp_threadgroup_size(64));

    KernelArgsBuilder args_builder(name().c_str());
    append_kernel_input_args(args_builder, 1, [&](size_t) { return src; }, name().c_str());
    args_builder.add_input_buffer(m_weights, "weights");
    args_builder.add_optional_input_buffer(bias);
    args_builder.add_optional_input_buffer(gamma);
    args_builder.add_optional_input_buffer(beta);
    args_builder.add_optional_input_buffer(mean);
    args_builder.add_optional_input_buffer(var);
    args_builder.add_output(&dst_tensor);
    args_builder.add_bytes(&params, sizeof(params));

    const auto args = args_builder.finalize(buffer_manager(), m_kernel.get());
    execute_kernel(*m_kernel, cmd_buf_handle, dispatch, args);

    dst_tensor.shape = output_shape();
    dst_tensor.expected_type = m_node->get_output_element_type(0);
}

}  // namespace gfx_plugin
}  // namespace ov
