// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef NO
#undef NO
#endif
#ifdef YES
#undef YES
#endif

#import "runtime/mlir_backend.hpp"

#import <Metal/Metal.h>
#ifdef NO
#undef NO
#endif
#ifdef YES
#undef YES
#endif

#include <exception>
#include <algorithm>
#include <functional>
#include <numeric>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <string>

#include "kernel_ir/kernel_ir_common.hpp"
#include "kernel_ir/add_kernel_ir.hpp"
#include "kernel_ir/mul_kernel_ir.hpp"
#include "kernel_ir/matmul_kernel_ir.hpp"
#include "kernel_ir/unary_kernel_ir.hpp"
#include "kernel_ir/softmax_kernel_ir.hpp"
#include "kernel_ir/pool_max_kernel_ir.hpp"
#include "kernel_ir/pool_avg_kernel_ir.hpp"
#include "kernel_ir/conv_kernel_ir.hpp"
#include "kernel_ir/batchnorm_kernel_ir.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_passes.hpp"
#include "mlir/mlir_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/tensor.hpp"
#include "mlir/IR/MLIRContext.h"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/group_conv.hpp"
#define OV_HAS_LAYER_NORM 0

#ifndef METAL_MLIR_DEBUG
#define LOG_CONV_DEBUG 1
#define METAL_MLIR_DEBUG 1
#endif

namespace ov {
namespace metal_plugin {

namespace {


std::string describe_shape(const KernelTensor* t) {
    if (!t) return "<null>";
    std::string r = "[";
    for (size_t i = 0; i < t->shape.size(); ++i) {
        if (i) r += ",";
        r += std::to_string(t->shape[i]);
    }
    r += "]";
    return r;
}

inline void debug_log(const std::string& msg) {
#if METAL_MLIR_DEBUG
    std::cerr << msg << "\n";
#endif
}

enum class DynamicSupport {
    FullMetal,
    FallbackCPU
};

static bool model_has_dynamic_shape(const std::shared_ptr<const ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            const auto ps = node->get_input_partial_shape(i);
            if (!ps.rank().is_static())
                return true;
            for (auto d : ps)
                if (!d.is_static())
                    return true;
        }
    }
    return false;
}

static bool softmax_has_dynamic_node(const std::shared_ptr<const ov::Model>& model) {
    for (const auto& node : model->get_ordered_ops()) {
        if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
            const auto ps = node->get_input_partial_shape(0);
            if (!ps.rank().is_static()) return true;
            for (auto d : ps)
                if (!d.is_static()) return true;
        } else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
            const auto ps = node->get_input_partial_shape(0);
            if (!ps.rank().is_static()) return true;
            for (auto d : ps)
                if (!d.is_static()) return true;
        }
    }
    return false;
}

static std::pair<bool, int64_t> single_softmax_axis(const std::shared_ptr<const ov::Model>& model) {
    int softmax_count = 0;
    int64_t axis = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Parameter>(node.get()) || ov::is_type<ov::op::v0::Result>(node.get()) ||
            ov::is_type<ov::op::v0::Constant>(node.get()))
            continue;
        if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
            ++softmax_count;
            axis = s1->get_axis();
        } else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
            ++softmax_count;
            axis = s8->get_axis();
        } else if (ov::as_type_ptr<const ov::op::v1::Reshape>(node) ||
                   ov::as_type_ptr<const ov::op::v1::Transpose>(node) ||
                   ov::as_type_ptr<const ov::op::v0::Squeeze>(node) ||
                   ov::as_type_ptr<const ov::op::v0::Unsqueeze>(node) ||
                   ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node) ||
                   ov::as_type_ptr<const ov::op::v0::Concat>(node) ||
                   ov::as_type_ptr<const ov::op::v0::Convert>(node)) {
            continue;  // shape/layout only
        } else {
            return {false, 0};  // other compute ops present
        }
    }
    return {softmax_count == 1, axis};
}

// Softmax shape/axis support:
// - rank 2–5 only
// - all dims static
// - axes: any for rank 2/3; for rank 4/5 only channel axis (1) or last axis
static bool is_softmax_shape_supported(const ov::PartialShape& pshape, int64_t axis) {
    if (!pshape.rank().is_static())
        return false;
    int64_t rank = pshape.rank().get_length();
    if (rank < 2 || rank > 5)
        return false;

    if (axis < 0)
        axis += rank;
    if (axis < 0 || axis >= rank)
        return false;

    if (rank == 4) {
        if (axis == rank - 1) return true;  // last
        if (axis == 1) return true;         // channel
        if (axis == 0) return true;         // batch
        if (axis == 2) return true;         // spatial axis H/W flatten
        return false;
    }
    if (rank == 5) {
        if (axis == rank - 1) return true;
        if (axis == 1) return true;
        if (axis == 0) return true;
        if (axis == 2) return true;
        if (axis == -3) return true;  // handled by normalization above, keep symmetrical intent
        return false;
    }

    return true;  // rank 2/3, allow dynamic dims
}

static DynamicSupport is_matmul_shape_supported_dynamic(const ov::PartialShape& a,
                                                        const ov::PartialShape& b,
                                                        bool transpose_a,
                                                        bool transpose_b) {
    if (!a.rank().is_static() || !b.rank().is_static())
        return DynamicSupport::FallbackCPU;
    auto ra = a.rank().get_length();
    auto rb = b.rank().get_length();
    // Support rank 2–4 MatMul (batched matmul flattens leading dims).
    if (ra < 2 || rb < 2 || ra > 4 || rb > 4)
        return DynamicSupport::FallbackCPU;

    auto dim_at = [&](const ov::PartialShape& ps, int idx_from_back) -> ov::Dimension {
        return ps[static_cast<int64_t>(ps.rank().get_length()) - 1 - idx_from_back];
    };
    auto K_a = transpose_a ? dim_at(a, 1) : dim_at(a, 0);
    auto K_b = transpose_b ? dim_at(b, 0) : dim_at(b, 1);
    if (K_a.is_static() && K_b.is_static() && K_a.get_length() != K_b.get_length())
        return DynamicSupport::FallbackCPU;

    // Leading batch dims must be broadcastable (for now require equality when known).
    size_t max_lead = static_cast<size_t>(std::max<int64_t>(ra, rb) - 2);
    for (size_t i = 0; i < max_lead; ++i) {
        ov::Dimension da = (i < ra - 2) ? a[i] : ov::Dimension(1);
        ov::Dimension db = (i < rb - 2) ? b[i] : ov::Dimension(1);
        if (da.is_static() && db.is_static() && da.get_length() != db.get_length() && da.get_length() != 1 &&
            db.get_length() != 1) {
            return DynamicSupport::FallbackCPU;
        }
    }

    return DynamicSupport::FullMetal;
}

static DynamicSupport is_conv2d_shape_supported_dynamic(const ov::PartialShape& input,
                                                        const ov::PartialShape& weights) {
    if (!input.rank().is_static() || !weights.rank().is_static())
        return DynamicSupport::FallbackCPU;
    if (input.rank().get_length() != 4 || weights.rank().get_length() != 4)
        return DynamicSupport::FallbackCPU;
    // channels must be static and compatible
    if (!input[1].is_static() || !weights[1].is_static() || !weights[0].is_static())
        return DynamicSupport::FallbackCPU;
    return DynamicSupport::FullMetal;  // allow N/H/W dynamic
}

static DynamicSupport is_eltwise_shape_supported_dynamic(const ov::PartialShape& a,
                                                         const ov::PartialShape& b) {
    if (!a.rank().is_static() || !b.rank().is_static())
        return DynamicSupport::FallbackCPU;
    if (a.rank().get_length() != b.rank().get_length())
        return DynamicSupport::FallbackCPU;  // restrict to equal rank for now
    return DynamicSupport::FullMetal;
}

// Utilities for temporary host-side type conversion (FP16 <-> FP32).
static ov::Tensor to_float32_tensor(const ov::Tensor& src) {
    if (src.get_element_type() == ov::element::f32) {
        return src;
    }
    OPENVINO_ASSERT(src.get_element_type() == ov::element::f16,
                    "to_float32_tensor supports only f16/f32 inputs");
    ov::Tensor dst{ov::element::f32, src.get_shape()};
    const auto* src_data = src.data<ov::float16>();
    auto* dst_data = dst.data<float>();
    const size_t count = src.get_size();
    for (size_t i = 0; i < count; ++i) {
        dst_data[i] = static_cast<float>(src_data[i]);
    }
    return dst;
}

static void copy_fp32_to_destination(const float* src, ov::Tensor& dst) {
    if (dst.get_element_type() == ov::element::f32) {
        std::memcpy(dst.data(), src, dst.get_byte_size());
        return;
    }
    OPENVINO_ASSERT(dst.get_element_type() == ov::element::f16,
                    "copy_fp32_to_destination supports only f16/f32 outputs");
    auto* dst_data = dst.data<ov::float16>();
    const size_t count = dst.get_size();
    for (size_t i = 0; i < count; ++i) {
        dst_data[i] = static_cast<ov::float16>(src[i]);
    }
}

// Simple host-side Conv2D reference in NCHW layout, OIHW weights.
static void cpu_conv2d_reference(const float* input,
                                 const float* weights,
                                 const float* bias,
                                 float* output,
                                 int N,
                                 int C_in,
                                 int H,
                                 int W,
                                 int C_out,
                                 int kH,
                                 int kW,
                                 int strideH,
                                 int strideW,
                                 int padTop,
                                 int padLeft,
                                 int dilationH,
                                 int dilationW,
                                 int groups,
                                 int outH,
                                 int outW) {
    const int cin_pg = C_in / groups;
    const int cout_pg = C_out / groups;
    auto out_index = [&](int n, int co, int oh, int ow) {
        return ((n * C_out + co) * outH + oh) * outW + ow;
    };
    auto in_index = [&](int n, int ci, int h, int w) {
        return ((n * C_in + ci) * H + h) * W + w;
    };
    auto w_index = [&](int g, int co_g, int ci_g, int kh, int kw) {
        return (((g * cout_pg + co_g) * cin_pg + ci_g) * kH + kh) * kW + kw;
    };

    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < C_out; ++co) {
            const int g = co / cout_pg;
            const int co_g = co - g * cout_pg;
            for (int oh = 0; oh < outH; ++oh) {
                const int in_h0 = oh * strideH - padTop;
                for (int ow = 0; ow < outW; ++ow) {
                    const int in_w0 = ow * strideW - padLeft;
                    float acc = bias ? bias[co] : 0.f;
                    for (int ci = 0; ci < cin_pg; ++ci) {
                        const int ci_global = g * cin_pg + ci;
                        for (int kh = 0; kh < kH; ++kh) {
                            const int ih = in_h0 + kh * dilationH;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < kW; ++kw) {
                                const int iw = in_w0 + kw * dilationW;
                                if (iw < 0 || iw >= W) continue;
                                const float in_v = input[in_index(n, ci_global, ih, iw)];
                                const float w_v = weights[w_index(g, co_g, ci, kh, kw)];
                                acc += in_v * w_v;
                            }
                        }
                    }
                    output[out_index(n, co, oh, ow)] = acc;
                }
            }
        }
    }
}

static void cpu_maxpool2d_reference(const float* input,
                                    float* output,
                                    int N,
                                    int C,
                                    int H,
                                    int W,
                                    int kH,
                                    int kW,
                                    int strideH,
                                    int strideW,
                                    int padTop,
                                    int padLeft,
                                    int outH,
                                    int outW) {
    auto out_index = [&](int n, int c, int oh, int ow) {
        return ((n * C + c) * outH + oh) * outW + ow;
    };
    auto in_index = [&](int n, int c, int h, int w) {
        return ((n * C + c) * H + h) * W + w;
    };
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < outH; ++oh) {
                const int in_h0 = oh * strideH - padTop;
                for (int ow = 0; ow < outW; ++ow) {
                    const int in_w0 = ow * strideW - padLeft;
                    float max_v = -std::numeric_limits<float>::infinity();
                    for (int kh = 0; kh < kH; ++kh) {
                        const int ih = in_h0 + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int iw = in_w0 + kw;
                            if (iw < 0 || iw >= W) continue;
                            float v = input[in_index(n, c, ih, iw)];
                            if (v > max_v) max_v = v;
                        }
                    }
                    output[out_index(n, c, oh, ow)] = max_v;
                }
            }
        }
    }
}

static void cpu_avgpool2d_reference(const float* input,
                                    float* output,
                                    int N,
                                    int C,
                                    int H,
                                    int W,
                                    int kH,
                                    int kW,
                                    int strideH,
                                    int strideW,
                                    int padTop,
                                    int padLeft,
                                    int outH,
                                    int outW,
                                    bool exclude_pad) {
    auto out_index = [&](int n, int c, int oh, int ow) {
        return ((n * C + c) * outH + oh) * outW + ow;
    };
    auto in_index = [&](int n, int c, int h, int w) {
        return ((n * C + c) * H + h) * W + w;
    };
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < outH; ++oh) {
                const int in_h0 = oh * strideH - padTop;
                for (int ow = 0; ow < outW; ++ow) {
                    const int in_w0 = ow * strideW - padLeft;
                    float sum = 0.f;
                    int count = 0;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int ih = in_h0 + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int iw = in_w0 + kw;
                            if (iw < 0 || iw >= W) continue;
                            sum += input[in_index(n, c, ih, iw)];
                            ++count;
                        }
                    }
                    float denom = exclude_pad ? static_cast<float>(count) : static_cast<float>(kH * kW);
                    output[out_index(n, c, oh, ow)] = (denom == 0.f) ? 0.f : (sum / denom);
                }
            }
        }
    }
}

struct MlirCapabilities {
    bool matmul = true;
    bool add = true;
    bool add_broadcast = true;
    bool mul = true;
    bool relu = true;
    bool sigmoid = true;
    bool tanh = true;
    bool elu = true;
    bool prelu = true;
    bool gelu = true;
    bool swish = true;
    bool softmax = true;
    bool maxpool = true;
    bool avgpool = true;
    bool conv2d = true;
    bool batch_norm = true;
    bool layer_norm = false;
};

MlirCapabilities default_capabilities() {
    MlirCapabilities caps;
    // Keep new ops disabled by default until lowering and kernels land.
    return caps;
}

struct ModelAnalysis {
    bool has_matmul = false;
    bool has_add = false;
    bool has_mul = false;
    bool has_add_broadcast = false;
    bool has_unary = false;
    bool has_softmax = false;
    bool has_maxpool = false;
    bool has_avgpool = false;
    bool has_conv2d = false;
    bool has_batchnorm = false;
    ActivationKind unary_kind = ActivationKind::Relu;
    float unary_alpha = 0.0f;
    size_t compute_ops = 0;
    bool has_future_ops = false;  // Ops recognized but not yet lowered in current backend.
    bool has_disabled_ops = false;  // Ops known but capability flag is off.
    bool has_unsupported_ops = false;  // Completely unknown to the backend.
    std::vector<std::string> disabled_list;
    std::vector<std::string> unsupported_list;
    std::vector<std::string> future_list;
};

std::string describe_node(const std::shared_ptr<const ov::Node>& node) {
    return node->get_friendly_name() + "(" + std::string(node->get_type_info().name) + ")";
}

ModelAnalysis analyze_model_for_mlir(const std::shared_ptr<const ov::Model>& model,
                                     const MlirCapabilities& caps) {
    ModelAnalysis res;

    auto mark_disabled = [&](const std::shared_ptr<const ov::Node>& node) {
        res.has_disabled_ops = true;
        res.disabled_list.push_back(describe_node(node));
        debug_log("[METAL MLIR] Fallback reason: disabled op " + describe_node(node));
    };
    auto mark_future = [&](const std::shared_ptr<const ov::Node>& node) {
        res.has_future_ops = true;
        res.future_list.push_back(describe_node(node));
        debug_log("[METAL MLIR] Fallback reason: future op " + describe_node(node));
    };
    auto mark_unsupported = [&](const std::shared_ptr<const ov::Node>& node) {
        res.has_unsupported_ops = true;
        res.unsupported_list.push_back(describe_node(node));
        debug_log("[METAL MLIR] Fallback reason: unsupported op " + describe_node(node));
    };

    for (const auto& node : model->get_ordered_ops()) {
        if (ov::as_type_ptr<const ov::op::v0::Parameter>(node) ||
            ov::as_type_ptr<const ov::op::v0::Constant>(node) ||
            ov::as_type_ptr<const ov::op::v0::Result>(node)) {
            continue;
        } else if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
            if (!caps.matmul) {
                mark_disabled(node);
                continue;
            }
            auto a = mm->get_input_partial_shape(0);
            auto b = mm->get_input_partial_shape(1);
            if (is_matmul_shape_supported_dynamic(a, b, mm->get_transpose_a(), mm->get_transpose_b()) ==
                DynamicSupport::FullMetal) {
                res.has_matmul = true;
                res.compute_ops++;
            } else {
                mark_future(node);
                continue;
            }
        } else if (auto add = ov::as_type_ptr<const ov::op::v1::Add>(node)) {
            bool broadcast = true;
            const auto ps0 = add->get_input_partial_shape(0);
            const auto ps1 = add->get_input_partial_shape(1);
            if (ps0.is_static() && ps1.is_static()) {
                broadcast = (ps0.to_shape() != ps1.to_shape());
            }
            if (broadcast) {
                if (!caps.add_broadcast) {
                    mark_disabled(node);
                    continue;
                }
                res.has_add_broadcast = true;
                res.compute_ops++;
            } else {
                if (!caps.add) {
                    mark_disabled(node);
                    continue;
                }
                if (is_eltwise_shape_supported_dynamic(ps0, ps1) == DynamicSupport::FullMetal) {
                    res.has_add = true;
                    res.compute_ops++;
                } else {
                    mark_future(node);
                    continue;
                }
            }
        } else if (auto mul = ov::as_type_ptr<const ov::op::v1::Multiply>(node)) {
            // Detect Swish/SiLU pattern: x * sigmoid(x)
            auto lhs = mul->get_input_node_shared_ptr(0);
            auto rhs = mul->get_input_node_shared_ptr(1);
            auto maybe_sigmoid = [&](const std::shared_ptr<const ov::Node>& n) {
                return ov::as_type_ptr<const ov::op::v0::Sigmoid>(n);
            };
            bool is_swish = false;
            if (auto s = maybe_sigmoid(rhs)) {
                is_swish = lhs.get() == s->input_value(0).get_node();
            } else if (auto s = maybe_sigmoid(lhs)) {
                is_swish = rhs.get() == s->input_value(0).get_node();
            }
            if (is_swish) {
                if (!caps.swish) {
                    mark_disabled(node);
                    continue;
                }
                res.has_unary = true;
                res.unary_kind = ActivationKind::Swish;
                res.compute_ops++;
                continue;
            }
            bool broadcast = true;
            const auto ps0 = mul->get_input_partial_shape(0);
            const auto ps1 = mul->get_input_partial_shape(1);
            if (ps0.is_static() && ps1.is_static()) {
                broadcast = (ps0.to_shape() != ps1.to_shape());
            }
            if (broadcast) {
                if (!caps.mul) {
                    mark_disabled(node);
                } else {
                    res.has_mul = true;
                    res.compute_ops++;
                }
            } else {
                if (!caps.mul) {
                    mark_disabled(node);
                } else {
                    if (is_eltwise_shape_supported_dynamic(ps0, ps1) == DynamicSupport::FullMetal) {
                        res.has_mul = true;
                        res.compute_ops++;
                    } else {
                        mark_future(node);
                        continue;
                    }
                }
            }
        } else if (ov::as_type_ptr<const ov::op::v0::Relu>(node)) {
            if (!caps.relu) {
                mark_disabled(node);
            } else {
                res.has_unary = true;
                res.unary_kind = ActivationKind::Relu;
                res.compute_ops++;
            }
        } else if (ov::as_type_ptr<const ov::op::v4::Swish>(node)) {
            if (!caps.swish) {
                mark_disabled(node);
            } else {
                res.has_unary = true;
                res.unary_kind = ActivationKind::Swish;
                res.compute_ops++;
            }
        } else if (ov::as_type_ptr<const ov::op::v0::Sigmoid>(node)) {
            if (!caps.sigmoid) {
                mark_disabled(node);
            } else {
                res.has_unary = true;
                res.unary_kind = ActivationKind::Sigmoid;
                res.compute_ops++;
            }
        } else if (ov::as_type_ptr<const ov::op::v0::Tanh>(node)) {
            if (!caps.tanh) {
                mark_disabled(node);
            } else {
                res.has_unary = true;
                res.unary_kind = ActivationKind::Tanh;
                res.compute_ops++;
            }
        } else if (ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
            if (!caps.elu) {
                mark_disabled(node);
            } else {
                res.has_unary = true;
                res.unary_kind = ActivationKind::Elu;
                if (auto elu = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
                    res.unary_alpha = static_cast<float>(elu->get_alpha());
                }
                res.compute_ops++;
            }
        } else if (ov::as_type_ptr<const ov::op::v0::PRelu>(node)) {
            if (!caps.prelu) {
                mark_disabled(node);
            } else {
                // For now support only scalar slope constant; otherwise fallback.
                auto slope_node = node->get_input_node_shared_ptr(1);
                if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(slope_node)) {
                    if (c->get_shape().size() == 0 || ov::shape_size(c->get_shape()) == 1) {
                        auto vec = c->cast_vector<float>();
                        res.unary_alpha = vec[0];
                        res.has_unary = true;
                        res.unary_kind = ActivationKind::Prelu;
                        res.compute_ops++;
                    } else {
                        mark_future(node);
                    }
                } else {
                    mark_future(node);
                }
            }
        } else if (ov::as_type_ptr<const ov::op::v0::Gelu>(node) ||
                   ov::as_type_ptr<const ov::op::v7::Gelu>(node)) {
            if (!caps.gelu) {
                mark_disabled(node);
            } else {
                res.has_unary = true;
                res.unary_kind = ActivationKind::Gelu;
                res.compute_ops++;
            }
        } else if (ov::as_type_ptr<const ov::op::v1::Softmax>(node) ||
                   ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
            if (!caps.softmax) {
                mark_disabled(node);
            } else {
                const auto ps = node->get_input_partial_shape(0);
                int64_t axis = -1;
                if (auto sm1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
                    axis = sm1->get_axis();
                } else if (auto sm8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
                    axis = sm8->get_axis();
                }
                if (!is_softmax_shape_supported(ps, axis)) {
                    mark_future(node);
                    continue;
                }
                if (axis < 0)
                    axis += static_cast<int64_t>(ps.rank().get_length());
                res.has_softmax = true;
                res.compute_ops++;
            }
        } else if (auto mp = ov::as_type_ptr<const ov::op::v1::MaxPool>(node)) {
            if (!caps.maxpool) {
                mark_disabled(node);
            } else if (mp->get_input_partial_shape(0).rank().is_static() &&
                       mp->get_input_shape(0).size() == 4) {
                res.has_maxpool = true;
                res.compute_ops++;
            } else {
                mark_future(node);
            }
        } else if (auto ap = ov::as_type_ptr<const ov::op::v1::AvgPool>(node)) {
            if (!caps.avgpool) {
                mark_disabled(node);
            } else if (ap->get_input_partial_shape(0).rank().is_static() &&
                       ap->get_input_shape(0).size() == 4) {
                res.has_avgpool = true;
                res.compute_ops++;
            } else {
                mark_future(node);
            }
        } else if (auto cv = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
            if (!caps.conv2d) {
                mark_disabled(node);
            } else {
                auto a = cv->get_input_partial_shape(0);
                auto w = cv->get_input_partial_shape(1);
                if (is_conv2d_shape_supported_dynamic(a, w) == DynamicSupport::FullMetal) {
                    res.has_conv2d = true;
                    res.compute_ops++;
                } else {
                    mark_future(node);
                }
            }
        } else if (ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node)) {
            if (!caps.batch_norm) {
                mark_disabled(node);
            } else {
                res.has_batchnorm = true;
                res.compute_ops++;
            }
        } else if (ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
            caps.conv2d ? mark_future(node) : mark_disabled(node);
        } else if (ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node) ||
                   ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node)) {
            caps.batch_norm ? mark_future(node) : mark_disabled(node);
        } else if (ov::as_type_ptr<const ov::op::v1::Multiply>(node)) {
            // Allow multiplies to pass when batchnorm path is enabled (BN pattern uses multiplies).
            if (!caps.batch_norm) {
                mark_unsupported(node);
            }
        } else if (ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
            if (!caps.conv2d) {
                mark_disabled(node);
            } else {
                res.has_conv2d = true;
                res.compute_ops++;
            }
        } else if (ov::as_type_ptr<const ov::op::v1::Reshape>(node) ||
                   ov::as_type_ptr<const ov::op::v1::Transpose>(node) ||
                   ov::as_type_ptr<const ov::op::v0::Squeeze>(node) ||
                   ov::as_type_ptr<const ov::op::v0::Unsqueeze>(node) ||
                   ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node) ||
                   ov::as_type_ptr<const ov::op::v0::Concat>(node) ||
                   ov::as_type_ptr<const ov::op::v0::Interpolate>(node) ||
                   ov::as_type_ptr<const ov::op::v4::Interpolate>(node) ||
                   ov::as_type_ptr<const ov::op::v0::Convert>(node)) {
            // Treat as layout/shape ops; do not trigger fallback.
            continue;
        } else {
            mark_unsupported(node);
        }
    }

    return res;
}

void log_list(const char* prefix, const std::vector<std::string>& items) {
    if (items.empty()) return;
    std::cerr << prefix;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i) std::cerr << ", ";
        std::cerr << items[i];
    }
    std::cerr << "\n";
}

}  // namespace

// Flat scheduling segment (future use for multi-op execution).
struct Segment {
    size_t first_op_index = 0;
    size_t op_count = 0;
    // Anchors into the OV graph; today we only use them for IO-aligned segments.
    std::vector<ov::Output<const ov::Node>> input_ports;
    std::vector<ov::Output<const ov::Node>> output_ports;
};

// compile() note (pre-flat-IR):
//  - Examines the model and matches a small set of hard-coded templates:
//    * single MatMul (optional MLIR lowering)
//    * MatMul -> {Softmax|Unary}
//    * MatMul -> Softmax -> MatMul (attention)
//    * Conv2D (+ optional BN) -> Unary
//    * Pool (Max/Avg)
//  - On a successful match it fills a tiny m_ops (1–3 KernelOp), creates m_pipelines
//    and returns immediately. For larger chains no IR is built and the code falls back to CPU.
//
// run() note (pre-flat-IR):
//  - Assumes m_ops has 1–3 ops chosen by compile().
//  - Sets up fixed Metal buffers (buf_in0, buf_in1, optional const buffers, a single tmp/out)
//    and dispatches ops in order; buffer roles are hard-wired to the first/second/third op.
//  - Any model that didn’t match a short template in compile() uses CPU fallback.

class MlirBackend::Impl {
public:
    explicit Impl(const std::shared_ptr<const ov::Model>& model,
                  const std::shared_ptr<const ov::Model>& original_model,
                  ov::element::Type inference_precision)
        : m_model(model),
          m_original_model(original_model),
          m_inference_precision(inference_precision) {
        m_device = MTLCreateSystemDefaultDevice();
        OPENVINO_ASSERT(m_device, "MlirBackend: failed to create Metal device");
        m_queue = [m_device newCommandQueue];
        OPENVINO_ASSERT(m_queue, "MlirBackend: failed to create command queue");
        const char* partial_env = std::getenv("OV_METAL_EXPERIMENTAL_PARTIAL_OFFLOAD");
        m_allow_partial_offload = partial_env && std::string(partial_env) == "1";
        m_force_fallback = false;

        compile(model);
    }

    void run(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) {
        auto cpu_fallback_error = [&]() {
            OPENVINO_THROW("METAL: CPU fallback is disabled in pure device mode");
        };

        auto run_fallback = [&]() {
            if (!m_allow_partial_offload) {
                cpu_fallback_error();
            }
            ensure_fallback();
            OPENVINO_ASSERT(m_fallback_request, "Fallback request is null");
            OPENVINO_ASSERT(inputs.size() == m_fallback_inputs.size(), "Fallback: input count mismatch");
            for (size_t i = 0; i < inputs.size(); ++i) {
                m_fallback_request->set_input_tensor(i, inputs[i]);
            }
            m_fallback_request->infer();
            OPENVINO_ASSERT(outputs.size() == m_fallback_outputs.size(), "Fallback: output count mismatch");
            for (size_t i = 0; i < outputs.size(); ++i) {
                auto tmp = m_fallback_request->get_output_tensor(i);
                if (outputs[i].get_shape() != tmp.get_shape() ||
                    outputs[i].get_element_type() != tmp.get_element_type()) {
                    outputs[i] = ov::Tensor{tmp.get_element_type(), tmp.get_shape()};
                }
                std::memcpy(outputs[i].data(), tmp.data(), tmp.get_byte_size());
            }
        };

        debug_log(std::string("[METAL MLIR] run: force_fallback=") + (m_force_fallback ? "true" : "false"));
        if (m_softmax_dynamic_only) {
            if (run_host_softmax(inputs, outputs))
                return;
        }
        if (m_force_fallback) {
            run_fallback();
            return;
        }

        // Patch missing dtype info (common for dynamic-typed test graphs).
        for (auto& op : m_ops) {
            if (op.kind == KernelOpKind::Softmax) {
                auto ty = static_cast<ov::element::Type_t>(op.element_type);
                if (ty == ov::element::Type_t::dynamic) {
                    op.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(m_inference_precision));
                }
            }
        }

        // Prefer previously built flat/template segment; otherwise CPU fallback.
        if (m_fallback_request) {
            run_fallback();
            return;
        }

        const auto& ops = m_ops;
        if (ops.empty() || m_segments.empty()) {
            run_fallback();
            return;
        }

        // Execute first segment (may be full flat IR with many ops)
        const Segment& seg = m_segments.front();
        if (seg.op_count == 0 || seg.first_op_index + seg.op_count > ops.size()) {
            debug_log("[METAL MLIR] Segment guard failed, falling back to CPU");
            run_fallback();
            return;
        }
        if (m_pipelines.empty() || m_pipelines.size() < seg.first_op_index + seg.op_count) {
            debug_log("[METAL MLIR] Missing pipelines for segment, fallback to CPU");
            run_fallback();
            return;
        }

        auto seg_op = [&](size_t i) -> const KernelOp& { return ops[seg.first_op_index + i]; };
        auto seg_pipe = [&](size_t i) -> id<MTLComputePipelineState> { return m_pipelines[seg.first_op_index + i]; };

        size_t expected_inputs = inputs.size();  // allow variable inputs for flat execution
        const auto& ops_ref = ops;  // alias to suppress shadowing warnings
        auto first_kind = seg_op(0).kind;
        OPENVINO_ASSERT(outputs.size() == 1, "MlirBackend: expected 1 output");

        auto inputs_ok = [](const std::vector<ov::Tensor>& ts) {
            for (const auto& t : ts) {
                if (t.get_element_type() != ov::element::f32)
                    continue;
                const float* p = t.data<const float>();
                for (size_t i = 0; i < t.get_size(); ++i) {
                    if (!std::isfinite(p[i]) || std::abs(p[i]) > 1e6f) {
                        return false;
                    }
                }
            }
            return true;
        };
        if (!inputs_ok(inputs)) {
            std::cerr << "[MlirBackend] Inputs invalid, CPU fallback disabled\n";
            cpu_fallback_error();
        }

        auto resize_output = [&](const std::vector<size_t>& out_shape, ov::element::Type et) {
            if (outputs.empty())
                return;
            ov::Shape osh(out_shape.begin(), out_shape.end());
            if (outputs[0].get_shape() != osh || outputs[0].get_element_type() != et) {
                outputs[0] = ov::Tensor{et, osh};
            }
        };
        auto resize_output_i64 = [&](const std::vector<int64_t>& out_shape, ov::element::Type et) {
            std::vector<size_t> tmp;
            tmp.reserve(out_shape.size());
            for (auto d : out_shape) tmp.push_back(static_cast<size_t>(d));
            resize_output(tmp, et);
        };

        auto shape_from_tensor = [](const ov::Tensor& t) {
            const auto& s = t.get_shape();
            return std::vector<int64_t>(s.begin(), s.end());
        };

        // Runtime-shape aware op parameters (all segment lengths).
        std::vector<KernelOp> runtime_ops;
        runtime_ops.reserve(seg.op_count);
        std::vector<int64_t> in0_shape = shape_from_tensor(inputs[0]);
        std::vector<int64_t> in1_shape = inputs.size() > 1 ? shape_from_tensor(inputs[1]) : std::vector<int64_t>{};
        std::vector<int64_t> last_out_shape;
        bool runtime_shapes_ok = true;

        auto update_output_shape = [](KernelOp& op, const std::vector<int64_t>& shp) {
            if (op.output) {
                op.output->shape = shp;
            }
            if (op.kind == KernelOpKind::ElementwiseAdd || op.kind == KernelOpKind::ElementwiseMul) {
                op.out_shape = shp;
            }
        };

        int64_t sm_rows = 0, sm_cols = 0, sm_inner = 0;
        bool sm_params_ok = false;

        auto softmax_params_from_shape = [&](const KernelOp& op,
                                             const std::vector<int64_t>& shape) -> std::tuple<int64_t, int64_t, int64_t, bool> {
            if (shape.empty() || op.softmax_axis == -1)
                return {0, 0, 0, false};
            int64_t rank = static_cast<int64_t>(shape.size());
            int64_t axis = op.softmax_axis;
            if (axis < 0) axis += rank;
            if (axis < 0 || axis >= rank)
                return {0, 0, 0, false};
            int64_t cols = shape[static_cast<size_t>(axis)];
            int64_t inner = 1;
            for (int64_t i = axis + 1; i < rank; ++i) inner *= shape[static_cast<size_t>(i)];
            int64_t outer = 1;
            for (int64_t i = 0; i < axis; ++i) outer *= shape[static_cast<size_t>(i)];
            int64_t rows = outer * inner;
            return {rows, cols, inner, true};
        };

        for (size_t idx = 0; idx < seg.op_count; ++idx) {
            KernelOp op_rt = seg_op(idx);
            const auto& cur_in0 = (idx == 0) ? in0_shape : last_out_shape;
            const auto& cur_in1 = (idx == 0) ? in1_shape : std::vector<int64_t>{};

            switch (op_rt.kind) {
                case KernelOpKind::MatMul: {
                    // Support batched MatMul: A[..., M, K] x B[..., K, N]
                    auto collapse = [](const std::vector<int64_t>& s) {
                        int64_t batch = 1;
                        for (size_t i = 0; i + 2 < s.size(); ++i) batch *= s[i];
                        return batch;
                    };
                    if (cur_in0.size() < 2 || (!cur_in1.empty() && cur_in1.size() < 2)) {
                        runtime_shapes_ok = false; break;
                    }
                    bool ta = op_rt.a_transpose;
                    bool tb = op_rt.b_transpose;
                    int64_t M = ta ? cur_in0.back() : cur_in0[cur_in0.size() - 2];
                    int64_t K = ta ? cur_in0[cur_in0.size() - 2] : cur_in0.back();
                    int64_t N = op_rt.N;
                    if (!cur_in1.empty()) {
                        int64_t kb = tb ? cur_in1.back() : cur_in1[cur_in1.size() - 2];
                        int64_t nb = tb ? cur_in1[cur_in1.size() - 2] : cur_in1.back();
                        if (kb != K) { runtime_shapes_ok = false; break; }
                        N = nb;
                    } else {
                        if (op_rt.K != 0 && op_rt.K != K) { runtime_shapes_ok = false; break; }
                    }
                    int64_t batch = std::max<int64_t>(1, collapse(cur_in0));
                    op_rt.batch = batch;
                    op_rt.M = M;
                    op_rt.K = K;
                    op_rt.N = N;
                    std::vector<int64_t> out_shape;
                    if (cur_in0.size() > 2) {
                        out_shape.assign(cur_in0.begin(), cur_in0.end() - 2);
                        out_shape.push_back(M);
                        out_shape.push_back(N);
                    } else {
                        out_shape = {M, N};
                    }
                    update_output_shape(op_rt, out_shape);
                    last_out_shape = out_shape;
                    break;
                }
                case KernelOpKind::Conv2D: {
                    if (cur_in0.size() != 4) { runtime_shapes_ok = false; break; }
                    int64_t N = cur_in0[0];
                    int64_t C = cur_in0[1];
                    int64_t H = cur_in0[2];
                    int64_t W = cur_in0[3];
                    auto& c = op_rt.conv2d;
                    if (C != static_cast<int64_t>(c.C_in)) { runtime_shapes_ok = false; break; }
                    int64_t outH = (H + c.padTop + c.padBottom - static_cast<int64_t>(c.dilationH) * (c.kernelH - 1) - 1) /
                                   static_cast<int64_t>(c.strideH) +
                                   1;
                    int64_t outW = (W + c.padLeft + c.padRight - static_cast<int64_t>(c.dilationW) * (c.kernelW - 1) - 1) /
                                   static_cast<int64_t>(c.strideW) +
                                   1;
                    if (outH <= 0 || outW <= 0) { runtime_shapes_ok = false; break; }
                    c.N = static_cast<uint32_t>(N);
                    c.H = static_cast<uint32_t>(H);
                    c.W = static_cast<uint32_t>(W);
                    c.outH = static_cast<uint32_t>(outH);
                    c.outW = static_cast<uint32_t>(outW);
                    op_rt.conv2d = c;
                    std::vector<int64_t> out_shape{N, static_cast<int64_t>(c.C_out), outH, outW};
                    update_output_shape(op_rt, out_shape);
                    last_out_shape = out_shape;
                    break;
                }
                case KernelOpKind::ElementwiseAdd:
                case KernelOpKind::ElementwiseMul: {
                    const auto& bshape = !cur_in1.empty() ? cur_in1 : cur_in0;
                    if (cur_in0 != bshape) { runtime_shapes_ok = false; break; }
                    update_output_shape(op_rt, cur_in0);
                    last_out_shape = cur_in0;
                    break;
                }
                case KernelOpKind::Unary: {
                    if (cur_in0.empty()) { runtime_shapes_ok = false; break; }
                    update_output_shape(op_rt, cur_in0);
                    last_out_shape = cur_in0;
                    break;
                }
                case KernelOpKind::Softmax: {
                    auto params = softmax_params_from_shape(op_rt, cur_in0);
                    sm_rows = std::get<0>(params);
                    sm_cols = std::get<1>(params);
                    sm_inner = std::get<2>(params);
                    sm_params_ok = std::get<3>(params);
                    if (!sm_params_ok || sm_rows <= 0 || sm_cols <= 0) { runtime_shapes_ok = false; break; }
                    update_output_shape(op_rt, cur_in0);
                    last_out_shape = cur_in0;
                    break;
                }
                case KernelOpKind::MaxPool2D:
                case KernelOpKind::AvgPool2D:
                case KernelOpKind::BatchNorm2D: {
                    // Keep static shapes for now
                    if (!op_rt.output || op_rt.output->shape.empty()) { runtime_shapes_ok = false; break; }
                    last_out_shape = op_rt.output->shape;
                    break;
                }
            }
            runtime_ops.push_back(op_rt);
            if (!runtime_shapes_ok) break;
        }

        if (!runtime_shapes_ok) {
            run_fallback();
            return;
        }

        // Commit runtime-updated ops back to storage
        for (size_t i = 0; i < runtime_ops.size(); ++i) {
            m_ops[seg.first_op_index + i] = runtime_ops[i];
        }

        // Final output shape
        if (!last_out_shape.empty()) {
            resize_output_i64(last_out_shape, outputs[0].get_element_type());
        }

        if (!ops_ref.empty() && seg_op(0).kind == KernelOpKind::MatMul &&
            inputs[0].get_element_type() == ov::element::f32 &&
            inputs.size() > 1 && inputs[1].get_element_type() == ov::element::f32) {
            auto log_tensor = [](const char* tag, const ov::Tensor& t) {
                std::cerr << "[MlirBackend run] " << tag << " size=" << t.get_size() << " first:";
                const float* p = t.data<const float>();
                for (size_t i = 0; i < std::min<size_t>(t.get_size(), 8); ++i) std::cerr << " " << p[i];
                std::cerr << "\n";
            };
            log_tensor("A", inputs[0]);
            if (inputs.size() > 1) log_tensor("B", inputs[1]);
        }

        const auto& pipelines = m_pipelines;
        if (pipelines.empty()) {
            run_fallback();
            return;
        }

        // Flat IR execution path (includes Conv/Pool/BN and handles FP16 up/down-cast).
        if (m_ops_from_flat_segment) {
            auto seg_outputs = run_segment(seg, inputs);
            if (seg_outputs.empty()) {
                run_fallback();
                return;
            }
            OPENVINO_ASSERT(outputs.size() == seg_outputs.size(), "Output count mismatch in run_segment");
            for (size_t i = 0; i < outputs.size(); ++i) {
                outputs[i] = seg_outputs[i];
            }
            return;
        }

        std::cerr << "[MlirBackend run] segment ops=" << seg.op_count << " kinds:";
        for (size_t i = 0; i < seg.op_count; ++i) std::cerr << " " << static_cast<int>(seg_op(i).kind);
        std::cerr << "\n";

        auto tensor_num_elems = [](const KernelTensor* t) -> size_t {
            if (!t) return 0;
            size_t elems = 1;
            for (auto d : t->shape) elems *= static_cast<size_t>(d);
            return elems;
        };

        auto tensor_bytes = [&](const KernelTensor* t) -> size_t {
            return tensor_num_elems(t) * sizeof(float);
        };

        // Buffer mapping for flat execution
        std::unordered_map<const KernelTensor*, id<MTLBuffer>> buf_map;
        auto make_buffer_from_tensor = [&](const ov::Tensor& t) -> id<MTLBuffer> {
            ov::Tensor tmp = t;
            if (t.get_element_type() == ov::element::f16) {
                tmp = to_float32_tensor(t);
            }
            return [m_device newBufferWithBytes:tmp.data() length:tmp.get_byte_size() options:MTLResourceStorageModeShared];
        };
        if (!inputs.empty()) {
            if (seg_op(0).input0) buf_map[seg_op(0).input0] = make_buffer_from_tensor(inputs[0]);
            if (inputs.size() > 1 && seg_op(0).input1) buf_map[seg_op(0).input1] = make_buffer_from_tensor(inputs[1]);
        }
        auto get_buffer = [&](const KernelTensor* t) -> id<MTLBuffer> {
            auto it = buf_map.find(t);
            return it == buf_map.end() ? nil : it->second;
        };

        id<MTLCommandBuffer> cmd = [m_queue commandBuffer];

        auto alloc_out_buffer = [&](const KernelOp& op) -> id<MTLBuffer> {
            if (!op.output) return nil;
            size_t bytes = tensor_bytes(op.output);
            if (bytes == 0) bytes = 1;
            return [m_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        };

        auto dispatch_op = [&](const KernelOp& op,
                               id<MTLComputePipelineState> pipeline,
                               id<MTLBuffer> src0,
                               id<MTLBuffer> src1,
                               id<MTLBuffer> dst,
                               id<MTLCommandBuffer> cmdBuf) {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:pipeline];

            switch (op.kind) {
                case KernelOpKind::MatMul: {
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    const NSUInteger total = static_cast<NSUInteger>(op.M * op.N * op.batch);
                    MTLSize grid = MTLSizeMake(total, 1, 1);
                    MTLSize tg = MTLSizeMake(128, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::ElementwiseAdd:
                case KernelOpKind::ElementwiseMul: {
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    const NSUInteger elems = static_cast<NSUInteger>(tensor_num_elems(op.output));
                    MTLSize grid = MTLSizeMake(elems, 1, 1);
                    MTLSize tg = MTLSizeMake(64, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Unary: {
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    const NSUInteger elems = static_cast<NSUInteger>(tensor_num_elems(op.output));
                    MTLSize grid = MTLSizeMake(elems, 1, 1);
                    MTLSize tg = MTLSizeMake(64, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Softmax: {
                    struct SoftmaxParams { uint32_t rows, cols, inner; } params;
                    params.rows = static_cast<uint32_t>(op.rows);
                    params.cols = static_cast<uint32_t>(op.cols);
                    params.inner = static_cast<uint32_t>(op.inner);
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    [enc setBytes:&params length:sizeof(params) atIndex:2];
                    const NSUInteger rows = static_cast<NSUInteger>(op.rows);
                    MTLSize grid = MTLSizeMake(rows, 1, 1);
                    MTLSize tg = MTLSizeMake(64, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                default:
                    break;
            }
            [enc endEncoding];
        };

        id<MTLBuffer> final_buf = nil;
        for (size_t i = 0; i < seg.op_count; ++i) {
            const auto& op = seg_op(i);
            id<MTLBuffer> src0 = op.input0 ? get_buffer(op.input0) : nil;
            id<MTLBuffer> src1 = op.input1 ? get_buffer(op.input1) : nil;
            if (!src0 && inputs.size() > 0 && i == 0) src0 = make_buffer_from_tensor(inputs[0]);
            if (!src0) { run_fallback(); return; }
            if ((op.kind == KernelOpKind::MatMul || op.kind == KernelOpKind::ElementwiseAdd || op.kind == KernelOpKind::ElementwiseMul) &&
                !src1 && inputs.size() > 1) {
                src1 = make_buffer_from_tensor(inputs[1]);
            }
            if ((op.kind == KernelOpKind::MatMul || op.kind == KernelOpKind::ElementwiseAdd || op.kind == KernelOpKind::ElementwiseMul) &&
                !src1) { run_fallback(); return; }

            id<MTLBuffer> dst = alloc_out_buffer(op);
            if (!dst) { run_fallback(); return; }
            dispatch_op(op, seg_pipe(i), src0, src1, dst, cmd);
            buf_map[op.output] = dst;
            final_buf = dst;
        }

        [cmd commit];
        [cmd waitUntilCompleted];

        if (final_buf) {
            std::memcpy(outputs[0].data(), [final_buf contents], outputs[0].get_byte_size());
        }

        for (auto& kv : buf_map) [kv.second release];
        return;
    }

private:
    void compile(const std::shared_ptr<const ov::Model>& model) {
        MetalKernelCompiler compiler(m_device);
        std::string log;
        m_force_fallback = false;
        m_has_const_b = false;
        m_has_const_w = false;
        m_has_const_mul = false;
        m_const_b.clear();
        m_const_w.clear();
        m_const_mul.clear();
        m_const_bn.clear();
        m_has_const_mm0 = m_has_const_mm1 = false;
        m_const_mm0.clear();
        m_const_mm1.clear();
        m_ops_from_flat_segment = false;
        m_add_const_input_index = -1;
        m_ops.clear();
        m_segments.clear();

        auto softmax_supported_globally = [&](const std::shared_ptr<const ov::Model>& m) -> bool {
            for (const auto& node : m->get_ordered_ops()) {
                if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) {
                    if (!is_softmax_shape_supported(node->get_input_partial_shape(0), s1->get_axis()))
                        return false;
                } else if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) {
                    if (!is_softmax_shape_supported(node->get_input_partial_shape(0), s8->get_axis()))
                        return false;
                }
            }
            return true;
        };

        if (model_has_dynamic_shape(model)) {
            auto sm = single_softmax_axis(model);
            if (sm.first) {
                m_softmax_dynamic_only = true;
                m_softmax_axis = sm.second;
                debug_log("[METAL MLIR] Detected dynamic Softmax graph; host softmax path enabled");
            }
        }

        if (softmax_has_dynamic_node(model)) {
            debug_log("[METAL MLIR] Softmax has dynamic dims; enabling host softmax path and continuing");
            auto sm = single_softmax_axis(model);
            if (sm.first) {
                m_softmax_dynamic_only = true;
                m_softmax_axis = sm.second;
            }
        }

        if (!softmax_supported_globally(model)) {
            debug_log("[METAL MLIR] Softmax unsupported shape/axis detected");
            if (!m_allow_partial_offload) {
                OPENVINO_THROW("METAL: model contains unsupported Softmax configuration for this device");
            }
            m_force_fallback = true;
            ensure_fallback(m_original_model);
            return;
        }

        const auto caps = default_capabilities();
        const auto analysis = analyze_model_for_mlir(model, caps);
        const bool segments_enabled = true;  // enable flat segments by default
        auto force_cpu_fallback = [&](const std::shared_ptr<const ov::Model>& m) {
            if (!m_allow_partial_offload) {
                OPENVINO_THROW("METAL: model contains unsupported ops for this device");
            }
            m_force_fallback = true;
            ensure_fallback(m);
        };
        auto force_cpu_fallback_default = [&]() { force_cpu_fallback(m_original_model); };

        auto precision_supported = [](ov::element::Type t) {
            return t == ov::element::f32 || t == ov::element::f16;
        };

        if (!precision_supported(m_inference_precision)) {
            debug_log("[METAL MLIR] Unsupported inference_precision " +
                      std::string(m_inference_precision.get_type_name()));
            force_cpu_fallback_default();
            return;
        }

        auto set_ops_from_ir = [&]() {
            m_ops = m_ir.ops;
            m_segments.clear();
            if (!m_ops.empty()) {
                m_segments.push_back(Segment{0, m_ops.size()});
            }
            m_ops_from_flat_segment = false;
        };
        if (analysis.has_unsupported_ops) {
            log_list("[MlirBackend] Unsupported ops → CPU fallback: ", analysis.unsupported_list);
            force_cpu_fallback(m_original_model);
            return;
        }

        if (analysis.has_disabled_ops) {
            log_list("[MlirBackend] Ops disabled by capability flags → CPU fallback: ", analysis.disabled_list);
            force_cpu_fallback(m_original_model);
            return;
        }

        if (analysis.has_future_ops) {
            log_list("[MlirBackend] Ops recognized but not lowered yet → CPU fallback: ", analysis.future_list);
            force_cpu_fallback(m_original_model);
            return;
        }

        debug_log("[MlirBackend] analysis compute_ops=" + std::to_string(analysis.compute_ops) +
                  " matmul=" + (analysis.has_matmul ? "1" : "0") +
                  " softmax=" + (analysis.has_softmax ? "1" : "0"));

#if METAL_MLIR_DEBUG
        build_flat_ir(model);
        build_segments_from_flat_ir();
#endif

        auto fill_constants_for_segment = [&](const Segment& seg) {
            // Walk ordered_ops in lockstep with m_flat_ops to harvest constants
            size_t flat_idx = 0;
            size_t mm_seen = 0;
            auto tensor_num_elems = [](const KernelTensor* t) -> size_t {
                if (!t) return 0;
                size_t elems = 1;
                for (auto d : t->shape) elems *= static_cast<size_t>(d);
                return elems;
            };
            for (const auto& node : model->get_ordered_ops()) {
                if (ov::is_type<ov::op::v0::Parameter>(node.get()) ||
                    ov::is_type<ov::op::v0::Result>(node.get()) ||
                    ov::is_type<ov::op::v0::Constant>(node.get())) {
                continue;
                }

                auto node_kind = [&](const std::shared_ptr<const ov::Node>& n, KernelOpKind& kind_out) -> bool {
                    if (ov::as_type_ptr<const ov::op::v0::MatMul>(n)) { kind_out = KernelOpKind::MatMul; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Add>(n)) { kind_out = KernelOpKind::ElementwiseAdd; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Multiply>(n)) { kind_out = KernelOpKind::ElementwiseMul; return true; }
                    if (ov::as_type_ptr<const ov::op::v0::Relu>(n) || ov::as_type_ptr<const ov::op::v0::Sigmoid>(n) ||
                        ov::as_type_ptr<const ov::op::v0::Tanh>(n) || ov::as_type_ptr<const ov::op::v0::Elu>(n) ||
                        ov::as_type_ptr<const ov::op::v0::PRelu>(n) || ov::as_type_ptr<const ov::op::v0::Gelu>(n) ||
                        ov::as_type_ptr<const ov::op::v4::Swish>(n)) { kind_out = KernelOpKind::Unary; return true; }
                    if (ov::is_type<ov::op::v1::Softmax>(n.get()) || ov::is_type<ov::op::v8::Softmax>(n.get())) { kind_out = KernelOpKind::Softmax; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::MaxPool>(n)) { kind_out = KernelOpKind::MaxPool2D; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::AvgPool>(n)) { kind_out = KernelOpKind::AvgPool2D; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Convolution>(n) || ov::as_type_ptr<const ov::op::v1::GroupConvolution>(n)) {
                        kind_out = KernelOpKind::Conv2D; return true; }
                    if (ov::as_type_ptr<const ov::op::v0::BatchNormInference>(n) || ov::as_type_ptr<const ov::op::v5::BatchNormInference>(n)) {
                        kind_out = KernelOpKind::BatchNorm2D; return true; }
                    return false;
                };

                KernelOpKind kind;
                if (!node_kind(node, kind))
                    continue;  // shape-only or unsupported

                if (flat_idx < seg.first_op_index) {
                    ++flat_idx;
                    continue;
                }
                if (flat_idx >= seg.first_op_index + seg.op_count)
                    break;

                auto& op = m_flat_ops[flat_idx];
                switch (kind) {
                    case KernelOpKind::MatMul: {
                        auto weight = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(1));
                        auto mm_node = std::dynamic_pointer_cast<const ov::op::v0::MatMul>(node);
                        const bool transpose_b = mm_node ? mm_node->get_transpose_b() : false;
                        // Reflect transpose_b in op flag
                        op.b_is_nk_layout = transpose_b;
                        m_flat_ops[flat_idx].b_is_nk_layout = transpose_b;
                        if (weight) {
                            auto vec = weight->cast_vector<float>();
                            const auto& wshape = weight->get_shape();  // original constant shape
                            auto desired_K = op.K;
                            auto desired_N = op.N;
                            OPENVINO_ASSERT(desired_K > 0 && desired_N > 0, "MatMul weights: invalid K/N");
                            const bool produce_nk = transpose_b;  // true -> store N x K layout
                            auto normalize_layout = [&](std::vector<float>& data) {
                                size_t batches = 1;
                                if (wshape.size() == 3)
                                    batches = wshape[0];
                                const size_t K = static_cast<size_t>(desired_K);
                                const size_t N = static_cast<size_t>(desired_N);
                                std::vector<float> dst(static_cast<size_t>(batches * K * N), 0.f);

                                auto read_at = [&](size_t b, size_t r, size_t c) -> float {
                                    // r,c follow source ordering as stored
                                    if (wshape.size() == 2) {
                                        if (r >= wshape[0] || c >= wshape[1]) return 0.f;
                                        return data[r * wshape[1] + c];
                                    } else {  // rank-3: [B, R, C]
                                        if (b >= wshape[0] || r >= wshape[1] || c >= wshape[2]) return 0.f;
                                        return data[(b * wshape[1] + r) * wshape[2] + c];
                                    }
                                };

                                auto write_dst = [&](size_t b, size_t k, size_t n, float v) {
                                    if (produce_nk) {
                                        // store as [N, K] row-major (n major)
                                        dst[(b * N + n) * K + k] = v;
                                    } else {
                                        // store as [K, N] row-major (k major)
                                        dst[(b * K + k) * N + n] = v;
                                    }
                                };

                                auto fill_from = [&](bool source_is_kxn) {
                                    for (size_t b = 0; b < batches; ++b) {
                                        for (size_t k = 0; k < K; ++k) {
                                            for (size_t n = 0; n < N; ++n) {
                                                float v = source_is_kxn ? read_at(b, k, n) : read_at(b, n, k);
                                                write_dst(b, k, n, v);
                                            }
                                        }
                                    }
                                };

                                if (wshape.size() == 2) {
                                    const size_t W0 = wshape[0], W1 = wshape[1];
                                    if (W0 == K && W1 == N) {
                                        fill_from(true);
                                    } else if (W0 == N && W1 == K) {
                                        fill_from(false);
                                    } else {
                                        OPENVINO_ASSERT(false, "MatMul weights shape unsupported");
                                    }
                                } else if (wshape.size() == 3) {
                                    const size_t W1 = wshape[1], W2 = wshape[2];
                                    if (W1 == K && W2 == N) {
                                        fill_from(true);
                                    } else if (W1 == N && W2 == K) {
                                        fill_from(false);
                                    } else {
                                        OPENVINO_ASSERT(false, "MatMul weights rank-3 shape unsupported");
                                    }
                                } else {
                                    OPENVINO_ASSERT(false, "MatMul weights rank not supported");
                                }
                                data.swap(dst);
                            };

                            normalize_layout(vec);
                            if (mm_seen == 0) {
                                m_const_mm0 = vec;
                                m_has_const_mm0 = true;
                            } else {
                                m_const_mm1 = vec;
                                m_has_const_mm1 = true;
                            }
                            m_const_b = vec;  // reused for first MatMul weight
                            m_has_const_b = true;
                        }
                        ++mm_seen;
                        break;
                    }
                    case KernelOpKind::ElementwiseMul: {
                        for (size_t i = 0; i < node->get_input_size(); ++i) {
                            if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(i))) {
                                m_const_mul = c->cast_vector<float>();
                                m_has_const_mul = true;
                                break;
                            }
                        }
                        break;
                    }
                    case KernelOpKind::ElementwiseAdd: {
                        size_t add_const_elems = 0;
                        for (size_t i = 0; i < node->get_input_size(); ++i) {
                            if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(i))) {
                                const auto& c_shape = c->get_shape();
                                add_const_elems = std::accumulate(c_shape.begin(), c_shape.end(), static_cast<size_t>(1),
                                                                  std::multiplies<size_t>());
                                if (!m_has_const_b) {
                                    m_const_b = c->cast_vector<float>();
                                    m_has_const_b = true;
                                    m_add_const_input_index = static_cast<int>(i);
                                }
                                break;
                            }
                        }
                        break;
                    }
                    case KernelOpKind::Conv2D: {
                        if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(node->get_input_node_shared_ptr(1))) {
                            m_const_w = c->cast_vector<float>();
                            m_has_const_w = true;
                        }
                        break;
                    }
                    case KernelOpKind::BatchNorm2D: {
                        const auto bn_v5 = ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node);
                        const auto bn_v0 = ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node);
                        const bool use_v5 = static_cast<bool>(bn_v5);
                        auto get_const = [&](size_t idx) {
                            return std::dynamic_pointer_cast<const ov::op::v0::Constant>(use_v5
                                ? bn_v5->get_input_node_shared_ptr(idx)
                                : bn_v0->get_input_node_shared_ptr(idx));
                        };
                        auto gamma = get_const(1);
                        auto beta = get_const(2);
                        auto mean = get_const(3);
                        auto var = get_const(4);
                        if (gamma && beta && mean && var) {
                            const size_t C = node->get_input_shape(0)[1];
                            op.bn_params.assign(4 * C + 1, 0.f);
                            auto copy_vec = [&](const std::shared_ptr<const ov::op::v0::Constant>& cst, size_t offset) {
                                auto vec = cst->cast_vector<float>();
                                if (vec.size() == C)
                                    std::copy(vec.begin(), vec.end(), op.bn_params.begin() + offset);
                            };
                            copy_vec(gamma, 0 * C);
                            copy_vec(beta, 1 * C);
                            copy_vec(mean, 2 * C);
                            copy_vec(var, 3 * C);
                            op.bn_params[4 * C] = static_cast<float>(use_v5 ? bn_v5->get_eps_value() : bn_v0->get_eps_value());
                            m_const_bn = op.bn_params;
                        }
                        break;
                    }
                    default:
                        break;
                }

                ++flat_idx;
            }
        };

        auto try_pick_single_flat_segment = [&]() {
            if (!m_ops.empty())
                return;
            auto allow_kind = [](KernelOpKind k) {
                switch (k) {
                    case KernelOpKind::Unary:
                    case KernelOpKind::Softmax:
                    case KernelOpKind::Conv2D:
                    case KernelOpKind::BatchNorm2D:
                    case KernelOpKind::MatMul:
                    case KernelOpKind::ElementwiseAdd:
                    case KernelOpKind::ElementwiseMul:
                        return true;
                    default:
                        return false;
                }
            };
            for (const auto& seg : m_flat_segments) {
                if (seg.op_count != 1)
                    continue;
                // only if covers full model
                if (seg.first_op_index != 0 || seg.op_count != m_flat_ops.size())
                    continue;
                const auto& op = m_flat_ops[seg.first_op_index];
                if (!allow_kind(op.kind))
                    continue;
                m_ops.clear();
                m_segments.clear();
                fill_constants_for_segment(seg);
                m_ops.push_back(op);
                m_segments.push_back(Segment{0, 1});
                m_ops_from_flat_segment = true;
#if METAL_MLIR_DEBUG
                debug_log("[METAL MLIR] Using single-op flat segment kind=" + std::to_string(static_cast<int>(op.kind)));
#endif
                // compile pipeline for this single op
                MetalKernelCompiler compiler(m_device);
                std::string log;
                m_pipelines.clear();
                switch (op.kind) {
                    case KernelOpKind::Unary:
                        m_pipelines.push_back(compiler.compile_unary_kernel(op, log));
                        break;
                    case KernelOpKind::Softmax:
                        m_pipelines.push_back(compiler.compile_softmax_kernel(op, log));
                        break;
                    case KernelOpKind::Conv2D:
                        m_pipelines.push_back(compiler.compile_conv2d_kernel(op, log));
                        break;
                    case KernelOpKind::BatchNorm2D:
                        m_pipelines.push_back(compiler.compile_batchnorm2d_kernel(op, log));
                        break;
                    case KernelOpKind::MatMul:
                        m_pipelines.push_back(compiler.compile_matmul_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseAdd:
                        m_pipelines.push_back(compiler.compile_add_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseMul:
                        m_pipelines.push_back(compiler.compile_mul_kernel(op, log));
                        break;
                    default:
                        break;
                }
                break;
            }
        };

        if (segments_enabled) {
            try_pick_single_flat_segment();
        }
        auto try_pick_multi_flat_segment = [&]() {
            if (!m_ops.empty())
                return;
            // Scan flat ops for the first MatMul -> Softmax -> MatMul chain.
            for (size_t i = 0; i + 2 < m_flat_ops.size(); ++i) {
                const auto& op0 = m_flat_ops[i];
                const auto& op1 = m_flat_ops[i + 1];
                const auto& op2 = m_flat_ops[i + 2];
                if (!(op0.kind == KernelOpKind::MatMul &&
                      op1.kind == KernelOpKind::Softmax &&
                      op2.kind == KernelOpKind::MatMul)) {
                    continue;
                }
                // Basic chaining: outputs/inputs must be the same tensor objects.
                if (!(op0.output && op1.input0 && op0.output == op1.input0))
                    continue;
                if (!(op1.output && op2.input0 && op1.output == op2.input0))
                    continue;
                m_ops.clear();
                m_segments.clear();
                m_pipelines.clear();
                m_ops_from_flat_segment = true;
                Segment seg{i, 3};
                if (!m_model->inputs().empty())
                    seg.input_ports.push_back(m_model->inputs()[0]);
                if (!m_model->outputs().empty())
                    seg.output_ports.push_back(m_model->outputs()[0]);
                fill_constants_for_segment(seg);
                MetalKernelCompiler compiler(m_device);
                std::string log;
                m_ops.push_back(m_flat_ops[i + 0]);
                m_ops.push_back(m_flat_ops[i + 1]);
                m_ops.push_back(m_flat_ops[i + 2]);
                // Keep MatMul weights in KxN layout for this conservative triple path
                m_ops[0].b_is_nk_layout = false;
                m_ops[2].b_is_nk_layout = false;
                m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[0], log));
                m_pipelines.push_back(compiler.compile_softmax_kernel(m_ops[1], log));
                m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[2], log));
                m_segments.push_back(Segment{0, 3});
#if METAL_MLIR_DEBUG
                debug_log("[METAL MLIR] Using flat segment MatMul->Softmax->MatMul len=3");
#endif
                break;
            }
        };
        if (segments_enabled) {
            try_pick_multi_flat_segment();
            // If no template matched, try to run full flat IR as one segment
            if (m_ops.empty() && !m_flat_ops.empty() && !m_flat_segments.empty()) {
                m_ops = m_flat_ops;
                m_segments = m_flat_segments;
                m_ops_from_flat_segment = true;
                MetalKernelCompiler compiler(m_device);
                std::string log;
                m_pipelines.clear();
                bool compile_ok = true;
                for (const auto& op : m_ops) {
                    switch (op.kind) {
                        case KernelOpKind::MatMul:
                            m_pipelines.push_back(compiler.compile_matmul_kernel(op, log));
                            break;
                        case KernelOpKind::Softmax:
                            m_pipelines.push_back(compiler.compile_softmax_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwiseAdd:
                            m_pipelines.push_back(compiler.compile_add_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwiseMul:
                            m_pipelines.push_back(compiler.compile_mul_kernel(op, log));
                            break;
                        case KernelOpKind::Unary:
                            m_pipelines.push_back(compiler.compile_unary_kernel(op, log));
                            break;
                        default:
                            compile_ok = false;
                            break;
                    }
                }
                if (!compile_ok || m_pipelines.size() != m_ops.size()) {
                    debug_log("[METAL MLIR] Fallback: flat IR contains unsupported op for Metal dispatch");
                    m_ops.clear();
                    m_segments.clear();
                } else {
#if METAL_MLIR_DEBUG
                    debug_log("[METAL MLIR] Using full flat IR segment count=" + std::to_string(m_ops.size()));
#endif
                }
            }
        }

        if (!m_ops.empty()) {
            return;
        }

        if (analysis.compute_ops > 32) {
            debug_log("[MlirBackend] Warning: large compute_ops=" + std::to_string(analysis.compute_ops) +
                      " (not forcing fallback)");
        }

        bool has_matmul = analysis.has_matmul;
        bool has_add = analysis.has_add;
        bool has_mul = analysis.has_mul;
        bool has_add_broadcast = analysis.has_add_broadcast;
        bool has_maxpool = analysis.has_maxpool;
        bool has_avgpool = analysis.has_avgpool;
        bool has_conv2d = analysis.has_conv2d;
        bool has_batchnorm = analysis.has_batchnorm;
        // Probe for constant B to support single-input matmuls
        std::shared_ptr<const ov::op::v0::MatMul> matmul_node;
        for (const auto& node : model->get_ordered_ops()) {
            if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
                matmul_node = mm;
                break;
            }
        }
        if (!has_matmul && !has_add && !has_add_broadcast && !has_mul && !analysis.has_unary && !analysis.has_softmax &&
            !analysis.has_maxpool && !analysis.has_avgpool && !analysis.has_conv2d && !analysis.has_batchnorm) {
            force_cpu_fallback(m_original_model);
            return;
        }
        if (matmul_node) {
            auto input1 = matmul_node->get_input_node_shared_ptr(1);
            if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(input1)) {
                m_has_const_b = true;
                m_const_b = c->cast_vector<float>();
            }
        }

        bool built_from_mlir = false;
        if (has_matmul && analysis.compute_ops == 1) {
            debug_log("[MlirBackend compile] single MatMul path");
            try {
            mlir::MLIRContext ctx;
            auto module = build_mlir_module_from_model(model, ctx);
            run_mlir_pipeline(module);

            int64_t M = 0, N = 0, K = 0;
            extract_matmul_shape(module, M, N, K);
            OPENVINO_ASSERT(M > 0 && N > 0 && K > 0, "MlirBackend: invalid MatMul dims");

            MetalKernelIR ir;
            ir.tensors.push_back({"a", {M, K}});
            ir.tensors.push_back({"b", {K, N}});
            ir.tensors.push_back({"c", {M, N}});

            KernelOp op{};
            op.kind = KernelOpKind::MatMul;
            op.input0 = &ir.tensors[0];
            op.input1 = &ir.tensors[1];
            op.output = &ir.tensors[2];
            op.M = M;
            op.N = N;
            op.K = K;
            op.batch = 1;
            op.batch_a = 1;
            op.batch_b = 1;
            op.b_is_nk_layout = false;
            ir.ops.push_back(op);

            m_ir = std::move(ir);
            set_ops_from_ir();
            m_pipelines.clear(); m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[0], log));
            built_from_mlir = true;
            } catch (const std::exception&) {
                debug_log("[MlirBackend compile] MLIR MatMul path failed, fallback to manual MatMul");
                // fall through to manual MatMul handling (no Add fallback)
            }
        }
        if (has_matmul && built_from_mlir) {
            return;
        }

        if (has_matmul && !built_from_mlir && analysis.compute_ops == 1) {
            // Manual shape extraction for MatMul (including simple batch broadcast)
            MetalKernelIR ir = build_kernel_ir_for_matmul(model);
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_pipelines.clear(); m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[0], log));
            debug_log("MlirBackend MatMul: M=" + std::to_string(m_ops[0].M) + " N=" +
                      std::to_string(m_ops[0].N) + " K=" + std::to_string(m_ops[0].K) +
                      " batch=" + std::to_string(m_ops[0].batch) + " batch_a=" +
                      std::to_string(m_ops[0].batch_a) + " batch_b=" + std::to_string(m_ops[0].batch_b));
            return;
        }

        if (has_matmul && (analysis.has_softmax || analysis.has_unary) && analysis.compute_ops == 2) {
            MetalKernelIR ir = build_kernel_ir_for_matmul(model);
            KernelTensor out_tensor = ir.tensors.back();
            KernelTensor second_out = {"stage1_out", out_tensor.shape};
            ir.tensors.push_back(second_out);

            KernelOp second{};
            if (analysis.has_softmax) {
                second.kind = KernelOpKind::Softmax;
                if (out_tensor.shape.size() == 2) {
                    second.rows = out_tensor.shape[0];
                    second.cols = out_tensor.shape[1];
                    second.inner = 1;
                } else if (out_tensor.shape.size() == 3) {
                    second.rows = out_tensor.shape[0] * out_tensor.shape[1];
                    second.cols = out_tensor.shape[2];
                    second.inner = 1;
                } else {
                    second.rows = 1;
                    for (size_t i = 0; i + 1 < out_tensor.shape.size(); ++i)
                        second.rows *= out_tensor.shape[i];
                    second.cols = out_tensor.shape.back();
                    second.inner = 1;
                }
            } else {
                second.kind = KernelOpKind::Unary;
                second.activation = analysis.unary_kind;
                second.alpha = analysis.unary_alpha;
            }
            second.input0 = &ir.tensors[ir.tensors.size() - 2];
            second.output = &ir.tensors.back();
            ir.ops.push_back(second);

            m_ir = std::move(ir);
            set_ops_from_ir();
            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[0], log));
            if (analysis.has_softmax)
            m_pipelines.push_back(compiler.compile_softmax_kernel(m_ops[1], log));
            else
                m_pipelines.push_back(compiler.compile_unary_kernel(m_ops[1], log));
            return;
        }

        if (has_matmul && analysis.has_softmax && !has_add && !has_add_broadcast && !analysis.has_unary &&
            !has_maxpool && !has_avgpool && !has_conv2d && !has_batchnorm && analysis.compute_ops == 3) {
            std::shared_ptr<const ov::op::v0::MatMul> mm0, mm1;
            std::shared_ptr<const ov::Node> sm;
            for (const auto& node : model->get_ordered_ops()) {
                if (!mm0) {
                    if (auto m = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) { mm0 = m; continue; }
                } else if (!sm) {
                    if (ov::is_type<ov::op::v1::Softmax>(node.get()) || ov::is_type<ov::op::v8::Softmax>(node.get())) {
                        sm = node; continue;
                    }
                } else if (!mm1) {
                    if (auto m = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) { mm1 = m; break; }
                }
            }
            if (!mm0 || !sm || !mm1) {
                force_cpu_fallback(m_original_model);
                return;
            }

            auto expand_to_3d = [](const ov::Shape& s) {
                OPENVINO_ASSERT(s.size() == 2 || s.size() == 3, "MatMul attention: only rank-2/3 inputs supported");
                if (s.size() == 3) return std::vector<int64_t>{static_cast<int64_t>(s[0]), static_cast<int64_t>(s[1]), static_cast<int64_t>(s[2])};
                return std::vector<int64_t>{1, static_cast<int64_t>(s[0]), static_cast<int64_t>(s[1])};
            };
            auto make_matmul_op = [&](const ov::Shape& a_shape,
                                      const ov::Shape& b_shape,
                                      KernelTensor* a_tensor,
                                      KernelTensor* b_tensor,
                                      KernelTensor* c_tensor) -> KernelOp {
                KernelOp op;
                op.kind = KernelOpKind::MatMul;
                op.input0 = a_tensor;
                op.input1 = b_tensor;
                op.output = c_tensor;
                auto a3 = expand_to_3d(a_shape);
                auto b3 = expand_to_3d(b_shape);
                const int64_t batch_a = a3[0];
                const int64_t batch_b = b3[0];
                const int64_t batch = std::max(batch_a, batch_b);
                OPENVINO_ASSERT(batch_a == batch_b || batch_a == 1 || batch_b == 1, "MatMul attention: batch broadcast mismatch");
                const int64_t M = a3[1];
                const int64_t K_a = a3[2];
                int64_t K_b = b3[1];
                int64_t N = b3[2];
                bool b_is_nk = false;
                if (K_b != K_a && b3[2] == K_a) {
                    K_b = b3[2];
                    N = b3[1];
                    b_is_nk = true;
                }
                // For attention weight matrices (typically laid out [D,D]) force NK layout to match kernel expectations.
                b_is_nk = true;
                OPENVINO_ASSERT(K_a == K_b, "MatMul attention: K mismatch");
                op.M = M;
                op.N = N;
                op.K = K_a;
                op.batch = batch;
                op.batch_a = batch_a;
                op.batch_b = batch_b;
                op.b_is_nk_layout = b_is_nk;
                return op;
            };

            const auto a0_shape = mm0->get_input_shape(0);
            const auto b0_shape = mm0->get_input_shape(1);
            const auto out0_shape = mm0->get_output_shape(0);
            const auto sm_shape = out0_shape;
            const auto a1_shape = mm1->get_input_shape(0);
            const auto b1_shape = mm1->get_input_shape(1);
            const auto out1_shape = mm1->get_output_shape(0);

            KernelTensor t_in{"in", {a0_shape.begin(), a0_shape.end()}};
            KernelTensor t_w0{"w0", {b0_shape.begin(), b0_shape.end()}};
            KernelTensor t_tmp1{"tmp1", {out0_shape.begin(), out0_shape.end()}};
            KernelTensor t_tmp2{"tmp2", {sm_shape.begin(), sm_shape.end()}};
            KernelTensor t_w1{"w1", {b1_shape.begin(), b1_shape.end()}};
            KernelTensor t_out{"out", {out1_shape.begin(), out1_shape.end()}};

            MetalKernelIR ir;
            ir.tensors = {t_in, t_w0, t_tmp1, t_tmp2, t_w1, t_out};

            KernelOp op0 = make_matmul_op(a0_shape, b0_shape, &ir.tensors[0], &ir.tensors[1], &ir.tensors[2]);
            KernelOp op1;
            op1.kind = KernelOpKind::Softmax;
            op1.input0 = &ir.tensors[2];
            op1.output = &ir.tensors[3];
            size_t elems = 1;
            for (auto d : sm_shape) elems *= d;
            const size_t cols = sm_shape.back();
            const size_t rows = elems / cols;
            op1.rows = static_cast<int64_t>(rows);
            op1.cols = static_cast<int64_t>(cols);
            op1.inner = 1;
            KernelOp op2 = make_matmul_op(a1_shape, b1_shape, &ir.tensors[3], &ir.tensors[4], &ir.tensors[5]);

            ir.ops = {op0, op1, op2};
            m_ir = std::move(ir);
            set_ops_from_ir();

            m_has_const_mm0 = false; m_has_const_mm1 = false;
            if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(mm0->get_input_node_shared_ptr(1))) {
                m_const_mm0 = c->cast_vector<float>();
                m_has_const_mm0 = true;
            }
            if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(mm1->get_input_node_shared_ptr(1))) {
                m_const_mm1 = c->cast_vector<float>();
                m_has_const_mm1 = true;
            }

            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[0], log));
            m_pipelines.push_back(compiler.compile_softmax_kernel(m_ops[1], log));
            m_pipelines.push_back(compiler.compile_matmul_kernel(m_ops[2], log));
            return;
        }

        if (has_add && !has_matmul && !has_add_broadcast && !analysis.has_unary) {
            MetalKernelIR ir = build_kernel_ir_for_add(model);
            // detect const second input
            for (const auto& node : model->get_ordered_ops()) {
                if (auto add = ov::as_type_ptr<const ov::op::v1::Add>(node)) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(add->get_input_node_shared_ptr(1))) {
                        m_has_const_b = true;
                        m_const_b = c->cast_vector<float>();
                    }
                    break;
                }
            }
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_pipelines.clear(); m_pipelines.push_back(compiler.compile_add_kernel(m_ops[0], log));
            return;
        }

        if (has_add_broadcast && !has_mul && !has_matmul && !analysis.has_unary && !analysis.has_softmax) {
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_broadcast_add_from_model(model, ctx);
                run_mlir_pipeline(module);
            } catch (const std::exception& e) {
                std::cerr << "[MlirBackend] Broadcast Add MLIR build failed, continue with kernel path: " << e.what()
                          << "\n";
            }

            for (const auto& node : model->get_ordered_ops()) {
                if (auto add = ov::as_type_ptr<const ov::op::v1::Add>(node)) {
                    for (int idx = 0; idx < 2; ++idx) {
                        if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(add->get_input_node_shared_ptr(idx))) {
                            m_has_const_b = true;
                            m_const_b = c->cast_vector<float>();
                            break;
                        }
                    }
                    break;
                }
            }

            MetalKernelIR ir = build_kernel_ir_for_broadcast_add(model);
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_pipelines.clear(); m_pipelines.push_back(compiler.compile_add_kernel(m_ops[0], log));
            return;
        }

        if (has_mul && analysis.compute_ops == 1 && !has_add && !has_add_broadcast && !analysis.has_unary &&
            !analysis.has_softmax && !has_maxpool && !has_avgpool && !has_conv2d && !has_batchnorm) {
            // Single multiply (broadcast or equal shape)
            MetalKernelIR ir = build_kernel_ir_for_broadcast_mul(model);
            m_ir = std::move(ir);
            set_ops_from_ir();
            for (const auto& node : model->get_ordered_ops()) {
                if (auto mul = ov::as_type_ptr<const ov::op::v1::Multiply>(node)) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(mul->get_input_node_shared_ptr(1))) {
                        m_const_mul = c->cast_vector<float>();
                        m_has_const_mul = true;
                    }
                    break;
                }
            }
            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_mul_kernel(m_ops[0], log));
            return;
        }

        if (analysis.has_softmax && !has_matmul && !has_add && !has_add_broadcast && !analysis.has_unary) {
            try {
                // Dynamic shapes or unsupported axis → CPU fallback.
                const auto pshape = model->input(0).get_partial_shape();
                if (!pshape.rank().is_static()) {
                    debug_log("[MlirBackend] Softmax dynamic rank, fallback to CPU");
                    force_cpu_fallback_default();
                    return;
                }
                for (auto d : pshape) {
                    if (!d.is_static()) {
                        debug_log("[MlirBackend] Softmax dynamic dimension, fallback to CPU");
                        force_cpu_fallback_default();
                        return;
                    }
                }
                int64_t axis = -1;
                for (const auto& node : model->get_ordered_ops()) {
                    if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) { axis = s1->get_axis(); break; }
                    if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) { axis = s8->get_axis(); break; }
                }
                if (!is_softmax_shape_supported(pshape, axis)) {
                    debug_log("[MlirBackend] Softmax shape/axis unsupported, fallback to CPU");
                    force_cpu_fallback_default();
                    return;
                }
                mlir::MLIRContext ctx;
                auto module = build_mlir_softmax_from_model(model, ctx);
                run_mlir_pipeline(module);
                MetalKernelIR ir = build_kernel_ir_for_softmax(model);
                m_ir = std::move(ir);
                set_ops_from_ir();
                m_pipelines.clear(); m_pipelines.push_back(compiler.compile_softmax_kernel(m_ops[0], log));
                return;
            } catch (const std::exception& e) {
                std::cerr << "[MlirBackend] Softmax compile failed, fallback: " << e.what() << "\n";
                force_cpu_fallback_default();
                return;
            }
        }


        if (has_mul && (has_add || has_add_broadcast) && analysis.compute_ops == 2) {
            // Pattern: per-channel affine y = scale * x + shift (BatchNorm decomposed by front-end).
            // Re-map to BatchNorm2D kernel with mean=0, var=1, eps=0 so that
            // y = gamma * (x - 0) / sqrt(1 + 0) + beta = scale * x + shift.
            std::vector<float> mul_const, add_const;
            bool mul_const_found = false, add_const_found = false;
            for (const auto& node : model->get_ordered_ops()) {
                if (auto mul = ov::as_type_ptr<const ov::op::v1::Multiply>(node)) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(mul->get_input_node_shared_ptr(1))) {
                        mul_const = c->cast_vector<float>();
                        mul_const_found = true;
                    }
                } else if (auto add = ov::as_type_ptr<const ov::op::v1::Add>(node)) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(add->get_input_node_shared_ptr(1))) {
                        add_const = c->cast_vector<float>();
                        add_const_found = true;
                    }
                }
            }
            if (!mul_const_found || !add_const_found) {
                force_cpu_fallback(m_original_model);
                return;
            }

            MetalKernelIR ir;
            // Infer shapes from multiply input and add output
            std::shared_ptr<const ov::Node> mul_node;
            std::shared_ptr<const ov::Node> add_node;
            for (const auto& node : model->get_ordered_ops()) {
                if (!mul_node && ov::as_type_ptr<const ov::op::v1::Multiply>(node)) mul_node = node;
                if (!add_node && ov::as_type_ptr<const ov::op::v1::Add>(node)) add_node = node;
            }
            if (!mul_node || !add_node) {
                force_cpu_fallback(m_original_model);
                return;
            }
            const auto in_shape = mul_node->get_input_shape(0);  // assume NCHW
            const auto out_shape = add_node->get_output_shape(0);
            KernelTensor in_t{"in", {in_shape.begin(), in_shape.end()}};
            KernelTensor out_t{"out", {out_shape.begin(), out_shape.end()}};
            ir.tensors.push_back(in_t);
            ir.tensors.push_back(out_t);

            KernelOp bn_op;
            bn_op.kind = KernelOpKind::BatchNorm2D;
            bn_op.input0 = &ir.tensors[0];
            bn_op.output = &ir.tensors[1];
            OPENVINO_ASSERT(in_shape.size() == 4, "Mul+Add BN pattern expects rank-4 input");
            const size_t C = in_shape[1];
            OPENVINO_ASSERT(mul_const.size() == C || mul_const.size() == 1,
                            "Mul constant size must match channels or be scalar");
            OPENVINO_ASSERT(add_const.size() == C || add_const.size() == 1,
                            "Add constant size must match channels or be scalar");
            bn_op.batchnorm.N = static_cast<uint32_t>(in_shape[0]);
            bn_op.batchnorm.C = static_cast<uint32_t>(C);
            bn_op.batchnorm.H = static_cast<uint32_t>(in_shape[2]);
            bn_op.batchnorm.W = static_cast<uint32_t>(in_shape[3]);
            bn_op.batchnorm.eps = 0.f;

            bn_op.bn_params.assign(4 * C + 1, 0.f);
            auto fill_broadcast = [&](const std::vector<float>& src, size_t offset) {
                if (src.size() == 1) {
                    std::fill(bn_op.bn_params.begin() + offset, bn_op.bn_params.begin() + offset + C, src[0]);
                } else {
                    std::copy(src.begin(), src.end(), bn_op.bn_params.begin() + offset);
                }
            };
            fill_broadcast(mul_const, 0 * C);  // gamma/scale
            fill_broadcast(add_const, 1 * C);  // beta/shift
            std::fill(bn_op.bn_params.begin() + 2 * C, bn_op.bn_params.begin() + 3 * C, 0.f);  // mean=0
            std::fill(bn_op.bn_params.begin() + 3 * C, bn_op.bn_params.begin() + 4 * C, 1.f);  // var=1
            bn_op.bn_params[4 * C] = 0.f;  // eps

            ir.ops.push_back(bn_op);
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_ops_from_flat_segment = true;
            if (!m_ops.empty()) {
                m_const_bn = m_ops[0].bn_params;
            }
            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_batchnorm2d_kernel(m_ops[0], log));
            return;
        }

        if (has_conv2d && !has_add && !has_add_broadcast && !analysis.has_unary && !analysis.has_softmax &&
            !has_maxpool && !has_avgpool && !has_batchnorm) {
            bool const_w = false;
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_conv2d_from_model(model, ctx);
                run_mlir_pipeline(module);
            } catch (const std::exception& e) {
                std::cerr << "[MlirBackend] Conv2D MLIR build failed, fallback: " << e.what() << "\n";
                force_cpu_fallback(m_original_model);
                return;
            }
            MetalKernelIR ir = build_kernel_ir_for_conv2d(model, const_w);
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_ops_from_flat_segment = true;
            debug_log("[METAL MLIR] Conv2D selected: input " + describe_shape(m_ir.ops[0].input0) +
                      " weights " + describe_shape(m_ir.ops[0].input1) +
                      " output " + describe_shape(m_ir.ops[0].output));
            if (const_w && !m_ops.empty()) {
                std::shared_ptr<const ov::Node> conv_node;
                for (const auto& node : model->get_ordered_ops()) {
                    if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) { conv_node = c; break; }
                    if (auto g = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) { conv_node = g; break; }
                }
                if (conv_node) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(conv_node->get_input_node_shared_ptr(1))) {
                        m_const_w = c->cast_vector<float>();
                        m_has_const_w = true;
                        debug_log("[METAL MLIR] Conv2D const weights size=" + std::to_string(m_const_w.size()));

                        if (!m_const_w.empty()) {
                            const auto* wptr = m_const_w.data();
                            std::ostringstream oss;
                            oss << "[METAL MLIR] Conv2D W first: ";
                            for (size_t i = 0; i < std::min<size_t>(m_const_w.size(), 8); ++i) oss << wptr[i] << " ";
                            debug_log(oss.str());
                        }

                    }
                }
            }
            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_conv2d_kernel(m_ops[0], log));
            return;
        }

        if (has_conv2d && analysis.has_unary && analysis.compute_ops == 2) {
            bool const_w = false;
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_conv2d_from_model(model, ctx);
                run_mlir_pipeline(module);
            } catch (const std::exception& e) {
                std::cerr << "[MlirBackend] Conv2D+Unary MLIR build failed, fallback: " << e.what() << "\n";
                force_cpu_fallback(m_original_model);
                return;
            }
            MetalKernelIR ir = build_kernel_ir_for_conv2d(model, const_w);
            // Append unary op
            KernelOp unary_op;
            unary_op.kind = KernelOpKind::Unary;
            unary_op.activation = analysis.unary_kind;
            unary_op.alpha = analysis.unary_alpha;
            ir.ops.push_back(unary_op);
            // Wire tensors: reuse conv output as unary input/output
            ir.ops[1].input0 = ir.ops[0].output;
            ir.ops[1].output = ir.ops[0].output;
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_ops_from_flat_segment = true;

            if (const_w && !m_ops.empty()) {
                std::shared_ptr<const ov::Node> conv_node;
                for (const auto& node : model->get_ordered_ops()) {
                    if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) { conv_node = c; break; }
                    if (auto g = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) { conv_node = g; break; }
                }
                if (conv_node) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(conv_node->get_input_node_shared_ptr(1))) {
                        m_const_w = c->cast_vector<float>();
                        m_has_const_w = true;
                    }
                }
            }

            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_conv2d_kernel(m_ops[0], log));
            m_pipelines.push_back(compiler.compile_unary_kernel(m_ops[1], log));
            return;
        }

        if (has_conv2d && has_batchnorm && analysis.has_unary && analysis.compute_ops == 3) {
            bool const_w = false;
            bool const_bn = false;
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_conv2d_from_model(model, ctx);
                run_mlir_pipeline(module);
                auto module_bn = build_mlir_batchnorm_from_model(model, ctx);
                run_mlir_pipeline(module_bn);
            } catch (const std::exception& e) {
                std::cerr << "[MlirBackend] Conv2D+BN+Unary MLIR build failed, fallback: " << e.what() << "\n";
                force_cpu_fallback(m_original_model);
                return;
            }

            MetalKernelIR conv_ir = build_kernel_ir_for_conv2d(model, const_w);
            MetalKernelIR bn_ir = build_kernel_ir_for_batchnorm(model, const_bn);
            if (!const_bn) {
                force_cpu_fallback(m_original_model);
                return;
            }

            MetalKernelIR ir;
            // tensors: input, tmp1 (conv out), tmp2 (bn out)
            ir.tensors.push_back(conv_ir.tensors[0]);  // input
            KernelTensor tmp1 = {"tmp1", conv_ir.ops[0].output ? conv_ir.ops[0].output->shape : conv_ir.tensors[0].shape};
            KernelTensor tmp2 = {"tmp2", bn_ir.ops[0].output ? bn_ir.ops[0].output->shape : conv_ir.tensors[0].shape};
            ir.tensors.push_back(tmp1);
            ir.tensors.push_back(tmp2);

            KernelOp conv_op = conv_ir.ops[0];
            conv_op.input0 = &ir.tensors[0];
            conv_op.output = &ir.tensors[1];

            KernelOp bn_op = bn_ir.ops[0];
            bn_op.input0 = &ir.tensors[1];
            bn_op.output = &ir.tensors[2];

            KernelOp unary_op;
            unary_op.kind = KernelOpKind::Unary;
            unary_op.activation = analysis.unary_kind;
            unary_op.alpha = analysis.unary_alpha;
            unary_op.input0 = &ir.tensors[2];
            unary_op.output = &ir.tensors[2];

            ir.ops.push_back(conv_op);
            ir.ops.push_back(bn_op);
            ir.ops.push_back(unary_op);
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_ops_from_flat_segment = true;

            if (const_w && !conv_ir.ops.empty()) {
                std::shared_ptr<const ov::Node> conv_node;
                for (const auto& node : model->get_ordered_ops()) {
                    if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) { conv_node = c; break; }
                    if (auto g = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) { conv_node = g; break; }
                }
                if (conv_node) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(conv_node->get_input_node_shared_ptr(1))) {
                        m_const_w = c->cast_vector<float>();
                        m_has_const_w = true;
                    }
                }
            }
            if (const_bn && !bn_ir.ops.empty()) {
                m_const_bn = bn_ir.ops[0].bn_params;
            }

            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_conv2d_kernel(m_ops[0], log));
            m_pipelines.push_back(compiler.compile_batchnorm2d_kernel(m_ops[1], log));
            m_pipelines.push_back(compiler.compile_unary_kernel(m_ops[2], log));
            return;
        }

        if (has_batchnorm && !has_matmul && !has_add && !has_add_broadcast && !analysis.has_unary &&
            !analysis.has_softmax && !has_maxpool && !has_avgpool && !has_conv2d) {
            bool const_params = false;
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_batchnorm_from_model(model, ctx);
                run_mlir_pipeline(module);
            } catch (const std::exception& e) {
                std::cerr << "[MlirBackend] BatchNorm MLIR build failed, fallback: " << e.what() << "\n";
                force_cpu_fallback(m_original_model);
                return;
            }
            MetalKernelIR ir = build_kernel_ir_for_batchnorm(model, const_params);
            if (!const_params) {
                force_cpu_fallback(m_original_model);
                return;
            }
            if (!ir.ops.empty()) {
                m_const_bn = ir.ops[0].bn_params;
            }
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_pipelines.clear(); m_pipelines.push_back(compiler.compile_batchnorm2d_kernel(m_ops[0], log));
            return;
        }

#if OV_HAS_LAYER_NORM
#endif

        if (analysis.has_maxpool && !has_matmul && !has_add && !has_add_broadcast && !analysis.has_unary &&
            !analysis.has_softmax && !analysis.has_avgpool) {
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_maxpool_from_model(model, ctx);
                run_mlir_pipeline(module);
            } catch (const std::exception& e) {
                std::cerr << "[MlirBackend] MaxPool MLIR build failed, fallback: " << e.what() << "\n";
                force_cpu_fallback(m_original_model);
                return;
            }

            MetalKernelIR ir = build_kernel_ir_for_maxpool(model);
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_ops_from_flat_segment = true;
            m_pipelines.clear(); m_pipelines.push_back(compiler.compile_maxpool2d_kernel(m_ops[0], log));
            return;
        }

        if (analysis.has_avgpool && !has_matmul && !has_add && !has_add_broadcast && !analysis.has_unary &&
            !analysis.has_softmax && !analysis.has_maxpool) {
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_avgpool_from_model(model, ctx);
                run_mlir_pipeline(module);
            } catch (const std::exception& e) {
                std::cerr << "[MlirBackend] AvgPool MLIR build failed, fallback: " << e.what() << "\n";
                force_cpu_fallback(m_original_model);
                return;
            }

            MetalKernelIR ir = build_kernel_ir_for_avgpool(model);
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_ops_from_flat_segment = true;
            m_pipelines.clear(); m_pipelines.push_back(compiler.compile_avgpool2d_kernel(m_ops[0], log));
            return;
        }

        if (analysis.has_unary && analysis.compute_ops == 1) {
            // Build MLIR module to mirror unary op for pipeline consistency (currently not inspected further).
            try {
                mlir::MLIRContext ctx;
                auto op_node = analysis.has_unary ? model->get_ordered_ops().back() : nullptr;
                auto module = build_mlir_unary_from_node(op_node, ctx, analysis.unary_kind, analysis.unary_alpha);
                run_mlir_pipeline(module);
            } catch (const std::exception& e) {
                std::cerr << "[MlirBackend] Unary MLIR build failed, fallback: " << e.what() << "\n";
                force_cpu_fallback(m_original_model);
                return;
            }

            // Build kernel IR directly from node
            const std::shared_ptr<const ov::Node> unary_node = [&]() -> std::shared_ptr<const ov::Node> {
                for (const auto& node : model->get_ordered_ops()) {
                    if (!ov::is_type<ov::op::v0::Parameter>(node.get()) &&
                        !ov::is_type<ov::op::v0::Constant>(node.get()) &&
                        !ov::is_type<ov::op::v0::Result>(node.get())) {
                        return node;
                    }
                }
                return nullptr;
            }();
            if (!unary_node) {
                force_cpu_fallback(m_original_model);
                return;
            }
            MetalKernelIR ir = build_kernel_ir_for_unary(unary_node, analysis.unary_kind, analysis.unary_alpha);
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_pipelines.clear(); m_pipelines.push_back(compiler.compile_unary_kernel(m_ops[0], log));
            return;
        }

        if (m_ops.empty()) {
            debug_log("[MlirBackend] Fallback: no supported template patterns matched");
            force_cpu_fallback(m_original_model);
            return;
        }
        OPENVINO_THROW("MlirBackend: no supported ops found for METAL backend");
    }

    id<MTLDevice> m_device = nil;
    id<MTLCommandQueue> m_queue = nil;
    MetalKernelIR m_ir;
    std::vector<KernelOp> m_ops;               // flat view of ops (prep for segmented scheduling)
    std::vector<Segment> m_segments;           // segments over m_ops (currently single segment)
    // Diagnostic flat IR (not used for execution yet)
    std::vector<KernelOp> m_flat_ops;
    std::vector<KernelTensor> m_flat_tensors;
    std::vector<Segment> m_flat_segments;
    std::vector<id<MTLComputePipelineState>> m_pipelines;
    bool m_force_fallback = false;
    bool m_has_const_b = false;
    bool m_has_const_w = false;
    bool m_has_const_mul = false;
    std::vector<float> m_const_w;
    std::vector<float> m_const_bn;
    std::vector<float> m_const_b;
    std::vector<float> m_const_mul;
    // MatMul weight constants for chained attention blocks
    bool m_has_const_mm0 = false;
    bool m_has_const_mm1 = false;
    std::vector<float> m_const_mm0;
    std::vector<float> m_const_mm1;
    bool m_ops_from_flat_segment = false;
    int m_add_const_input_index = -1;  // 0 or 1 if Add constant detected during flat harvest
    ov::CompiledModel m_fallback_model;
    std::unique_ptr<ov::InferRequest> m_fallback_request;
    std::vector<ov::Output<const ov::Node>> m_fallback_inputs;
    std::vector<ov::Output<const ov::Node>> m_fallback_outputs;
    std::shared_ptr<const ov::Model> m_model;
    std::shared_ptr<const ov::Model> m_original_model;
    ov::element::Type m_inference_precision{ov::element::f32};
    bool m_allow_partial_offload = false;
    bool m_softmax_dynamic_only = false;
    int64_t m_softmax_axis = 0;

    void ensure_fallback(const std::shared_ptr<const ov::Model>& model) {
        if (m_fallback_request)
            return;
        ov::Core core;
        auto src_model = model ? model : m_original_model;
        m_fallback_model = core.compile_model(src_model, "CPU");
        m_fallback_inputs = m_fallback_model.inputs();
        m_fallback_outputs = m_fallback_model.outputs();
        m_fallback_request = std::make_unique<ov::InferRequest>(m_fallback_model.create_infer_request());
    }

    void ensure_fallback() {
        ensure_fallback(m_original_model);
    }

    // High-precision host softmax for dynamic-only graphs (single Softmax op).
    bool run_host_softmax(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) const {
        if (!m_softmax_dynamic_only || inputs.empty())
            return false;

        debug_log("[METAL MLIR] Host softmax path executing");
        const auto& in = inputs[0];
        auto shape = in.get_shape();
        if (shape.empty())
            return false;

        const int64_t rank = static_cast<int64_t>(shape.size());
        int64_t axis = m_softmax_axis;
        if (axis < 0)
            axis += rank;
        if (axis < 0 || axis >= rank)
            return false;

        int64_t cols = static_cast<int64_t>(shape[axis]);
        int64_t inner = 1;
        for (int64_t i = axis + 1; i < rank; ++i)
            inner *= static_cast<int64_t>(shape[static_cast<size_t>(i)]);
        int64_t outer = 1;
        for (int64_t i = 0; i < axis; ++i)
            outer *= static_cast<int64_t>(shape[static_cast<size_t>(i)]);
        int64_t rows = outer * inner;

        ov::Tensor out{in.get_element_type(), shape};
        const bool is_f16 = in.get_element_type() == ov::element::f16;
        const auto* src_f16 = is_f16 ? in.data<ov::float16>() : nullptr;
        const auto* src_f32 = !is_f16 ? in.data<const float>() : nullptr;
        auto* dst_f16 = is_f16 ? out.data<ov::float16>() : nullptr;
        auto* dst_f32 = !is_f16 ? out.data<float>() : nullptr;

        for (int64_t r = 0; r < rows; ++r) {
            int64_t outer_idx = r / inner;
            int64_t inner_idx = r - outer_idx * inner;
            int64_t base = outer_idx * cols * inner + inner_idx;

            double max_v = -std::numeric_limits<double>::infinity();
            for (int64_t c = 0; c < cols; ++c) {
                double v = is_f16 ? static_cast<double>(src_f16[base + c * inner])
                                  : static_cast<double>(src_f32[base + c * inner]);
                if (v > max_v)
                    max_v = v;
            }

            double sum = 0.0;
            for (int64_t c = 0; c < cols; ++c) {
                double v = is_f16 ? static_cast<double>(src_f16[base + c * inner])
                                  : static_cast<double>(src_f32[base + c * inner]);
                double e = std::exp(v - max_v);
                if (is_f16)
                    dst_f16[base + c * inner] = static_cast<ov::float16>(e);
                else
                    dst_f32[base + c * inner] = static_cast<float>(e);
                sum += e;
            }
            double inv_sum = 1.0 / sum;
            for (int64_t c = 0; c < cols; ++c) {
                if (is_f16) {
                    double v = static_cast<double>(dst_f16[base + c * inner]);
                    dst_f16[base + c * inner] = static_cast<ov::float16>(v * inv_sum);
                } else {
                    dst_f32[base + c * inner] = static_cast<float>(static_cast<double>(dst_f32[base + c * inner]) * inv_sum);
                }
            }
        }

        if (!outputs.empty()) {
            outputs[0] = out;
        }
        return true;
    }

public:
    bool has_segment() const {
        return !m_segments.empty();
    }

    bool segment_io_is_model_io() const {
        if (m_segments.empty())
            return false;
        const auto& seg = m_segments.front();
        // Require single input/output and that they correspond to model IO
        if (seg.input_ports.size() != 1 || seg.output_ports.size() != 1)
            return false;
        if (m_model->inputs().empty() || m_model->outputs().empty())
            return false;
        return seg.input_ports[0] == m_model->inputs()[0] && seg.output_ports[0] == m_model->outputs()[0];
    }

    const Segment& get_segment() const {
        OPENVINO_ASSERT(!m_segments.empty(), "MlirBackend: no segments available");
        return m_segments.front();
    }

    std::vector<ov::Tensor> run_segment(const Segment& seg, const std::vector<ov::Tensor>& inputs) {
        auto run_fallback = [&]() -> std::vector<ov::Tensor> {
            if (!m_allow_partial_offload) {
                OPENVINO_THROW("METAL: CPU fallback is disabled in pure device mode");
            }
            ensure_fallback();
            OPENVINO_ASSERT(m_fallback_request, "Fallback request is null");
            OPENVINO_ASSERT(inputs.size() == m_fallback_inputs.size(), "Fallback: input count mismatch");
            for (size_t i = 0; i < inputs.size(); ++i) {
                m_fallback_request->set_input_tensor(i, inputs[i]);
            }
            m_fallback_request->infer();
            std::vector<ov::Tensor> res;
            res.reserve(m_fallback_outputs.size());
            for (size_t i = 0; i < m_fallback_outputs.size(); ++i) {
                auto tmp = m_fallback_request->get_output_tensor(i);
                ov::Tensor out(tmp.get_element_type(), tmp.get_shape());
                std::memcpy(out.data(), tmp.data(), tmp.get_byte_size());
                res.emplace_back(std::move(out));
            }
            return res;
        };

        if (seg.op_count == 0 || m_ops.empty() ||
            seg.first_op_index + seg.op_count > m_ops.size()) {
            debug_log("[METAL MLIR] run_segment guard failed → fallback");
            return run_fallback();
        }
        if (m_pipelines.empty() || m_pipelines.size() < seg.first_op_index + seg.op_count) {
            debug_log("[METAL MLIR] run_segment missing pipelines → fallback");
            return run_fallback();
        }

        // For now we assume single-input / single-output IO-aligned segment.
        OPENVINO_ASSERT(inputs.size() == 1, "run_segment expects exactly one input tensor");

        auto seg_op = [&](size_t i) -> const KernelOp& { return m_ops[seg.first_op_index + i]; };
        auto seg_pipe = [&](size_t i) -> id<MTLComputePipelineState> { return m_pipelines[seg.first_op_index + i]; };

        auto tensor_num_elems = [](const KernelTensor* t) -> size_t {
            if (!t) return 0;
            size_t elems = 1;
            for (auto d : t->shape) elems *= static_cast<size_t>(d);
            return elems;
        };
        auto tensor_bytes = [&](const KernelTensor* t) -> size_t {
            return tensor_num_elems(t) * sizeof(float);
        };

        ov::Tensor input_f32 = to_float32_tensor(inputs[0]);

        const KernelOp& last_op = seg_op(seg.op_count - 1);
        ov::Shape out_shape;
        if (last_op.output && !last_op.output->shape.empty()) {
            out_shape.assign(last_op.output->shape.begin(), last_op.output->shape.end());
        } else {
            out_shape = inputs[0].get_shape();
        }
        const ov::element::Type out_elem_type = inputs[0].get_element_type();
        size_t out_bytes = 0;
        if (last_op.output) {
            out_bytes = tensor_num_elems(last_op.output) * sizeof(float);
        } else {
            size_t elems = 1;
            for (auto d : out_shape) elems *= d;
            out_bytes = elems * sizeof(float);
        }

#if METAL_MLIR_DEBUG
        auto shape_to_str = [](const ov::Shape& s) {
            std::string r = "[";
            for (size_t i = 0; i < s.size(); ++i) {
                if (i) r += ",";
                r += std::to_string(s[i]);
            }
            r += "]";
            return r;
        };
        debug_log("[METAL MLIR] run_segment ops=" + std::to_string(seg.op_count) +
                  " input_shape=" + shape_to_str(inputs[0].get_shape()) +
                  " output_shape=" + shape_to_str(out_shape));
        if (seg.op_count == 1) {
            const auto& op = seg_op(0);
            if (op.kind == KernelOpKind::Conv2D) {
                const auto& c = op.conv2d;
                std::ostringstream oss;
                oss << "[METAL MLIR] Conv2D params N=" << c.N << " C_in=" << c.C_in
                    << " H=" << c.H << " W=" << c.W
                    << " C_out=" << c.C_out
                    << " k=" << c.kernelH << "x" << c.kernelW
                    << " stride=" << c.strideH << "x" << c.strideW
                    << " pad=(" << c.padTop << "," << c.padLeft << "," << c.padBottom << "," << c.padRight << ")"
                    << " dil=" << c.dilationH << "x" << c.dilationW
                    << " groups=" << c.groups
                    << " out=" << c.outH << "x" << c.outW;
                debug_log(oss.str());
            } else if (op.kind == KernelOpKind::MaxPool2D || op.kind == KernelOpKind::AvgPool2D) {
                const auto& p = op.pool;
                std::ostringstream oss;
                oss << "[METAL MLIR] Pool params N=" << p.N << " C=" << p.C
                    << " H=" << p.H << " W=" << p.W
                    << " k=" << p.kernelH << "x" << p.kernelW
                    << " stride=" << p.strideH << "x" << p.strideW
                    << " padTop=" << p.padTop << " padLeft=" << p.padLeft
                    << " out=" << p.outH << "x" << p.outW
                    << " exclude_pad=" << (op.kind == KernelOpKind::AvgPool2D ? (p.exclude_pad ? 1 : 0) : 0);
                debug_log(oss.str());
            }
        }
#endif

        size_t max_tmp_bytes = 0;
        for (size_t i = 0; i < seg.op_count; ++i) {
            const auto* t = seg_op(i).output;
            max_tmp_bytes = std::max(max_tmp_bytes, t ? tensor_bytes(t) : inputs[0].get_byte_size());
        }
        if (max_tmp_bytes == 0)
            max_tmp_bytes = input_f32.get_byte_size();

        id<MTLBuffer> buf_in = [m_device newBufferWithBytes:input_f32.data()
                                                     length:input_f32.get_byte_size()
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_tmp = [m_device newBufferWithLength:max_tmp_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_out = [m_device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];

        auto make_const_buffer = [&](const std::vector<float>& data) -> id<MTLBuffer> {
            if (data.empty())
                return nil;
            return [m_device newBufferWithBytes:data.data()
                                         length:data.size() * sizeof(float)
                                        options:MTLResourceStorageModeShared];
        };

        id<MTLBuffer> buf_const_add = m_has_const_b ? make_const_buffer(m_const_b) : nil;
        id<MTLBuffer> buf_const_mul = m_has_const_mul ? make_const_buffer(m_const_mul) : nil;
        id<MTLBuffer> buf_const_bn = !m_const_bn.empty() ? make_const_buffer(m_const_bn) : nil;
        id<MTLBuffer> buf_const_w = m_has_const_w ? make_const_buffer(m_const_w) : nil;
        id<MTLBuffer> buf_const_mm0 = m_has_const_mm0 ? make_const_buffer(m_const_mm0) : nil;
        id<MTLBuffer> buf_const_mm1 = m_has_const_mm1 ? make_const_buffer(m_const_mm1) : nil;
        std::vector<id<MTLBuffer>> temp_const_buffers;

        auto maybe_broadcast_const = [&](const KernelOp& op, const std::vector<float>& src, std::vector<float>& dst) -> bool {
            if (!op.is_broadcast || !op.output) return false;
            const size_t out_elems = tensor_num_elems(op.output);
            if (out_elems == 0) return false;
            // Scalar
            if (src.size() == 1) {
                dst.assign(out_elems, src[0]);
                return true;
            }
            // Channel broadcast heuristic
            if (op.out_shape.size() >= 2) {
                size_t Cdim = 1;
                size_t C = static_cast<size_t>(op.out_shape[Cdim]);
                if (src.size() == C) {
                    dst.assign(out_elems, 0.f);
                    if (op.out_shape.size() == 2) {  // N,C
                        size_t N = static_cast<size_t>(op.out_shape[0]);
                        for (size_t n = 0; n < N; ++n)
                            for (size_t c = 0; c < C; ++c)
                                dst[n * C + c] = src[c];
                        return true;
                    } else if (op.out_shape.size() == 3) {  // N,C,L
                        size_t N = static_cast<size_t>(op.out_shape[0]);
                        size_t L = static_cast<size_t>(op.out_shape[2]);
                        for (size_t n = 0; n < N; ++n)
                            for (size_t c = 0; c < C; ++c)
                                for (size_t l = 0; l < L; ++l)
                                    dst[(n * C + c) * L + l] = src[c];
                        return true;
                    } else if (op.out_shape.size() == 4) {  // N,C,H,W
                        size_t N = static_cast<size_t>(op.out_shape[0]);
                        size_t H = static_cast<size_t>(op.out_shape[2]);
                        size_t W = static_cast<size_t>(op.out_shape[3]);
                        for (size_t n = 0; n < N; ++n)
                            for (size_t c = 0; c < C; ++c)
                                for (size_t h = 0; h < H; ++h)
                                    for (size_t w = 0; w < W; ++w)
                                        dst[((n * C + c) * H + h) * W + w] = src[c];
                        return true;
                    }
                }
            }
            if (src.size() == out_elems) {
                dst = src;
                return true;
            }
            if (op.out_shape.empty() || op.stride1.empty())
                return false;
            const size_t rank = op.out_shape.size();
            std::vector<size_t> out_stride(rank, 1);
            for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                out_stride[i] = out_stride[i + 1] * static_cast<size_t>(op.out_shape[i + 1]);
            }
            dst.assign(out_elems, 0.f);
            for (size_t idx = 0; idx < out_elems; ++idx) {
                size_t tmp = idx;
                size_t src_idx = 0;
                for (size_t d = 0; d < rank; ++d) {
                    size_t coord = tmp / out_stride[d];
                    tmp -= coord * out_stride[d];
                    if (d < op.stride1.size()) {
                        src_idx += coord * static_cast<size_t>(op.stride1[d]);
                    }
                }
                if (src_idx < src.size()) {
                    dst[idx] = src[src_idx];
                }
            }
            return true;
        };

        if (!buf_in || !buf_out || !buf_tmp) {
            debug_log("[METAL MLIR] run_segment buffer allocation failed → fallback");
            if (buf_in) [buf_in release];
            if (buf_tmp) [buf_tmp release];
            if (buf_out) [buf_out release];
            if (buf_const_add) [buf_const_add release];
            if (buf_const_mul) [buf_const_mul release];
            if (buf_const_bn) [buf_const_bn release];
            if (buf_const_mm0) [buf_const_mm0 release];
            if (buf_const_mm1) [buf_const_mm1 release];
            return run_fallback();
        }

        auto dispatch_op = [&](const KernelOp& op,
                               id<MTLComputePipelineState> pipeline,
                               id<MTLBuffer> src0,
                               id<MTLBuffer> src1,
                               id<MTLBuffer> dst,
                               id<MTLCommandBuffer> cmdBuf) {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            if (!pipeline) {
                debug_log("[METAL MLIR] Null pipeline in run_segment dispatch");
                [enc endEncoding];
                return;
            }
            switch (op.kind) {
                case KernelOpKind::MatMul: {
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    const NSUInteger total = static_cast<NSUInteger>(op.M * op.N * op.batch);
                    const NSUInteger threads_per_tg = 128;
                    MTLSize grid = MTLSizeMake(total, 1, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::ElementwiseAdd:
                case KernelOpKind::ElementwiseMul: {
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    const NSUInteger elems = static_cast<NSUInteger>(op.is_broadcast
                        ? tensor_num_elems(op.output)
                        : tensor_num_elems(op.input0));
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(elems, 1, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Unary: {
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    const NSUInteger elems = static_cast<NSUInteger>(tensor_num_elems(op.output));
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(elems, 1, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Softmax: {
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    const NSUInteger rows = static_cast<NSUInteger>(op.rows);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(rows, 1, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Conv2D: {
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
                    } params;
                    params.N = op.conv2d.N;
                    params.C_in = op.conv2d.C_in;
                    params.H = op.conv2d.H;
                    params.W = op.conv2d.W;
                    params.C_out = op.conv2d.C_out;
                    params.groups = op.conv2d.groups;
                    params.C_in_pg = op.conv2d.C_in_per_group;
                    params.C_out_pg = op.conv2d.C_out_per_group;
                    params.kH = op.conv2d.kernelH;
                    params.kW = op.conv2d.kernelW;
                    params.strideH = op.conv2d.strideH;
                    params.strideW = op.conv2d.strideW;
                    params.dilationH = op.conv2d.dilationH;
                    params.dilationW = op.conv2d.dilationW;
                    params.padTop = op.conv2d.padTop;
                    params.padLeft = op.conv2d.padLeft;
                    params.padBottom = op.conv2d.padBottom;
                    params.padRight = op.conv2d.padRight;
                    params.outH = op.output && op.output->shape.size() == 4 ? static_cast<uint32_t>(op.output->shape[2]) : 0;
                    params.outW = op.output && op.output->shape.size() == 4 ? static_cast<uint32_t>(op.output->shape[3]) : 0;
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    [enc setBytes:&params length:sizeof(params) atIndex:3];
                    const NSUInteger total = static_cast<NSUInteger>(params.N) *
                                             static_cast<NSUInteger>(params.outH) *
                                             static_cast<NSUInteger>(params.outW) *
                                             static_cast<NSUInteger>(params.C_out);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(total, 1, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::MaxPool2D:
                case KernelOpKind::AvgPool2D: {
                    struct PoolParams {
                        uint32_t N, H, W, C;
                        uint32_t outH, outW;
                        uint32_t kH, kW;
                        uint32_t strideH, strideW;
                        uint32_t padTop, padLeft;
                        uint32_t exclude_pad;
                    } params;
                    params.N = op.pool.N;
                    params.H = op.pool.H;
                    params.W = op.pool.W;
                    params.C = op.pool.C;
                    params.outH = op.pool.outH;
                    params.outW = op.pool.outW;
                    params.kH = op.pool.kernelH;
                    params.kW = op.pool.kernelW;
                    params.strideH = op.pool.strideH;
                    params.strideW = op.pool.strideW;
                    params.padTop = op.pool.padTop;
                    params.padLeft = op.pool.padLeft;
                    params.exclude_pad = op.kind == KernelOpKind::AvgPool2D ? (op.pool.exclude_pad ? 1u : 0u) : 0u;
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    [enc setBytes:&params length:sizeof(params) atIndex:2];
                    const NSUInteger total = static_cast<NSUInteger>(op.pool.N) *
                                             static_cast<NSUInteger>(op.pool.outH) *
                                             static_cast<NSUInteger>(op.pool.outW) *
                                             static_cast<NSUInteger>(op.pool.C);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(total, 1, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::BatchNorm2D: {
                    struct BNParams {
                        uint32_t N, C, H, W;
                    } params;
                    params.N = op.batchnorm.N;
                    params.C = op.batchnorm.C;
                    params.H = op.batchnorm.H;
                    params.W = op.batchnorm.W;
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    [enc setBytes:&params length:sizeof(params) atIndex:3];
                    const NSUInteger total = static_cast<NSUInteger>(params.N) *
                                             static_cast<NSUInteger>(params.C) *
                                             static_cast<NSUInteger>(params.H) *
                                             static_cast<NSUInteger>(params.W);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(total, 1, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
            }
            [enc endEncoding];
        };

        id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
        OPENVINO_ASSERT(cmd, "MlirBackend: failed to create command buffer");
        size_t mm_seen = 0;
        auto select_src1 = [&](const KernelOp& op) -> id<MTLBuffer> {
            switch (op.kind) {
                case KernelOpKind::MatMul: {
                    id<MTLBuffer> chosen = nil;
                    if (mm_seen == 0 && buf_const_mm0) {
                        chosen = buf_const_mm0;
                    } else if (mm_seen == 1 && buf_const_mm1) {
                        chosen = buf_const_mm1;
                    } else if (mm_seen == 0 && buf_const_add) {  // legacy weight reuse
                        chosen = buf_const_add;
                    }
                    ++mm_seen;
                    return chosen;
                }
                case KernelOpKind::ElementwiseAdd: {
                    if (!m_has_const_b) return nil;
                    std::vector<float> broadcasted;
                    const std::vector<float>* src_vec = &m_const_b;
                    if (op.is_broadcast && maybe_broadcast_const(op, m_const_b, broadcasted)) {
                        src_vec = &broadcasted;
                    }
                    id<MTLBuffer> buf = make_const_buffer(*src_vec);
                    if (buf) temp_const_buffers.push_back(buf);
                    return buf;
                }
                case KernelOpKind::ElementwiseMul: {
                    if (!m_has_const_mul) return nil;
                    std::vector<float> broadcasted;
                    const std::vector<float>* src_vec = &m_const_mul;
                    if (op.is_broadcast && maybe_broadcast_const(op, m_const_mul, broadcasted)) {
                        src_vec = &broadcasted;
                    }
                    id<MTLBuffer> buf = make_const_buffer(*src_vec);
                    if (buf) temp_const_buffers.push_back(buf);
                    return buf;
                }
                case KernelOpKind::BatchNorm2D: return buf_const_bn;
                case KernelOpKind::Conv2D: return buf_const_w;
                default: return nil;
            }
        };

        id<MTLBuffer> current = buf_in;
        id<MTLBuffer> final_buf = nil;

        auto dst_for_step = [&](size_t idx) -> id<MTLBuffer> {
            if (seg.op_count == 1) return buf_out;
            if (seg.op_count == 2) return (idx == 0) ? buf_tmp : buf_out;
            if (seg.op_count == 3) return (idx == 0) ? buf_tmp : (idx == 1 ? buf_out : buf_tmp);
            // fallback alternating
            return (idx == seg.op_count - 1) ? buf_out : buf_tmp;
        };

        for (size_t idx = 0; idx < seg.op_count; ++idx) {
            const auto& op = seg_op(idx);
            id<MTLBuffer> dst = dst_for_step(idx);
            id<MTLBuffer> src1 = select_src1(op);
            if (op.kind == KernelOpKind::MatMul || op.kind == KernelOpKind::BatchNorm2D ||
                op.kind == KernelOpKind::ElementwiseAdd || op.kind == KernelOpKind::ElementwiseMul ||
                op.kind == KernelOpKind::Conv2D) {
                OPENVINO_ASSERT(src1, "run_segment: missing secondary buffer for op");
                if (op.kind == KernelOpKind::Conv2D) {
                    float* wptr = static_cast<float*>([src1 contents]);
                    std::cerr << "[METAL Conv2D] src0=" << (void*)[current contents]
                              << " w=" << (void*)wptr
                              << " dst=" << (void*)[dst contents]
                              << " len_w_bytes=" << [src1 length] << " first_w:";
                    if (wptr) {
                        size_t n = std::min<NSUInteger>(8, [src1 length]/sizeof(float));
                        for (size_t i = 0; i < n; ++i) std::cerr << " " << wptr[i];
                    }
                    std::cerr << "\n";
                }
            }
            // For ops without second input, keep nil; dispatch will ignore.
            dispatch_op(op, seg_pipe(idx), current, src1, dst, cmd);
            current = dst;
        }

        final_buf = current;

        [cmd commit];
        [cmd waitUntilCompleted];

        ov::Tensor seg_output_f32{ov::element::f32, out_shape};

#if METAL_MLIR_DEBUG
        if (!seg_output_f32.get_shape().empty()) {
            const float* p = seg_output_f32.data<const float>();
            size_t n = std::min<size_t>(seg_output_f32.get_size(), 8);
            std::ostringstream oss;
            oss << "[METAL MLIR] seg out first:";
            for (size_t i = 0; i < n; ++i) oss << " " << p[i];
            debug_log(oss.str());
        }
#endif

        std::memcpy(seg_output_f32.data(), [final_buf contents], out_bytes);

#if METAL_MLIR_DEBUG
        if (seg.op_count == 1) {
            const auto& op = seg_op(0);
            if (op.kind == KernelOpKind::Conv2D && m_has_const_w) {
                std::vector<float> ref(seg_output_f32.get_size(), 0.f);
                const auto& c = op.conv2d;
                cpu_conv2d_reference(input_f32.data<const float>(),
                                     m_const_w.data(),
                                     nullptr,
                                     ref.data(),
                                     static_cast<int>(c.N),
                                     static_cast<int>(c.C_in),
                                     static_cast<int>(c.H),
                                     static_cast<int>(c.W),
                                     static_cast<int>(c.C_out),
                                     static_cast<int>(c.kernelH),
                                     static_cast<int>(c.kernelW),
                                     static_cast<int>(c.strideH),
                                     static_cast<int>(c.strideW),
                                     static_cast<int>(c.padTop),
                                     static_cast<int>(c.padLeft),
                                     static_cast<int>(c.dilationH),
                                     static_cast<int>(c.dilationW),
                                     static_cast<int>(c.groups),
                                     static_cast<int>(c.outH),
                                     static_cast<int>(c.outW));
                const float* metal = seg_output_f32.data<const float>();
                float max_diff = 0.f;
                size_t max_idx = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    float d = std::fabs(ref[i] - metal[i]);
                    if (d > max_diff) {
                        max_diff = d;
                        max_idx = i;
                    }
                }
                std::ostringstream oss;
                oss << "[METAL MLIR] Conv2D max_abs_diff=" << max_diff
                    << " at idx=" << max_idx
                    << " ref=" << ref[max_idx]
                    << " metal=" << metal[max_idx];
                debug_log(oss.str());
            } else if (op.kind == KernelOpKind::MaxPool2D) {
                std::vector<float> ref(seg_output_f32.get_size(), 0.f);
                const auto& p = op.pool;
                cpu_maxpool2d_reference(input_f32.data<const float>(),
                                        ref.data(),
                                        static_cast<int>(p.N),
                                        static_cast<int>(p.C),
                                        static_cast<int>(p.H),
                                        static_cast<int>(p.W),
                                        static_cast<int>(p.kernelH),
                                        static_cast<int>(p.kernelW),
                                        static_cast<int>(p.strideH),
                                        static_cast<int>(p.strideW),
                                        static_cast<int>(p.padTop),
                                        static_cast<int>(p.padLeft),
                                        static_cast<int>(p.outH),
                                        static_cast<int>(p.outW));
                const float* metal = seg_output_f32.data<const float>();
                float max_diff = 0.f;
                size_t max_idx = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    float d = std::fabs(ref[i] - metal[i]);
                    if (d > max_diff) {
                        max_diff = d;
                        max_idx = i;
                    }
                }
                std::ostringstream oss;
                oss << "[METAL MLIR] MaxPool2D max_abs_diff=" << max_diff
                    << " at idx=" << max_idx
                    << " ref=" << ref[max_idx]
                    << " metal=" << metal[max_idx];
                debug_log(oss.str());
            } else if (op.kind == KernelOpKind::AvgPool2D) {
                std::vector<float> ref(seg_output_f32.get_size(), 0.f);
                const auto& p = op.pool;
                cpu_avgpool2d_reference(input_f32.data<const float>(),
                                        ref.data(),
                                        static_cast<int>(p.N),
                                        static_cast<int>(p.C),
                                        static_cast<int>(p.H),
                                        static_cast<int>(p.W),
                                        static_cast<int>(p.kernelH),
                                        static_cast<int>(p.kernelW),
                                        static_cast<int>(p.strideH),
                                        static_cast<int>(p.strideW),
                                        static_cast<int>(p.padTop),
                                        static_cast<int>(p.padLeft),
                                        static_cast<int>(p.outH),
                                        static_cast<int>(p.outW),
                                        p.exclude_pad);
                const float* metal = seg_output_f32.data<const float>();
                float max_diff = 0.f;
                size_t max_idx = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    float d = std::fabs(ref[i] - metal[i]);
                    if (d > max_diff) {
                        max_diff = d;
                        max_idx = i;
                    }
                }
                std::ostringstream oss;
                oss << "[METAL MLIR] AvgPool2D max_abs_diff=" << max_diff
                    << " at idx=" << max_idx
                    << " ref=" << ref[max_idx]
                    << " metal=" << metal[max_idx];
                debug_log(oss.str());
            }
        }
#endif

        ov::Tensor seg_output{out_elem_type, out_shape};
        copy_fp32_to_destination(seg_output_f32.data<const float>(), seg_output);

        [buf_in release];
        [buf_tmp release];
        [buf_out release];
        if (buf_const_add) [buf_const_add release];
        if (buf_const_mul) [buf_const_mul release];
        if (buf_const_bn) [buf_const_bn release];
        if (buf_const_w) [buf_const_w release];
        if (buf_const_mm0) [buf_const_mm0 release];
        if (buf_const_mm1) [buf_const_mm1 release];
        for (auto b : temp_const_buffers) {
            [b release];
        }

        return {std::move(seg_output)};
    }

private:
    void build_flat_ir(const std::shared_ptr<const ov::Model>& model) {
        m_flat_ops.clear();
        m_flat_tensors.clear();
        m_flat_segments.clear();

        struct OutputKey {
            const ov::Node* node = nullptr;
            size_t index = 0;
            bool operator==(const OutputKey& other) const {
                return node == other.node && index == other.index;
            }
        };
        struct OutputKeyHash {
            size_t operator()(const OutputKey& k) const noexcept {
                return std::hash<const void*>()(k.node) ^ (k.index * 0x9e3779b97f4a7c15ULL);
            }
        };
        std::unordered_map<OutputKey, size_t, OutputKeyHash> tensor_index;

        auto make_tensor = [&](const ov::Output<ov::Node>& out) -> size_t {
            OutputKey key{out.get_node(), out.get_index()};
            auto it = tensor_index.find(key);
            if (it != tensor_index.end()) return it->second;
            const auto pshape = out.get_partial_shape();
            KernelTensor t;
            t.name = out.get_node()->get_friendly_name();
            if (pshape.is_static()) {
                const auto shape = pshape.to_shape();
                t.shape.assign(shape.begin(), shape.end());
            } else if (pshape.rank().is_static()) {
                t.shape.assign(static_cast<size_t>(pshape.rank().get_length()), -1);
            }
            m_flat_tensors.push_back(t);
            size_t idx = m_flat_tensors.size() - 1;
            tensor_index.emplace(key, idx);
            return idx;
        };

        const auto& ordered = model->get_ordered_ops();
        for (const auto& node : ordered) {
            // Skip Parameter/Result/Constant early
            if (ov::is_type<ov::op::v0::Parameter>(node.get()) ||
                ov::is_type<ov::op::v0::Result>(node.get()) ||
                ov::is_type<ov::op::v0::Constant>(node.get())) {
                continue;
            }

            auto add_input_indices = [&](KernelOp& op) {
                if (node->get_input_size() > 0) {
                    size_t in0 = make_tensor(node->input_value(0));
                    op.input0 = &m_flat_tensors[in0];
                }
                if (node->get_input_size() > 1) {
                    size_t in1 = make_tensor(node->input_value(1));
                    op.input1 = &m_flat_tensors[in1];
                }
            };

            auto set_output = [&](KernelOp& op) {
                if (node->get_output_size() == 0) return;
                size_t out_idx = make_tensor(node->output(0));
                op.output = &m_flat_tensors[out_idx];
            };

            KernelOp op{};
            bool supported = false;

            if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
                // reuse existing helper for shapes
                auto a_shape = mm->get_input_shape(0);
                auto b_shape = mm->get_input_shape(1);
                auto out_shape = mm->get_output_shape(0);
#if METAL_MLIR_DEBUG
                debug_log("[METAL MLIR] MatMul build_flat_ir A shape=" + std::to_string(a_shape.size()) +
                          " dims0=" + (a_shape.size() > 0 ? std::to_string(a_shape[0]) : "0") +
                          " dims1=" + (a_shape.size() > 1 ? std::to_string(a_shape[1]) : "0") +
                          " B shape dims=" + (b_shape.size() > 0 ? std::to_string(b_shape[0]) : "0") + "," +
                          (b_shape.size() > 1 ? std::to_string(b_shape[1]) : "0"));
#endif
                auto to3d = [](const ov::Shape& s) {
                    if (s.size() == 2) return std::vector<int64_t>{1, (int64_t)s[0], (int64_t)s[1]};
                    return std::vector<int64_t>{(int64_t)s[0], (int64_t)s[1], (int64_t)s[2]};
                };
                auto a3 = to3d(a_shape);
                auto b3 = to3d(b_shape);
                op.kind = KernelOpKind::MatMul;
                // Trust output shape for M/N; trust A for K to avoid backend-inserted transposes on weights.
                op.M = (out_shape.size() >= 2) ? static_cast<int64_t>(out_shape[out_shape.size() - 2])
                                               : a3[1];
                op.N = (out_shape.size() >= 1) ? static_cast<int64_t>(out_shape.back()) : b3[2];
                op.K = (a_shape.size() >= 1) ? static_cast<int64_t>(a_shape.back()) : a3[2];
                op.b_is_nk_layout = false;
                op.batch_a = a3[0];
                op.batch_b = b3[0];
                op.batch = std::max(op.batch_a, op.batch_b);
                add_input_indices(op);
                set_output(op);
                supported = true;
            } else if (ov::as_type_ptr<const ov::op::v1::Add>(node)) {
                op.kind = KernelOpKind::ElementwiseAdd;
                // Detect broadcast for diagnostics (used later in dispatch)
                op.is_broadcast = node->get_input_shape(0) != node->get_input_shape(1);
                op.out_shape.clear();
                for (auto d : node->get_output_shape(0)) op.out_shape.push_back(static_cast<int64_t>(d));
                if (node->get_input_partial_shape(0).is_static() && node->get_input_partial_shape(1).is_static()) {
                    auto a_shape_vec = node->get_input_shape(0);
                    auto b_shape_vec = node->get_input_shape(1);
                    size_t rank = op.out_shape.size();
                    auto normalize_rank = [&](const ov::Shape& s) {
                        std::vector<int64_t> r(rank, 1);
                        size_t off = rank - s.size();
                        for (size_t i = 0; i < s.size(); ++i) r[off + i] = static_cast<int64_t>(s[i]);
                        return r;
                    };
                    auto a_norm = normalize_rank(a_shape_vec);
                    auto b_norm = normalize_rank(b_shape_vec);
                    auto make_stride = [&](const std::vector<int64_t>& s) {
                        std::vector<int64_t> st(s.size(), 1);
                        for (int i = static_cast<int>(s.size()) - 2; i >= 0; --i) {
                            st[i] = st[i + 1] * s[i + 1];
                        }
                        for (size_t i = 0; i < s.size(); ++i) {
                            if (s[i] == 1) st[i] = 0;  // broadcast axis
                        }
                        return st;
                    };
                    op.stride0 = make_stride(a_norm);
                    op.stride1 = make_stride(b_norm);
                }
                add_input_indices(op);
                set_output(op);
                supported = true;
            } else if (ov::as_type_ptr<const ov::op::v1::Multiply>(node)) {
                op.kind = KernelOpKind::ElementwiseMul;
                add_input_indices(op);
                set_output(op);
                supported = true;
            } else if (ov::as_type_ptr<const ov::op::v0::Relu>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Sigmoid>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Tanh>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Elu>(node) ||
                       ov::as_type_ptr<const ov::op::v0::PRelu>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Gelu>(node) ||
                       ov::as_type_ptr<const ov::op::v4::Swish>(node)) {
                op.kind = KernelOpKind::Unary;
                add_input_indices(op);
                set_output(op);
                supported = true;
            } else if (ov::is_type<ov::op::v1::Softmax>(node.get()) || ov::is_type<ov::op::v8::Softmax>(node.get())) {
                op.kind = KernelOpKind::Softmax;
                add_input_indices(op);
                set_output(op);
                int64_t axis = -1;
                if (auto sm1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node))
                    axis = sm1->get_axis();
                else if (auto sm8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node))
                    axis = sm8->get_axis();
                op.softmax_axis = axis;

                // Guard unsupported axes for rank 4/5 early to force CPU fallback.
                const auto in_pshape = node->get_input_partial_shape(0);
                if (!is_softmax_shape_supported(in_pshape, axis)) {
                    debug_log("[METAL MLIR] Softmax unsupported axis/shape in flat IR build, using host softmax");
                    m_softmax_dynamic_only = false;  // force CPU fallback for unsupported static axes
                    if (!m_allow_partial_offload) {
                        OPENVINO_THROW("METAL: unsupported Softmax axis/shape in pure device mode");
                    }
                    m_force_fallback = true;
                    ensure_fallback(m_original_model);
                    return;
                }

                if (node->get_output_size() > 0) {
                    const auto pshape = node->get_output_partial_shape(0);
                    if (pshape.rank().is_static()) {
                        const auto rank = pshape.rank().get_length();
                        if (axis < 0) axis += rank;
                        if (pshape.is_static()) {
                            auto shape = pshape.to_shape();
                            int64_t cols = static_cast<int64_t>(shape[static_cast<size_t>(axis)]);
                            int64_t outer = 1;
                            for (int64_t i = 0; i < axis; ++i) outer *= static_cast<int64_t>(shape[static_cast<size_t>(i)]);
                            int64_t inner = 1;
                            for (size_t i = static_cast<size_t>(axis) + 1; i < shape.size(); ++i) inner *= static_cast<int64_t>(shape[i]);
                            int64_t rows = outer * inner;
                            op.rows = rows;
                            op.cols = cols;
                            op.inner = inner;
                        } else {
                            op.rows = 0;
                            op.cols = 0;
                            op.inner = 1;
                        }
                    }
                }
                supported = true;
            } else if (auto mp = ov::as_type_ptr<const ov::op::v1::MaxPool>(node)) {
                op.kind = KernelOpKind::MaxPool2D;
                add_input_indices(op);
                set_output(op);
                const auto in_shape = mp->get_input_shape(0);
                const auto out_shape = mp->get_output_shape(0);
                const auto k = mp->get_kernel();
                const auto s = mp->get_strides();
                const auto pb = mp->get_pads_begin();
                const auto pe = mp->get_pads_end();
                if (in_shape.size() == 4 && out_shape.size() == 4 && k.size() == 2 && s.size() == 2 && pb.size() == 2 && pe.size() == 2) {
                    op.pool.N = static_cast<uint32_t>(in_shape[0]);
                    op.pool.C = static_cast<uint32_t>(in_shape[1]);
                    op.pool.H = static_cast<uint32_t>(in_shape[2]);
                    op.pool.W = static_cast<uint32_t>(in_shape[3]);
                    op.pool.outH = static_cast<uint32_t>(out_shape[2]);
                    op.pool.outW = static_cast<uint32_t>(out_shape[3]);
                    op.pool.kernelH = static_cast<uint32_t>(k[0]);
                    op.pool.kernelW = static_cast<uint32_t>(k[1]);
                    op.pool.strideH = static_cast<uint32_t>(s[0]);
                    op.pool.strideW = static_cast<uint32_t>(s[1]);
                    op.pool.padTop = static_cast<uint32_t>(pb[0]);
                    op.pool.padLeft = static_cast<uint32_t>(pb[1]);
                }
                supported = true;
            } else if (auto ap = ov::as_type_ptr<const ov::op::v1::AvgPool>(node)) {
                op.kind = KernelOpKind::AvgPool2D;
                add_input_indices(op);
                set_output(op);
                const auto in_shape = ap->get_input_shape(0);
                const auto out_shape = ap->get_output_shape(0);
                const auto k = ap->get_kernel();
                const auto s = ap->get_strides();
                const auto pb = ap->get_pads_begin();
                const auto pe = ap->get_pads_end();
                op.pool.exclude_pad = ap->get_exclude_pad();
                if (in_shape.size() == 4 && out_shape.size() == 4 && k.size() == 2 && s.size() == 2 && pb.size() == 2 && pe.size() == 2) {
                    op.pool.N = static_cast<uint32_t>(in_shape[0]);
                    op.pool.C = static_cast<uint32_t>(in_shape[1]);
                    op.pool.H = static_cast<uint32_t>(in_shape[2]);
                    op.pool.W = static_cast<uint32_t>(in_shape[3]);
                    op.pool.outH = static_cast<uint32_t>(out_shape[2]);
                    op.pool.outW = static_cast<uint32_t>(out_shape[3]);
                    op.pool.kernelH = static_cast<uint32_t>(k[0]);
                    op.pool.kernelW = static_cast<uint32_t>(k[1]);
                    op.pool.strideH = static_cast<uint32_t>(s[0]);
                    op.pool.strideW = static_cast<uint32_t>(s[1]);
                    op.pool.padTop = static_cast<uint32_t>(pb[0]);
                    op.pool.padLeft = static_cast<uint32_t>(pb[1]);
                }
                supported = true;
            } else if (ov::as_type_ptr<const ov::op::v1::Convolution>(node) ||
                       ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
                op.kind = KernelOpKind::Conv2D;
                add_input_indices(op);
                set_output(op);
                bool is_group = static_cast<bool>(ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node));
                const auto in_shape = node->get_input_shape(0);
                const auto w_shape = node->get_input_shape(1);
                const auto out_shape = node->get_output_shape(0);
                if (in_shape.size() == 4 && w_shape.size() >= 4) {
                    const auto& s = is_group
                        ? ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)->get_strides()
                        : ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_strides();
                    const auto& pb = is_group
                        ? ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)->get_pads_begin()
                        : ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_pads_begin();
                    const auto& pe = is_group
                        ? ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)->get_pads_end()
                        : ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_pads_end();
                    op.conv2d.N = static_cast<uint32_t>(in_shape[0]);
                    op.conv2d.H = static_cast<uint32_t>(in_shape[2]);
                    op.conv2d.W = static_cast<uint32_t>(in_shape[3]);
                    op.conv2d.strideH = static_cast<uint32_t>(s[0]);
                    op.conv2d.strideW = static_cast<uint32_t>(s[1]);
                    op.conv2d.padTop = static_cast<uint32_t>(pb[0]);
                    op.conv2d.padLeft = static_cast<uint32_t>(pb[1]);
                    op.conv2d.padBottom = static_cast<uint32_t>(pe[0]);
                    op.conv2d.padRight = static_cast<uint32_t>(pe[1]);
                    const auto& dil = is_group
                        ? ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)->get_dilations()
                        : ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_dilations();
                    if (dil.size() == 2) {
                        op.conv2d.dilationH = static_cast<uint32_t>(dil[0]);
                        op.conv2d.dilationW = static_cast<uint32_t>(dil[1]);
                    }
                    if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
                        op.conv2d.padType = static_cast<uint32_t>(c->get_auto_pad());
                    } else if (auto g = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node)) {
                        op.conv2d.padType = static_cast<uint32_t>(g->get_auto_pad());
                    }
                    op.conv2d.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(node->get_input_element_type(0)));
                    if (!is_group) {
                        op.conv2d.C_in = static_cast<uint32_t>(in_shape[1]);
                        op.conv2d.groups = 1;
                        op.conv2d.C_in_per_group = op.conv2d.C_in;
                        op.conv2d.C_out = static_cast<uint32_t>(w_shape[0]);
                        op.conv2d.C_out_per_group = op.conv2d.C_out;
                        op.conv2d.kernelH = static_cast<uint32_t>(w_shape[2]);
                        op.conv2d.kernelW = static_cast<uint32_t>(w_shape[3]);
                    } else {
                        uint32_t groups = static_cast<uint32_t>(w_shape[0]);
                        uint32_t c_out_pg = static_cast<uint32_t>(w_shape[1]);
                        uint32_t c_in_pg = static_cast<uint32_t>(w_shape[2]);
                        op.conv2d.groups = groups;
                        op.conv2d.C_in = static_cast<uint32_t>(in_shape[1]);
                        op.conv2d.C_in_per_group = c_in_pg;
                        op.conv2d.C_out = groups * c_out_pg;
                        op.conv2d.C_out_per_group = c_out_pg;
                        op.conv2d.kernelH = static_cast<uint32_t>(w_shape[3]);
                        op.conv2d.kernelW = static_cast<uint32_t>(w_shape[4]);
                    }
                    if (out_shape.size() == 4) {
                        KernelTensor* out_t = op.output;
                        if (out_t) {
                            out_t->shape = {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                            static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3])};
                        }
                    }
                }
                supported = true;
            } else if (ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node)) {
                op.kind = KernelOpKind::BatchNorm2D;
                add_input_indices(op);
                set_output(op);
                const auto in_shape = node->get_input_shape(0);
                if (in_shape.size() == 4) {
                    op.batchnorm.N = static_cast<uint32_t>(in_shape[0]);
                    op.batchnorm.C = static_cast<uint32_t>(in_shape[1]);
                    op.batchnorm.H = static_cast<uint32_t>(in_shape[2]);
                    op.batchnorm.W = static_cast<uint32_t>(in_shape[3]);
                }
                const auto bn_v5 = ov::as_type_ptr<const ov::op::v5::BatchNormInference>(node);
                const auto bn_v0 = ov::as_type_ptr<const ov::op::v0::BatchNormInference>(node);
                const bool use_v5 = static_cast<bool>(bn_v5);
                op.batchnorm.eps = static_cast<float>(use_v5 ? bn_v5->get_eps_value() : bn_v0->get_eps_value());
                supported = true;
            } else if (ov::as_type_ptr<const ov::op::v1::Reshape>(node) ||
                       ov::as_type_ptr<const ov::op::v1::Transpose>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Squeeze>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Unsqueeze>(node) ||
                       ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Concat>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Interpolate>(node) ||
                       ov::as_type_ptr<const ov::op::v4::Interpolate>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Convert>(node)) {
                // Shape/type only: ensure outputs recorded
                set_output(op);
                continue;  // no KernelOp added
            } else {
                debug_log("[METAL MLIR] Flat IR: unsupported node " + node->get_friendly_name() +
                          " type=" + node->get_type_name());
                continue;
            }

            if (supported) {
                m_flat_ops.push_back(op);
            }
        }

        if (!m_flat_ops.empty()) {
            m_flat_segments.push_back(Segment{0, m_flat_ops.size()});
        }

#if METAL_MLIR_DEBUG
        debug_log("[METAL MLIR] Flat IR ops count: " + std::to_string(m_flat_ops.size()));
        const size_t dump_n = std::min<size_t>(m_flat_ops.size(), 8);
        for (size_t i = 0; i < dump_n; ++i) {
            const auto& op = m_flat_ops[i];
            auto kind_int = static_cast<int>(op.kind);
            std::string in0 = op.input0 ? op.input0->name : "-";
            std::string in1 = op.input1 ? op.input1->name : "-";
            std::string out = op.output ? op.output->name : "-";
            debug_log("[METAL MLIR] Flat IR op[" + std::to_string(i) + "]: kind=" + std::to_string(kind_int) +
                      " in0=" + in0 + " in1=" + in1 + " out=" + out);
        }
#endif
    }

    void build_segments_from_flat_ir() {
        m_flat_segments.clear();
        if (m_flat_ops.empty())
            return;

        auto is_segmentable = [](KernelOpKind k) {
            switch (k) {
                case KernelOpKind::Conv2D:
                case KernelOpKind::BatchNorm2D:
                case KernelOpKind::Unary:
                case KernelOpKind::MatMul:
                case KernelOpKind::ElementwiseAdd:
                case KernelOpKind::ElementwiseMul:
                case KernelOpKind::Softmax:
                case KernelOpKind::MaxPool2D:
                case KernelOpKind::AvgPool2D:
                    return true;
                default:
                    return false;
            }
        };

        size_t seg_start = 0;
        size_t seg_len = 0;
        auto push_seg = [&](size_t start, size_t len) {
            if (len == 0) return;
            // chunk into pieces of at most 3 to be executable today
            size_t offset = 0;
            while (offset < len) {
                size_t chunk = std::min<size_t>(3, len - offset);
                m_flat_segments.push_back(Segment{start + offset, chunk});
                offset += chunk;
            }
        };

        for (size_t i = 0; i < m_flat_ops.size(); ++i) {
            if (is_segmentable(m_flat_ops[i].kind)) {
                if (seg_len == 0) seg_start = i;
                ++seg_len;
            } else {
                push_seg(seg_start, seg_len);
                seg_len = 0;
            }
        }
        push_seg(seg_start, seg_len);

#if METAL_MLIR_DEBUG
        debug_log("[METAL MLIR] Segments built: " + std::to_string(m_flat_segments.size()) +
                  " over flat ops: " + std::to_string(m_flat_ops.size()));
        const size_t dump_n = std::min<size_t>(m_flat_segments.size(), 4);
        for (size_t i = 0; i < dump_n; ++i) {
            auto s = m_flat_segments[i];
            debug_log("[METAL MLIR] Segment[" + std::to_string(i) + "] start=" + std::to_string(s.first_op_index) +
                      " len=" + std::to_string(s.op_count));
        }
#endif
    }
};

MlirBackend::MlirBackend(const std::shared_ptr<const ov::Model>& model,
                         const std::shared_ptr<const ov::Model>& original_model,
                         ov::element::Type inference_precision)
    : m_impl(std::make_unique<MlirBackend::Impl>(model, original_model, inference_precision)) {}

MlirBackend::~MlirBackend() = default;

void MlirBackend::run(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) {
    m_impl->run(inputs, outputs);
}

bool MlirBackend::has_segment() const {
    return m_impl->has_segment();
}

bool MlirBackend::segment_io_is_model_io() const {
    return m_impl->segment_io_is_model_io();
}

const Segment& MlirBackend::get_segment() const {
    return m_impl->get_segment();
}

std::vector<ov::Tensor> MlirBackend::run_segment(const Segment& seg, const std::vector<ov::Tensor>& inputs) {
    return m_impl->run_segment(seg, inputs);
}

}  // namespace metal_plugin
}  // namespace ov
