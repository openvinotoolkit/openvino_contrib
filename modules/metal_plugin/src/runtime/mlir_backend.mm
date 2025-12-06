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

#include "runtime/metal_dtype.hpp"

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
#include <optional>

#include "kernel_ir/kernel_ir_common.hpp"
#include "kernel_ir/add_kernel_ir.hpp"
#include "kernel_ir/mul_kernel_ir.hpp"
#include "kernel_ir/matmul_kernel_ir.hpp"
#include "kernel_ir/unary_kernel_ir.hpp"
#include "kernel_ir/softmax_kernel_ir.hpp"
#include "kernel_ir/pool_max_kernel_ir.hpp"
#include "kernel_ir/pool_avg_kernel_ir.hpp"
#include "kernel_ir/conv_kernel_ir.hpp"
#include "kernel_ir/conv3d_kernel_ir.hpp"
#include "kernel_ir/batchnorm_kernel_ir.hpp"
#include "kernel_ir/slice_kernel_ir.hpp"
#include "kernel_codegen/metal_kernel_compiler.hpp"
#include "mlir/mlir_builder.hpp"
#include "mlir/mlir_passes.hpp"
#include "mlir_codegen/codegen_common.hpp"
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
#include "openvino/op/squared_difference.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
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
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/sign.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/floor_mod.hpp"
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

inline bool metal_debug_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("OV_METAL_DEBUG");
        if (!env) env = std::getenv("OV_METAL_TEST_DEBUG");
        return env && std::string(env) != "0";
    }();
    return enabled;
}

inline void debug_log(const std::string& msg) {
    if (!metal_debug_enabled())
        return;
    std::cerr << msg << "\n";
}

static bool use_handwritten_msl() {
    static const bool flag = []() {
        const char* env = std::getenv("OV_METAL_USE_HANDWRITTEN_MSL");
        return env && std::string(env) != "0";
    }();
    return flag;
}

struct BroadcastResult {
    ov::Shape out_shape;
    std::vector<int64_t> stride0;
    std::vector<int64_t> stride1;
    bool success = false;
};

static BroadcastResult compute_broadcast(const ov::Shape& a_shape, const ov::Shape& b_shape) {
    BroadcastResult res;
    const size_t rank = std::max(a_shape.size(), b_shape.size());
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
    // Row-major strides based on each input's own shape (not out), zero on broadcast axes
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
    res.out_shape = out;
    res.success = true;
    return res;
}

// Try to detect the decomposition pattern that OpenVINO produces for Mod/FloorMod:
// result = Sign(A) * (Abs(A) - something_using_B)
// where "something" typically contains Divide(Abs(A), Abs(B)) with extra converts/mults.
// If matched, returns original inputs A,B and whether the friendly name hints FloorMod.
static std::optional<std::tuple<ov::Output<const ov::Node>,
                                ov::Output<const ov::Node>,
                                bool>>
match_mod_decomposition(const std::shared_ptr<const ov::Node>& node) {
    auto mul = ov::as_type_ptr<const ov::op::v1::Multiply>(node);
    if (!mul) return std::nullopt;
    const bool log_mod = metal_debug_enabled() &&
                         node->get_friendly_name().find("Mod") != std::string::npos;

    auto get_input_if = [](const std::shared_ptr<const ov::Node>& n,
                           auto predicate) -> ov::Output<const ov::Node> {
        for (size_t i = 0; i < n->get_input_size(); ++i) {
            auto src = n->input_value(i);
            if (predicate(src.get_node_shared_ptr()))
                return src;
        }
        return {};
    };

    std::shared_ptr<const ov::op::v0::Sign> sign;
    std::shared_ptr<const ov::Node> sub_like;
    // mul inputs can be swapped
    for (int swap = 0; swap < 2; ++swap) {
        auto in0 = mul->get_input_node_shared_ptr(swap ? 1 : 0);
        auto in1 = mul->get_input_node_shared_ptr(swap ? 0 : 1);
        if (log_mod) {
            debug_log("[METAL MLIR] Mod pattern check: mul in0=" + std::string(in0 ? in0->get_type_info().name : "nil") +
                      " in1=" + std::string(in1 ? in1->get_type_info().name : "nil"));
        }
        sign = ov::as_type_ptr<const ov::op::v0::Sign>(in0);
        if (sign) {
            if (auto s = ov::as_type_ptr<const ov::op::v1::Subtract>(in1))
                sub_like = s;
            else if (auto a = ov::as_type_ptr<const ov::op::v1::Add>(in1))
                sub_like = a;
        }
        if (sign && sub_like) break;
    }
    if (!sign || !sub_like) {
        if (log_mod) debug_log("[METAL MLIR] Mod pattern missed: no Sign/Sub/Add combination");
        return std::nullopt;
    }

    auto abs_a = ov::as_type_ptr<const ov::op::v0::Abs>(sub_like->get_input_node_shared_ptr(0));
    if (!abs_a) {
        if (log_mod) debug_log("[METAL MLIR] Mod pattern missed: first arg not Abs");
        return std::nullopt;
    }
    auto a_out = abs_a->input_value(0);

    // Traverse subtract->input1 to locate Divide node; fallback to that input directly.
    std::function<ov::Output<const ov::Node>(const ov::Output<const ov::Node>&)> find_div;
    find_div = [&](const ov::Output<const ov::Node>& out) -> ov::Output<const ov::Node> {
        if (!out.get_node()) return {};
        auto n = out.get_node_shared_ptr();
        if (ov::is_type<ov::op::v1::Divide>(n.get()))
            return out;
        for (size_t i = 0; i < n->get_input_size(); ++i) {
            auto res = find_div(n->input_value(i));
            if (res.get_node()) return res;
        }
        return {};
    };

    ov::Output<const ov::Node> div_out = find_div(sub_like->input_value(1));
    ov::Output<const ov::Node> b_out;
    if (div_out.get_node()) {
        auto div_node = div_out.get_node_shared_ptr();
        auto rhs = div_node->input_value(1);
        if (auto abs_b = ov::as_type_ptr<const ov::op::v0::Abs>(rhs.get_node_shared_ptr())) {
            b_out = abs_b->input_value(0);
        } else {
            b_out = rhs;
        }
    } else {
        // Fallback: try to grab Abs/Parameter/Constant somewhere under subtract->input1
        auto rhs_abs = get_input_if(sub_like, [](const std::shared_ptr<const ov::Node>& n) {
            return ov::is_type<ov::op::v0::Abs>(n.get());
        });
        if (rhs_abs.get_node()) {
            auto abs_node = rhs_abs.get_node_shared_ptr();
            b_out = abs_node->input_value(0);
        } else {
            b_out = sub_like->input_value(1);
        }
    }
    if (!a_out.get_node() || !b_out.get_node()) return std::nullopt;

    bool is_floor = node->get_friendly_name().find("FloorMod") != std::string::npos;
    return std::make_tuple(a_out, b_out, is_floor);
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
// - axes: any within rank
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
    // rank 2/3/4/5: allow any axis in range
    return true;
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

static void copy_fp32_to_destination(const float* src, ov::Tensor& dst) {
    const auto et = dst.get_element_type();
    if (et == ov::element::f32) {
        std::memcpy(dst.data(), src, dst.get_byte_size());
        return;
    }
    if (et == ov::element::f16) {
        auto* dst_data = dst.data<ov::float16>();
        const size_t count = dst.get_size();
        for (size_t i = 0; i < count; ++i) {
            dst_data[i] = static_cast<ov::float16>(src[i]);
        }
        return;
    }
    if (et == ov::element::i32) {
        auto* dst_data = dst.data<int32_t>();
        const size_t count = dst.get_size();
        for (size_t i = 0; i < count; ++i) {
            dst_data[i] = static_cast<int32_t>(std::lrint(src[i]));
        }
        return;
    }
    if (et == ov::element::i64) {
        auto* dst_data = dst.data<int64_t>();
        const size_t count = dst.get_size();
        for (size_t i = 0; i < count; ++i) {
            dst_data[i] = static_cast<int64_t>(std::llrint(src[i]));
        }
        return;
    }
    OPENVINO_THROW("copy_fp32_to_destination supports only f16/f32/i32/i64 outputs, got ",
                   et.get_type_name());
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
    bool pow = true;
    bool pow_broadcast = true;
    bool mul = true;
    bool mod = true;
    bool relu = true;
    bool sigmoid = true;
    bool tanh = true;
    bool elu = true;
    bool prelu = true;
    bool gelu = true;
    bool swish = true;
    bool softmax = true;
    bool split = true;
    bool maxpool = true;
    bool avgpool = true;
    bool conv2d = true;
    bool conv3d = false;
    bool batch_norm = true;
    bool layer_norm = false;
};

MlirCapabilities default_capabilities() {
    MlirCapabilities caps;
    caps.conv3d = true;
    caps.pow = true;
    caps.pow_broadcast = true;
    caps.mod = true;
    return caps;
}

struct ModelAnalysis {
    bool has_matmul = false;
    bool has_add = false;
    bool has_mul = false;
    bool has_div = false;
    bool has_mod = false;
    bool has_floor_mod = false;
    bool has_pow = false;
    bool has_add_broadcast = false;
    bool has_unary = false;
    bool has_softmax = false;
    bool has_maxpool = false;
    bool has_avgpool = false;
    bool has_conv2d = false;
    bool has_conv3d = false;
    bool has_batchnorm = false;
    bool has_split = false;
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
        } else if (ov::as_type_ptr<const ov::op::v1::Split>(node) ||
                   ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
            if (!caps.split) {
                mark_disabled(node);
                continue;
            }
            res.has_split = true;
            res.compute_ops++;
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
        } else if (auto sub = ov::as_type_ptr<const ov::op::v1::Subtract>(node)) {
            bool broadcast = true;
            const auto ps0 = sub->get_input_partial_shape(0);
            const auto ps1 = sub->get_input_partial_shape(1);
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
        } else if (ov::is_type<ov::op::v0::SquaredDifference>(node.get())) {
            // Treat as two ops: Sub + Mul; both supported if Mul is supported
            const auto ps0 = node->get_input_partial_shape(0);
            const auto ps1 = node->get_input_partial_shape(1);
            bool broadcast = true;
            if (ps0.is_static() && ps1.is_static()) {
                broadcast = (ps0.to_shape() != ps1.to_shape());
            }
            if (broadcast) {
                if (!caps.mul) {
                    mark_disabled(node);
                    continue;
                }
                res.has_add_broadcast = true;  // reuse broadcast flag
            } else {
                if (!caps.mul) {
                    mark_disabled(node);
                    continue;
                }
                if (is_eltwise_shape_supported_dynamic(ps0, ps1) != DynamicSupport::FullMetal) {
                    mark_future(node);
                    continue;
                }
                res.has_mul = true;
            }
            res.compute_ops += 2;  // Sub + Mul
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
        } else if (auto div = ov::as_type_ptr<const ov::op::v1::Divide>(node)) {
            bool broadcast = true;
            const auto ps0 = div->get_input_partial_shape(0);
            const auto ps1 = div->get_input_partial_shape(1);
            if (ps0.is_static() && ps1.is_static()) {
                broadcast = (ps0.to_shape() != ps1.to_shape());
            }
            if (broadcast) {
                if (!caps.mul) {
                    mark_disabled(node);
                } else {
                    res.has_div = true;
                    res.compute_ops++;
                }
            } else {
                if (!caps.mul) {
                    mark_disabled(node);
                } else {
                    if (is_eltwise_shape_supported_dynamic(ps0, ps1) == DynamicSupport::FullMetal) {
                        res.has_div = true;
                        res.compute_ops++;
                    } else {
                        mark_future(node);
                        continue;
                    }
                }
            }
        } else if (auto pow = ov::as_type_ptr<const ov::op::v1::Power>(node)) {
            bool broadcast = true;
            const auto ps0 = pow->get_input_partial_shape(0);
            const auto ps1 = pow->get_input_partial_shape(1);
            if (ps0.is_static() && ps1.is_static()) {
                broadcast = (ps0.to_shape() != ps1.to_shape());
            }
            if (broadcast) {
                if (!caps.pow_broadcast) {
                    mark_disabled(node);
                    continue;
                }
                res.has_pow = true;
                res.compute_ops++;
            } else {
                if (!caps.pow) {
                    mark_disabled(node);
                    continue;
                }
                if (is_eltwise_shape_supported_dynamic(ps0, ps1) == DynamicSupport::FullMetal) {
                    res.has_pow = true;
                    res.compute_ops++;
                } else {
                    mark_future(node);
                    continue;
                }
            }
        } else if (ov::as_type_ptr<const ov::op::v1::Mod>(node) ||
                   ov::as_type_ptr<const ov::op::v1::FloorMod>(node)) {
            bool broadcast = true;
            const auto ps0 = node->get_input_partial_shape(0);
            const auto ps1 = node->get_input_partial_shape(1);
            if (ps0.is_static() && ps1.is_static()) {
                broadcast = (ps0.to_shape() != ps1.to_shape());
            }
            if (broadcast) {
                if (!caps.mod) {
                    mark_disabled(node);
                    continue;
                }
                res.has_mod = true;
                res.compute_ops++;
            } else {
                if (!caps.mod) {
                    mark_disabled(node);
                    continue;
                }
                if (is_eltwise_shape_supported_dynamic(ps0, ps1) == DynamicSupport::FullMetal) {
                    if (ov::as_type_ptr<const ov::op::v1::Mod>(node))
                        res.has_mod = true;
                    else
                        res.has_floor_mod = true;
                    res.compute_ops++;
                } else {
                    mark_future(node);
                    continue;
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
        } else if (ov::as_type_ptr<const ov::op::v0::Abs>(node)) {
            res.has_unary = true;
            res.unary_kind = ActivationKind::Abs;
            res.compute_ops++;
        } else if (ov::as_type_ptr<const ov::op::v0::Sign>(node)) {
            res.has_unary = true;
            res.unary_kind = ActivationKind::Sign;
            res.compute_ops++;
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
            const auto rank = cv->get_input_partial_shape(0).rank();
            if (rank.is_static() && rank.get_length() == 5) {
                if (!caps.conv3d) {
                    mark_disabled(node);
                } else {
                    res.has_conv3d = true;
                    res.compute_ops++;
                }
            } else {
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
        } else if (auto sub = ov::as_type_ptr<const ov::op::v1::Subtract>(node)) {
            bool broadcast = true;
            const auto ps0 = sub->get_input_partial_shape(0);
            const auto ps1 = sub->get_input_partial_shape(1);
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
                   ov::as_type_ptr<const ov::op::v8::Slice>(node) ||
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
        // Strict mode: no CPU fallback.
        m_allow_partial_offload = false;
        m_force_fallback = false;

        // Reject unsupported input types early (allow f16/f32/i32/i64).
        for (const auto& p : model->get_parameters()) {
            auto et = p->get_element_type();
            if (!(et == ov::element::f16 || et == ov::element::f32 || et == ov::element::i32 || et == ov::element::i64)) {
                OPENVINO_THROW("METAL supports only f16/f32/i32/i64 model inputs");
            }
        }

        compile(model);
    }

    ~Impl() {
        // Release Metal resources to avoid exhausting command queue handles across tests.
        for (auto pipeline : m_pipelines) {
            if (pipeline) {
                [pipeline release];
            }
        }
        if (m_queue) {
            [m_queue release];
            m_queue = nil;
        }
        if (m_device) {
            [m_device release];
            m_device = nil;
        }
    }

    void run(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) {
        m_last_inputs = inputs;
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
                if (op.dtype.ov_type == ov::element::dynamic) {
                    try {
                        op.dtype = resolve_metal_dtype(m_inference_precision);
                    } catch (const ov::Exception&) {
                        // leave as dynamic; will fallback
                    }
                }
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
            // Handle shape-only graphs like single Reshape without CPU fallback.
            auto reshape_only = [&]() -> std::shared_ptr<const ov::op::v1::Reshape> {
                std::shared_ptr<const ov::op::v1::Reshape> reshape;
                for (const auto& node : m_model->get_ordered_ops()) {
                    if (ov::as_type_ptr<const ov::op::v0::Parameter>(node) ||
                        ov::as_type_ptr<const ov::op::v0::Result>(node) ||
                        ov::as_type_ptr<const ov::op::v0::Constant>(node)) {
                        continue;
                    }
                    if (auto r = ov::as_type_ptr<const ov::op::v1::Reshape>(node)) {
                        if (reshape)
                            return nullptr;  // more than one non-IO op
                        reshape = r;
                        continue;
                    }
                    // Any other compute/layout op → not reshape-only
                    return nullptr;
                }
                return reshape;
            }();

            if (reshape_only && inputs.size() == 1 && !outputs.empty()) {
                const auto target_pshape = reshape_only->get_output_partial_shape(0);
                OPENVINO_ASSERT(target_pshape.is_static(), "Reshape output shape must be static");
                const ov::Shape target_shape = target_pshape.to_shape();
                const auto& in = inputs[0];
                const size_t in_elems = in.get_size();
                size_t out_elems = 1;
                for (auto d : target_shape) out_elems *= d;
                OPENVINO_ASSERT(in_elems == out_elems,
                                "Reshape element count mismatch in Metal reshape-only path");
                ov::Tensor out{in.get_element_type(), target_shape};
                std::memcpy(out.data(), in.data(), in.get_byte_size());
                outputs[0] = std::move(out);
                return;
            }

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

        // Special-case Split: single op with multiple outputs → dedicated GPU kernel.
        if (seg.op_count == 1 && seg_op(0).kind == KernelOpKind::Split) {
            const auto& op = seg_op(0);
            const auto& desc = op.split;
            const size_t outputs_count = desc.split_sizes.size();
            if (inputs.empty()) cpu_fallback_error();
            if (outputs.size() != outputs_count) {
                if (!m_allow_partial_offload) cpu_fallback_error();
                ensure_fallback(m_original_model);
                run_fallback();
                return;
            }

            auto element_type = static_cast<ov::element::Type_t>(desc.element_type);
            if (element_type == ov::element::Type_t::dynamic)
                element_type = inputs[0].get_element_type();
            size_t elem_size = element_type == ov::element::f16 ? sizeof(ov::float16) : sizeof(float);

            // Prepare outputs shapes
            for (size_t i = 0; i < outputs_count; ++i) {
                ov::Shape shp;
                shp.reserve(desc.input_shape.size());
                for (size_t d = 0; d < desc.input_shape.size(); ++d) {
                    size_t val = static_cast<size_t>(desc.input_shape[d]);
                    if (d == static_cast<size_t>(desc.axis)) val = desc.split_sizes[i];
                    shp.push_back(val);
                }
                if (outputs[i].get_shape() != shp || outputs[i].get_element_type() != element_type) {
                    outputs[i] = ov::Tensor{element_type, shp};
                }
            }

            id<MTLBuffer> buf_in = [m_device newBufferWithBytesNoCopy:const_cast<void*>(inputs[0].data())
                                                            length:inputs[0].get_byte_size()
                                                           options:MTLResourceStorageModeShared
                                                       deallocator:nil];
            std::vector<id<MTLBuffer>> out_bufs(outputs_count, nil);
            for (size_t i = 0; i < outputs_count; ++i) {
                out_bufs[i] = [m_device newBufferWithBytesNoCopy:const_cast<void*>(outputs[i].data())
                                                          length:outputs[i].get_byte_size()
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
            }

            id<MTLCommandBuffer> cmd = [m_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:seg_pipe(0)];
            [enc setBuffer:buf_in offset:0 atIndex:0];
            for (size_t i = 0; i < outputs_count; ++i) {
                [enc setBuffer:out_bufs[i] offset:0 atIndex:(1 + i)];
            }

            auto total_elems = [&]() {
                size_t t = 1;
                for (auto d : desc.input_shape) t *= static_cast<size_t>(d);
                return t;
            }();
            const NSUInteger threads_per_tg = 128;
            MTLSize grid = MTLSizeMake(static_cast<NSUInteger>(total_elems), 1, 1);
            MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            [buf_in release];
            for (auto b : out_bufs) if (b) [b release];
            return;
        }

        // Special-case multi-output Split → Slice expansion: run lightweight host slicing.
        auto run_host_slices = [&](const Segment& s) {
            if (inputs.empty()) cpu_fallback_error();
            const ov::Tensor& src = inputs[0];
            const bool is_f16 = src.get_element_type() == ov::element::f16;

            auto product = [](const std::vector<int64_t>& dims) -> size_t {
                size_t v = 1;
                for (auto d : dims) v *= static_cast<size_t>(d);
                return v;
            };

            auto copy_slice = [&](const KernelOp& op, ov::Tensor& dst) {
                const auto& desc = op.slice;
                const size_t rank = desc.out_shape.size();
                std::vector<size_t> out_strides(rank, 1);
                for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                    out_strides[i] = out_strides[i + 1] * static_cast<size_t>(desc.out_shape[i + 1]);
                }
                const size_t elems = product(desc.out_shape);
                if (is_f16) {
                    const auto* src_ptr = src.data<const ov::float16>();
                    auto* dst_ptr = dst.data<ov::float16>();
                    for (size_t idx = 0; idx < elems; ++idx) {
                        size_t tmp = idx;
                        int64_t in_index = 0;
                        for (size_t d = 0; d < rank; ++d) {
                            size_t coord = tmp / out_strides[d];
                            tmp -= coord * out_strides[d];
                            int64_t in_coord = desc.starts[d] + static_cast<int64_t>(coord) * desc.steps[d];
                            in_index += in_coord * desc.in_strides[d];
                        }
                        dst_ptr[idx] = src_ptr[in_index];
                    }
                } else {
                    const auto* src_ptr = src.data<const float>();
                    auto* dst_ptr = dst.data<float>();
                    for (size_t idx = 0; idx < elems; ++idx) {
                        size_t tmp = idx;
                        int64_t in_index = 0;
                        for (size_t d = 0; d < rank; ++d) {
                            size_t coord = tmp / out_strides[d];
                            tmp -= coord * out_strides[d];
                            int64_t in_coord = desc.starts[d] + static_cast<int64_t>(coord) * desc.steps[d];
                            in_index += in_coord * desc.in_strides[d];
                        }
                        dst_ptr[idx] = src_ptr[in_index];
                    }
                }
            };

            OPENVINO_ASSERT(outputs.size() == s.op_count, "Split host path expects one output per Slice op");
            for (size_t i = 0; i < s.op_count; ++i) {
                const auto& op = seg_op(i);
                OPENVINO_ASSERT(op.kind == KernelOpKind::Slice, "Split host path supports Slice-only segment");
                auto& out = outputs[i];
                const auto& oshp = op.output && !op.output->shape.empty()
                    ? op.output->shape
                    : op.slice.out_shape;
                ov::Shape shape;
                shape.assign(oshp.begin(), oshp.end());
                if (out.get_shape() != shape || out.get_element_type() != src.get_element_type()) {
                    out = ov::Tensor{src.get_element_type(), shape};
                }
                copy_slice(op, out);
            }
        };

        if (outputs.size() > 1) {
            bool slice_only = true;
            for (size_t i = 0; i < seg.op_count; ++i) {
                if (seg_op(i).kind != KernelOpKind::Slice) {
                    slice_only = false;
                    break;
                }
            }
            if (slice_only) {
                run_host_slices(seg);
                return;
            }
            if (m_allow_partial_offload) {
                ensure_fallback(m_original_model);
                run_fallback();
                return;
            }
            cpu_fallback_error();
        }

        OPENVINO_ASSERT(outputs.size() == 1, "MlirBackend: expected 1 output");

        auto inputs_ok = [](const std::vector<ov::Tensor>& ts) {
            for (const auto& t : ts) {
                if (t.get_element_type() != ov::element::f32)
                    continue;
                const float* p = t.data<const float>();
                for (size_t i = 0; i < t.get_size(); ++i) {
                    if (!std::isfinite(p[i])) {
                        return false;
                    }
                }
            }
            return true;
        };
        if (!inputs_ok(inputs)) {
            debug_log("[MlirBackend] Inputs non-finite; attempting CPU fallback");
            if (m_allow_partial_offload) {
                ensure_fallback(m_original_model);
                run_fallback();
                return;
            } else {
                cpu_fallback_error();
            }
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
            if (op.kind == KernelOpKind::ElementwiseAdd || op.kind == KernelOpKind::ElementwiseSub ||
                op.kind == KernelOpKind::ElementwiseMul || op.kind == KernelOpKind::ElementwiseDiv ||
                op.kind == KernelOpKind::ElementwisePow) {
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
                case KernelOpKind::Conv3D: {
                    if (cur_in0.size() != 5) { runtime_shapes_ok = false; break; }
                    int64_t N = cur_in0[0];
                    int64_t C = cur_in0[1];
                    int64_t D = cur_in0[2];
                    int64_t H = cur_in0[3];
                    int64_t W = cur_in0[4];
                    auto& c = op_rt.conv3d;
                    if (C != static_cast<int64_t>(c.C_in)) { runtime_shapes_ok = false; break; }
                    int64_t outD = (D + c.padFront + c.padBack - static_cast<int64_t>(c.dilationD) * (c.kernelD - 1) - 1) /
                                   static_cast<int64_t>(c.strideD) + 1;
                    int64_t outH = (H + c.padTop + c.padBottom - static_cast<int64_t>(c.dilationH) * (c.kernelH - 1) - 1) /
                                   static_cast<int64_t>(c.strideH) + 1;
                    int64_t outW = (W + c.padLeft + c.padRight - static_cast<int64_t>(c.dilationW) * (c.kernelW - 1) - 1) /
                                   static_cast<int64_t>(c.strideW) + 1;
                    if (outD <= 0 || outH <= 0 || outW <= 0) { runtime_shapes_ok = false; break; }
                    c.N = static_cast<uint32_t>(N);
                    c.D = static_cast<uint32_t>(D);
                    c.H = static_cast<uint32_t>(H);
                    c.W = static_cast<uint32_t>(W);
                    c.outD = static_cast<uint32_t>(outD);
                    c.outH = static_cast<uint32_t>(outH);
                    c.outW = static_cast<uint32_t>(outW);
                    op_rt.conv3d = c;
                    std::vector<int64_t> out_shape{N, static_cast<int64_t>(c.C_out), outD, outH, outW};
                    update_output_shape(op_rt, out_shape);
                    last_out_shape = out_shape;
                    break;
                }
                case KernelOpKind::ElementwiseAdd:
                case KernelOpKind::ElementwiseSub:
                case KernelOpKind::ElementwiseMul:
                case KernelOpKind::ElementwiseDiv:
                case KernelOpKind::ElementwisePow:
                case KernelOpKind::ElementwiseMod:
                case KernelOpKind::ElementwiseFloorMod: {
                    auto compute_broadcast = [](std::vector<int64_t> a, std::vector<int64_t> b) -> std::vector<int64_t> {
                        if (a.empty()) return b;
                        if (b.empty()) return a;
                        const size_t rank = std::max(a.size(), b.size());
                        auto pad = [&](std::vector<int64_t>& s) {
                            if (s.size() < rank) s.insert(s.begin(), rank - s.size(), 1);
                        };
                        pad(a);
                        pad(b);
                        std::vector<int64_t> out(rank, 1);
                        for (size_t i = 0; i < rank; ++i) {
                            if (a[i] == b[i] || a[i] == 1) out[i] = b[i];
                            else if (b[i] == 1) out[i] = a[i];
                            else return {};  // incompatible
                        }
                        return out;
                    };
                    std::vector<int64_t> out_shape;
                    if (!op_rt.out_shape.empty()) {
                        out_shape = op_rt.out_shape;
                    } else if (!cur_in0.empty() && !cur_in1.empty()) {
                        out_shape = compute_broadcast(cur_in0, cur_in1);
                    } else {
                        out_shape = cur_in0;
                    }
                    if (out_shape.empty()) { runtime_shapes_ok = false; break; }
                    update_output_shape(op_rt, out_shape);
                    last_out_shape = out_shape;
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
                    op_rt.rows = sm_rows;
                    op_rt.cols = sm_cols;
                    op_rt.inner = sm_inner;
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

        auto compute_size = [](const MetalDType& dt) -> size_t {
            switch (dt.compute) {
                case MetalDType::ComputeType::F32: return sizeof(float);
                case MetalDType::ComputeType::I32: return sizeof(int32_t);
                case MetalDType::ComputeType::I64: return sizeof(int64_t);
                default: return sizeof(float);
            }
        };
        auto element_size = [&](const KernelTensor* t) -> size_t {
            if (!t) return sizeof(float);
            if (t->dtype.ov_type == ov::element::dynamic)
                return sizeof(float);
            return compute_size(t->dtype);
        };
        auto tensor_bytes = [&](const KernelTensor* t) -> size_t {
            return tensor_num_elems(t) * element_size(t);
        };

        // Buffer mapping for flat execution
        std::unordered_map<const KernelTensor*, id<MTLBuffer>> buf_map;
        std::vector<id<MTLBuffer>> temp_const_buffers;

        auto make_const_buffer = [&](const KernelTensor* t) -> id<MTLBuffer> {
            if (!t || !t->from_constant || t->const_data.empty())
                return nil;
            switch (t->dtype.storage) {
                case MetalDType::StorageType::I32: {
                    std::vector<int32_t> tmp(t->const_data.size());
                    for (size_t i = 0; i < t->const_data.size(); ++i)
                        tmp[i] = static_cast<int32_t>(t->const_data[i]);
                    return [m_device newBufferWithBytes:tmp.data()
                                                 length:tmp.size() * sizeof(int32_t)
                                                options:MTLResourceStorageModeShared];
                }
                case MetalDType::StorageType::I64: {
                    std::vector<int64_t> tmp(t->const_data.size());
                    for (size_t i = 0; i < t->const_data.size(); ++i)
                        tmp[i] = static_cast<int64_t>(t->const_data[i]);
                    return [m_device newBufferWithBytes:tmp.data()
                                                 length:tmp.size() * sizeof(int64_t)
                                                options:MTLResourceStorageModeShared];
                }
                case MetalDType::StorageType::F16:
                case MetalDType::StorageType::F32:
                default: {
                    return [m_device newBufferWithBytes:t->const_data.data()
                                                 length:t->const_data.size() * sizeof(float)
                                                options:MTLResourceStorageModeShared];
                }
            }
        };

        auto load_const_input = [&](const KernelOp& op) -> id<MTLBuffer> {
            if (!op.input1 || !op.input1->from_constant)
                return nil;
            id<MTLBuffer> buf = make_const_buffer(op.input1);
            if (buf) temp_const_buffers.push_back(buf);
            return buf;
        };

        auto make_buffer_from_tensor = [&](const ov::Tensor& t) -> id<MTLBuffer> {
            if (t.get_element_type().is_integral()) {
                return [m_device newBufferWithBytes:t.data()
                                             length:t.get_byte_size()
                                            options:MTLResourceStorageModeShared];
            }
            ov::Tensor tmp = t;
            if (t.get_element_type() == ov::element::f16) {
                tmp = to_float32_tensor(t);
            }
            return [m_device newBufferWithBytes:tmp.data()
                                         length:tmp.get_byte_size()
                                        options:MTLResourceStorageModeShared];
        };
        const std::vector<ov::Tensor>* in_vec = inputs.empty() && !m_last_inputs.empty() ? &m_last_inputs : &inputs;
        if (!in_vec->empty()) {
            const auto& in0 = (*in_vec)[0];
            if (metal_debug_enabled() && !in0.get_shape().empty()) {
                auto in0_f32 = to_float32_tensor(in0);
                auto* p = in0_f32.data<const float>();
                size_t n = std::min<size_t>(8, in0_f32.get_size());
                std::cerr << "[METAL MLIR] input0 first:";
                for (size_t i = 0; i < n; ++i) std::cerr << " " << p[i];
                std::cerr << "\n";
            } else if (metal_debug_enabled() && in0.get_size() > 0) {
                auto in0_f32 = to_float32_tensor(in0);
                auto* p = in0_f32.data<const float>();
                std::cerr << "[METAL MLIR] input0 scalar=" << p[0] << "\n";
            }
            if (seg_op(0).input0) buf_map[seg_op(0).input0] = make_buffer_from_tensor(in0);
            if (in_vec->size() > 1 && seg_op(0).input1) {
                const auto& in1 = (*in_vec)[1];
                if (metal_debug_enabled()) {
                    auto in1_f32 = to_float32_tensor(in1);
                    auto* p1 = in1_f32.data<const float>();
                    size_t n1 = std::min<size_t>(8, in1_f32.get_size());
                    std::cerr << "[METAL MLIR] input1 first:";
                    for (size_t i = 0; i < n1; ++i) std::cerr << " " << p1[i];
                    std::cerr << "\n";
                }
                buf_map[seg_op(0).input1] = make_buffer_from_tensor(in1);
            }
        }
        // Pre-create buffers for constant tensors referenced in the segment (helps broadcast const exponents).
        for (size_t i = seg.first_op_index; i < seg.first_op_index + seg.op_count; ++i) {
            const auto& op = m_ops[i];
            auto try_bind_const = [&](const KernelTensor* t) {
                if (!t || !t->from_constant || t->const_data.empty())
                    return;
                if (buf_map.find(t) != buf_map.end())
                    return;
                id<MTLBuffer> buf = make_const_buffer(t);
                if (buf) temp_const_buffers.push_back(buf);
                buf_map[t] = buf;
            };
            try_bind_const(op.input0);
            try_bind_const(op.input1);
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
            if (!pipeline) {
                debug_log("[METAL MLIR] Null pipeline in run_segment dispatch -> fallback");
                [enc endEncoding];
                m_force_fallback = true;
                ensure_fallback(m_original_model);
                return;
            }
            [enc setComputePipelineState:pipeline];
            switch (op.kind) {
                case KernelOpKind::MatMul: {
                    if (!src0 || !src1 || !dst) {
                        debug_log("[METAL MLIR] MatMul missing buffer -> fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
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
                case KernelOpKind::Conv3D: {
                    std::cerr << "[METAL MLIR] entering Conv3D case\n";
                    if (!src0 || !src1 || !dst) {
                        debug_log("[METAL MLIR] Conv3D missing buffer -> fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
                    struct Conv3DParams {
                        uint32_t N, C_in, D, H, W;
                        uint32_t C_out;
                        uint32_t kD, kH, kW;
                        uint32_t strideD, strideH, strideW;
                        uint32_t dilationD, dilationH, dilationW;
                        uint32_t padFront, padTop, padLeft, padBack, padBottom, padRight;
                        uint32_t outD, outH, outW;
                    } params;
                    params.N = op.conv3d.N;
                    params.C_in = op.conv3d.C_in;
                    params.D = op.conv3d.D;
                    params.H = op.conv3d.H;
                    params.W = op.conv3d.W;
                    params.C_out = op.conv3d.C_out;
                    params.kD = op.conv3d.kernelD;
                    params.kH = op.conv3d.kernelH;
                    params.kW = op.conv3d.kernelW;
                    params.strideD = op.conv3d.strideD;
                    params.strideH = op.conv3d.strideH;
                    params.strideW = op.conv3d.strideW;
                    params.dilationD = op.conv3d.dilationD;
                    params.dilationH = op.conv3d.dilationH;
                    params.dilationW = op.conv3d.dilationW;
                    params.padFront = op.conv3d.padFront;
                    params.padTop = op.conv3d.padTop;
                    params.padLeft = op.conv3d.padLeft;
                    params.padBack = op.conv3d.padBack;
                    params.padBottom = op.conv3d.padBottom;
                    params.padRight = op.conv3d.padRight;
                    params.outD = op.conv3d.outD;
                    params.outH = op.conv3d.outH;
                    params.outW = op.conv3d.outW;
                    if (std::getenv("METAL_MLIR_DEBUG")) {
                        std::cerr << "[METAL MLIR] conv3d params N=" << params.N
                                  << " C_in=" << params.C_in << " D/H/W=" << params.D << "/" << params.H << "/" << params.W
                                  << " C_out=" << params.C_out << " k=" << params.kD << "x" << params.kH << "x" << params.kW
                                  << " stride=" << params.strideD << "," << params.strideH << "," << params.strideW
                                  << " dilation=" << params.dilationD << "," << params.dilationH << "," << params.dilationW
                                  << " padF/T/L/B/Bot/R=" << params.padFront << "," << params.padTop << "," << params.padLeft
                                  << "/" << params.padBack << "," << params.padBottom << "," << params.padRight
                                  << " out=" << params.outD << "x" << params.outH << "x" << params.outW << "\n";
                    }
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    [enc setBytes:&params length:sizeof(params) atIndex:3];
                    std::cerr << "[METAL MLIR] conv3d debug block\n";
                    auto* wptr = static_cast<const float*>([src1 contents]);
                    std::cerr << "[METAL MLIR] conv3d weights ptr=" << wptr << " sample: ";
                    auto wcount = std::min<uint32_t>(8, params.C_out * params.C_in * params.kD * params.kH * params.kW);
                    for (uint32_t i = 0; i < wcount; ++i) std::cerr << wptr[i] << (i + 1 < wcount ? " " : "\n");
                    auto* iptr = static_cast<const float*>([src0 contents]);
                    auto icount = std::min<uint32_t>(8, params.N * params.C_in * params.D * params.H * params.W);
                    std::cerr << "[METAL MLIR] conv3d input ptr=" << iptr << " sample: ";
                    for (uint32_t i = 0; i < icount; ++i) std::cerr << iptr[i] << (i + 1 < icount ? " " : "\n");
                    MTLSize grid = MTLSizeMake(params.C_out, params.N, 1);
                    MTLSize tg = MTLSizeMake(1, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::ElementwiseAdd:
                case KernelOpKind::ElementwiseSub:
                case KernelOpKind::ElementwiseMul:
                case KernelOpKind::ElementwiseDiv:
                case KernelOpKind::ElementwisePow:
                case KernelOpKind::ElementwiseMod:
                case KernelOpKind::ElementwiseFloorMod: {
                    if (!src0 || !src1 || !dst) {
                        debug_log("[METAL MLIR] Elementwise op missing buffer -> fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
                    if (metal_debug_enabled()) {
                        const float* p0 = static_cast<const float*>([src0 contents]);
                        const float* p1 = static_cast<const float*>([src1 contents]);
                        std::cerr << "[METAL MLIR] eltwise src0[0]=" << (p0 ? p0[0] : 0.f)
                                  << " src1[0]=" << (p1 ? p1[0] : 0.f)
                                  << " kind=" << static_cast<int>(op.kind) << "\n";
                    }
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    const NSUInteger elems = static_cast<NSUInteger>(tensor_num_elems(op.output));
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(elems, 1, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Unary: {
                    if (!src0 || !dst) {
                        debug_log("[METAL MLIR] Unary missing buffer -> fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
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
                    if (!src0 || !dst) {
                        debug_log("[METAL MLIR] Softmax missing buffer -> fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
                    if (metal_debug_enabled()) {
                        std::cerr << "[METAL MLIR] softmax dispatch rows=" << op.rows
                                  << " cols=" << op.cols << " inner=" << op.inner << "\n";
                    }
                    std::cerr << "[METAL MLIR] softmax dispatch rows=" << op.rows
                              << " cols=" << op.cols << " inner=" << op.inner << "\n";
                    if (src0) {
                        const float* p = static_cast<const float*>([src0 contents]);
                        size_t n = std::min<size_t>(8, tensor_num_elems(op.input0));
                        std::cerr << "[METAL MLIR] softmax src first:";
                        for (size_t i = 0; i < n; ++i) std::cerr << " " << p[i];
                        std::cerr << "\n";
                    }
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    struct SoftmaxParams {
                        uint32_t rows;
                        uint32_t cols;
                        uint32_t inner;
                    } params;
                    params.rows = static_cast<uint32_t>(op.rows);
                    params.cols = static_cast<uint32_t>(op.cols);
                    params.inner = static_cast<uint32_t>(op.inner);
                    [enc setBytes:&params length:sizeof(params) atIndex:2];
                    const NSUInteger rows = static_cast<NSUInteger>(op.rows);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(rows, 1, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Slice: {
                    if (!src0 || !dst) {
                        debug_log("[METAL MLIR] Slice missing buffer -> fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    const NSUInteger elems = static_cast<NSUInteger>(tensor_num_elems(op.output));
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(elems, 1, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Conv2D: {
                    if (!src0 || !src1 || !dst) {
                        debug_log("[METAL MLIR] Conv2D missing buffer -> fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
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
            if ((op.kind == KernelOpKind::MatMul || op.kind == KernelOpKind::ElementwiseAdd ||
                 op.kind == KernelOpKind::ElementwiseSub || op.kind == KernelOpKind::ElementwiseMul ||
                 op.kind == KernelOpKind::ElementwiseDiv || op.kind == KernelOpKind::ElementwisePow) &&
                !src1 && inputs.size() > 1) {
                src1 = make_buffer_from_tensor(inputs[1]);
            }
            if (!src1) src1 = load_const_input(op);
            if ((op.kind == KernelOpKind::MatMul || op.kind == KernelOpKind::ElementwiseAdd ||
                 op.kind == KernelOpKind::ElementwiseSub || op.kind == KernelOpKind::ElementwiseMul ||
                 op.kind == KernelOpKind::ElementwiseDiv || op.kind == KernelOpKind::ElementwisePow) &&
                !src1) { run_fallback(); return; }

            id<MTLBuffer> dst = alloc_out_buffer(op);
            if (!dst) { run_fallback(); return; }
            dispatch_op(op, seg_pipe(i), src0, src1, dst, cmd);
            buf_map[op.output] = dst;
            final_buf = dst;
        }

        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.error) {
            debug_log(std::string("[METAL MLIR] Command buffer error: ") +
                      [[cmd.error localizedDescription] UTF8String]);
        }

        if (final_buf) {
            if (outputs[0].get_element_type().is_integral()) {
                std::memcpy(outputs[0].data(), [final_buf contents], outputs[0].get_byte_size());
            } else {
                ov::Tensor tmp_f32{ov::element::f32, outputs[0].get_shape()};
                std::memcpy(tmp_f32.data(), [final_buf contents], tmp_f32.get_byte_size());
                copy_fp32_to_destination(tmp_f32.data<const float>(), outputs[0]);
            }
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

        // Softmax always compiled for METAL; no dynamic-only host path, no CPU fallback.
        m_softmax_dynamic_only = false;

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

        // Always harvest a flat IR so segmented execution and op-specific lowering
        // can reuse the same preparation path (even in release builds).
        build_flat_ir(model);
        build_segments_from_flat_ir();


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
                    if (ov::as_type_ptr<const ov::op::v1::Subtract>(n)) { kind_out = KernelOpKind::ElementwiseSub; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Divide>(n)) { kind_out = KernelOpKind::ElementwiseDiv; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Power>(n)) { kind_out = KernelOpKind::ElementwisePow; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Multiply>(n)) { kind_out = KernelOpKind::ElementwiseMul; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::Mod>(n)) { kind_out = KernelOpKind::ElementwiseMod; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::FloorMod>(n)) { kind_out = KernelOpKind::ElementwiseFloorMod; return true; }
                    if (ov::as_type_ptr<const ov::op::v0::SquaredDifference>(n)) { kind_out = KernelOpKind::ElementwiseMul; return true; }
                    if (ov::as_type_ptr<const ov::op::v0::Relu>(n) || ov::as_type_ptr<const ov::op::v0::Sigmoid>(n) ||
                        ov::as_type_ptr<const ov::op::v0::Tanh>(n) || ov::as_type_ptr<const ov::op::v0::Elu>(n) ||
                        ov::as_type_ptr<const ov::op::v0::PRelu>(n) || ov::as_type_ptr<const ov::op::v0::Gelu>(n) ||
                        ov::as_type_ptr<const ov::op::v4::Swish>(n)) { kind_out = KernelOpKind::Unary; return true; }
                    if (ov::is_type<ov::op::v1::Softmax>(n.get()) || ov::is_type<ov::op::v8::Softmax>(n.get())) { kind_out = KernelOpKind::Softmax; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::MaxPool>(n)) { kind_out = KernelOpKind::MaxPool2D; return true; }
                    if (ov::as_type_ptr<const ov::op::v1::AvgPool>(n)) { kind_out = KernelOpKind::AvgPool2D; return true; }
                    if (auto cv = ov::as_type_ptr<const ov::op::v1::Convolution>(n)) {
                        const auto rank = cv->get_input_partial_shape(0).rank();
                        if (rank.is_static() && rank.get_length() == 5) { kind_out = KernelOpKind::Conv3D; return true; }
                        kind_out = KernelOpKind::Conv2D; return true;
                    }
                    if (auto gcv = ov::as_type_ptr<const ov::op::v1::GroupConvolution>(n)) {
                        const auto rank = gcv->get_input_partial_shape(0).rank();
                        if (rank.is_static() && rank.get_length() == 5) { kind_out = KernelOpKind::Conv3D; return true; }
                        kind_out = KernelOpKind::Conv2D; return true;
                    }
                    if (ov::as_type_ptr<const ov::op::v0::BatchNormInference>(n) || ov::as_type_ptr<const ov::op::v5::BatchNormInference>(n)) {
                        kind_out = KernelOpKind::BatchNorm2D; return true; }
                    if (ov::as_type_ptr<const ov::op::v8::Slice>(n)) { kind_out = KernelOpKind::Slice; return true; }
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
                                    // If this is a pure broadcastable constant, pre-broadcast it and disable kernel-side broadcast.
                                    auto try_expand_const = [&](const ov::Shape& shape) {
                                        if (!m_const_b.empty()) {
                                            std::ostringstream dbg;
                                            dbg << "[METAL MLIR] try_expand_const: original first=" << m_const_b[0]
                                                << " size=" << m_const_b.size();
                                            debug_log(dbg.str());
                                        }
                                        if (!op.output) return false;
                                        const size_t out_elems = tensor_num_elems(op.output);
                                        if (out_elems == 0) return false;
                                        // Scalar
                                        if (m_const_b.size() == 1) {
                                            float v = m_const_b[0];
                                            m_const_b.assign(out_elems, v);
                                            return true;
                                        }
                                        // Channel broadcast for NCHW/NCL/NC
                                        if (op.out_shape.size() >= 2) {
                                            size_t Cdim = 1;
                                            size_t C = static_cast<size_t>(op.out_shape[Cdim]);
                                            if (m_const_b.size() == C) {
                                                m_const_b.assign(out_elems, 0.f);
                                                if (op.out_shape.size() == 2) {
                                                    size_t N = static_cast<size_t>(op.out_shape[0]);
                                                    for (size_t n = 0; n < N; ++n)
                                                        for (size_t cidx = 0; cidx < C; ++cidx)
                                                            m_const_b[n * C + cidx] = c->cast_vector<float>()[cidx];
                                                } else if (op.out_shape.size() == 3) {
                                                    size_t N = static_cast<size_t>(op.out_shape[0]);
                                                    size_t L = static_cast<size_t>(op.out_shape[2]);
                                                    for (size_t n = 0; n < N; ++n)
                                                        for (size_t cidx = 0; cidx < C; ++cidx)
                                                            for (size_t l = 0; l < L; ++l)
                                                                m_const_b[(n * C + cidx) * L + l] = c->cast_vector<float>()[cidx];
                                                } else if (op.out_shape.size() == 4) {
                                                    size_t N = static_cast<size_t>(op.out_shape[0]);
                                                    size_t H = static_cast<size_t>(op.out_shape[2]);
                                                    size_t W = static_cast<size_t>(op.out_shape[3]);
                                                    for (size_t n = 0; n < N; ++n)
                                                        for (size_t cidx = 0; cidx < C; ++cidx)
                                                            for (size_t h = 0; h < H; ++h)
                                                                for (size_t w = 0; w < W; ++w)
                                                                    m_const_b[((n * C + cidx) * H + h) * W + w] = c->cast_vector<float>()[cidx];
                                                }
                                                return true;
                                            }
                                        }
                                        if (m_const_b.size() == out_elems)
                                            return true;
                                        return false;
                                    };
                                    if (op.is_broadcast) {
                                        bool expanded = try_expand_const(c_shape);
                                        if (expanded) {
                                            op.is_broadcast = false;
                                            op.stride0.clear();
                                            op.stride1.clear();
                                        }
                                    }
                                }
                                break;
                            }
                        }
                        break;
                    }
                    case KernelOpKind::Conv2D:
                    case KernelOpKind::Conv3D: {
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

        if (!m_flat_ops.empty()) {
            Segment all_seg{0, m_flat_ops.size()};
            fill_constants_for_segment(all_seg);
        }

        auto try_pick_single_flat_segment = [&]() {
            if (!m_ops.empty())
                return;
            auto allow_kind = [](KernelOpKind k) {
                switch (k) {
                    case KernelOpKind::Unary:
                    case KernelOpKind::Softmax:
                    case KernelOpKind::Conv2D:
                    case KernelOpKind::Conv3D:
                    case KernelOpKind::BatchNorm2D:
                    case KernelOpKind::MatMul:
                    case KernelOpKind::ElementwiseAdd:
                    case KernelOpKind::ElementwiseSub:
                    case KernelOpKind::ElementwiseDiv:
                    case KernelOpKind::ElementwiseMul:
                    case KernelOpKind::ElementwiseMod:
                    case KernelOpKind::ElementwiseFloorMod:
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
                Segment io_seg{};
                io_seg.first_op_index = 0;
                io_seg.op_count = 1;
                for (const auto& in_port : model->inputs()) io_seg.input_ports.push_back(in_port);
                for (const auto& out_port : model->outputs()) io_seg.output_ports.push_back(out_port);
                m_segments.push_back(std::move(io_seg));
                m_ops_from_flat_segment = true;
#if METAL_MLIR_DEBUG
                debug_log("[METAL MLIR] Using single-op flat segment kind=" + std::to_string(static_cast<int>(op.kind)));
#endif
                // compile pipeline for this single op
                MetalKernelCompiler compiler(m_device);
                std::string log;
                m_pipelines.clear();
                bool compile_ok = true;
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
                    case KernelOpKind::Conv3D: {
                        try {
                            mlir::MLIRContext ctx;
                            auto module = build_mlir_conv3d_from_model(m_model, ctx);
                            run_mlir_pipeline(module);
                            Conv3DCodegenDesc desc;
                            const auto& kop = op;
                            desc.kind = KernelOpKind::Conv3D;
                            desc.N = kop.conv3d.N;
                            desc.C_in = kop.conv3d.C_in;
                            desc.D = kop.conv3d.D;
                            desc.H = kop.conv3d.H;
                            desc.W = kop.conv3d.W;
                            desc.C_out = kop.conv3d.C_out;
                            desc.kD = kop.conv3d.kernelD;
                            desc.kH = kop.conv3d.kernelH;
                            desc.kW = kop.conv3d.kernelW;
                            desc.strideD = kop.conv3d.strideD;
                            desc.strideH = kop.conv3d.strideH;
                            desc.strideW = kop.conv3d.strideW;
                            desc.dilationD = kop.conv3d.dilationD;
                            desc.dilationH = kop.conv3d.dilationH;
                            desc.dilationW = kop.conv3d.dilationW;
                            desc.padFront = kop.conv3d.padFront;
                            desc.padTop = kop.conv3d.padTop;
                            desc.padLeft = kop.conv3d.padLeft;
                            desc.padBack = kop.conv3d.padBack;
                            desc.padBottom = kop.conv3d.padBottom;
                            desc.padRight = kop.conv3d.padRight;
                            desc.outD = kop.conv3d.outD;
                            desc.outH = kop.conv3d.outH;
                            desc.outW = kop.conv3d.outW;
                            auto source = generate_msl_from_mlir(module, desc);
                            m_pipelines.push_back(compiler.compile_msl_from_source(source, "conv3d_kernel", log));
                        } catch (const std::exception& e) {
                            debug_log(std::string("[METAL MLIR] Conv3D flat segment compile failed: ") + e.what());
                            compile_ok = false;
                        }
                        break;
                    }
                    case KernelOpKind::BatchNorm2D:
                        m_pipelines.push_back(compiler.compile_batchnorm2d_kernel(op, log));
                        break;
                    case KernelOpKind::MatMul:
                        m_pipelines.push_back(compiler.compile_matmul_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseAdd:
                        m_pipelines.push_back(compiler.compile_add_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseSub:
                        m_pipelines.push_back(compiler.compile_sub_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseDiv:
                        m_pipelines.push_back(compiler.compile_div_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseMul:
                        m_pipelines.push_back(compiler.compile_mul_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseMod:
                        m_pipelines.push_back(compiler.compile_mod_kernel(op, log));
                        break;
                    case KernelOpKind::ElementwiseFloorMod:
                        m_pipelines.push_back(compiler.compile_floor_mod_kernel(op, log));
                        break;
                    case KernelOpKind::Slice:
                        m_pipelines.push_back(compiler.compile_slice_kernel(op, log));
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
                m_ops.clear();
                m_segments.clear();
                for (const auto& op : m_flat_ops) {
                    m_ops.push_back(op);
                }
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
                        case KernelOpKind::ElementwiseSub:
                            m_pipelines.push_back(compiler.compile_sub_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwiseDiv:
                            m_pipelines.push_back(compiler.compile_div_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwisePow:
                            m_pipelines.push_back(compiler.compile_pow_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwiseMul:
                            m_pipelines.push_back(compiler.compile_mul_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwiseMod:
                            m_pipelines.push_back(compiler.compile_mod_kernel(op, log));
                            break;
                        case KernelOpKind::ElementwiseFloorMod:
                            m_pipelines.push_back(compiler.compile_floor_mod_kernel(op, log));
                            break;
                        case KernelOpKind::Unary:
                            m_pipelines.push_back(compiler.compile_unary_kernel(op, log));
                            break;
                        case KernelOpKind::Conv2D:
                            m_pipelines.push_back(compiler.compile_conv2d_kernel(op, log));
                            break;
                        case KernelOpKind::BatchNorm2D:
                            m_pipelines.push_back(compiler.compile_batchnorm2d_kernel(op, log));
                            break;
                        case KernelOpKind::MaxPool2D:
                            m_pipelines.push_back(compiler.compile_maxpool2d_kernel(op, log));
                            break;
                        case KernelOpKind::AvgPool2D:
                            m_pipelines.push_back(compiler.compile_avgpool2d_kernel(op, log));
                            break;
                        case KernelOpKind::Slice:
                            m_pipelines.push_back(compiler.compile_slice_kernel(op, log));
                            break;
                        case KernelOpKind::Split:
                            m_pipelines.push_back(compiler.compile_split_kernel(op, log));
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
        bool has_conv3d = analysis.has_conv3d;
        bool has_batchnorm = analysis.has_batchnorm;
        // Probe for constant B to support single-input matmuls
        std::shared_ptr<const ov::op::v0::MatMul> matmul_node;
        for (const auto& node : model->get_ordered_ops()) {
            if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
                matmul_node = mm;
                break;
            }
        }
        if (!has_matmul && !has_add && !has_add_broadcast && !has_mul && !analysis.has_pow &&
            !analysis.has_unary && !analysis.has_softmax && !analysis.has_maxpool && !analysis.has_avgpool &&
            !analysis.has_conv2d && !analysis.has_batchnorm && !analysis.has_split) {
            if (analysis.compute_ops == 0) {
                debug_log("[MlirBackend] shape-only graph after template probe; skipping Metal pipelines");
                return;
            }
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
            m_pipelines.clear();
            MatMulCodegenDesc desc;
            const auto& kop = m_ops[0];
            desc.M = kop.M;
            desc.N = kop.N;
            desc.K = kop.K;
            desc.batch = kop.batch;
            desc.batch_a = kop.batch_a;
            desc.batch_b = kop.batch_b;
            desc.a_transpose = kop.a_transpose;
            desc.b_transpose = kop.b_transpose;
            desc.b_is_nk_layout = kop.b_is_nk_layout;
            auto source = generate_msl_from_mlir(module, desc);
            m_pipelines.push_back(compiler.compile_msl_from_source(source, "matmul_kernel", log));
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
            m_pipelines.clear();
            const auto& op = m_ops[0];
            if (!op.is_broadcast && op.output) {
                try {
                    mlir::MLIRContext ctx;
                    auto module = build_mlir_broadcast_add_from_model(model, ctx);  // simple generic add
                    run_mlir_pipeline(module);
                    EltwiseCodegenDesc desc;
                    desc.kind = KernelOpKind::ElementwiseAdd;
                    desc.eltwise_kind = KernelOpKind::ElementwiseAdd;
                    uint32_t elems = 1;
                    if (op.output) {
                        for (auto d : op.output->shape) elems *= static_cast<uint32_t>(d);
                    }
                    desc.num_elements = elems;
                    desc.is_broadcast = false;
                    auto source = generate_msl_from_mlir(module, desc);
                    m_pipelines.push_back(compiler.compile_msl_from_source(source, "eltwise_kernel", log));
                } catch (const std::exception& e) {
                    debug_log(std::string("[METAL MLIR] Add MLIR→MSL failed, fallback: ") + e.what());
                    m_pipelines.push_back(compiler.compile_add_kernel(m_ops[0], log));
                }
            } else {
                m_pipelines.push_back(compiler.compile_add_kernel(m_ops[0], log));
            }
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

        if (has_mul && !analysis.has_div && !analysis.has_mod && !analysis.has_floor_mod &&
            analysis.compute_ops == 1 && !has_add && !has_add_broadcast && !analysis.has_unary &&
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
            // Always generate a Metal softmax kernel (no CPU fallback).
            // Build a minimal IR manually (one input/one output softmax).
            int64_t axis = -1;
            for (const auto& node : model->get_ordered_ops()) {
                if (auto s1 = ov::as_type_ptr<const ov::op::v1::Softmax>(node)) { axis = s1->get_axis(); break; }
                if (auto s8 = ov::as_type_ptr<const ov::op::v8::Softmax>(node)) { axis = s8->get_axis(); break; }
            }

            auto pshape = model->input().get_partial_shape();
            if (!pshape.rank().is_static()) {
                OPENVINO_THROW("METAL Softmax expects static rank");
            }
            const int64_t rank = pshape.rank().get_length();
            if (axis < 0) axis += rank;

            std::vector<int64_t> in_shape;
            in_shape.reserve(static_cast<size_t>(rank));
            for (int64_t i = 0; i < rank; ++i) {
                const auto& d = pshape[static_cast<size_t>(i)];
                in_shape.push_back(d.is_static() ? d.get_length() : int64_t{-1});
            }

            int64_t cols = 0, inner = 0, outer = 0, rows = 0;
            if (pshape.is_static()) {
                cols = in_shape[static_cast<size_t>(axis)];
                inner = 1;
                for (int64_t i = axis + 1; i < rank; ++i) inner *= in_shape[static_cast<size_t>(i)];
                outer = 1;
                for (int64_t i = 0; i < axis; ++i) outer *= in_shape[static_cast<size_t>(i)];
                rows = outer * inner;
            }

            MetalKernelIR ir;
            ir.tensors.resize(2);
            ir.ops.clear();

            auto dtype = resolve_metal_dtype(model->input().get_element_type());
            ir.tensors[0].name = "input0";
            ir.tensors[0].shape = pshape.is_static() ? in_shape : std::vector<int64_t>{};
            ir.tensors[0].dtype = dtype;
            ir.tensors[1].name = "output0";
            ir.tensors[1].shape = pshape.is_static() ? in_shape : std::vector<int64_t>{};
            ir.tensors[1].dtype = dtype;

            KernelOp op{};
            op.kind = KernelOpKind::Softmax;
            op.input0 = &ir.tensors[0];
            op.output = &ir.tensors[1];
            op.softmax_axis = axis;
            op.rows = rows;
            op.cols = cols;
            op.inner = inner;
            op.out_shape = pshape.is_static() ? in_shape : std::vector<int64_t>{};
            op.dtype = dtype;
            ir.ops.push_back(op);

            m_ir = std::move(ir);
            set_ops_from_ir();
            m_ops_from_flat_segment = true;
            m_pipelines.clear();
            m_pipelines.push_back(compiler.compile_softmax_kernel(m_ops[0], log));
            return;
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
            !has_maxpool && !has_avgpool && !has_batchnorm && !analysis.has_conv3d) {
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
            if (use_handwritten_msl()) {
                debug_log("[METAL MLIR] Conv2D: OV_METAL_USE_HANDWRITTEN_MSL=1, using fallback kernel");
                m_pipelines.push_back(compiler.compile_conv2d_kernel(m_ops[0], log));
            } else {
                try {
                    mlir::MLIRContext ctx;
                    auto module = build_mlir_conv2d_from_model(model, ctx);
                    run_mlir_pipeline(module);
                    Conv2DCodegenDesc desc;
                    const auto& kop = m_ops[0];
                    desc.kind = KernelOpKind::Conv2D;
                    desc.N = kop.conv2d.N;
                    desc.C_in = kop.conv2d.C_in;
                    desc.H = kop.conv2d.H;
                    desc.W = kop.conv2d.W;
                    desc.C_out = kop.conv2d.C_out;
                    desc.kH = kop.conv2d.kernelH;
                    desc.kW = kop.conv2d.kernelW;
                    desc.strideH = kop.conv2d.strideH;
                    desc.strideW = kop.conv2d.strideW;
                    desc.dilationH = kop.conv2d.dilationH;
                    desc.dilationW = kop.conv2d.dilationW;
                    desc.padTop = kop.conv2d.padTop;
                    desc.padLeft = kop.conv2d.padLeft;
                    desc.padBottom = kop.conv2d.padBottom;
                    desc.padRight = kop.conv2d.padRight;
                    desc.outH = kop.output && kop.output->shape.size() > 2 ? static_cast<uint32_t>(kop.output->shape[2]) : 0;
                    desc.outW = kop.output && kop.output->shape.size() > 3 ? static_cast<uint32_t>(kop.output->shape[3]) : 0;
                    desc.groups = kop.conv2d.groups;
                    auto source = generate_msl_from_mlir(module, desc);
                    m_pipelines.push_back(compiler.compile_msl_from_source(source, "conv2d_kernel", log));
                } catch (const std::exception& e) {
                    debug_log(std::string("[METAL MLIR] Conv2D MLIR→MSL failed, fallback to hand-written kernel: ") + e.what());
                    m_pipelines.push_back(compiler.compile_conv2d_kernel(m_ops[0], log));
                }
            }
            return;
        }

        if (analysis.has_conv3d && !has_matmul && !has_add && !has_add_broadcast && !analysis.has_unary &&
            !analysis.has_softmax && !has_maxpool && !has_avgpool && !has_batchnorm) {
            bool const_w = false;
            MetalKernelIR ir = build_kernel_ir_for_conv3d(model, const_w);
            m_ir = std::move(ir);
            set_ops_from_ir();
            m_ops_from_flat_segment = true;
            debug_log("[METAL MLIR] Conv3D selected: input " + describe_shape(m_ir.ops[0].input0) +
                      " weights " + describe_shape(m_ir.ops[0].input1) +
                      " output " + describe_shape(m_ir.ops[0].output));
            if (const_w && !m_ops.empty()) {
                std::shared_ptr<const ov::Node> conv_node;
                for (const auto& node : model->get_ordered_ops()) {
                    if (auto c = ov::as_type_ptr<const ov::op::v1::Convolution>(node)) {
                        if (c->get_input_shape(0).size() == 5) { conv_node = c; break; }
                    }
                }
                if (conv_node) {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(conv_node->get_input_node_shared_ptr(1))) {
                        m_const_w = c->cast_vector<float>();
                        m_has_const_w = true;
                        debug_log("[METAL MLIR] Conv3D const weights size=" + std::to_string(m_const_w.size()));
                    }
                }
            }
            m_pipelines.clear();
            MetalKernelCompiler compiler(m_device);
            std::string log;
            try {
                mlir::MLIRContext ctx;
                auto module = build_mlir_conv3d_from_model(model, ctx);
                run_mlir_pipeline(module);
                Conv3DCodegenDesc desc;
                const auto& kop = m_ops[0];
                desc.kind = KernelOpKind::Conv3D;
                desc.N = kop.conv3d.N;
                desc.C_in = kop.conv3d.C_in;
                desc.D = kop.conv3d.D;
                desc.H = kop.conv3d.H;
                desc.W = kop.conv3d.W;
                desc.C_out = kop.conv3d.C_out;
                desc.kD = kop.conv3d.kernelD;
                desc.kH = kop.conv3d.kernelH;
                desc.kW = kop.conv3d.kernelW;
                desc.strideD = kop.conv3d.strideD;
                desc.strideH = kop.conv3d.strideH;
                desc.strideW = kop.conv3d.strideW;
                desc.dilationD = kop.conv3d.dilationD;
                desc.dilationH = kop.conv3d.dilationH;
                desc.dilationW = kop.conv3d.dilationW;
                desc.padFront = kop.conv3d.padFront;
                desc.padTop = kop.conv3d.padTop;
                desc.padLeft = kop.conv3d.padLeft;
                desc.padBack = kop.conv3d.padBack;
                desc.padBottom = kop.conv3d.padBottom;
                desc.padRight = kop.conv3d.padRight;
                desc.outD = kop.conv3d.outD;
                desc.outH = kop.conv3d.outH;
                desc.outW = kop.conv3d.outW;
                auto source = generate_msl_from_mlir(module, desc);
                m_pipelines.push_back(compiler.compile_msl_from_source(source, "conv3d_kernel", log));
            } catch (const std::exception& e) {
                debug_log(std::string("[METAL MLIR] Conv3D MLIR→MSL failed, fallback to CPU: ") + e.what());
                force_cpu_fallback(m_original_model);
                return;
            }
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
            m_pipelines.clear();
            if (use_handwritten_msl()) {
                debug_log("[METAL MLIR] MaxPool: OV_METAL_USE_HANDWRITTEN_MSL=1, using fallback kernel");
                m_pipelines.push_back(compiler.compile_maxpool2d_kernel(m_ops[0], log));
            } else {
                try {
                    mlir::MLIRContext ctx;
                    auto module = build_mlir_maxpool_from_model(model, ctx);
                    run_mlir_pipeline(module);
                    Pool2DCodegenDesc desc;
                    const auto& kop = m_ops[0];
                    desc.kind = KernelOpKind::MaxPool2D;
                    desc.N = kop.pool.N;
                    desc.C = kop.pool.C;
                    desc.H = kop.pool.H;
                    desc.W = kop.pool.W;
                    desc.kH = kop.pool.kernelH;
                    desc.kW = kop.pool.kernelW;
                    desc.strideH = kop.pool.strideH;
                    desc.strideW = kop.pool.strideW;
                    desc.padTop = kop.pool.padTop;
                    desc.padLeft = kop.pool.padLeft;
                    desc.padBottom = 0;
                    desc.padRight = 0;
                    desc.outH = kop.pool.outH;
                    desc.outW = kop.pool.outW;
                    desc.is_avg = false;
                    desc.exclude_pad = kop.pool.exclude_pad;
                    auto source = generate_msl_from_mlir(module, desc);
                    m_pipelines.push_back(compiler.compile_msl_from_source(source, "pool2d_kernel", log));
                } catch (const std::exception& e) {
                    debug_log(std::string("[METAL MLIR] MaxPool MLIR→MSL failed, fallback: ") + e.what());
                    m_pipelines.push_back(compiler.compile_maxpool2d_kernel(m_ops[0], log));
                }
            }
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
            m_pipelines.clear();
            if (use_handwritten_msl()) {
                debug_log("[METAL MLIR] AvgPool: OV_METAL_USE_HANDWRITTEN_MSL=1, using fallback kernel");
                m_pipelines.push_back(compiler.compile_avgpool2d_kernel(m_ops[0], log));
            } else {
                try {
                    mlir::MLIRContext ctx;
                    auto module = build_mlir_avgpool_from_model(model, ctx);
                    run_mlir_pipeline(module);
                    Pool2DCodegenDesc desc;
                    const auto& kop = m_ops[0];
                    desc.kind = KernelOpKind::AvgPool2D;
                    desc.N = kop.pool.N;
                    desc.C = kop.pool.C;
                    desc.H = kop.pool.H;
                    desc.W = kop.pool.W;
                    desc.kH = kop.pool.kernelH;
                    desc.kW = kop.pool.kernelW;
                    desc.strideH = kop.pool.strideH;
                    desc.strideW = kop.pool.strideW;
                    desc.padTop = kop.pool.padTop;
                    desc.padLeft = kop.pool.padLeft;
                    desc.padBottom = 0;
                    desc.padRight = 0;
                    desc.outH = kop.pool.outH;
                    desc.outW = kop.pool.outW;
                    desc.is_avg = true;
                    desc.exclude_pad = kop.pool.exclude_pad;
                    auto source = generate_msl_from_mlir(module, desc);
                    m_pipelines.push_back(compiler.compile_msl_from_source(source, "pool2d_kernel", log));
                } catch (const std::exception& e) {
                    debug_log(std::string("[METAL MLIR] AvgPool MLIR→MSL failed, fallback: ") + e.what());
                    m_pipelines.push_back(compiler.compile_avgpool2d_kernel(m_ops[0], log));
                }
            }
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
            if (analysis.compute_ops == 0) {
                debug_log("[MlirBackend] shape-only graph; no compute ops to lower");
                // Leave m_ops empty so run() can handle reshape/transpose host path without fallback.
                return;
            }
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
    std::vector<ov::Tensor> m_last_inputs;
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
        @autoreleasepool {
        auto run_fallback = [&]() -> std::vector<ov::Tensor> {
            if (metal_debug_enabled()) {
                debug_log("[METAL MLIR] run_segment -> run_fallback");
            }
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

        // Ensure ops are present; if compile filled m_ir but not m_ops, sync them.
        if (m_ops.empty() && !m_ir.ops.empty()) {
            m_ops = m_ir.ops;
            m_segments.clear();
            m_segments.push_back(Segment{0, m_ops.size()});
        }

        if (seg.op_count == 0 || seg.first_op_index + seg.op_count > m_ops.size()) {
            debug_log("[METAL MLIR] run_segment guard failed; regenerating ops from IR");
            if (!m_ir.ops.empty()) {
                m_ops = m_ir.ops;
                m_segments.clear();
                m_segments.push_back(Segment{0, m_ops.size()});
            }
        }
        if (seg.op_count == 0 || seg.first_op_index + seg.op_count > m_ops.size()) {
            debug_log("[METAL MLIR] run_segment unable to recover ops → fallback");
            return run_fallback();
        }

        if (m_pipelines.size() < seg.first_op_index + seg.op_count) {
            debug_log("[METAL MLIR] run_segment missing pipelines → try lazy compile");
            MetalKernelCompiler compiler(m_device);
            std::string log;
            // Lazy-compile softmax (and other single-op segments) on demand.
            for (size_t i = seg.first_op_index; i < seg.first_op_index + seg.op_count; ++i) {
                if (m_pipelines.size() <= i) {
                    m_pipelines.resize(i + 1);
                }
                if (m_pipelines[i] != nil)
                    continue;
                const auto& kop = m_ops[i];
                switch (kop.kind) {
                    case KernelOpKind::Softmax:
                        m_pipelines[i] = compiler.compile_softmax_kernel(kop, log);
                        break;
                    case KernelOpKind::ElementwiseAdd:     m_pipelines[i] = compiler.compile_add_kernel(kop, log); break;
                    case KernelOpKind::ElementwiseSub:     m_pipelines[i] = compiler.compile_sub_kernel(kop, log); break;
                    case KernelOpKind::ElementwiseMul:     m_pipelines[i] = compiler.compile_mul_kernel(kop, log); break;
                    case KernelOpKind::ElementwiseDiv:     m_pipelines[i] = compiler.compile_div_kernel(kop, log); break;
                    case KernelOpKind::ElementwisePow:     m_pipelines[i] = compiler.compile_pow_kernel(kop, log); break;
                    case KernelOpKind::ElementwiseMod:     m_pipelines[i] = compiler.compile_mod_kernel(kop, log); break;
                    case KernelOpKind::ElementwiseFloorMod:m_pipelines[i] = compiler.compile_floor_mod_kernel(kop, log); break;
                    case KernelOpKind::Unary:
                        m_pipelines[i] = compiler.compile_unary_kernel(kop, log);
                        break;
                    default:
                        break;
                }
            }
            if (m_pipelines.empty() || m_pipelines.size() < seg.first_op_index + seg.op_count) {
                debug_log("[METAL MLIR] run_segment still missing pipelines → fallback");
                return run_fallback();
            }
        }

        // Buffers for temporary constant materialization (e.g., scalar/vector pow exponents)
        std::vector<id<MTLBuffer>> temp_const_buffers;
        auto make_const_buffer = [&](const KernelTensor* t) -> id<MTLBuffer> {
            if (!t || !t->from_constant || t->const_data.empty())
                return nil;
            switch (t->dtype.storage) {
                case MetalDType::StorageType::I32: {
                    std::vector<int32_t> tmp(t->const_data.size());
                    for (size_t i = 0; i < t->const_data.size(); ++i)
                        tmp[i] = static_cast<int32_t>(t->const_data[i]);
                    return [m_device newBufferWithBytes:tmp.data()
                                                 length:tmp.size() * sizeof(int32_t)
                                                options:MTLResourceStorageModeShared];
                }
                case MetalDType::StorageType::I64: {
                    std::vector<int64_t> tmp(t->const_data.size());
                    for (size_t i = 0; i < t->const_data.size(); ++i)
                        tmp[i] = static_cast<int64_t>(t->const_data[i]);
                    return [m_device newBufferWithBytes:tmp.data()
                                                 length:tmp.size() * sizeof(int64_t)
                                                options:MTLResourceStorageModeShared];
                }
                case MetalDType::StorageType::F16:
                case MetalDType::StorageType::F32:
                default:
                    return [m_device newBufferWithBytes:t->const_data.data()
                                                 length:t->const_data.size() * sizeof(float)
                                                options:MTLResourceStorageModeShared];
            }
        };
        auto load_const_input = [&](const KernelOp& op) -> id<MTLBuffer> {
            if (!op.input1 || !op.input1->from_constant)
                return nil;
            id<MTLBuffer> buf = make_const_buffer(op.input1);
            if (buf) temp_const_buffers.push_back(buf);
            return buf;
        };

        // For now we assume single-input / single-output IO-aligned segment.
        if (!seg.input_ports.empty()) {
            OPENVINO_ASSERT(inputs.size() == seg.input_ports.size(),
                            "run_segment expects inputs matching segment IO");
        } else {
            OPENVINO_ASSERT(!inputs.empty(), "run_segment expects at least one input tensor");
        }

        auto seg_op = [&](size_t i) -> const KernelOp& { return m_ops[seg.first_op_index + i]; };
        auto seg_pipe = [&](size_t i) -> id<MTLComputePipelineState> { return m_pipelines[seg.first_op_index + i]; };

#if METAL_MLIR_DEBUG
        {
            std::ostringstream dbg;
            dbg << "[METAL MLIR] run_segment ops=" << seg.op_count << " kinds:";
            for (size_t i = 0; i < seg.op_count; ++i) dbg << " " << static_cast<int>(seg_op(i).kind);
            debug_log(dbg.str());
        }
#endif

        auto tensor_num_elems = [](const KernelTensor* t) -> size_t {
            if (!t) return 0;
            size_t elems = 1;
            for (auto d : t->shape) elems *= static_cast<size_t>(d);
            return elems;
        };
        auto compute_size = [](const MetalDType& dt) -> size_t {
            switch (dt.compute) {
                case MetalDType::ComputeType::F32: return sizeof(float);
                case MetalDType::ComputeType::I32: return sizeof(int32_t);
                case MetalDType::ComputeType::I64: return sizeof(int64_t);
                default: return sizeof(float);
            }
        };
        auto element_size = [&](const KernelTensor* t) -> size_t {
            if (!t) return sizeof(float);
            if (t->dtype.ov_type == ov::element::dynamic)
                return sizeof(float);
            return compute_size(t->dtype);
        };
        auto tensor_bytes = [&](const KernelTensor* t) -> size_t {
            return tensor_num_elems(t) * element_size(t);
        };

        ov::Tensor input0_tensor = inputs[0];
        ov::Tensor input_f32;
        if (inputs[0].get_element_type().is_real()) {
            input0_tensor = to_float32_tensor(inputs[0]);
            input_f32 = input0_tensor;
        }
        if (metal_debug_enabled()) {
            auto dump_tensor = [](const ov::Tensor& t, const char* tag) {
                if (!t.get_element_type().is_real()) {
                    std::cerr << "[METAL MLIR] input dump " << tag << " (non-float type) skipped\n";
                    return;
                }
                std::cerr << "[METAL MLIR] input dump " << tag << " shape=";
                auto s = t.get_shape();
                std::cerr << "[";
                for (size_t i = 0; i < s.size(); ++i) {
                    if (i) std::cerr << ",";
                    std::cerr << s[i];
                }
                std::cerr << "] first:";
                ov::Tensor as_f32 = to_float32_tensor(t);
                const float* p = as_f32.data<const float>();
                size_t n = std::min<size_t>(8, as_f32.get_size());
                for (size_t i = 0; i < n; ++i) std::cerr << " " << p[i];
                std::cerr << "\n";
            };
            dump_tensor(input_f32, "in0");
            if (inputs.size() > 1) dump_tensor(inputs[1], "in1");
        }

        const KernelOp& last_op = seg_op(seg.op_count - 1);
        ov::Shape out_shape;
        if (!m_model->outputs().empty() && m_model->output(0).get_partial_shape().is_static()) {
            out_shape = m_model->output(0).get_shape();
        } else if (last_op.output && !last_op.output->shape.empty()) {
            out_shape.assign(last_op.output->shape.begin(), last_op.output->shape.end());
        } else {
            out_shape = inputs[0].get_shape();
        }
        ov::element::Type out_elem_type = inputs[0].get_element_type();
        if (last_op.output) {
            out_elem_type = last_op.output->dtype.ov_type;
        }
        size_t out_bytes = last_op.output ? tensor_bytes(last_op.output) : inputs[0].get_byte_size();
        MetalDType out_dtype = last_op.output ? last_op.output->dtype : resolve_metal_dtype(out_elem_type);
        auto num_elems_from_shape = [](const ov::Shape& s) {
            size_t n = 1;
            for (auto d : s) n *= d;
            return n;
        };
        out_bytes = num_elems_from_shape(out_shape) * compute_size(out_dtype);

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
            max_tmp_bytes = input0_tensor.get_byte_size();

        // Buffer mapping for flat execution
        std::unordered_map<const KernelTensor*, id<MTLBuffer>> buf_map;

        auto make_buffer_from_tensor = [&](const ov::Tensor& t, const MetalDType& dtype) -> id<MTLBuffer> {
            if (dtype.storage == MetalDType::StorageType::I32 || dtype.storage == MetalDType::StorageType::I64) {
                return [m_device newBufferWithBytes:t.data()
                                             length:t.get_byte_size()
                                            options:MTLResourceStorageModeShared];
            }
            ov::Tensor tmp = t;
            if (t.get_element_type() == ov::element::f16) {
                tmp = to_float32_tensor(t);
            }
            return [m_device newBufferWithBytes:tmp.data()
                                         length:tmp.get_byte_size()
                                        options:MTLResourceStorageModeShared];
        };

        MetalDType in0_dt = resolve_metal_dtype(inputs[0].get_element_type());
        ov::Tensor buf_src0 = (in0_dt.storage == MetalDType::StorageType::F16 || in0_dt.storage == MetalDType::StorageType::F32)
                                  ? to_float32_tensor(inputs[0])
                                  : inputs[0];
        id<MTLBuffer> buf_in = make_buffer_from_tensor(buf_src0, in0_dt);
        id<MTLBuffer> buf_tmp = [m_device newBufferWithLength:max_tmp_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_out = [m_device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_in1 = nil;
        if (inputs.size() > 1) {
            MetalDType in1_dt = resolve_metal_dtype(inputs[1].get_element_type());
            ov::Tensor buf_src1 = (in1_dt.storage == MetalDType::StorageType::F16 || in1_dt.storage == MetalDType::StorageType::F32)
                                      ? to_float32_tensor(inputs[1])
                                      : inputs[1];
            buf_in1 = make_buffer_from_tensor(buf_src1, in1_dt);
        }

        auto make_const_buffer_vec = [&](const std::vector<float>& data, MetalDType::StorageType storage) -> id<MTLBuffer> {
            if (data.empty())
                return nil;
            if (storage == MetalDType::StorageType::I32) {
                std::vector<int32_t> tmp(data.size());
                std::transform(data.begin(), data.end(), tmp.begin(), [](float v) { return static_cast<int32_t>(v); });
                return [m_device newBufferWithBytes:tmp.data()
                                             length:tmp.size() * sizeof(int32_t)
                                            options:MTLResourceStorageModeShared];
            } else if (storage == MetalDType::StorageType::I64) {
                std::vector<int64_t> tmp(data.size());
                std::transform(data.begin(), data.end(), tmp.begin(), [](float v) { return static_cast<int64_t>(v); });
                return [m_device newBufferWithBytes:tmp.data()
                                             length:tmp.size() * sizeof(int64_t)
                                            options:MTLResourceStorageModeShared];
            }
            return [m_device newBufferWithBytes:data.data()
                                         length:data.size() * sizeof(float)
                                        options:MTLResourceStorageModeShared];
        };

        id<MTLBuffer> buf_const_add = m_has_const_b ? make_const_buffer_vec(m_const_b, last_op.dtype.storage) : nil;
        id<MTLBuffer> buf_const_mul = m_has_const_mul ? make_const_buffer_vec(m_const_mul, last_op.dtype.storage) : nil;
        id<MTLBuffer> buf_const_pow = nil;
        id<MTLBuffer> buf_const_bn = !m_const_bn.empty() ? make_const_buffer_vec(m_const_bn, last_op.dtype.storage) : nil;
        id<MTLBuffer> buf_const_w = m_has_const_w ? make_const_buffer_vec(m_const_w, last_op.dtype.storage) : nil;
        id<MTLBuffer> buf_const_mm0 = m_has_const_mm0 ? make_const_buffer_vec(m_const_mm0, last_op.dtype.storage) : nil;
        id<MTLBuffer> buf_const_mm1 = m_has_const_mm1 ? make_const_buffer_vec(m_const_mm1, last_op.dtype.storage) : nil;
        if (std::getenv("METAL_MLIR_DEBUG")) {
            std::cerr << "[METAL MLIR] const weights size=" << m_const_w.size()
                      << " has_const_w=" << (m_has_const_w ? "yes" : "no") << "\n";
        }
        // load_const_input defined earlier (before first dispatch loop)

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
                    if (d < op.stride1.size() && op.stride1[d] != 0) {
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
            if (!pipeline) {
                debug_log("[METAL MLIR] Null pipeline in run_segment dispatch → lazy compile");
                MetalKernelCompiler compiler(m_device);
                std::string log;
                switch (op.kind) {
                    case KernelOpKind::Softmax:
                        pipeline = compiler.compile_softmax_kernel(op, log);
                        break;
                    case KernelOpKind::ElementwiseAdd:     pipeline = compiler.compile_add_kernel(op, log); break;
                    case KernelOpKind::ElementwiseSub:     pipeline = compiler.compile_sub_kernel(op, log); break;
                    case KernelOpKind::ElementwiseMul:     pipeline = compiler.compile_mul_kernel(op, log); break;
                    case KernelOpKind::ElementwiseDiv:     pipeline = compiler.compile_div_kernel(op, log); break;
                    case KernelOpKind::ElementwisePow:     pipeline = compiler.compile_pow_kernel(op, log); break;
                    case KernelOpKind::ElementwiseMod:     pipeline = compiler.compile_mod_kernel(op, log); break;
                    case KernelOpKind::ElementwiseFloorMod:pipeline = compiler.compile_floor_mod_kernel(op, log); break;
                    case KernelOpKind::Unary:
                        pipeline = compiler.compile_unary_kernel(op, log);
                        break;
                    default:
                        break;
                }
                // Persist the newly compiled pipeline for future dispatches.
                size_t idx = seg.first_op_index + static_cast<size_t>(&op - &m_ops[seg.first_op_index]);
                if (m_pipelines.size() <= idx)
                    m_pipelines.resize(idx + 1);
                m_pipelines[idx] = pipeline;
                if (!pipeline) {
                    [enc endEncoding];
                    m_force_fallback = true;
                    ensure_fallback(m_original_model);
                    return;
                }
            }
            [enc setComputePipelineState:pipeline];
            switch (op.kind) {
                case KernelOpKind::MatMul: {
                    if (!src0 || !src1 || !dst) {
                        debug_log("[METAL MLIR] MatMul missing buffer → fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
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
                case KernelOpKind::ElementwiseSub:
                case KernelOpKind::ElementwiseMul:
                case KernelOpKind::ElementwiseDiv:
                case KernelOpKind::ElementwisePow:
                case KernelOpKind::ElementwiseMod:
                case KernelOpKind::ElementwiseFloorMod: {
                    if (!src0 || !src1 || !dst) {
                        debug_log("[METAL MLIR] Elementwise op missing buffer → fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
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
                    if (!src0 || !dst) {
                        debug_log("[METAL MLIR] Unary missing buffer → fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
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
                    if (!src0 || !dst) {
                        debug_log("[METAL MLIR] Softmax missing buffer → fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
                    struct SoftmaxParams {
                        uint32_t rows;
                        uint32_t cols;
                        uint32_t inner;
                    } params;
                    params.rows = static_cast<uint32_t>(op.rows);
                    params.cols = static_cast<uint32_t>(op.cols);
                    params.inner = static_cast<uint32_t>(op.inner);
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    [enc setBytes:&params length:sizeof(params) atIndex:2];
                    const NSUInteger rows = static_cast<NSUInteger>(op.rows);
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(rows, 1, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Conv3D: {
                    if (!src0 || !src1 || !dst) {
                        debug_log("[METAL MLIR] Conv3D missing buffer → fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
                    struct Conv3DParams {
                        uint32_t N, C_in, D, H, W;
                        uint32_t C_out;
                        uint32_t kD, kH, kW;
                        uint32_t strideD, strideH, strideW;
                        uint32_t dilationD, dilationH, dilationW;
                        uint32_t padFront, padTop, padLeft, padBack, padBottom, padRight;
                        uint32_t outD, outH, outW;
                    } params;
                    params.N = op.conv3d.N;
                    params.C_in = op.conv3d.C_in;
                    params.D = op.conv3d.D;
                    params.H = op.conv3d.H;
                    params.W = op.conv3d.W;
                    params.C_out = op.conv3d.C_out;
                    params.kD = op.conv3d.kernelD;
                    params.kH = op.conv3d.kernelH;
                    params.kW = op.conv3d.kernelW;
                    params.strideD = op.conv3d.strideD;
                    params.strideH = op.conv3d.strideH;
                    params.strideW = op.conv3d.strideW;
                    params.dilationD = op.conv3d.dilationD;
                    params.dilationH = op.conv3d.dilationH;
                    params.dilationW = op.conv3d.dilationW;
                    params.padFront = op.conv3d.padFront;
                    params.padTop = op.conv3d.padTop;
                    params.padLeft = op.conv3d.padLeft;
                    params.padBack = op.conv3d.padBack;
                    params.padBottom = op.conv3d.padBottom;
                    params.padRight = op.conv3d.padRight;
                    params.outD = op.conv3d.outD;
                    params.outH = op.conv3d.outH;
                    params.outW = op.conv3d.outW;
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:src1 offset:0 atIndex:1];
                    [enc setBuffer:dst offset:0 atIndex:2];
                    [enc setBytes:&params length:sizeof(params) atIndex:3];
                    const NSUInteger threads_per_tg = 1;
                    MTLSize grid = MTLSizeMake(params.C_out, params.N, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Slice: {
                    if (!src0 || !dst) {
                        debug_log("[METAL MLIR] Slice missing buffer → fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
                    [enc setBuffer:src0 offset:0 atIndex:0];
                    [enc setBuffer:dst offset:0 atIndex:1];
                    const NSUInteger elems = static_cast<NSUInteger>(tensor_num_elems(op.output));
                    const NSUInteger threads_per_tg = 64;
                    MTLSize grid = MTLSizeMake(elems, 1, 1);
                    MTLSize tg = MTLSizeMake(threads_per_tg, 1, 1);
                    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
                    break;
                }
                case KernelOpKind::Conv2D: {
                    if (!src0 || !src1 || !dst) {
                        debug_log("[METAL MLIR] Conv2D missing buffer → fallback");
                        [enc endEncoding];
                        m_force_fallback = true;
                        ensure_fallback(m_original_model);
                        return;
                    }
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
                case KernelOpKind::ElementwiseAdd:
                case KernelOpKind::ElementwiseSub: {
                    // Prefer per-op constant captured in flat IR (after broadcast expansion if any).
                    if (op.input1 && op.input1->from_constant && !op.input1->const_data.empty()) {
                        std::vector<float> broadcasted;
                        const std::vector<float>* src_vec = &op.input1->const_data;
                        if (op.is_broadcast && maybe_broadcast_const(op, op.input1->const_data, broadcasted)) {
                            src_vec = &broadcasted;
                        }
                        if (!src_vec->empty()) {
                            std::ostringstream l;
                            l << "[METAL MLIR] const add/sub first=" << (*src_vec)[0] << " size=" << src_vec->size();
                            debug_log(l.str());
                        }
                        id<MTLBuffer> buf = make_const_buffer_vec(*src_vec, op.dtype.storage);
                        if (buf) temp_const_buffers.push_back(buf);
                        return buf;
                    }
                    if (!m_has_const_b) return nil;
                    std::vector<float> broadcasted;
                    const std::vector<float>* src_vec = &m_const_b;
                    if (op.is_broadcast && maybe_broadcast_const(op, m_const_b, broadcasted)) {
                        src_vec = &broadcasted;
                    }
                    if (!src_vec->empty()) {
                        std::ostringstream l;
                        l << "[METAL MLIR] const add first=" << (*src_vec)[0] << " size=" << src_vec->size();
                        debug_log(l.str());
                    }
                    id<MTLBuffer> buf = make_const_buffer_vec(*src_vec, op.dtype.storage);
                    if (buf) temp_const_buffers.push_back(buf);
                    return buf;
                }
                case KernelOpKind::ElementwiseDiv:
                case KernelOpKind::ElementwiseMul: {
                    if (!op.input1 || !op.input1->from_constant)
                        return nil;
                    std::vector<float> broadcasted;
                    const std::vector<float>* src_vec = &op.input1->const_data;
                    if (op.is_broadcast && maybe_broadcast_const(op, op.input1->const_data, broadcasted)) {
                        src_vec = &broadcasted;
                    }
                    id<MTLBuffer> buf = make_const_buffer_vec(*src_vec, op.dtype.storage);
                    if (buf) temp_const_buffers.push_back(buf);
                    return buf;
                }
                case KernelOpKind::BatchNorm2D: return buf_const_bn;
                case KernelOpKind::Conv2D:
                case KernelOpKind::Conv3D:
                    return buf_const_w;
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

        // Map input tensors to flat IR tensors
        auto map_input = [&](size_t idx, id<MTLBuffer> buf) {
            if (!buf || idx >= inputs.size())
                return;
            const size_t elems_in = inputs[idx].get_size();
            const ov::element::Type et_in = inputs[idx].get_element_type();
            const auto& shape_in = inputs[idx].get_shape();
            const bool have_ports = idx < seg.input_ports.size();
            const ov::Node* param_node = have_ports ? seg.input_ports[idx].get_node() : nullptr;
            auto shapes_equal = [&](const KernelTensor& kt) {
                return kt.shape.size() == shape_in.size() &&
                       std::equal(kt.shape.begin(), kt.shape.end(), shape_in.begin());
            };
            // 1) Exact match: this Parameter node
            if (param_node) {
                for (auto& kt : m_flat_tensors) {
                    if (kt.from_parameter && kt.source_node == param_node &&
                        kt.dtype.ov_type == et_in &&
                        tensor_num_elems(&kt) == elems_in &&
                        shapes_equal(kt) &&
                        buf_map.find(&kt) == buf_map.end()) {
                        buf_map[&kt] = buf;
                        if (metal_debug_enabled()) {
                            std::cerr << "[METAL MLIR] map_input idx=" << idx << " matched Parameter node "
                                      << kt.name << " elems=" << elems_in << " (exact)\n";
                        }
                        return;
                    }
                }
            }
            // 2) Any Parameter with the same shape (helps if graph was rewritten)
            for (auto& kt : m_flat_tensors) {
                if (kt.from_parameter &&
                    kt.dtype.ov_type == et_in &&
                    tensor_num_elems(&kt) == elems_in &&
                    shapes_equal(kt) &&
                    buf_map.find(&kt) == buf_map.end()) {
                    buf_map[&kt] = buf;
                    if (metal_debug_enabled()) {
                        std::cerr << "[METAL MLIR] map_input idx=" << idx << " matched Parameter any "
                                  << kt.name << " elems=" << elems_in << "\n";
                    }
                    return;
                }
            }
            // 3) Last resort: match by element count and dtype
            for (auto& kt : m_flat_tensors) {
                if (kt.dtype.ov_type == et_in &&
                    tensor_num_elems(&kt) == elems_in &&
                    buf_map.find(&kt) == buf_map.end()) {
                    buf_map[&kt] = buf;
                    if (metal_debug_enabled()) {
                        std::cerr << "[METAL MLIR] map_input idx=" << idx << " matched by count "
                                  << kt.name << " elems=" << elems_in << "\n";
                    }
                    break;
                }
            }
        };
        ov::Tensor input1_f32;
        if (!inputs.empty()) {
            map_input(0, buf_in);
            if (inputs.size() > 1) {
                if (inputs[1].get_element_type().is_real()) {
                    input1_f32 = to_float32_tensor(inputs[1]);
                } else {
                    input1_f32 = ov::Tensor{ov::element::f32, inputs[1].get_shape()};
                    if (inputs[1].get_element_type() == ov::element::i32) {
                        const int32_t* src = inputs[1].data<const int32_t>();
                        float* dst = input1_f32.data<float>();
                        for (size_t i = 0; i < inputs[1].get_size(); ++i) dst[i] = static_cast<float>(src[i]);
                    } else if (inputs[1].get_element_type() == ov::element::i64) {
                        const int64_t* src = inputs[1].data<const int64_t>();
                        float* dst = input1_f32.data<float>();
                        for (size_t i = 0; i < inputs[1].get_size(); ++i) dst[i] = static_cast<float>(src[i]);
                    }
                }
                map_input(1, buf_in1);
            }
        }
        auto get_buffer = [&](const KernelTensor* kt) -> id<MTLBuffer> {
            auto it = buf_map.find(kt);
            return it == buf_map.end() ? nil : it->second;
        };

            for (size_t idx = 0; idx < seg.op_count; ++idx) {
                const auto& op = seg_op(idx);
                id<MTLBuffer> dst = dst_for_step(idx);
                id<MTLBuffer> src1 = select_src1(op);
                if (!src1 && op.input1) src1 = get_buffer(op.input1);
                if (!src1) src1 = load_const_input(op);
                if (!src1 && idx > 0) {
                    const auto& prev = seg_op(idx - 1);
                    if (prev.output && prev.output == op.input1) {
                        src1 = current;  // reuse previous op output buffer
                    }
                }
                id<MTLBuffer> srcA = current;
            if (op.input0) {
                auto buf0 = get_buffer(op.input0);
                if (buf0) srcA = buf0;
            }
            if (op.kind == KernelOpKind::ElementwiseAdd || op.kind == KernelOpKind::ElementwiseSub ||
                op.kind == KernelOpKind::ElementwiseMul || op.kind == KernelOpKind::ElementwisePow ||
                op.kind == KernelOpKind::ElementwiseMod || op.kind == KernelOpKind::ElementwiseFloorMod ||
                op.kind == KernelOpKind::ElementwiseDiv) {
                if (metal_debug_enabled()) {
                    std::ostringstream oss;
                    oss << "[METAL MLIR] elementwise dispatch: in0=" << describe_shape(op.input0)
                        << " in1=" << describe_shape(op.input1)
                        << " out=" << describe_shape(op.output)
                        << " srcA=" << (srcA ? "ok" : "nil")
                        << " src1=" << (src1 ? "ok" : "nil")
                        << " same_ptr=" << ((srcA == src1) ? "yes" : "no")
                        << " dst=" << (dst ? "ok" : "nil");
                    std::cerr << oss.str() << "\n";
                    auto dump_vec = [](const std::vector<int64_t>& v, const char* tag) {
                        std::cerr << "  " << tag << "=[";
                        for (size_t i = 0; i < v.size(); ++i) {
                            if (i) std::cerr << ",";
                            std::cerr << v[i];
                        }
                        std::cerr << "]\n";
                    };
                    if (!op.stride0.empty() || !op.stride1.empty()) {
                        dump_vec(op.stride0, "stride0");
                        dump_vec(op.stride1, "stride1");
                        dump_vec(op.out_shape, "out_shape");
                    }
                    if (srcA) {
                        const float* p0 = static_cast<const float*>([srcA contents]);
                        size_t n0 = std::min<size_t>(8, op.input0 ? tensor_num_elems(op.input0) : 0);
                        std::cerr << "[METAL MLIR] elementwise src0 first:";
                        for (size_t i = 0; i < n0; ++i) std::cerr << " " << p0[i];
                        std::cerr << "\n";
                    }
                    if (src1) {
                        const float* p = static_cast<const float*>([src1 contents]);
                        size_t n = std::min<size_t>(8, op.is_broadcast && op.output ? tensor_num_elems(op.output)
                                                                                    : (op.input1 ? tensor_num_elems(op.input1) : 0));
                        std::cerr << "[METAL MLIR] elementwise src1 first:";
                        for (size_t i = 0; i < n; ++i) std::cerr << " " << p[i];
                        std::cerr << "\n";
                        if ((op.input1 ? tensor_num_elems(op.input1) : 0) > 100) {
                            std::cerr << "[METAL MLIR] elementwise src1[90]=" << p[90] << " src1[123]=" << p[123] << "\n";
                        }
                        std::cerr << "[METAL MLIR] elementwise const size="
                                  << (op.is_broadcast ? tensor_num_elems(op.output) : (op.input1 ? tensor_num_elems(op.input1) : 0))
                                  << " m_const_b_size=" << m_const_b.size() << "\n";
                    }
                }
            }
            if (metal_debug_enabled()) {
                std::cerr << "[METAL MLIR] dispatch kind=" << static_cast<int>(op.kind)
                          << " srcA=" << (srcA ? "ok" : "nil")
                          << " src1=" << (src1 ? "ok" : "nil")
                          << " dst=" << (dst ? "ok" : "nil") << "\n";
                if (!src1 && op.input1 && op.input1->from_constant) {
                    std::cerr << "[METAL MLIR] const input size=" << op.input1->const_data.size() << "\n";
                }
                std::cerr.flush();
            }
            // If elementwise inputs missing, drop to fallback
            if ((op.kind == KernelOpKind::ElementwiseAdd || op.kind == KernelOpKind::ElementwiseSub ||
                 op.kind == KernelOpKind::ElementwiseMul || op.kind == KernelOpKind::ElementwiseDiv ||
                 op.kind == KernelOpKind::ElementwisePow || op.kind == KernelOpKind::ElementwiseMod ||
                 op.kind == KernelOpKind::ElementwiseFloorMod) &&
                (!srcA || !src1 || !dst)) {
                debug_log("[METAL MLIR] run_segment: elementwise missing buffer -> fallback");
                [cmd release];
                for (auto* b : temp_const_buffers) if (b) [b release];
                return run_fallback();
            }
            dispatch_op(op, seg_pipe(idx), srcA, src1, dst, cmd);
            current = dst;
            if (op.output && dst) {
                buf_map[op.output] = dst;
            }
            if (metal_debug_enabled() && op.kind == KernelOpKind::ElementwisePow && dst) {
                const float* p = static_cast<const float*>([dst contents]);
                size_t n = std::min<size_t>(8, tensor_num_elems(op.output));
                std::cerr << "[METAL MLIR] pow output first:";
                for (size_t i = 0; i < n; ++i) std::cerr << " " << p[i];
                std::cerr << "\n";
            }
            std::cerr << "[METAL MLIR] dispatch complete kind=" << static_cast<int>(op.kind) << "\n";
        }

        final_buf = current;

        [cmd commit];
        [cmd waitUntilCompleted];

        if (out_dtype.storage == MetalDType::StorageType::I32 ||
            out_dtype.storage == MetalDType::StorageType::I64) {
            ov::Tensor seg_output{out_elem_type, out_shape};
            std::memcpy(seg_output.data(), [final_buf contents], out_bytes);

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

        ov::Tensor seg_output_f32{ov::element::f32, out_shape};

        std::memcpy(seg_output_f32.data(), [final_buf contents], out_bytes);

#if METAL_MLIR_DEBUG
        if (seg.op_count == 1) {
            const auto& op = seg_op(0);
            // Spot-check div mismatches to hunt f16 constant issues
            if (op.kind == KernelOpKind::ElementwiseDiv) {
                const float* a_ptr = input_f32.data<const float>();
                const float* b_ptr = nullptr;
                if (op.input1 && op.input1->from_constant && !op.input1->const_data.empty()) {
                    b_ptr = op.input1->const_data.data();
                } else if (inputs.size() > 1) {
                    b_ptr = input1_f32.data<const float>();
                }
                const float* out_ptr = seg_output_f32.data<const float>();
                if (b_ptr) {
                    const size_t rank = op.out_shape.size();
                    std::vector<size_t> out_stride(rank, 1);
                    for (int i = static_cast<int>(rank) - 2; i >= 0; --i)
                        out_stride[i] = out_stride[i + 1] * static_cast<size_t>(op.out_shape[i + 1]);
                    auto idx_to_offset = [&](size_t idx, const std::vector<int64_t>& stride) {
                        if (stride.empty()) return idx;
                        size_t tmp = idx;
                        size_t off = 0;
                        for (size_t d = 0; d < rank; ++d) {
                            size_t coord = tmp / out_stride[d];
                            tmp -= coord * out_stride[d];
                            if (d < stride.size() && stride[d] != 0)
                                off += coord * static_cast<size_t>(stride[d]);
                        }
                        return off;
                    };
                    size_t total = seg_output_f32.get_size();
                    for (size_t i = 0; i < std::min<size_t>(total, 256); ++i) {
                        size_t off0 = idx_to_offset(i, op.stride0);
                        size_t off1 = idx_to_offset(i, op.stride1);
                        float ref = b_ptr[off1] != 0.0f ? a_ptr[off0] / b_ptr[off1] : std::numeric_limits<float>::infinity();
                        if (std::fabs(ref - out_ptr[i]) > 1e-3f) {
                            std::cerr << "[METAL MLIR] div mismatch idx=" << i
                                      << " a=" << a_ptr[off0] << " b=" << b_ptr[off1]
                                      << " ref=" << ref << " out=" << out_ptr[i] << "\n";
                            break;
                        }
                    }
                }
            }
        }
        if (seg.op_count == 1 && metal_debug_enabled()) {
            const auto& op = seg_op(0);
            if ((op.kind == KernelOpKind::ElementwiseFloorMod || op.kind == KernelOpKind::ElementwiseMod) &&
                inputs.size() > 1) {
                auto a_f32 = to_float32_tensor(inputs[0]);
                auto b_f32 = to_float32_tensor(inputs[1]);
                const float* a_ptr = a_f32.data<const float>();
                const float* b_ptr = b_f32.data<const float>();
                const float* out_ptr = seg_output_f32.data<const float>();
                size_t total = seg_output_f32.get_size();
                for (size_t i = 0; i < total; ++i) {
                    float ref = b_ptr[i] != 0.0f ? a_ptr[i] - std::floor(a_ptr[i] / b_ptr[i]) * b_ptr[i] : a_ptr[i];
                    if (std::fabs(ref - out_ptr[i]) > 1e-3f) {
                        std::cerr << "[METAL MLIR] eltwise mismatch i=" << i
                                  << " a=" << a_ptr[i] << " b=" << b_ptr[i]
                                  << " out=" << out_ptr[i] << " ref=" << ref << "\n";
                        size_t lo = (i > 4) ? i - 4 : 0;
                        size_t hi = std::min(total, i + 5);
                        for (size_t j = lo; j < hi; ++j) {
                            float rj = b_ptr[j] != 0.0f ? a_ptr[j] - std::floor(a_ptr[j] / b_ptr[j]) * b_ptr[j] : a_ptr[j];
                            std::cerr << "  idx=" << j << " a=" << a_ptr[j] << " b=" << b_ptr[j]
                                      << " ref=" << rj << " out=" << out_ptr[j] << "\n";
                        }
                        break;
                    }
                }
            }
        }
        if (!seg_output_f32.get_shape().empty()) {
            const float* p = seg_output_f32.data<const float>();
            size_t n = std::min<size_t>(seg_output_f32.get_size(), 8);
            std::ostringstream oss;
            oss << "[METAL MLIR] seg out first:";
            for (size_t i = 0; i < n; ++i) oss << " " << p[i];
            debug_log(oss.str());
            if (seg.op_count == 1) {
                const auto& op = seg_op(0);
                if ((op.kind == KernelOpKind::ElementwiseMod || op.kind == KernelOpKind::ElementwiseFloorMod) &&
                    seg_output_f32.get_size() > 90 &&
                    inputs.size() > 1 && inputs[0].get_size() > 90 && inputs[1].get_size() > 90) {
                    auto a_dbg = to_float32_tensor(inputs[0]);
                    auto b_dbg = to_float32_tensor(inputs[1]);
                    const float* a_ptr = a_dbg.data<const float>();
                    const float* b_ptr = b_dbg.data<const float>();
                    const float out_v = seg_output_f32.data<const float>()[90];
                    float ref_v = 0.f;
                    if (b_ptr[90] != 0.0f) {
                        ref_v = a_ptr[90] - std::floor(a_ptr[90] / b_ptr[90]) * b_ptr[90];
                    }
                    std::ostringstream os2;
                    os2 << "[METAL MLIR] eltwise idx=90 A=" << a_ptr[90] << " B=" << b_ptr[90]
                        << " out=" << out_v << " ref=" << ref_v;
                    debug_log(os2.str());
                }
            }
        }
#endif
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
        }  // @autoreleasepool
    }

private:
    void build_flat_ir(const std::shared_ptr<const ov::Model>& model) {
        m_flat_ops.clear();
        m_flat_tensors.clear();
        m_flat_segments.clear();
        // Reserve to keep KernelTensor pointers stable while push_back is used.
        m_flat_tensors.reserve(model->get_ordered_ops().size() * 2);

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

        auto make_tensor = [&](const ov::Output<const ov::Node>& out) -> size_t {
            OutputKey key{out.get_node(), out.get_index()};
            auto it = tensor_index.find(key);
            if (it != tensor_index.end()) return it->second;
            const auto pshape = out.get_partial_shape();
            KernelTensor t;
            t.name = out.get_node()->get_friendly_name();
            t.from_parameter = ov::is_type<ov::op::v0::Parameter>(out.get_node());
            t.from_constant = ov::is_type<ov::op::v0::Constant>(out.get_node());
            if (t.from_constant) {
                if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(out.get_node_shared_ptr())) {
                    // Store as float for simplicity; integers are converted to float.
                    t.const_data = c->cast_vector<float>();
                }
            }
            const auto ov_et = out.get_element_type();
            if (ov_et != ov::element::dynamic) {
                try {
                    t.dtype = resolve_metal_dtype(ov_et);
                } catch (const ov::Exception&) {
                    // Keep default dtype for unsupported types; fallback will handle later.
                }
            }
            t.source_node = out.get_node();
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
            if (metal_debug_enabled()) {
                std::ostringstream oss;
                oss << "[METAL MLIR] visiting node " << node->get_friendly_name()
                    << " type=" << node->get_type_info().name;
                debug_log(oss.str());
            }
            // Ensure Parameters have corresponding tensors for input mapping
            if (ov::is_type<ov::op::v0::Parameter>(node.get())) {
                if (node->get_output_size() > 0)
                    make_tensor(node->output(0));
                continue;
            }
            // Skip Result/Constant early
            if (ov::is_type<ov::op::v0::Result>(node.get()) ||
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

            auto propagate_dtype = [&](KernelOp& op) {
                if (op.dtype.ov_type == ov::element::dynamic && node->get_output_size() > 0) {
                    auto et = node->get_output_element_type(0);
                    if (et != ov::element::dynamic) {
                        try {
                            op.dtype = resolve_metal_dtype(et);
                        } catch (const ov::Exception&) {
                            // unsupported dtype will be handled by fallback paths
                        }
                    }
                }
                switch (op.kind) {
                    case KernelOpKind::Split:
                        if (op.split.dtype.ov_type == ov::element::dynamic) {
                            op.split.dtype = op.dtype;
                        }
                        break;
                    case KernelOpKind::Conv2D:
                        if (op.conv2d.dtype.ov_type == ov::element::dynamic) {
                            op.conv2d.dtype = op.dtype;
                        }
                        break;
                    case KernelOpKind::Conv3D:
                        if (op.conv3d.dtype.ov_type == ov::element::dynamic) {
                            op.conv3d.dtype = op.dtype;
                        }
                        break;
                    case KernelOpKind::BatchNorm2D:
                        if (op.batchnorm.dtype.ov_type == ov::element::dynamic) {
                            op.batchnorm.dtype = op.dtype;
                        }
                        break;
                    default:
                        break;
                }
            };

            KernelOp op{};
            bool supported = false;

            // Collapse OpenVINO Mod/FloorMod decomposition (Sign/Abs/Divide/Sub/Mul) into a single eltwise op.
            if (auto mod_pat = match_mod_decomposition(node)) {
                auto [a_out, b_out, is_floor] = *mod_pat;
                if (a_out.get_partial_shape().is_static() && b_out.get_partial_shape().is_static()) {
                    auto a_shape_vec = a_out.get_shape();
                    auto b_shape_vec = b_out.get_shape();
                    auto br = compute_broadcast(a_shape_vec, b_shape_vec);
                    if (!br.success) {
                        debug_log("[METAL MLIR] Mod pattern broadcast mismatch → skipping");
                    } else {
                        op.kind = is_floor ? KernelOpKind::ElementwiseFloorMod : KernelOpKind::ElementwiseMod;
                        op.is_broadcast = (a_shape_vec != b_shape_vec);
                        op.out_shape.assign(br.out_shape.begin(), br.out_shape.end());
                        op.stride0 = br.stride0;
                        op.stride1 = br.stride1;
                        auto normalize_stride = [&](std::vector<int64_t>& st) {
                            if (st.size() < op.out_shape.size())
                                st.insert(st.begin(), op.out_shape.size() - st.size(), 0);
                        };
                        normalize_stride(op.stride0);
                        normalize_stride(op.stride1);
                        size_t in0_idx = make_tensor(a_out);
                        size_t in1_idx = make_tensor(b_out);
                        op.input0 = &m_flat_tensors[in0_idx];
                        op.input1 = &m_flat_tensors[in1_idx];
                        set_output(op);
                        if (op.output) op.output->shape = op.out_shape;
                        // If RHS is Constant and requires broadcast, pre-expand to out_shape
                        if (op.input1 && op.input1->from_constant && op.is_broadcast && !op.out_shape.empty()) {
                            const size_t out_elems = ov::shape_size(op.out_shape);
                            if (out_elems > 0 && !op.input1->const_data.empty()) {
                                const size_t rank = op.out_shape.size();
                                // normalize RHS shape to rank
                                std::vector<size_t> rhs_shape_norm(rank, 1);
                                {
                                    auto rhs_shape = op.input1->shape;
                                    size_t off = rank - rhs_shape.size();
                                    for (size_t i = 0; i < rhs_shape.size(); ++i)
                                        rhs_shape_norm[off + i] = static_cast<size_t>(rhs_shape[i]);
                                }
                                std::vector<size_t> rhs_stride(rank, 1);
                                for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                                    rhs_stride[i] = rhs_stride[i + 1] * rhs_shape_norm[i + 1];
                                }
                                std::vector<int64_t> out_stride(rank, 1);
                                for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                                    out_stride[i] = out_stride[i + 1] * static_cast<int64_t>(op.out_shape[i + 1]);
                                }
                                std::vector<float> broadcasted(out_elems, 0.f);
                                for (size_t idx = 0; idx < out_elems; ++idx) {
                                    size_t tmp = idx;
                                    size_t src_idx = 0;
                                    for (size_t d = 0; d < rank; ++d) {
                                        size_t coord = tmp / static_cast<size_t>(out_stride[d]);
                                        tmp -= coord * static_cast<size_t>(out_stride[d]);
                                        if (rhs_shape_norm[d] == 1) continue;
                                        src_idx += coord * rhs_stride[d];
                                    }
                                    broadcasted[idx] = op.input1->const_data[std::min(src_idx, op.input1->const_data.size() - 1)];
                                }
                                op.input1->const_data.swap(broadcasted);
                                op.input1->shape = op.out_shape;
                                const size_t rank2 = op.out_shape.size();
                                // LHS normalized shape/strides
                                std::vector<size_t> lhs_shape_norm(rank2, 1);
                                if (op.input0) {
                                    auto lhs_shape = op.input0->shape;
                                    size_t off_l = rank2 - lhs_shape.size();
                                    for (size_t i = 0; i < lhs_shape.size(); ++i)
                                        lhs_shape_norm[off_l + i] = static_cast<size_t>(lhs_shape[i]);
                                }
                                std::vector<size_t> lhs_stride(rank2, 1);
                                for (int i = static_cast<int>(rank2) - 2; i >= 0; --i) {
                                    lhs_stride[i] = lhs_stride[i + 1] * lhs_shape_norm[i + 1];
                                }
                                op.stride0.resize(rank2);
                                op.stride1.assign(rank2, 1);
                                for (int i = static_cast<int>(rank2) - 2; i >= 0; --i) {
                                    op.stride1[i] = op.stride1[i + 1] * op.out_shape[i + 1];
                                }
                                for (size_t k = 0; k < rank2; ++k) {
                                    op.stride0[k] = (lhs_shape_norm[k] == 1) ? 0
                                                                              : static_cast<int64_t>(lhs_stride[k]);
                                    if (rhs_shape_norm[k] == 1)
                                        op.stride1[k] = 0;
                                }
                                // Broadcast is still needed if lhs differs from out_shape
                                bool lhs_match = true;
                                for (size_t k = 0; k < rank2; ++k) {
                                    if (lhs_shape_norm[k] != static_cast<size_t>(op.out_shape[k])) {
                                        lhs_match = false;
                                        break;
                                    }
                                }
                                op.is_broadcast = !lhs_match;
                            }
                        }
                        supported = true;
                        if (metal_debug_enabled()) {
                            std::ostringstream oss;
                            oss << "[METAL MLIR] Collapsed Mod pattern to single op. kind="
                                << (is_floor ? "FloorMod" : "Mod")
                                << " A=" << describe_shape(op.input0)
                                << " B=" << describe_shape(op.input1)
                                << " out=" << describe_shape(op.output);
                            debug_log(oss.str());
                        }
                    }
                }
                if (supported) {
                    propagate_dtype(op);
                    m_flat_ops.push_back(op);
                    continue;
                }
            }

            if (auto split = ov::as_type_ptr<const ov::op::v1::Split>(node)) {
                const auto in_shape = split->get_input_shape(0);
                if (in_shape.empty()) continue;
                auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(split->input_value(1).get_node_shared_ptr());
                if (!axis_const) {
                    debug_log("[METAL MLIR] Split axis not constant; skipping");
                    continue;
                }
                auto axis_vec = axis_const->cast_vector<int64_t>();
                if (axis_vec.empty()) continue;
                int64_t axis = axis_vec[0];
                int64_t rank = static_cast<int64_t>(in_shape.size());
                int64_t axis_norm = axis >= 0 ? axis : axis + rank;
                if (axis_norm < 0 || axis_norm >= rank) {
                    debug_log("[METAL MLIR] Split axis out of range; skipping");
                    continue;
                }
                size_t parts = split->get_num_splits();
                auto dim = in_shape[static_cast<size_t>(axis_norm)];
                if (dim % parts != 0) {
                    debug_log("[METAL MLIR] Split dim not divisible by parts; skipping");
                    continue;
                }
                op.kind = KernelOpKind::Split;
                op.split.axis = axis_norm;
                op.split.input_shape.assign(in_shape.begin(), in_shape.end());
                op.split.split_sizes.assign(parts, dim / parts);
                op.split.dtype = resolve_metal_dtype(node->get_output_element_type(0));
                op.split.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(node->get_output_element_type(0)));
                size_t in_idx = make_tensor(node->input_value(0));
                op.input0 = &m_flat_tensors[in_idx];
                for (size_t i = 0; i < parts; ++i) {
                    size_t out_idx = make_tensor(node->output(i));
                    op.outputs.push_back(&m_flat_tensors[out_idx]);
                }
                supported = true;
            } else if (auto vs = ov::as_type_ptr<const ov::op::v1::VariadicSplit>(node)) {
                const auto in_shape = vs->get_input_shape(0);
                if (in_shape.empty()) continue;
                auto axis_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(1).get_node_shared_ptr());
                auto len_const = ov::as_type_ptr<const ov::op::v0::Constant>(vs->input_value(2).get_node_shared_ptr());
                if (!axis_const || !len_const) {
                    debug_log("[METAL MLIR] VariadicSplit inputs not constant; skipping");
                    continue;
                }
                auto axis_vec = axis_const->cast_vector<int64_t>();
                if (axis_vec.empty()) continue;
                int64_t axis = axis_vec[0];
                int64_t rank = static_cast<int64_t>(in_shape.size());
                int64_t axis_norm = axis >= 0 ? axis : axis + rank;
                if (axis_norm < 0 || axis_norm >= rank) {
                    debug_log("[METAL MLIR] VariadicSplit axis out of range; skipping");
                    continue;
                }
                auto lengths = len_const->cast_vector<int64_t>();
                op.kind = KernelOpKind::Split;
                op.split.axis = axis_norm;
                op.split.input_shape.assign(in_shape.begin(), in_shape.end());
                op.split.split_sizes.clear();
                for (auto v : lengths) op.split.split_sizes.push_back(static_cast<size_t>(v));
                op.split.dtype = resolve_metal_dtype(node->get_output_element_type(0));
                op.split.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(node->get_output_element_type(0)));
                size_t in_idx = make_tensor(node->input_value(0));
                op.input0 = &m_flat_tensors[in_idx];
                for (size_t i = 0; i < lengths.size(); ++i) {
                    size_t out_idx = make_tensor(node->output(i));
                    op.outputs.push_back(&m_flat_tensors[out_idx]);
                }
                supported = true;
            } else if (ov::as_type_ptr<const ov::op::v0::Convert>(node)) {
                KernelOp unary{};
                unary.kind = KernelOpKind::Unary;
                unary.activation = ActivationKind::Identity;
                add_input_indices(unary);
                set_output(unary);
                if (unary.output && unary.input0) {
                    unary.output->shape = unary.input0->shape;
                }
                m_flat_ops.push_back(unary);
                supported = true;
                continue;
            } else if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
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
            } else if (auto sq = ov::as_type_ptr<const ov::op::v0::SquaredDifference>(node)) {
                // Decompose SqDiff into Sub then Mul(sub, sub)
                auto a_shape_vec = node->get_input_shape(0);
                auto b_shape_vec = node->get_input_shape(1);
                auto br = compute_broadcast(a_shape_vec, b_shape_vec);
                if (!br.success) {
                    debug_log("[METAL MLIR] SqDiff broadcast mismatch → unsupported");
                    continue;
                }
                // Create temp tensor for sub output
                KernelTensor tmp;
                tmp.name = node->get_friendly_name() + "_sqdiff_tmp";
                tmp.shape.assign(br.out_shape.begin(), br.out_shape.end());
                try {
                    tmp.dtype = resolve_metal_dtype(node->get_output_element_type(0));
                } catch (const ov::Exception&) {
                    // leave dynamic; fallback will handle if unsupported
                }
                m_flat_tensors.push_back(tmp);
                KernelTensor* tmp_ptr = &m_flat_tensors.back();

                // Sub op
                KernelOp sub_op{};
                sub_op.kind = KernelOpKind::ElementwiseSub;
                sub_op.is_broadcast = (a_shape_vec != b_shape_vec);
                sub_op.out_shape.assign(br.out_shape.begin(), br.out_shape.end());
                sub_op.stride0 = br.stride0;
                sub_op.stride1 = br.stride1;
                auto norm = [&](std::vector<int64_t>& st) {
                    if (st.size() < sub_op.out_shape.size())
                        st.insert(st.begin(), sub_op.out_shape.size() - st.size(), 0);
                };
                norm(sub_op.stride0);
                norm(sub_op.stride1);
                size_t in0_idx = make_tensor(node->input_value(0));
                size_t in1_idx = make_tensor(node->input_value(1));
                sub_op.input0 = &m_flat_tensors[in0_idx];
                sub_op.input1 = &m_flat_tensors[in1_idx];
                sub_op.output = tmp_ptr;
                sub_op.output->shape = sub_op.out_shape;
                // If RHS is a Constant and broadcast is needed, pre-expand to full out_shape
                if (sub_op.input1 && sub_op.input1->from_constant && sub_op.is_broadcast && !sub_op.out_shape.empty()) {
                    std::vector<float> broadcasted;
                    const size_t out_elems = ov::shape_size(sub_op.out_shape);
                    if (out_elems > 0 && !sub_op.input1->const_data.empty()) {
                        const size_t rank = sub_op.out_shape.size();
                        // row-major strides for output
                        std::vector<size_t> out_stride(rank, 1);
                        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                            out_stride[i] = out_stride[i + 1] * static_cast<size_t>(sub_op.out_shape[i + 1]);
                        }
                        // normalize RHS shape to out rank
                        std::vector<size_t> rhs_shape(rank, 1);
                        size_t off = rank - sub_op.input1->shape.size();
                        for (size_t i = 0; i < sub_op.input1->shape.size(); ++i)
                            rhs_shape[off + i] = static_cast<size_t>(sub_op.input1->shape[i]);
                        std::vector<size_t> rhs_stride(rank, 1);
                        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                            rhs_stride[i] = rhs_stride[i + 1] * rhs_shape[i + 1];
                        }
                        broadcasted.assign(out_elems, 0.f);
                        for (size_t idx = 0; idx < out_elems; ++idx) {
                            size_t tmp = idx;
                            size_t src_idx = 0;
                            for (size_t d = 0; d < rank; ++d) {
                                size_t coord = tmp / out_stride[d];
                                tmp -= coord * out_stride[d];
                                if (rhs_shape[d] == 1) continue;
                                src_idx += coord * rhs_stride[d];
                            }
                            broadcasted[idx] = sub_op.input1->const_data[std::min(src_idx, sub_op.input1->const_data.size() - 1)];
                        }
                        sub_op.input1->const_data.swap(broadcasted);
                        sub_op.input1->shape = sub_op.out_shape;
                        sub_op.stride1.assign(rank, 1);
                        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                            sub_op.stride1[i] = sub_op.stride1[i + 1] * sub_op.out_shape[i + 1];
                        }
                    }
                } else if (sub_op.is_broadcast && sub_op.input1 && sub_op.input1->shape.size() == 1 && sub_op.out_shape.size() > 1) {
                    // Scalar RHS that wasn’t constant-expanded: make stride1 all zeros except last
                    sub_op.input1->shape.assign(sub_op.out_shape.size(), 1);
                    sub_op.stride1.assign(sub_op.out_shape.size(), 0);
                }

                // Mul op: tmp * tmp -> original output
                KernelOp mul_op{};
                mul_op.kind = KernelOpKind::ElementwiseMul;
                mul_op.is_broadcast = false;
                mul_op.out_shape = sub_op.out_shape;
                mul_op.stride0.assign(sub_op.out_shape.size(), 1);
                mul_op.stride1.assign(sub_op.out_shape.size(), 1);
                for (int i = static_cast<int>(mul_op.out_shape.size()) - 2; i >= 0; --i) {
                    mul_op.stride0[i] = mul_op.stride0[i + 1] * mul_op.out_shape[i + 1];
                    mul_op.stride1[i] = mul_op.stride0[i];
                }
                mul_op.input0 = tmp_ptr;
                mul_op.input1 = tmp_ptr;
                size_t out_idx = make_tensor(node->output(0));
                mul_op.output = &m_flat_tensors[out_idx];
                mul_op.output->shape = mul_op.out_shape;
                if (mul_op.output->shape.empty()) {
                    mul_op.output->shape.assign(mul_op.out_shape.begin(), mul_op.out_shape.end());
                }
                // Propagate dtype to temporary/output tensors explicitly
                try {
                    auto out_dtype = resolve_metal_dtype(node->get_output_element_type(0));
                    tmp_ptr->dtype = out_dtype;
                    mul_op.output->dtype = out_dtype;
                    sub_op.dtype = out_dtype;
                    mul_op.dtype = out_dtype;
                } catch (const ov::Exception&) {
                }

                propagate_dtype(sub_op);
                propagate_dtype(mul_op);
                m_flat_ops.push_back(sub_op);
                m_flat_ops.push_back(mul_op);
                supported = true;
                continue;  // already emitted decomposed ops

            } else if (ov::as_type_ptr<const ov::op::v1::Add>(node) ||
                       ov::as_type_ptr<const ov::op::v1::Subtract>(node) ||
                       ov::as_type_ptr<const ov::op::v1::Multiply>(node) ||
                       ov::as_type_ptr<const ov::op::v1::Power>(node) ||
                       ov::is_type<ov::op::v1::Divide>(node.get()) ||
                       ov::as_type_ptr<const ov::op::v1::Mod>(node) ||
                       ov::as_type_ptr<const ov::op::v1::FloorMod>(node)) {
                auto et0 = node->get_input_element_type(0);
                auto et1 = node->get_input_element_type(1);
                auto supported_num = [](const ov::element::Type& t) {
                    return t.is_real() || t == ov::element::i32 || t == ov::element::i64;
                };
                if (!supported_num(et0) || !supported_num(et1)) {
                    OPENVINO_THROW("METAL eltwise supports only f16/f32/i32/i64 inputs");
                }
                // Unified binary eltwise handling (Add/Sub/Mul/Div/Pow/Mod/FloorMod) with proper NumPy broadcast
                const bool is_mul_node = static_cast<bool>(ov::as_type_ptr<const ov::op::v1::Multiply>(node));
                const bool name_says_div = node->get_friendly_name().find("Divide") != std::string::npos;

                // Detect Multiply + Power(x, -1) pattern that represents true division by parameter.
                auto is_neg_one_const = [](const std::shared_ptr<const ov::Node>& n) {
                    auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(n);
                    if (!c) return false;
                    auto vals = c->cast_vector<float>();
                    if (vals.empty()) return false;
                    for (float v : vals) {
                        if (std::fabs(v + 1.0f) > 1e-6f) return false;
                    }
                    return true;
                };

                ov::Output<const ov::Node> lhs_value = node->input_value(0);
                ov::Output<const ov::Node> rhs_value = node->input_value(1);
                auto a_shape_vec = node->get_input_shape(0);
                auto b_shape_vec = node->get_input_shape(1);

                bool div_pow_rhs = false;
                bool div_pow_lhs = false;

                if (is_mul_node) {
                    if (auto pow_rhs = ov::as_type_ptr<const ov::op::v1::Power>(rhs_value.get_node_shared_ptr())) {
                        if (is_neg_one_const(pow_rhs->get_input_node_shared_ptr(1))) {
                            div_pow_rhs = true;
                            rhs_value = pow_rhs->input_value(0);  // true divisor
                            b_shape_vec = pow_rhs->get_input_shape(0);
                        }
                    }
                    if (!div_pow_rhs) {
                        if (auto pow_lhs = ov::as_type_ptr<const ov::op::v1::Power>(lhs_value.get_node_shared_ptr())) {
                            if (is_neg_one_const(pow_lhs->get_input_node_shared_ptr(1))) {
                                div_pow_lhs = true;
                                lhs_value = rhs_value;  // numerator is the other input
                                a_shape_vec = node->get_input_shape(1);
                                rhs_value = pow_lhs->input_value(0);
                                b_shape_vec = pow_lhs->get_input_shape(0);
                            }
                        }
                    }
                }

                if (ov::as_type_ptr<const ov::op::v1::Add>(node)) {
                    op.kind = KernelOpKind::ElementwiseAdd;
                } else if (ov::as_type_ptr<const ov::op::v1::Subtract>(node)) {
                    op.kind = KernelOpKind::ElementwiseSub;
                } else if (ov::is_type<ov::op::v1::Divide>(node.get()) || (is_mul_node && (name_says_div || div_pow_rhs || div_pow_lhs))) {
                    op.kind = KernelOpKind::ElementwiseDiv;
                } else if (is_mul_node) {
                    op.kind = KernelOpKind::ElementwiseMul;
                } else if (ov::as_type_ptr<const ov::op::v1::Mod>(node)) {
                    op.kind = KernelOpKind::ElementwiseMod;
                } else if (ov::as_type_ptr<const ov::op::v1::FloorMod>(node)) {
                    op.kind = KernelOpKind::ElementwiseFloorMod;
                } else {
                    op.kind = KernelOpKind::ElementwisePow;
                }

                auto br = compute_broadcast(a_shape_vec, b_shape_vec);
                if (!br.success) {
                    debug_log("[METAL MLIR] Eltwise broadcast mismatch → mark unsupported");
                    continue;
                }
                op.is_broadcast = (a_shape_vec != b_shape_vec);
                op.out_shape.assign(br.out_shape.begin(), br.out_shape.end());
                op.stride0 = br.stride0;
                op.stride1 = br.stride1;
                // Normalize stride arrays to same rank length as out_shape
                auto normalize_stride = [&](std::vector<int64_t>& st) {
                    if (st.size() < op.out_shape.size()) {
                        st.insert(st.begin(), op.out_shape.size() - st.size(), 0);
                    }
                };
                normalize_stride(op.stride0);
                normalize_stride(op.stride1);

                if (div_pow_rhs || div_pow_lhs) {
                    size_t in0 = make_tensor(lhs_value);
                    size_t in1 = make_tensor(rhs_value);
                    op.input0 = &m_flat_tensors[in0];
                    op.input1 = &m_flat_tensors[in1];
                } else {
                    add_input_indices(op);
                }
                set_output(op);
                if (op.output) op.output->shape = op.out_shape;
                if (metal_debug_enabled()) {
                    std::ostringstream oss;
                    oss << "[METAL MLIR] elementwise node=" << node->get_friendly_name()
                        << " kind=" << static_cast<int>(op.kind);
                    if (div_pow_rhs || div_pow_lhs) oss << " div_from_pow=-1";
                    debug_log(oss.str());
                }
                // If RHS is Constant and requires broadcast, pre-expand it after inputs are wired
                // This helps all binary eltwise ops (Add/Sub/Mul/Div/Pow) keep simple row-major reads.
                if (op.input1 && op.input1->from_constant && op.is_broadcast && !op.out_shape.empty()) {
                    const size_t out_elems = ov::shape_size(op.out_shape);
                    if (out_elems > 0 && !op.input1->const_data.empty()) {
                        const auto rhs_orig_shape = op.input1->shape;  // save original
                        // Normalize RHS shape to rank of output
                        const size_t rank = op.out_shape.size();
                        ov::Shape rhs_shape_norm(rank, 1);
                        {
                            auto rhs_shape = rhs_orig_shape;
                            size_t off = rank - rhs_shape.size();
                            for (size_t i = 0; i < rhs_shape.size(); ++i)
                                rhs_shape_norm[off + i] = static_cast<size_t>(rhs_shape[i]);
                        }
                        // Strides for normalized RHS
                        std::vector<size_t> rhs_stride(rank, 1);
                        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                            rhs_stride[i] = rhs_stride[i + 1] * rhs_shape_norm[i + 1];
                        }
                        std::vector<float> broadcasted(out_elems, 0.f);
                        std::vector<int64_t> out_stride(rank, 1);
                        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                            out_stride[i] = out_stride[i + 1] * static_cast<int64_t>(op.out_shape[i + 1]);
                        }
                        debug_log("[METAL MLIR] rhs_shape_norm dims: " + std::to_string(rhs_shape_norm.size()));
                        for (size_t i = 0; i < rhs_shape_norm.size(); ++i) {
                            debug_log("  rhs_shape_norm[" + std::to_string(i) + "]=" + std::to_string(rhs_shape_norm[i]) +
                                      " rhs_stride=" + std::to_string(rhs_stride[i]) +
                                      " out_stride=" + std::to_string(out_stride[i]));
                        }
                        for (size_t idx = 0; idx < out_elems; ++idx) {
                            size_t tmp = idx;
                            size_t src_idx = 0;
                            for (size_t d = 0; d < rank; ++d) {
                                const size_t coord = tmp / static_cast<size_t>(out_stride[d]);
                                tmp -= coord * static_cast<size_t>(out_stride[d]);
                                if (rhs_shape_norm[d] == 1) continue;  // broadcast axis
                                src_idx += coord * rhs_stride[d];
                            }
                            broadcasted[idx] = op.input1->const_data[std::min(src_idx, op.input1->const_data.size() - 1)];
                        }
                        op.input1->const_data.swap(broadcasted);
                        op.input1->shape = op.out_shape;
                        // Recompute strides after expansion
                        auto make_row_major = [&](const std::vector<int64_t>& shape) {
                            std::vector<int64_t> st(shape.size(), 1);
                            for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
                                st[i] = st[i + 1] * shape[i + 1];
                            return st;
                        };
                        std::vector<size_t> lhs_shape_norm(rank, 1);
                        if (op.input0) {
                            auto lhs_shape = op.input0->shape;
                            size_t off_l = rank - lhs_shape.size();
                            for (size_t i = 0; i < lhs_shape.size(); ++i)
                                lhs_shape_norm[off_l + i] = static_cast<size_t>(lhs_shape[i]);
                        }
                        // Decide whether broadcast kernel is still needed: lhs may still require it
                        bool lhs_matches_out = true;
                        for (size_t k = 0; k < rank; ++k) {
                            if (lhs_shape_norm[k] != static_cast<size_t>(op.out_shape[k])) {
                                lhs_matches_out = false;
                                break;
                            }
                        }
                        bool rhs_matches_out = true;
                        for (size_t k = 0; k < rank; ++k) {
                            if (rhs_shape_norm[k] != static_cast<size_t>(op.out_shape[k])) {
                                rhs_matches_out = false;
                                break;
                            }
                        }
                        op.is_broadcast = !(lhs_matches_out && rhs_matches_out);
                        // Use row-major strides derived from original shapes; zero stride on broadcast axes
                        // lhs strides use original layout, rhs uses expanded (row-major out_shape)
                        std::vector<size_t> lhs_stride(rank, 1);
                        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
                            lhs_stride[i] = lhs_stride[i + 1] * lhs_shape_norm[i + 1];
                        }
                        op.stride0.resize(rank);
                        op.stride1.assign(out_stride.begin(), out_stride.end());
                        for (size_t k = 0; k < rank; ++k) {
                            op.stride0[k] = (lhs_shape_norm[k] == 1) ? 0
                                                                     : static_cast<int64_t>(lhs_stride[k]);
                            if (rhs_shape_norm[k] == 1) op.stride1[k] = 0;
                        }
                        // Ensure stride0/1 are padded to rank
                        auto pad_stride = [&](std::vector<int64_t>& st) {
                            if (st.size() < rank) st.insert(st.begin(), rank - st.size(), 0);
                        };
                        pad_stride(op.stride0);
                        pad_stride(op.stride1);
                        if (metal_debug_enabled()) {
                            std::ostringstream oss;
                            oss << "[METAL MLIR] rhs after expand first: ";
                            for (size_t i = 0; i < std::min<size_t>(8, op.input1->const_data.size()); ++i) {
                                oss << op.input1->const_data[i] << " ";
                            }
                            debug_log(oss.str());
                        }
                    }
                }
                // Some front-end passes turn Divide by constant into Multiply by reciprocal.
                // If we see Div with Constant and the node type is Multiply, recover the original divisor
                // to improve f16 accuracy (compute in half then cast).
                if (op.kind == KernelOpKind::ElementwiseDiv &&
                    op.input1 && op.input1->from_constant &&
                    node->get_type_info().name == std::string("Multiply")) {
                    bool invert_ok = true;
                    for (float v : op.input1->const_data) {
                        if (v == 0.f) { invert_ok = false; break; }
                    }
                    if (invert_ok) {
                        for (auto& v : op.input1->const_data) {
                            v = 1.0f / v;
                        }
                        if (metal_debug_enabled()) {
                            std::cerr << "[METAL MLIR] Detected Div decomposed to Multiply; restored divisor, first="
                                      << op.input1->const_data[0] << "\n";
                        }
                    }
                }
                supported = true;
            } else if (auto slice = ov::as_type_ptr<const ov::op::v8::Slice>(node)) {
                op.kind = KernelOpKind::Slice;
                add_input_indices(op);
                set_output(op);
                auto in_shape = slice->get_input_shape(0);
                auto out_shape = slice->get_output_shape(0);
                op.slice.in_shape.assign(in_shape.begin(), in_shape.end());
                op.slice.out_shape.assign(out_shape.begin(), out_shape.end());

                auto get_const_vec = [](const std::shared_ptr<const ov::Node>& n) -> std::vector<int64_t> {
                    if (auto c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(n)) {
                        return c->cast_vector<int64_t>();
                    }
                    return {};
                };
                auto starts_raw = get_const_vec(slice->get_input_node_shared_ptr(1));
                auto stops_raw  = get_const_vec(slice->get_input_node_shared_ptr(2));
                auto steps_raw  = get_const_vec(slice->get_input_node_shared_ptr(3));
                std::vector<int64_t> axes;
                if (slice->get_input_size() >= 5) {
                    axes = get_const_vec(slice->get_input_node_shared_ptr(4));
                }
                if (axes.empty()) {
                    axes.resize(starts_raw.size());
                    std::iota(axes.begin(), axes.end(), 0);
                }
                if (steps_raw.empty())
                    steps_raw.assign(starts_raw.size(), 1);
                op.slice.starts.assign(in_shape.size(), 0);
                op.slice.steps.assign(in_shape.size(), 1);
                op.slice.axes = axes;
                for (size_t i = 0; i < axes.size(); ++i) {
                    size_t ax = static_cast<size_t>(axes[i]);
                    if (ax >= in_shape.size() || i >= starts_raw.size())
                        continue;
                    int64_t dim = static_cast<int64_t>(in_shape[ax]);
                    int64_t start = starts_raw[i];
                    int64_t step  = (i < steps_raw.size() ? steps_raw[i] : 1);
                    if (step == 0) step = 1;
                    if (start < 0) start += dim;
                    start = std::max<int64_t>(0, std::min<int64_t>(start, dim - 1));
                    op.slice.starts[ax] = start;
                    op.slice.steps[ax] = step;
                }
                op.slice.in_strides.resize(in_shape.size(), 1);
                for (int i = static_cast<int>(in_shape.size()) - 2; i >= 0; --i) {
                    op.slice.in_strides[i] = op.slice.in_strides[i + 1] * static_cast<int64_t>(in_shape[i + 1]);
                }
                supported = true;
            } else if (ov::as_type_ptr<const ov::op::v0::Relu>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Sigmoid>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Tanh>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Elu>(node) ||
                       ov::as_type_ptr<const ov::op::v0::PRelu>(node) ||
                       ov::as_type_ptr<const ov::op::v0::Gelu>(node) ||
                       ov::as_type_ptr<const ov::op::v4::Swish>(node) ||
                       node->get_type_info().name == std::string("Sign") ||
                       node->get_type_info().name == std::string("Abs")) {
                op.kind = KernelOpKind::Unary;
                if (ov::as_type_ptr<const ov::op::v0::Relu>(node)) op.activation = ActivationKind::Relu;
                else if (ov::as_type_ptr<const ov::op::v0::Sigmoid>(node)) op.activation = ActivationKind::Sigmoid;
                else if (ov::as_type_ptr<const ov::op::v0::Tanh>(node)) op.activation = ActivationKind::Tanh;
                else if (auto e = ov::as_type_ptr<const ov::op::v0::Elu>(node)) {
                    op.activation = ActivationKind::Elu;
                    op.alpha = static_cast<float>(e->get_alpha());
                } else if (auto p = ov::as_type_ptr<const ov::op::v0::PRelu>(node)) {
                    op.activation = ActivationKind::Prelu;
                    // store slope in alpha for now when it is scalar constant
                    if (auto slope_c = std::dynamic_pointer_cast<const ov::op::v0::Constant>(p->get_input_node_shared_ptr(1))) {
                        auto vals = slope_c->cast_vector<float>();
                        if (!vals.empty()) op.alpha = vals[0];
                    }
                } else if (ov::as_type_ptr<const ov::op::v0::Gelu>(node)) {
                    op.activation = ActivationKind::Gelu;
                } else if (ov::as_type_ptr<const ov::op::v4::Swish>(node)) {
                    op.activation = ActivationKind::Swish;
                } else if (ov::as_type_ptr<const ov::op::v0::Abs>(node)) {
                    op.activation = ActivationKind::Abs;
                } else if (ov::as_type_ptr<const ov::op::v0::Sign>(node)) {
                    op.activation = ActivationKind::Sign;
                } else if (node->get_type_info().name == std::string("Sign")) {
                    op.activation = ActivationKind::Sign;
                } else if (node->get_type_info().name == std::string("Abs")) {
                    op.activation = ActivationKind::Abs;
                }
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
                const bool is_group = static_cast<bool>(ov::as_type_ptr<const ov::op::v1::GroupConvolution>(node));
                const auto in_shape = node->get_input_shape(0);
                const auto w_shape = node->get_input_shape(1);
                const auto out_shape = node->get_output_shape(0);

                // Rank-5 -> Conv3D, Rank-4 -> Conv2D
                if (in_shape.size() == 5 && w_shape.size() >= 5) {
                    if (is_group) {
                        debug_log("[METAL MLIR] Flat IR: GroupConvolution 3D not supported yet");
                        continue;
                    }
                    op.kind = KernelOpKind::Conv3D;
                    add_input_indices(op);
                    set_output(op);
                    const auto& s = ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_strides();
                    const auto& pb = ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_pads_begin();
                    const auto& pe = ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_pads_end();
                    const auto& dil = ov::as_type_ptr<const ov::op::v1::Convolution>(node)->get_dilations();
                    op.conv3d.N = static_cast<uint32_t>(in_shape[0]);
                    op.conv3d.C_in = static_cast<uint32_t>(in_shape[1]);
                    op.conv3d.D = static_cast<uint32_t>(in_shape[2]);
                    op.conv3d.H = static_cast<uint32_t>(in_shape[3]);
                    op.conv3d.W = static_cast<uint32_t>(in_shape[4]);
                    if (s.size() == 3) {
                        op.conv3d.strideD = static_cast<uint32_t>(s[0]);
                        op.conv3d.strideH = static_cast<uint32_t>(s[1]);
                        op.conv3d.strideW = static_cast<uint32_t>(s[2]);
                    }
                    if (dil.size() == 3) {
                        op.conv3d.dilationD = static_cast<uint32_t>(dil[0]);
                        op.conv3d.dilationH = static_cast<uint32_t>(dil[1]);
                        op.conv3d.dilationW = static_cast<uint32_t>(dil[2]);
                    }
                    if (pb.size() == 3 && pe.size() == 3) {
                        op.conv3d.padFront = static_cast<uint32_t>(pb[0]);
                        op.conv3d.padTop = static_cast<uint32_t>(pb[1]);
                        op.conv3d.padLeft = static_cast<uint32_t>(pb[2]);
                        op.conv3d.padBack = static_cast<uint32_t>(pe[0]);
                        op.conv3d.padBottom = static_cast<uint32_t>(pe[1]);
                        op.conv3d.padRight = static_cast<uint32_t>(pe[2]);
                    }
                    op.conv3d.C_out = static_cast<uint32_t>(w_shape[0]);
                    op.conv3d.kernelD = static_cast<uint32_t>(w_shape[2]);
                    op.conv3d.kernelH = static_cast<uint32_t>(w_shape[3]);
                    op.conv3d.kernelW = static_cast<uint32_t>(w_shape[4]);
                    op.conv3d.element_type = static_cast<uint32_t>(static_cast<ov::element::Type_t>(node->get_input_element_type(0)));
                    if (out_shape.size() == 5) {
                        op.conv3d.outD = static_cast<uint32_t>(out_shape[2]);
                        op.conv3d.outH = static_cast<uint32_t>(out_shape[3]);
                        op.conv3d.outW = static_cast<uint32_t>(out_shape[4]);
                        KernelTensor* out_t = op.output;
                        if (out_t) {
                            out_t->shape = {static_cast<int64_t>(out_shape[0]), static_cast<int64_t>(out_shape[1]),
                                            static_cast<int64_t>(out_shape[2]), static_cast<int64_t>(out_shape[3]),
                                            static_cast<int64_t>(out_shape[4])};
                        }
                    }
                    supported = true;
                } else if (in_shape.size() == 4 && w_shape.size() >= 4) {
                    op.kind = KernelOpKind::Conv2D;
                    add_input_indices(op);
                    set_output(op);
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
                    supported = true;
                } else {
                    debug_log("[METAL MLIR] Flat IR: Convolution rank not supported");
                    continue;
                }
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
                propagate_dtype(op);
                m_flat_ops.push_back(op);
            }
        }

        if (!m_flat_ops.empty()) {
            Segment seg{};
            seg.first_op_index = 0;
            seg.op_count = m_flat_ops.size();
            for (const auto& in : model->inputs()) seg.input_ports.push_back(in);
            for (const auto& out : model->outputs()) seg.output_ports.push_back(out);
            m_flat_segments.push_back(std::move(seg));
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

        bool all_slice = true;
        for (const auto& op : m_flat_ops) {
            if (op.kind != KernelOpKind::Slice) {
                all_slice = false;
                break;
            }
        }
        if (all_slice) {
            m_flat_segments.push_back(Segment{0, m_flat_ops.size()});
            m_ops_from_flat_segment = true;
            return;
        }

        auto is_segmentable = [](KernelOpKind k) {
            switch (k) {
                case KernelOpKind::Conv2D:
                case KernelOpKind::Conv3D:
                case KernelOpKind::BatchNorm2D:
                case KernelOpKind::Unary:
                case KernelOpKind::MatMul:
                case KernelOpKind::ElementwiseAdd:
                case KernelOpKind::ElementwiseMul:
                case KernelOpKind::ElementwiseSub:
                case KernelOpKind::ElementwiseDiv:
                case KernelOpKind::ElementwisePow:
                case KernelOpKind::ElementwiseMod:
                case KernelOpKind::ElementwiseFloorMod:
                case KernelOpKind::Softmax:
                case KernelOpKind::MaxPool2D:
                case KernelOpKind::AvgPool2D:
                case KernelOpKind::Slice:
                case KernelOpKind::Split:
                    return true;
                default:
                    return false;
            }
        };

        size_t seg_start = 0;
        size_t seg_len = 0;
        auto push_seg = [&](size_t start, size_t len) {
            if (len == 0) return;
            m_flat_segments.push_back(Segment{start, len});
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

        if (!m_flat_segments.empty()) {
            m_ops_from_flat_segment = true;
        }
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
