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

#include "runtime/metal_logger.hpp"
#include "runtime/metal_dtype.hpp"
#include "runtime/metal_memory.hpp"

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
#include <unordered_map>
#include <unordered_set>
#include <array>

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
#include "openvino/runtime/profiling_info.hpp"
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

static bool use_handwritten_msl() {
    // Force MLIR-based codegen only.
    return false;
}

struct TempBufferKey {
    KernelOpKind kind{};
    size_t slot = 0;
    ov::element::Type type = ov::element::dynamic;

    bool operator==(const TempBufferKey& other) const {
        return kind == other.kind && slot == other.slot && type == other.type;
    }
};

struct TempBufferKeyHash {
    size_t operator()(const TempBufferKey& k) const {
        size_t h1 = static_cast<size_t>(k.kind);
        size_t h2 = std::hash<size_t>{}(k.slot);
        auto te = static_cast<ov::element::Type_t>(k.type);
        size_t h3 = std::hash<int>{}(static_cast<int>(te));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct BroadcastResult {
    ov::Shape out_shape;
    std::vector<int64_t> stride0;
    std::vector<int64_t> stride1;
    bool success = false;
};

static std::string kind_to_string(KernelOpKind k) {
    switch (k) {
        case KernelOpKind::ElementwiseAdd: return "ElementwiseAdd";
        case KernelOpKind::ElementwiseSub: return "ElementwiseSub";
        case KernelOpKind::ElementwiseMul: return "ElementwiseMul";
        case KernelOpKind::ElementwiseDiv: return "ElementwiseDiv";
        case KernelOpKind::ElementwisePow: return "ElementwisePow";
        case KernelOpKind::ElementwiseMod: return "ElementwiseMod";
        case KernelOpKind::ElementwiseFloorMod: return "ElementwiseFloorMod";
        case KernelOpKind::Concat: return "Concat";
        case KernelOpKind::Interpolate: return "Interpolate";
        case KernelOpKind::MatMul: return "MatMul";
        case KernelOpKind::Split: return "Split";
        case KernelOpKind::Slice: return "Slice";
        case KernelOpKind::Convert: return "Convert";
        case KernelOpKind::Unary: return "Unary";
        case KernelOpKind::Softmax: return "Softmax";
        case KernelOpKind::MaxPool2D: return "MaxPool2D";
        case KernelOpKind::AvgPool2D: return "AvgPool2D";
        case KernelOpKind::Conv2D: return "Conv2D";
        case KernelOpKind::Conv3D: return "Conv3D";
        case KernelOpKind::BatchNorm2D: return "BatchNorm2D";
        case KernelOpKind::Transpose: return "Transpose";
        case KernelOpKind::Reshape: return "Reshape";
        default: return "Unknown";
    }
}

static BroadcastResult compute_broadcast(const ov::Shape& a_shape, const ov::Shape& b_shape) {
    BroadcastResult res;
    size_t rank = std::max(a_shape.size(), b_shape.size());
    // Treat rank-0 (scalar) as rank-1 with dim=1 to simplify downstream stride handling.
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
    const bool log_mod = node->get_friendly_name().find("Mod") != std::string::npos;

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
            METAL_LOG_DEBUG("mlir", "[METAL MLIR] Mod pattern check: mul in0=" + std::string(in0 ? in0->get_type_info().name : "nil") +
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
        if (log_mod) METAL_LOG_DEBUG("mlir", "[METAL MLIR] Mod pattern missed: no Sign/Sub/Add combination");
        return std::nullopt;
    }

    auto abs_a = ov::as_type_ptr<const ov::op::v0::Abs>(sub_like->get_input_node_shared_ptr(0));
    if (!abs_a) {
        if (log_mod) METAL_LOG_DEBUG("mlir", "[METAL MLIR] Mod pattern missed: first arg not Abs");
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

std::string describe_node(const std::shared_ptr<const ov::Node>& node) {
    return node->get_friendly_name() + "(" + std::string(node->get_type_info().name) + ")";
}

#include "runtime/mlir_backend_model_analysis.inc"
#include "runtime/mlir_backend_analysis.inc"



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
            res.has_squared_diff = true;
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
    METAL_LOGSTREAM_TRACE("mlir") << prefix;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i) METAL_LOGSTREAM_TRACE("mlir") << ", ";
        METAL_LOGSTREAM_TRACE("mlir") << items[i];
    }
    METAL_LOGSTREAM_TRACE("mlir") << "\n";
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

        // Reject unsupported input types early (allow f16/f32/i32/i64/u8).
        for (const auto& p : model->get_parameters()) {
            auto et = p->get_element_type();
            if (!(et == ov::element::f16 || et == ov::element::f32 || et == ov::element::i32 || et == ov::element::i64 ||
                  et == ov::element::u8)) {
                OPENVINO_THROW("METAL supports only f16/f32/i32/i64/u8 model inputs");
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
        if (m_transpose4d_pipeline) {
            [m_transpose4d_pipeline release];
            m_transpose4d_pipeline = nil;
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

    void run(const std::vector<ov::Tensor>& orig_inputs, std::vector<ov::Tensor>& outputs);
    void set_profiling_enabled(bool enable) { m_enable_profiling = enable; }
    std::vector<ov::ProfilingInfo> get_profiling() const { return m_last_profiling; }
    bool has_segment() const;
    bool segment_io_is_model_io() const;
    const Segment& get_segment() const;
    std::vector<ov::Tensor> run_segment(const Segment& seg, const std::vector<ov::Tensor>& inputs);
    bool run_device(MetalTensorMap& tensors, MetalBufferManager& mgr);
    void preload_constants(MetalBufferManager& mgr);

private:
    void ensure_fallback(const std::shared_ptr<const ov::Model>& model);
    void ensure_fallback();
    bool run_host_softmax(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) const;
    // Upload constant tensors once and cache device buffers.
    MetalBuffer ensure_const_buffer(const KernelTensor* t, MetalBufferManager& mgr);
#ifdef __OBJC__
    id<MTLComputePipelineState> get_transpose4d_pipeline();
#endif

    void compile(const std::shared_ptr<const ov::Model>& model);
    bool try_build_eltwise(const std::shared_ptr<const ov::Model>& model,
                           const ModelAnalysis& analysis,
                           MetalKernelCompiler& compiler,
                           std::string& log);
    bool try_build_softmax(const std::shared_ptr<const ov::Model>& model,
                           const ModelAnalysis& analysis,
                           MetalKernelCompiler& compiler,
                           std::string& log);
    bool try_build_matmul(const std::shared_ptr<const ov::Model>& model,
                          const ModelAnalysis& analysis,
                          MetalKernelCompiler& compiler,
                          std::string& log);
    bool try_build_conv(const std::shared_ptr<const ov::Model>& model,
                        const ModelAnalysis& analysis,
                        MetalKernelCompiler& compiler,
                        std::string& log);
    bool try_build_pool(const std::shared_ptr<const ov::Model>& model,
                        const ModelAnalysis& analysis,
                        MetalKernelCompiler& compiler,
                        std::string& log);
    bool try_build_unary(const std::shared_ptr<const ov::Model>& model,
                         const ModelAnalysis& analysis,
                         MetalKernelCompiler& compiler,
                         std::string& log);
    bool try_build_batchnorm(const std::shared_ptr<const ov::Model>& model,
                             const ModelAnalysis& analysis,
                             MetalKernelCompiler& compiler,
                             std::string& log);

    void build_flat_ir(const std::shared_ptr<const ov::Model>& model);
    bool try_run_split_segment(const Segment& seg,
                               const std::vector<KernelOp>& ops,
                               const std::vector<ov::Tensor>& inputs,
                               std::vector<ov::Tensor>& outputs,
                               MetalBufferManager& mgr,
                               const std::function<void(const std::string&)>& cpu_fallback_error);

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
    bool m_enable_profiling = false;
    std::vector<ov::ProfilingInfo> m_last_profiling;

    void build_segments_from_flat_ir();

    // Active buffer manager for current inference (set by run_device, otherwise lazily created).
    MetalBufferManager* m_active_mgr = nullptr;
    std::shared_ptr<MetalBufferManager> m_owned_mgr;
    MetalTensorMap* m_active_tensor_map = nullptr;
    // Persistent constant buffers keyed by KernelTensor*.
    std::unordered_map<const KernelTensor*, MetalBuffer> m_const_buffer_cache;
    // Dynamic handles for temporaries keyed by logical op slot/kind/type.
    std::unordered_map<TempBufferKey, BufferHandle, TempBufferKeyHash> m_temp_buffer_handles;
    // Owner of temp handles above to prevent reuse across different managers.
    MetalBufferManager* m_temp_mgr = nullptr;
    // Persistent constants harvested during compile (host vectors) mirrored on device.
    MetalBuffer m_const_w_buf;
    MetalBuffer m_const_bn_buf;
    MetalBuffer m_const_b_buf;
    MetalBuffer m_const_mul_buf;
    MetalBuffer m_const_mm0_buf;
    MetalBuffer m_const_mm1_buf;
#ifdef __OBJC__
    id<MTLComputePipelineState> m_transpose4d_pipeline = nil;
#endif

};

MetalBuffer MlirBackend::Impl::ensure_const_buffer(const KernelTensor* t, MetalBufferManager& mgr) {
    if (!t || !t->from_constant || t->const_data.empty())
        return {};
    auto it = m_const_buffer_cache.find(t);
    if (it != m_const_buffer_cache.end())
        return it->second;

    // Use storage size to match the kernel pointer type (half buffers must be 2B per element).
    const size_t elem_sz = storage_size(t->dtype);
    size_t bytes = t->const_data.size() * elem_sz;
    if (bytes == 0) bytes = elem_sz;

    MetalBuffer buf = mgr.allocate(bytes, t->dtype.ov_type, /*persistent=*/true, /*storageModePrivate=*/true);
    switch (t->dtype.storage) {
        case MetalDType::StorageType::I32: {
            std::vector<int32_t> tmp(t->const_data.size());
            for (size_t i = 0; i < t->const_data.size(); ++i)
                tmp[i] = static_cast<int32_t>(t->const_data[i]);
            mgr.upload(buf, tmp.data(), tmp.size() * sizeof(int32_t));
            break;
        }
        case MetalDType::StorageType::I64: {
            std::vector<int64_t> tmp(t->const_data.size());
            for (size_t i = 0; i < t->const_data.size(); ++i)
                tmp[i] = static_cast<int64_t>(t->const_data[i]);
            mgr.upload(buf, tmp.data(), tmp.size() * sizeof(int64_t));
            break;
        }
        case MetalDType::StorageType::U8: {
            std::vector<uint8_t> tmp(t->const_data.size());
            for (size_t i = 0; i < t->const_data.size(); ++i)
                tmp[i] = static_cast<uint8_t>(t->const_data[i]);
            mgr.upload(buf, tmp.data(), tmp.size() * sizeof(uint8_t));
            break;
        }
        case MetalDType::StorageType::I8: {
            std::vector<int8_t> tmp(t->const_data.size());
            for (size_t i = 0; i < t->const_data.size(); ++i)
                tmp[i] = static_cast<int8_t>(t->const_data[i]);
            mgr.upload(buf, tmp.data(), tmp.size() * sizeof(int8_t));
            break;
        }
        case MetalDType::StorageType::F16: {
            std::vector<ov::float16> tmp(t->const_data.size());
            for (size_t i = 0; i < t->const_data.size(); ++i)
                tmp[i] = ov::float16{t->const_data[i]};
            mgr.upload(buf, tmp.data(), tmp.size() * sizeof(ov::float16));
            break;
        }
        case MetalDType::StorageType::F32:
        default: {
            mgr.upload(buf, t->const_data.data(), bytes);
            break;
        }
    }
    if (std::getenv("METAL_DEBUG_DUMP_OUTPUT")) {
        auto mb = static_cast<id<MTLBuffer>>(buf.buffer);
        if (mb && [mb contents]) {
            fprintf(stderr, "[METAL][dbg] const upload first=%f size=%zu\n",
                    static_cast<const float*>([mb contents])[0],
                    t->const_data.size());
        }
    }
    m_const_buffer_cache[t] = buf;
    return buf;
}

#ifdef __OBJC__
id<MTLComputePipelineState> MlirBackend::Impl::get_transpose4d_pipeline() {
    if (m_transpose4d_pipeline)
        return m_transpose4d_pipeline;
    NSError* err = nil;
    static const char* kSrc =
        "using namespace metal;\n"
        "kernel void transpose4d_f32(const device float* src [[buffer(0)]],\n"
        "                             device float* dst [[buffer(1)]],\n"
        "                             constant uint4& in_dims [[buffer(2)]],\n"
        "                             constant uint4& perm [[buffer(3)]],\n"
        "                             constant uint4& out_dims [[buffer(4)]],\n"
        "                             uint gid [[thread_position_in_grid]]) {\n"
        "    uint total = out_dims.x * out_dims.y * out_dims.z * out_dims.w;\n"
        "    if (gid >= total) return;\n"
        "    uint idx = gid;\n"
        "    uint o3 = idx % out_dims.w; idx /= out_dims.w;\n"
        "    uint o2 = idx % out_dims.z; idx /= out_dims.z;\n"
        "    uint o1 = idx % out_dims.y; idx /= out_dims.y;\n"
        "    uint o0 = idx;\n"
        "    uint ic[4];\n"
        "    ic[perm.x] = o0;\n"
        "    ic[perm.y] = o1;\n"
        "    ic[perm.z] = o2;\n"
        "    ic[perm.w] = o3;\n"
        "    uint stride0 = in_dims.y * in_dims.z * in_dims.w;\n"
        "    uint stride1 = in_dims.z * in_dims.w;\n"
        "    uint stride2 = in_dims.w;\n"
        "    uint src_index = ic[0] * stride0 + ic[1] * stride1 + ic[2] * stride2 + ic[3];\n"
        "    dst[gid] = src[src_index];\n"
        "}\n";
    NSString* msl = [NSString stringWithUTF8String:kSrc];
    id<MTLDevice> dev = m_device;
    if (!dev) return nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:msl options:nil error:&err];
    if (!lib || err) {
        METAL_LOG_ERROR("mlir", "[METAL MLIR] failed to build transpose4d MSL");
        return nil;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"transpose4d_f32"];
    if (!fn) {
        [lib release];
        METAL_LOG_ERROR("mlir", "[METAL MLIR] transpose4d_f32 function missing");
        return nil;
    }
    id<MTLComputePipelineState> pipe = [dev newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    [lib release];
    if (!pipe || err) {
        METAL_LOG_ERROR("mlir", "[METAL MLIR] failed to create transpose pipeline");
        return nil;
    }
    m_transpose4d_pipeline = pipe;
    return m_transpose4d_pipeline;
}
#endif

void MlirBackend::Impl::preload_constants(MetalBufferManager& mgr) {
    for (auto& tensor : m_ir.tensors) {
        if (tensor.from_constant) {
            ensure_const_buffer(&tensor, mgr);
        }
    }
}

#include "runtime/mlir_backend_run_device.inc"
#include "runtime/mlir_backend_segments.inc"
#include "runtime/mlir_backend_fallback.inc"
#include "runtime/mlir_backend_compile.inc"
#include "runtime/mlir_backend_build_flat_ir.inc"
#include "runtime/mlir_backend_run.inc"
#include "runtime/mlir_backend_run_segment.inc"

MlirBackend::MlirBackend(const std::shared_ptr<const ov::Model>& model,
                         const std::shared_ptr<const ov::Model>& original_model,
                         ov::element::Type inference_precision)
    : m_impl(std::make_unique<MlirBackend::Impl>(model, original_model, inference_precision)) {}

MlirBackend::~MlirBackend() = default;

void MlirBackend::run(const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) {
    m_impl->run(inputs, outputs);
}

bool MlirBackend::run_device(MetalTensorMap& tensors, MetalBufferManager& mgr) {
    return m_impl->run_device(tensors, mgr);
}

std::shared_ptr<MetalBufferManager> MlirBackend::create_buffer_manager() {
#ifdef __OBJC__
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return std::make_shared<MetalBufferManager>(dev);
#else
    return nullptr;
#endif
}

void MlirBackend::preload_constants(MetalBufferManager& mgr) {
    m_impl->preload_constants(mgr);
}

void MlirBackend::set_profiling(bool enable) {
    m_impl->set_profiling_enabled(enable);
}

std::vector<ov::ProfilingInfo> MlirBackend::get_profiling_info() const {
    return m_impl->get_profiling();
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

#include "runtime/ops/mlir_backend_op_eltwise.inc"
#include "runtime/ops/mlir_backend_op_softmax.inc"
#include "runtime/ops/mlir_backend_op_matmul.inc"
#include "runtime/ops/mlir_backend_op_conv.inc"
#include "runtime/ops/mlir_backend_op_pool.inc"
#include "runtime/ops/mlir_backend_op_unary.inc"
#include "runtime/ops/mlir_backend_op_batchnorm.inc"
#include "runtime/ops/mlir_backend_op_split.inc"

}  // namespace metal_plugin
}  // namespace ov
