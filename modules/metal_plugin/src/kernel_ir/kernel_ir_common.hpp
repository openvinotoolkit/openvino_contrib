// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "runtime/metal_dtype.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
class Model;
class Node;

namespace metal_plugin {

enum class ActivationKind { Relu, Sigmoid, Tanh, Elu, Prelu, Gelu, Swish, Abs, Sign, Identity };

enum class KernelOpKind {
    ElementwiseAdd,
    ElementwiseMul,
    ElementwiseSub,
    ElementwiseDiv,
    ElementwisePow,
    ElementwiseMod,
    ElementwiseFloorMod,
    Concat,
    Interpolate,
    MatMul,
    Split,
    Slice,
    Unary,
    Softmax,
    MaxPool2D,
    AvgPool2D,
    Conv2D,
    Conv3D,
    BatchNorm2D
};

struct KernelTensor {
    std::string name;
    std::vector<int64_t> shape;
    MetalDType dtype;
    bool from_parameter = false;  // true if originated from ov::Parameter
    bool from_constant = false;   // true if originated from ov::Constant
    std::vector<float> const_data;  // only valid when from_constant == true (stored as f32)
    const void* source_node = nullptr;  // raw pointer to originating ov::Node for mapping
};

struct KernelOp {
    KernelOpKind kind;
    KernelTensor* input0 = nullptr;
    KernelTensor* input1 = nullptr;
    KernelTensor* output = nullptr;
    std::vector<KernelTensor*> outputs;  // for multi-output ops like Split
    MetalDType dtype;
    // Element type of tensors (ov::element::Type_t). For ops with a single dtype (softmax/unary/etc.).
    uint32_t element_type = static_cast<uint32_t>(ov::element::Type_t::dynamic);
    // Unary-specific
    ActivationKind activation = ActivationKind::Relu;
    float alpha = 0.0f;  // used for ELU/PReLU
    // Elementwise broadcast metadata
    bool is_broadcast = false;
    std::vector<int64_t> out_shape;
    std::vector<int64_t> stride0;
    std::vector<int64_t> stride1;
    // Softmax-specific (flattened to 2D: rows x cols)
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t inner = 1;  // stride between elements of softmax dimension when axis is not last
    int64_t softmax_axis = -1;  // original axis as provided by the op
    // MatMul-specific dims (M x K) * (K x N) = (M x N)
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    // Optional batch dimension for simple 3D batched matmul; 1 means no batch.
    int64_t batch = 1;
    int64_t batch_a = 1;
    int64_t batch_b = 1;
    // If true, B is laid out as [N, K] instead of [K, N] (e.g., a folded transpose).
    bool b_is_nk_layout = false;
    bool a_transpose = false;  // logical transpose of A
    bool b_transpose = false;  // logical transpose of B
    // Pool2D-specific
    struct Pool2DDesc {
        uint32_t N = 0;
        uint32_t H = 0;
        uint32_t W = 0;
        uint32_t C = 0;
        uint32_t outH = 0;
        uint32_t outW = 0;
        uint32_t kernelH = 0;
        uint32_t kernelW = 0;
        uint32_t strideH = 0;
        uint32_t strideW = 0;
        uint32_t padTop = 0;
        uint32_t padLeft = 0;
        bool exclude_pad = false;
    } pool;
    struct Conv2DDesc {
        MetalDType dtype;
        uint32_t N = 0;
        uint32_t C_in = 0;
        uint32_t H = 0;
        uint32_t W = 0;
        uint32_t C_out = 0;
        uint32_t groups = 1;
        uint32_t C_in_per_group = 0;
        uint32_t C_out_per_group = 0;
        uint32_t kernelH = 0;
        uint32_t kernelW = 0;
        uint32_t strideH = 0;
        uint32_t strideW = 0;
        uint32_t dilationH = 1;
        uint32_t dilationW = 1;
        uint32_t padTop = 0;
        uint32_t padLeft = 0;
        uint32_t padBottom = 0;
        uint32_t padRight = 0;
        uint32_t padType = 0;  // ov::op::PadType enum value
        uint32_t element_type = 0;  // ov::element::Type_t value
        uint32_t outH = 0;
        uint32_t outW = 0;
    } conv2d;
    struct Conv3DDesc {
        MetalDType dtype;
        uint32_t N = 0;
        uint32_t C_in = 0;
        uint32_t D = 0;
        uint32_t H = 0;
        uint32_t W = 0;
        uint32_t C_out = 0;
        uint32_t kernelD = 0;
        uint32_t kernelH = 0;
        uint32_t kernelW = 0;
        uint32_t strideD = 1;
        uint32_t strideH = 1;
        uint32_t strideW = 1;
        uint32_t dilationD = 1;
        uint32_t dilationH = 1;
        uint32_t dilationW = 1;
        uint32_t padFront = 0;
        uint32_t padTop = 0;
        uint32_t padLeft = 0;
        uint32_t padBack = 0;
        uint32_t padBottom = 0;
        uint32_t padRight = 0;
        uint32_t outD = 0;
        uint32_t outH = 0;
        uint32_t outW = 0;
        uint32_t element_type = 0;
    } conv3d;
    struct BatchNormDesc {
        MetalDType dtype;
        uint32_t N = 0;
        uint32_t C = 0;
        uint32_t H = 0;
        uint32_t W = 0;
        float eps = 0.f;
    } batchnorm;
    struct SliceDesc {
        MetalDType dtype;
        std::vector<int64_t> starts;
        std::vector<int64_t> steps;
        std::vector<int64_t> axes;
        std::vector<int64_t> in_shape;
        std::vector<int64_t> out_shape;
        std::vector<int64_t> in_strides;
    } slice;
    struct SplitDesc {
        MetalDType dtype;
        std::vector<int64_t> input_shape;
        int64_t axis = 0;
        std::vector<size_t> split_sizes;
        std::vector<uint64_t> axis_offsets;
        uint64_t inner = 1;
        uint32_t element_type = static_cast<uint32_t>(ov::element::Type_t::dynamic);
    } split;
    struct ConcatDesc {
        MetalDType dtype;
        int64_t axis = 0;
        uint64_t outer = 0;
        uint64_t inner = 0;
        std::vector<uint64_t> axis_sizes;
        std::vector<uint64_t> axis_offsets;
    } concat;
    struct InterpolateDesc {
        MetalDType dtype;
        uint32_t N = 0;
        uint32_t C = 0;
        uint32_t H_in = 0;
        uint32_t W_in = 0;
        uint32_t H_out = 0;
        uint32_t W_out = 0;
        float scale_h = 1.f;
        float scale_w = 1.f;
        bool nearest = true;   // nearest or linear
        bool align_corners = false;
    } interpolate;
    // Parameters for BatchNorm (gamma, beta, mean, var) flattened; length = 4*C + 1 (eps sentinel).
    std::vector<float> bn_params;
    // Variadic inputs (used by ops like Concat)
    std::vector<KernelTensor*> inputs;
};

struct MetalKernelIR {
    std::vector<KernelTensor> tensors;
    std::vector<KernelOp> ops;
};

}  // namespace metal_plugin
}  // namespace ov
