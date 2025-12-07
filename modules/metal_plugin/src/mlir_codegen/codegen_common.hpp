// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kernel_ir/kernel_ir_common.hpp"
#include "mlir/IR/BuiltinOps.h"

namespace ov {
namespace metal_plugin {

struct BaseCodegenDesc {
    KernelOpKind kind{};
    ov::element::Type element_type = ov::element::f32;
};

struct MatMulCodegenDesc : BaseCodegenDesc {
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int64_t batch = 1;
    int64_t batch_a = 1;
    int64_t batch_b = 1;
    bool a_transpose = false;
    bool b_transpose = false;
    bool b_is_nk_layout = false;
};

struct Conv2DCodegenDesc : BaseCodegenDesc {
    uint32_t N = 0;
    uint32_t C_in = 0;
    uint32_t H = 0;
    uint32_t W = 0;
    uint32_t C_out = 0;
    uint32_t kH = 0;
    uint32_t kW = 0;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t padTop = 0;
    uint32_t padLeft = 0;
    uint32_t padBottom = 0;
    uint32_t padRight = 0;
    uint32_t outH = 0;
    uint32_t outW = 0;
    uint32_t groups = 1;
};

struct Conv3DCodegenDesc : BaseCodegenDesc {
    uint32_t N = 0;
    uint32_t C_in = 0;
    uint32_t D = 0;
    uint32_t H = 0;
    uint32_t W = 0;
    uint32_t C_out = 0;
    uint32_t kD = 0;
    uint32_t kH = 0;
    uint32_t kW = 0;
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
};

struct EltwiseCodegenDesc : BaseCodegenDesc {
    KernelOpKind eltwise_kind{};
    uint32_t num_elements = 0;
    bool is_broadcast = false;
    bool use_half_compute = false;
    std::vector<int64_t> out_shape;
    std::vector<int64_t> stride0;
    std::vector<int64_t> stride1;
};

struct Pool2DCodegenDesc : BaseCodegenDesc {
    uint32_t N = 0;
    uint32_t C = 0;
    uint32_t H = 0;
    uint32_t W = 0;
    uint32_t kH = 0;
    uint32_t kW = 0;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t padTop = 0;
    uint32_t padLeft = 0;
    uint32_t padBottom = 0;
    uint32_t padRight = 0;
    uint32_t outH = 0;
    uint32_t outW = 0;
    bool is_avg = false;
    bool exclude_pad = true;
};

struct SoftmaxCodegenDesc : BaseCodegenDesc {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t inner = 1;
};

struct InterpolateCodegenDesc : BaseCodegenDesc {
    uint32_t N = 0;
    uint32_t C = 0;
    uint32_t H_in = 0;
    uint32_t W_in = 0;
    uint32_t H_out = 0;
    uint32_t W_out = 0;
    float scale_h = 1.f;
    float scale_w = 1.f;
    bool align_corners = false;
    bool nearest = true;  // false → bilinear
};

struct SplitCodegenDesc : BaseCodegenDesc {
    int64_t axis = 0;
    uint64_t inner = 1;
    uint64_t outer = 1;
    std::vector<int64_t> input_shape;
    std::vector<size_t> split_sizes;
};

struct ConcatCodegenDesc : BaseCodegenDesc {
    uint64_t outer = 0;
    uint64_t inner = 0;
    uint64_t axis_offset = 0;
    uint64_t axis_len = 0;
};

// Per-op emitters (msl generation only; MLIR module is currently unused for non-MatMul stubs).
std::string generate_msl_for_matmul(const MatMulCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_conv2d(const Conv2DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_conv3d(const Conv3DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_eltwise(const EltwiseCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_maxpool2d(const Pool2DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_avgpool2d(const Pool2DCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_softmax(const SoftmaxCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_concat(const ConcatCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_interpolate(const InterpolateCodegenDesc& desc, mlir::ModuleOp module);
std::string generate_msl_for_split(const SplitCodegenDesc& desc, mlir::ModuleOp module);

// Dispatcher by kind.
std::string generate_msl_from_mlir(mlir::ModuleOp module, const BaseCodegenDesc& desc);

}  // namespace metal_plugin
}  // namespace ov
