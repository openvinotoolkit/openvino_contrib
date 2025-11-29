// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ov {
class Model;
namespace metal_plugin {

enum class KernelOpKind { ElementwiseAdd, MatMul };

struct KernelTensor {
    std::string name;
    std::vector<int64_t> shape;
};

struct KernelOp {
    KernelOpKind kind;
    KernelTensor* input0 = nullptr;
    KernelTensor* input1 = nullptr;
    KernelTensor* output = nullptr;
    // MatMul-specific dims (M x K) * (K x N) = (M x N)
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
};

struct MetalKernelIR {
    std::vector<KernelTensor> tensors;
    std::vector<KernelOp> ops;
};

MetalKernelIR build_kernel_ir_for_add(const std::shared_ptr<const ov::Model>& model);
MetalKernelIR build_kernel_ir_for_matmul(const std::shared_ptr<const ov::Model>& model);

}  // namespace metal_plugin
}  // namespace ov
