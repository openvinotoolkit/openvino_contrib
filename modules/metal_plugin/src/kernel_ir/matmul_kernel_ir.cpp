// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_ir/matmul_kernel_ir.hpp"

#include <sstream>

#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace metal_plugin {

MetalKernelIR build_kernel_ir_for_matmul(const std::shared_ptr<const ov::Model>& model) {
    MetalKernelIR ir;
    std::shared_ptr<const ov::op::v0::MatMul> matmul;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto mm = ov::as_type_ptr<const ov::op::v0::MatMul>(node)) {
            matmul = mm;
            break;
        }
    }
    OPENVINO_ASSERT(matmul, "MatMul builder: MatMul op not found");
    auto shape_a = matmul->get_input_shape(0);
    auto shape_b = matmul->get_input_shape(1);
    OPENVINO_ASSERT(!shape_a.empty() && !shape_b.empty(), "MatMul: empty shapes");
    OPENVINO_ASSERT(matmul->get_output_element_type(0) == ov::element::f32, "MatMul only supports f32");

    auto flatten_to_3d = [](const ov::Shape& s) {
        OPENVINO_ASSERT(s.size() >= 2 && s.size() <= 4, "MatMul: rank 2–4 supported");
        int64_t batch = 1;
        for (size_t i = 0; i + 2 < s.size(); ++i) batch *= static_cast<int64_t>(s[i]);
        int64_t m = static_cast<int64_t>(s[s.size() - 2]);
        int64_t k = static_cast<int64_t>(s[s.size() - 1]);
        return std::vector<int64_t>{batch, m, k};
    };

    auto a3 = flatten_to_3d(shape_a);  // [batch_a, M, K]
    auto b3 = flatten_to_3d(shape_b);  // [batch_b, K, N]

    auto shape_to_string = [](const ov::Shape& s) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < s.size(); ++i) {
            if (i) oss << ",";
            oss << s[i];
        }
        oss << "]";
        return oss.str();
    };

    const bool ta = matmul->get_transpose_a();
    const bool tb = matmul->get_transpose_b();

    const int64_t batch_a = a3[0];
    const int64_t batch_b = b3[0];
    const int64_t batch = std::max(batch_a, batch_b);
    OPENVINO_ASSERT(batch_a == batch_b || batch_a == 1 || batch_b == 1,
                    "MatMul: incompatible batch dimensions for broadcast");

    const int64_t M = ta ? a3[2] : a3[1];
    const int64_t K_a = ta ? a3[1] : a3[2];

    int64_t K_b = tb ? b3[2] : b3[1];
    int64_t N = tb ? b3[1] : b3[2];
    bool b_is_nk_layout = tb;
    if (!tb && K_b != K_a && b3[2] == K_a) {
        // fallback to NK layout detection
        K_b = b3[2];
        N = b3[1];
        b_is_nk_layout = true;
    }

    if (K_a != K_b) {
        std::cerr << "MatMul builder K mismatch: A=" << shape_to_string(shape_a)
                  << " B=" << shape_to_string(shape_b) << " (Ka=" << K_a << " Kb=" << K_b << ")" << std::endl;
    }
    OPENVINO_ASSERT(K_a == K_b, "MatMul: K mismatch: A last dim = ", K_a, " B penultimate dim = ", K_b);
    const int64_t K = K_a;

    KernelTensor a{"a", {a3.begin(), a3.end()}};
    KernelTensor b{"b", {b3.begin(), b3.end()}};
    KernelTensor c{"c", batch == 1 ? std::vector<int64_t>{M, N} : std::vector<int64_t>{batch, M, N}};

    ir.tensors.push_back(a);
    ir.tensors.push_back(b);
    ir.tensors.push_back(c);

    KernelOp op;
    op.kind = KernelOpKind::MatMul;
    op.input0 = &ir.tensors[0];
    op.input1 = &ir.tensors[1];
    op.output = &ir.tensors[2];
    op.M = M;
    op.N = N;
    op.K = K;
    op.batch = batch;
    op.batch_a = batch_a;
    op.batch_b = batch_b;
    ir.ops.push_back(op);
    ir.ops.back().b_is_nk_layout = b_is_nk_layout;
    ir.ops.back().a_transpose = ta;
    ir.ops.back().b_transpose = tb || b_is_nk_layout;
    return ir;
}

}  // namespace metal_plugin
}  // namespace ov
