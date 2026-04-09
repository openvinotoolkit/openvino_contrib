// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/cc/pass/itt.hpp"
#include "matmul_transformations.hpp"

#include <cuda_op_buffers_extractor.hpp>
#include <gsl/span_ext>
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/transpose.hpp"

using namespace ov::pass::pattern;

namespace ov::nvidia_gpu::pass {
template <typename T>
bool verify_permutation(std::shared_ptr<ov::op::v0::Constant> permConstant) {
    const auto perm2D = std::vector<T>{1, 0};
    const auto perm3D = std::vector<T>{0, 2, 1};
    const auto perm4D = std::vector<T>{0, 1, 3, 2};
    const auto perm5D = std::vector<T>{0, 1, 2, 4, 3};
    auto span = gsl::make_span(
        static_cast<const T *>(permConstant->get_data_ptr()),
        ov::nvidia_gpu::OperationBuffersExtractor::GetTensorByteSize(permConstant->output(0)) / sizeof(T));
    switch (span.size()) {
        case 2:
            return span == gsl::make_span(perm2D);
        case 3:
            return span == gsl::make_span(perm3D);
        case 4:
            return span == gsl::make_span(perm4D);
        case 5:
            return span == gsl::make_span(perm5D);
        default:
            return false;
    }
}

bool verify_permutation(std::shared_ptr<ov::op::v0::Constant> permConstant) {
    using ov::element::Type_t;
    switch (permConstant->get_element_type()) {
        case Type_t::i8:
            return verify_permutation<std::int8_t>(permConstant);
        case Type_t::i16:
            return verify_permutation<std::int16_t>(permConstant);
        case Type_t::i32:
            return verify_permutation<std::int32_t>(permConstant);
        case Type_t::i64:
            return verify_permutation<std::int64_t>(permConstant);
        case Type_t::u8:
            return verify_permutation<std::uint8_t>(permConstant);
        case Type_t::u16:
            return verify_permutation<std::uint16_t>(permConstant);
        case Type_t::u32:
            return verify_permutation<std::uint32_t>(permConstant);
        case Type_t::u64:
            return verify_permutation<std::uint64_t>(permConstant);
        default:
            return false;
    }
}

bool fuse_transpose_with_matmul(Matcher &m) {
    auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(m.get_match_root());
    if (!matmul) {
        return false;
    }

    auto transpose0 =
        std::dynamic_pointer_cast<ov::op::v1::Transpose>(matmul->input(0).get_source_output().get_node_shared_ptr());
    auto transpose1 =
        std::dynamic_pointer_cast<ov::op::v1::Transpose>(matmul->input(1).get_source_output().get_node_shared_ptr());
    if (!transpose0 && !transpose1) {
        return false;
    }

    size_t input_idx;
    std::shared_ptr<ov::op::v1::Transpose> transpose;
    if (transpose0) {
        input_idx = 0;
        transpose = transpose0;
    } else {
        input_idx = 1;
        transpose = transpose1;
    }

    auto permConstant =
        std::dynamic_pointer_cast<ov::op::v0::Constant>(transpose->input(1).get_source_output().get_node_shared_ptr());
    if (!verify_permutation(permConstant)) {
        return false;
    }

    Output<Node> a;
    Output<Node> b;
    bool transpose_a = matmul->get_transpose_a();
    bool transpose_b = matmul->get_transpose_b();
    if (0 == input_idx) {
        a = transpose->input(0).get_source_output();
        b = matmul->input(1).get_source_output();
        transpose_a = !transpose_a;
    } else {
        a = matmul->input(0).get_source_output();
        b = transpose->input(0).get_source_output();
        transpose_b = !transpose_b;
    }

    auto newMatMul = std::make_shared<ov::op::v0::MatMul>(a, b, transpose_a, transpose_b);

    newMatMul->set_friendly_name(matmul->get_friendly_name());

    ov::copy_runtime_info({transpose, matmul}, newMatMul);
    ov::replace_node(matmul, newMatMul);

    return true;
}

TransposeMatMulTransformation::TransposeMatMulTransformation() {
    MATCHER_SCOPE(TransposeMatMulTransformation);
    auto matmul = wrap_type<ov::op::v0::MatMul>({any_input(), any_input()});
    matcher_pass_callback callback = [](Matcher &m) { return fuse_transpose_with_matmul(m); };

    auto m = std::make_shared<Matcher>(matmul, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::nvidia_gpu::pass
