// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "transformations/convert_mat_mul.hpp"

#include <numeric>

#include "opset/opset.hpp"
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ArmPlugin::pass::ConvertMatMulToGemm, "ConvertMatMulToGemm", 0);
ArmPlugin::pass::ConvertMatMulToGemm::ConvertMatMulToGemm() {
    auto matmul = pattern::wrap_type<opset::MatMul>({pattern::any_input(pattern::has_static_shape()),
                                                      pattern::any_input(pattern::has_static_shape())},
                                                     pattern::has_static_shape());

    matcher_pass_callback callback = [this](pattern::Matcher& m) {
        std::cout << "0" << std::endl;
        auto matmul = std::dynamic_pointer_cast<opset::MatMul>(m.get_match_root());
        if (!matmul) {
            std::cout << "1" << std::endl;
            return false;
        }

        auto input_a = matmul->input(0).get_source_output();
        auto input_b = matmul->input(1).get_source_output();

        auto shape_a = input_a.get_shape();
        auto shape_b = input_b.get_shape();
        auto output_shape = matmul->get_shape();

        auto fc_input_a = input_a, fc_input_b = input_b;
        NodeVector new_ops;

        if (shape_a.size() == 1) {
            // If the first input is 1D tensor, it is unsqueezed to 2D tensor (row vector)
            // by adding axes with size 1 at ROW_INDEX_DIM, to the left of the shape.
            // For example {S} will be reshaped to {1, S}.
            fc_input_a = std::make_shared<opset::Unsqueeze>(fc_input_a,
                                                                     opset::Constant::create(ov::element::i64, Shape{1}, {0}));
            shape_a = fc_input_a.get_shape();
            new_ops.push_back(fc_input_a.get_node_shared_ptr());
            // For 1D inputs transpose flag is expected to always act like `false`
            matmul->set_transpose_a(false);
        }
        if (shape_b.size() == 1) {
            // If the second input is 1D tensor, it is unsqueezed to 2D tensor (column vector)
            // by adding axes with size 1 at COL_INDEX_DIM, to the right of the shape.
            // For example {S} will be reshaped to {S, 1}.
            fc_input_b = std::make_shared<opset::Unsqueeze>(fc_input_b,
                                                                     opset::Constant::create(ov::element::i64, Shape{1}, {1}));
            shape_b = fc_input_b.get_shape();
            new_ops.push_back(fc_input_b.get_node_shared_ptr());
            // For 1D inputs transpose flag is expected to always act like `false`
            matmul->set_transpose_b(false);
        }

        // WA for IE that Gemm must have inputs with the same length.
        // If ranks of input arguments are still different,
        // the smaller tensor is unsqueezed from the left side of the shape
        // by necessary number of axes to make both shapes of the same rank.
        if (shape_a.size() < shape_b.size()) {
            // Reshape first input (fc_input_a)
            Shape reshape_shape(shape_b.size() - shape_a.size(), 1);
            reshape_shape.insert(reshape_shape.end(), shape_a.begin(), shape_a.end());
            fc_input_a = ov::op::util::reshapeTo(fc_input_a, reshape_shape);
            new_ops.push_back(fc_input_a.get_node_shared_ptr());
        } else if (shape_b.size() < shape_a.size()) {
            // Reshape second input (fc_input_b)
            Shape reshape_shape(shape_a.size() - shape_b.size(), 1);
            reshape_shape.insert(reshape_shape.end(), shape_b.begin(), shape_b.end());
            fc_input_b = ov::op::util::reshapeTo(fc_input_b, reshape_shape);
            new_ops.push_back(fc_input_b.get_node_shared_ptr());
        }

        auto gemm = matmul->clone_with_new_inputs({ fc_input_a, fc_input_b });
        new_ops.push_back(gemm);

        if (gemm->get_shape() != output_shape) {
            // This case is possible when one of the inputs has exactly 1 dimension (that is not supported by GEMM operation)
            // So to preserve output shape we insert additional reshape operation
            std::shared_ptr<Node> reshape_output;
            if (output_shape.size() == 0) {
                std::vector<int64_t> dim_indices(gemm->get_shape().size());
                std::iota(dim_indices.begin(), dim_indices.end(), 0);
                reshape_output = std::make_shared<opset::Squeeze>(gemm,
                                                                           opset::Constant::create(ov::element::i64, Shape{dim_indices.size()}, dim_indices));
            } else {
                reshape_output = ov::op::util::reshapeTo(gemm, output_shape);
            }

            new_ops.push_back(reshape_output);
            gemm->set_friendly_name(matmul->get_friendly_name() + "/gemm");
            reshape_output->set_friendly_name(matmul->get_friendly_name());
            copy_runtime_info(matmul, new_ops);
            replace_node(matmul, reshape_output);
        } else {
            gemm->set_friendly_name(matmul->get_friendly_name());
            copy_runtime_info(matmul, new_ops);
            replace_node(matmul, gemm);
        }

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(matmul, "ConvertMatMulToGemm");
    this->register_matcher(m, callback);
}