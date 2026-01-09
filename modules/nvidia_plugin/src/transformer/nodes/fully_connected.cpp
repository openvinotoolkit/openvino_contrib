// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected.hpp"

namespace ov::nvidia_gpu::nodes {

namespace matmul {

ov::PartialShape validate_matmul_output_shape(const ov::PartialShape& arg0_shape,
                                              const ov::PartialShape& arg1_shape,
                                              bool transpose_a,
                                              bool transpose_b);
}

FullyConnected::FullyConnected(const ov::Output<Node>& A,
                               const ov::Output<Node>& B,
                               const ov::Output<Node>& C,
                               const bool& transpose_a,
                               const bool& transpose_b)
    : ov::op::Op(ov::OutputVector{A, B, C}), m_transpose_a{transpose_a}, m_transpose_b{transpose_b} {
    constructor_validate_and_infer_types();
}

bool FullyConnected::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("transpose_a", m_transpose_a);
    visitor.on_attribute("transpose_b", m_transpose_b);
    return true;
}

std::shared_ptr<ov::Node> FullyConnected::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<FullyConnected>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_transpose_a, m_transpose_b);
}

void FullyConnected::validate_and_infer_types() {
    ov::element::Type result_et;

    NODE_VALIDATION_CHECK(this,
                          ov::element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)),
                          "Arguments do not have the same element type (arg0 element type: ",
                          get_input_element_type(0),
                          ", arg1 element type: ",
                          get_input_element_type(1),
                          ").");
    NODE_VALIDATION_CHECK(this,
                          ov::element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(2)),
                          "Arguments do not have the same element type (arg0 element type: ",
                          get_input_element_type(0),
                          ", arg2 element type: ",
                          get_input_element_type(2),
                          ").");

    const auto& A_partial_shape = get_input_partial_shape(0);
    const auto& B_partial_shape = get_input_partial_shape(1);
    if (A_partial_shape.rank().is_static() && B_partial_shape.rank().is_static()) {
        ov::PartialShape output_shape;

        const bool transpose_a = get_transpose_a();
        const bool transpose_b = get_transpose_b();

        output_shape = matmul::validate_matmul_output_shape(A_partial_shape, B_partial_shape, transpose_a, transpose_b);

        set_output_type(0, result_et, output_shape);
    } else {
        set_output_type(0, result_et, ov::PartialShape::dynamic());
    }
}

namespace matmul {

ov::PartialShape validate_matmul_output_shape(const ov::PartialShape& arg0_shape,
                                              const ov::PartialShape& arg1_shape,
                                              bool transpose_a,
                                              bool transpose_b) {
    auto arg0_rank = arg0_shape.rank().get_length();
    auto arg1_rank = arg1_shape.rank().get_length();

    OPENVINO_ASSERT((arg0_rank != 0 && arg1_rank != 0), "Scalars are not supported as MatMul inputs.");

    // Temporary Dimension vectors to calculate output shape
    std::vector<ov::Dimension> arg0_shape_tmp(arg0_shape);
    std::vector<ov::Dimension> arg1_shape_tmp(arg1_shape);

    // 1. Applying transpositions specified by optional `transpose_a` and `transpose_b`
    // Only two right-most dimensions are swapped, other dimensions remain the same.
    // Transpose attributes are ignored for 1D tensors.
    if (transpose_a && arg0_rank > 1) {
        std::swap(arg0_shape_tmp[arg0_rank - 2], arg0_shape_tmp[arg0_rank - 1]);
    }
    if (transpose_b && arg1_rank > 1) {
        std::swap(arg1_shape_tmp[arg1_rank - 2], arg1_shape_tmp[arg1_rank - 1]);
    }

    // 2. One-dimensional tensors unsqueezing is applied to each input independently.
    if (arg0_rank == 1) {
        // If the first input is 1D tensor, it is unsqueezed to 2D tensor (row vector)
        // by adding axes with size 1 at ROW_INDEX_DIM, to the left of the shape.
        // For example {S} will be reshaped to {1, S}.
        arg0_shape_tmp.insert(arg0_shape_tmp.begin(), 1);
        arg0_rank = arg0_shape_tmp.size();
    }
    if (arg1_rank == 1) {
        // If the second input is 1D tensor, it is unsqueezed to 2D tensor (column vector)
        // by adding axes with size 1 at COL_INDEX_DIM, to the right of the shape.
        // For example {S} will be reshaped to {S, 1}.
        arg1_shape_tmp.insert(arg1_shape_tmp.end(), 1);
        arg1_rank = arg1_shape_tmp.size();
    }

    // Check matrices dimensions compatibility,
    // COL_INDEX_DIM of the first matrix has to match ROW_INDEX_DIM of the second matrix.
    // Error is not thrown for dynamic dimensions bounds without intersection
    // to ensure MatMul backward compatibility.
    auto merged_dimension = ov::Dimension::dynamic();
    auto arg0_col_dim = arg0_shape_tmp[arg0_rank - 1];
    auto arg1_row_dim = arg1_shape_tmp[arg1_rank - 2];
    OPENVINO_ASSERT(ov::Dimension::merge(merged_dimension, arg0_col_dim, arg1_row_dim) || arg0_col_dim.is_dynamic() ||
                        arg1_row_dim.is_dynamic(),
                    "Incompatible MatMul matrix dimension. ",
                    "First input dimension=",
                    arg0_col_dim,
                    " at COL_INDEX_DIM=",
                    (arg0_rank - 1),
                    " doesn't match the second input dimension=",
                    arg1_row_dim,
                    " at ROW_INDEX_DIM=",
                    (arg1_rank - 2));

    // 3. If ranks of input arguments are different after steps 1 and 2,
    // the smaller tensor is unsqueezed from the left side of the shape
    // by necessary number of axes to make both shapes of the same rank.
    if (arg0_rank < arg1_rank)
        arg0_shape_tmp.insert(arg0_shape_tmp.begin(), arg1_rank - arg0_rank, 1);
    else if (arg0_rank > arg1_rank)
        arg1_shape_tmp.insert(arg1_shape_tmp.begin(), arg0_rank - arg1_rank, 1);
    // Both arg0_shape_tmp and arg1_shape_tmp have identical size now
    auto max_rank = arg0_shape_tmp.size();
    std::vector<ov::Dimension> output_shape(max_rank);

    // 4. Usual rules of the broadcasting are applied for batch dimensions.
    // Broadcast all batches (last two dimensions represent matrix),
    // expand dim with value 1 to bigger dim if dimensions are not equal.
    for (auto i = 0; i < max_rank - 2; i++) {
        auto min_dim_val = std::min(arg0_shape_tmp[i].get_min_length(), arg1_shape_tmp[i].get_min_length());

        // If both dimensions don't have 1 in range, usual merge is enough.
        if (min_dim_val > 1) {
            // Error is not thrown for dynamic dimensions bounds without intersection
            // to ensure MatMul backward compatibility.
            // Instead fully dynamic dimension is set as default for such a case.
            auto merged_dimension = ov::Dimension::dynamic();
            OPENVINO_ASSERT(ov::Dimension::merge(merged_dimension, arg0_shape_tmp[i], arg1_shape_tmp[i]) ||
                                arg0_shape_tmp[i].is_dynamic() || arg1_shape_tmp[i].is_dynamic(),
                            "Incompatible MatMul batch dimension. ",
                            "Can't merge first input dimension=",
                            arg0_shape_tmp[i],
                            " with second input dimension=",
                            arg1_shape_tmp[i],
                            " at index=",
                            i);

            output_shape[i] = merged_dimension;
        } else {
            // Dimension with value 1 can be expanded to any bigger.
            ov::Dimension::value_type lower_bound;  // The lowest possible value of output dimension
            ov::Dimension::value_type upper_bound;  // The highest possible value of output dimension

            // Output dimension lower_bound is a maximum of
            // corresponding input dimensions lower bounds.
            lower_bound = std::max(arg0_shape_tmp[i].get_min_length(), arg1_shape_tmp[i].get_min_length());
            if (lower_bound <= 1) {
                // If both of the dimensions have 1 in range, output dimension upper_bound
                // is a maximum of corresponding input dimensions upper bounds.
                upper_bound = std::max(arg0_shape_tmp[i].get_interval().get_max_val(),
                                       arg1_shape_tmp[i].get_interval().get_max_val());
            } else {
                // Otherwise output dimension upper_bound is same as upper bound of
                // the dimension without 1 in range.
                upper_bound = arg0_shape_tmp[i].get_min_length() <= 1 ? arg1_shape_tmp[i].get_max_length()
                                                                      : arg0_shape_tmp[i].get_max_length();
            }
            output_shape[i] = ov::Dimension(lower_bound, upper_bound);
        }
    }

    // In output_shape replace 2 last axes with ROW_INDEX_DIM from arg0 matrix
    // and COL_INDEX_DIM from arg1 matrix.
    output_shape.at(output_shape.size() - 2) = arg0_shape_tmp.at(arg0_shape_tmp.size() - 2);
    output_shape.at(output_shape.size() - 1) = arg1_shape_tmp.at(arg1_shape_tmp.size() - 1);

    // 5. Removing the temporary axes from originally 1D tensors.
    // Output shape of two 1D tensors multiplication will be a 0D tensor (scalar).
    if (arg0_shape.rank().get_length() == 1) {
        // arg0 input temporary axis inserted at ROW_INDEX_DIM is removed
        output_shape.erase(output_shape.begin() + output_shape.size() - 2);
    }
    if (arg1_shape.rank().get_length() == 1) {
        // arg1 input temporary axis inserted at COL_INDEX_DIM is removed
        output_shape.erase(output_shape.begin() + output_shape.size() - 1);
    }

    return ov::PartialShape(output_shape);
}

}  // namespace matmul

}  // namespace ov::nvidia_gpu::nodes
