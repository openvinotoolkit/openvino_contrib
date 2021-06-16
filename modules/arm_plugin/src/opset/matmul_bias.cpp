// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <vector>

#include "matmul_bias.hpp"

#include "ngraph/util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;
using namespace ArmPlugin;

NGRAPH_RTTI_DEFINITION(opset::MatMulBias, "MatMulBias", 0);

opset::MatMulBias::~MatMulBias() {}

opset::MatMulBias::MatMulBias(const ngraph::Output<ngraph::Node>& data,
                              const ngraph::Output<ngraph::Node>& weights,
                              const ngraph::Output<ngraph::Node>& bias,
                              const bool& transpose_b)
    : MatMul{data, weights, false, transpose_b}, m_transpose_b{transpose_b} {
    set_argument(2, bias);
    constructor_validate_and_infer_types();
}

shared_ptr<Node> opset::MatMulBias::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<MatMulBias>(new_args.at(0), new_args.at(1), new_args.at(2), m_transpose_b);
}
