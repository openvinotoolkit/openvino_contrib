// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_utils.hpp"

namespace ArmPlugin {
namespace pass {

using namespace ov;

static std::vector<int> nchw_to_nhwc{0, 2, 3, 1};
static std::vector<int> nhwc_to_nchw{0, 3, 1, 2};

static std::vector<int> ncdhw_to_ndhwc{0, 2, 3, 4, 1};
static std::vector<int> ndhwc_to_ncdhw{0, 4, 1, 2, 3};

std::shared_ptr<ArmPlugin::opset::Transpose> transpose_on_input(const Output<Node>& input, size_t rank) {
    switch (rank) {
    case 4:
        return std::make_shared<ArmPlugin::opset::Transpose>(input,
                ArmPlugin::opset::Constant::create(element::i32, Shape{nchw_to_nhwc.size()}, nchw_to_nhwc));
    case 5:
        return std::make_shared<ArmPlugin::opset::Transpose>(input,
                ArmPlugin::opset::Constant::create(element::i32, Shape{ncdhw_to_ndhwc.size()}, ncdhw_to_ndhwc));
    default:
        IE_THROW() << "ConvertLayout: unsupported rank";
    }
}

std::shared_ptr<ArmPlugin::opset::Transpose> transpose_on_output(const Output<Node>& input, size_t rank) {
    switch (rank) {
    case 4:
        return std::make_shared<ArmPlugin::opset::Transpose>(input,
                ArmPlugin::opset::Constant::create(element::i32, Shape{nhwc_to_nchw.size()}, nhwc_to_nchw));
    case 5:
        return std::make_shared<ArmPlugin::opset::Transpose>(input,
                ArmPlugin::opset::Constant::create(element::i32, Shape{ndhwc_to_ncdhw.size()}, ndhwc_to_ncdhw));
    default:
        IE_THROW() << "ConvertLayout: unsupported rank";
    }
}

PartialShape transpose_output_shape(const std::shared_ptr<Node>& node, size_t rank) {
    const auto& shape = node->get_output_partial_shape(0);
    PartialShape new_output_shape;
    new_output_shape.reserve(rank);
    const auto& perm = rank == 4 ? nchw_to_nhwc : ncdhw_to_ndhwc;
    for (size_t i = 0; i < rank; i++) {
        new_output_shape.push_back(shape[perm[i]]);
    }
    return new_output_shape;
}

} // pass
} // ArmPlugin
