// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::ArmStridedSlice::~ArmStridedSlice() {}

opset::ArmStridedSlice::ArmStridedSlice(const ngraph::Output<ngraph::Node>& data,
                                        const ngraph::Output<ngraph::Node>& begin,
                                        const ngraph::Output<ngraph::Node>& end,
                                        const ngraph::Output<ngraph::Node>& strides,
                                        const std::vector<int64_t>& begin_mask,
                                        const std::vector<int64_t>& end_mask,
                                        const std::vector<int64_t>& new_axis_mask,
                                        const std::vector<int64_t>& shrink_axis_mask,
                                        const std::vector<int64_t>& ellipsis_mask)
    : StridedSlice{data, begin, end, strides, begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask},
    m_begin_mask(begin_mask), m_end_mask(end_mask), m_new_axis_mask(new_axis_mask),
    m_shrink_axis_mask(shrink_axis_mask), m_ellipsis_mask(ellipsis_mask) {
    constructor_validate_and_infer_types();
}

opset::ArmStridedSlice::ArmStridedSlice(const ngraph::Output<ngraph::Node>& data,
                                        const ngraph::Output<ngraph::Node>& begin,
                                        const ngraph::Output<ngraph::Node>& end,
                                        const std::vector<int64_t>& begin_mask,
                                        const std::vector<int64_t>& end_mask,
                                        const std::vector<int64_t>& new_axis_mask,
                                        const std::vector<int64_t>& shrink_axis_mask,
                                        const std::vector<int64_t>& ellipsis_mask)
    : StridedSlice{data, begin, end, begin_mask, end_mask, new_axis_mask, shrink_axis_mask, ellipsis_mask},
    m_begin_mask(begin_mask), m_end_mask(end_mask), m_new_axis_mask(new_axis_mask),
    m_shrink_axis_mask(shrink_axis_mask), m_ellipsis_mask(ellipsis_mask) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ArmPlugin::opset::ArmStridedSlice::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto num_args = new_args.size();
    if (num_args == 4) {
        return std::make_shared<ArmStridedSlice>(new_args.at(0),
                                                 new_args.at(1),
                                                 new_args.at(2),
                                                 new_args.at(3),
                                                 m_begin_mask,
                                                 m_end_mask,
                                                 m_new_axis_mask,
                                                 m_shrink_axis_mask,
                                                 m_ellipsis_mask);
    } else {
        throw ngraph_error("Unsupported number of arguments for ArmStridedSlice operation");
    }
}
