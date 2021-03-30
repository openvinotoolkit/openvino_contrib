// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"

namespace ArmPlugin {
namespace opset {

class ArmStridedSlice : public StridedSlice {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"ArmStridedSlice", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info; }
    ArmStridedSlice() = default;
    ~ArmStridedSlice() override;

    ArmStridedSlice(const ngraph::Output<ngraph::Node>& data,
                    const ngraph::Output<ngraph::Node>& begin,
                    const ngraph::Output<ngraph::Node>& end,
                    const ngraph::Output<ngraph::Node>& strides,
                    const std::vector<int64_t>& begin_mask,
                    const std::vector<int64_t>& end_mask,
                    const std::vector<int64_t>& new_axis_mask = std::vector<int64_t>{},
                    const std::vector<int64_t>& shrink_axis_mask = std::vector<int64_t>{},
                    const std::vector<int64_t>& ellipsis_mask = std::vector<int64_t>{});

    ArmStridedSlice(const ngraph::Output<ngraph::Node>& data,
                    const ngraph::Output<ngraph::Node>& begin,
                    const ngraph::Output<ngraph::Node>& end,
                    const std::vector<int64_t>& begin_mask,
                    const std::vector<int64_t>& end_mask,
                    const std::vector<int64_t>& new_axis_mask = std::vector<int64_t>{},
                    const std::vector<int64_t>& shrink_axis_mask = std::vector<int64_t>{},
                    const std::vector<int64_t>& ellipsis_mask = std::vector<int64_t>{});

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

protected:
    std::vector<int64_t> m_begin_mask;
    std::vector<int64_t> m_end_mask;
    std::vector<int64_t> m_new_axis_mask;
    std::vector<int64_t> m_shrink_axis_mask;
    std::vector<int64_t> m_ellipsis_mask;
};
}  // namespace opset
}  // namespace ArmPlugin
