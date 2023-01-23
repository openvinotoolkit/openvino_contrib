// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <ngraph/op/constant.hpp>
#include "openvino/pass/graph_rewrite.hpp"
#include <string>

#include "transformations_visibility.hpp"

namespace ov {
namespace nvidia_gpu {
namespace pass {

class AddStdScale;

}  // namespace pass
}  // namespace nvidia_gpu
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Add `stdScale` preprocessing to input nodes
 */
class ov::nvidia_gpu::pass::AddStdScale : public ov::pass::MatcherPass {
public:
    using ScaleMap = std::map<std::string, std::shared_ptr<ov::op::v0::Constant>>;

    OPENVINO_RTTI("AddStdScale", "0");
    explicit AddStdScale(const ScaleMap& inputInfoMap);
};
