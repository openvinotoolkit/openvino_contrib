// Copyright (C) 2018-2022 Intel Corporation
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

class AddMeanSubtract;

}  // namespace pass
}  // namespace nvidia_gpu
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief Add `meanValue` or `meanImage` preprocessing to input nodes
 */
class ov::nvidia_gpu::pass::AddMeanSubtract : public ov::pass::MatcherPass {
public:
    using MeanMap = std::map<std::string, std::shared_ptr<ov::op::v0::Constant>>;

    OPENVINO_RTTI("AddMeanSubtract", "0");
    explicit AddMeanSubtract(const MeanMap& inputInfoMap);
};
