// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ArmPlugin {
namespace pass {

class ConvertPrecisionFP16ToFP32: public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ConvertPrecisionFP16ToFP32", "0");
    ConvertPrecisionFP16ToFP32() = default;;
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

}  // namespace pass
}  // namespace ArmPlugin
