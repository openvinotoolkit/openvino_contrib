// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include <transformations_visibility.hpp>

namespace ov::nvidia_gpu::pass {

class Convert2LSTMSequenceToBidirectionalLSTMSequence : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("Convert2LSTMSequenceToBidirectionalLSTMSequence", "0");
    Convert2LSTMSequenceToBidirectionalLSTMSequence();
};

class ConvertBidirectionalLSTMSequenceToBidirectionalLSTMSequenceOptimized : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertBidirectionalLSTMSequenceToBidirectionalLSTMSequenceOptimized", "0");
    ConvertBidirectionalLSTMSequenceToBidirectionalLSTMSequenceOptimized();
};

class BidirectionalSequenceComposition : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("BidirectionalSequenceComposition", "0");
    explicit BidirectionalSequenceComposition(std::shared_ptr<ov::pass::PassConfig> pass_config);
    bool run_on_model(const std::shared_ptr<ov::Model>& f) override;

private:
    std::shared_ptr<ov::pass::PassConfig> pass_config_;
};

}  // namespace ov::nvidia_gpu::pass
