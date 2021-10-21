// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph::pass {

class Convert2LSTMSequenceToBidirectionalLSTMSequence : public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    Convert2LSTMSequenceToBidirectionalLSTMSequence();
};

class BidirectionalSequenceComposition : public ngraph::pass::FunctionPass {
public:
    NGRAPH_RTTI_DECLARATION;
    explicit BidirectionalSequenceComposition(std::shared_ptr<PassConfig> pass_config);
    bool run_on_function(std::shared_ptr<ngraph::Function> f) override;

private:
    std::shared_ptr<PassConfig> pass_config_;
};

}  // namespace ngraph::pass
