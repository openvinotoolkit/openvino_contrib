// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::nvidia_gpu::pass {

class ReduceTransformation : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ReduceTransformation", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
};

class ReduceMaxTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReduceMaxTransformation", "0");
    ReduceMaxTransformation();
};

class ReduceMeanTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReduceMeanTransformation", "0");
    ReduceMeanTransformation();
};

class ReduceMinTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReduceMinTransformation", "0");
    ReduceMinTransformation();
};

class ReduceProdTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReduceProdTransformation", "0");
    ReduceProdTransformation();
};

class ReduceSumTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ReduceSumTransformation", "0");
    ReduceSumTransformation();
};

}  // namespace ov::nvidia_gpu::pass
