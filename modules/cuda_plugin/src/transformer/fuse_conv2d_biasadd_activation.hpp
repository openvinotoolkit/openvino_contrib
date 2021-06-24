// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph::pass {

class FuseConvolution2DWithBiasAdd;
class SinkReluToFusedConvolution;
class CudaFuseConv2DBiasAddActivation;

class FuseConvolution2DWithBiasAdd : public ngraph::pass::MatcherPass {
 public:
  NGRAPH_RTTI_DECLARATION;
  FuseConvolution2DWithBiasAdd();
};

class SinkReluToFusedConvolution : public ngraph::pass::MatcherPass {
 public:
  NGRAPH_RTTI_DECLARATION;
  SinkReluToFusedConvolution();
};

class CudaFuseConv2DBiasAddActivation: public ngraph::pass::GraphRewrite {
 public:
  NGRAPH_RTTI_DECLARATION;
  CudaFuseConv2DBiasAddActivation() {
    add_matcher<FuseConvolution2DWithBiasAdd>();
    add_matcher<SinkReluToFusedConvolution>();
  }
};

}  // namespace ngraph::pass
