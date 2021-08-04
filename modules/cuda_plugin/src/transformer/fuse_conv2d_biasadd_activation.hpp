// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph::pass {

class FuseConvolution2DWithBiasAdd : public ngraph::pass::MatcherPass {
 public:
  NGRAPH_RTTI_DECLARATION;
  FuseConvolution2DWithBiasAdd();
};

class FuseConvolution2DWithBiasaddAdd : public ngraph::pass::MatcherPass {
 public:
  NGRAPH_RTTI_DECLARATION;
  FuseConvolution2DWithBiasaddAdd();
};

class SinkReluToFusedConvolution : public ngraph::pass::MatcherPass {
 public:
  static constexpr auto Name = "SinkReluToFusedConvolution";

  NGRAPH_RTTI_DECLARATION;
  SinkReluToFusedConvolution();
};

class SinkSigmoidToFusedConvolution : public ngraph::pass::MatcherPass {
 public:
  static constexpr auto Name = "SinkSigmoidToFusedConvolution";

  NGRAPH_RTTI_DECLARATION;
  SinkSigmoidToFusedConvolution();
};

class CudaFuseConv2DBiasAddActivation: public ngraph::pass::GraphRewrite {
 public:
  NGRAPH_RTTI_DECLARATION;
  CudaFuseConv2DBiasAddActivation();
};

class CudaFuseConvBackpropData2DAdd : public ngraph::pass::MatcherPass {
 public:
  static constexpr auto Name = "FuseConvBackpropData2DAdd";

  NGRAPH_RTTI_DECLARATION;
  CudaFuseConvBackpropData2DAdd();
};

}  // namespace ngraph::pass
