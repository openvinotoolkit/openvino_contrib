// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_conv2d_biasadd_activation.hpp"

#include <exec_graph_info.hpp>
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <transformations/utils/utils.hpp>

#include "nodes/cuda_plugin_custom_node_types.hpp"
#include "nodes/convolution2d_biasadd_activation.hpp"

using namespace ngraph;
using FusedConv = CUDAPlugin::nodes::Conv2DBiasAddActivation;

template <class A, class B>
std::pair<std::shared_ptr<A>, std::shared_ptr<B>> parse_eltwise_inputs(
    std::shared_ptr<ngraph::Node> node) {
  auto eltwise = std::dynamic_pointer_cast<A>(
      node->input(0).get_source_output().get_node_shared_ptr());
  auto constant = std::dynamic_pointer_cast<B>(
      node->input(1).get_source_output().get_node_shared_ptr());

  if (!eltwise) {
    eltwise = std::dynamic_pointer_cast<A>(
        node->input(1).get_source_output().get_node_shared_ptr());
    constant = std::dynamic_pointer_cast<B>(
        node->input(0).get_source_output().get_node_shared_ptr());
  }

  if (!eltwise || !constant) {
    return {nullptr, nullptr};
  }

  return {eltwise, constant};
}

bool fuse_convolution2d_with_biasadd(ngraph::pattern::Matcher &m) {
  auto eltwise = m.get_match_root();
  auto [m_conv, m_const] =
      parse_eltwise_inputs<ngraph::opset1::Convolution,
                           ngraph::opset1::Constant>(eltwise);
  if (!m_conv || !m_const) {
    return false;
  }

  if (m_conv->inputs().size() != 2) {
    return false;
  }

  if (std::dynamic_pointer_cast<ngraph::opset1::Add>(eltwise) == nullptr) {
    return false;
  }

  const ngraph::Output<Node> &data = m_conv->input(0).get_source_output();
  const ngraph::Output<Node> &filters = m_conv->input(1).get_source_output();
  const ngraph::Output<Node> &bias = m_const->output(0);

  constexpr auto conv2d_bias_rank_min{3};
  constexpr auto conv2d_bias_rank_max{4};
  const auto& bias_shape = bias.get_shape();
  const auto bias_rank = bias_shape.size();
  if (bias_rank < conv2d_bias_rank_min || bias_rank > conv2d_bias_rank_max) {
    return false;
  }
  constexpr auto nchw_channel_dim_reverse_offset = 3;
  const auto output_shape = m_conv->get_output_shape(0);
  if (bias_shape.at(bias_shape.size() - nchw_channel_dim_reverse_offset) !=
      output_shape.at(output_shape.size() - nchw_channel_dim_reverse_offset)) {
    return false;
  }

  auto fused_conv = std::make_shared<FusedConv>(
      data,                      //
      filters,                   //
      bias,                      //
      m_conv->get_strides(),     //
      m_conv->get_pads_begin(),  //
      m_conv->get_pads_end(),    //
      m_conv->get_dilations(),   //
      m_conv->get_auto_pad(),    //
      CUDAPlugin::nodes::ActivationMode::NO_ACTIVATION);
  ngraph::Output<ngraph::Node> new_conv(fused_conv);

  fused_conv->set_friendly_name(eltwise->get_friendly_name());

  ngraph::copy_runtime_info({m_conv, eltwise}, new_conv.get_node_shared_ptr());

  const std::string originalLayers =
      eltwise->get_friendly_name() + "," + m_conv->get_friendly_name();
  fused_conv->get_rt_info()[ExecGraphInfoSerialization::ORIGINAL_NAMES] =
      std::make_shared<ngraph::VariantWrapper<std::string>>(originalLayers);

  ngraph::replace_node(m.get_match_root(), new_conv.get_node_shared_ptr());
  return true;
}

bool sink_relu_to_fused_convolution(ngraph::pattern::Matcher &m) {
  auto relu = m.get_match_root();
  auto fused_conv = std::dynamic_pointer_cast<FusedConv>(
      relu->input(0).get_source_output().get_node_shared_ptr());

  fused_conv->set_activation(CUDAPlugin::nodes::ActivationMode::RELU);
  fused_conv->set_friendly_name(relu->get_friendly_name());

  auto rt_info_layer_names =
      fused_conv->get_rt_info()[ExecGraphInfoSerialization::ORIGINAL_NAMES];
  const auto original_names =
      std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(
          rt_info_layer_names);
  const std::string original_names_with_activation =
      relu->get_friendly_name() + "," + original_names->get();
  rt_info_layer_names = std::make_shared<ngraph::VariantWrapper<std::string>>(
      original_names_with_activation);

  ngraph::replace_node(m.get_match_root(), fused_conv);
  return true;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::FuseConvolution2DWithBiasAdd,
                       "FuseConvolutionWithBiasAdd", 0);

ngraph::pass::FuseConvolution2DWithBiasAdd::FuseConvolution2DWithBiasAdd() {
  auto conv = ngraph::pattern::wrap_type<opset1::Convolution>(
      pattern::consumers_count(1));
  auto add =
      ngraph::pattern::wrap_type<opset1::Add>({conv, pattern::any_input()});

  matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
    return fuse_convolution2d_with_biasadd(m);
  };

  auto m = std::make_shared<ngraph::pattern::Matcher>(
      add, "FuseConvolutionWithBiasAdd");
  register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::SinkReluToFusedConvolution,
                       "SinkReluToFusedConvolution", 0);

pass::SinkReluToFusedConvolution::SinkReluToFusedConvolution() {
  auto fused_convolution =
      ngraph::pattern::wrap_type<FusedConv>(pattern::consumers_count(1));
  auto activation =
      ngraph::pattern::wrap_type<opset1::Relu>({fused_convolution});

  matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
    return sink_relu_to_fused_convolution(m);
  };

  auto m = std::make_shared<ngraph::pattern::Matcher>(
      activation, "SinkReluToFusedConvolution");
  register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::CudaFuseConv2DBiasAddActivation,
                       "CudaFuseConvBiasAddActivation", 0);
