// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_conv_biasadd_activation.hpp"

#include <exec_graph_info.hpp>
#include <ngraph/node.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <transformations/utils/utils.hpp>

#include "nodes/cuda_plugin_custom_node_types.hpp"
#include "nodes/fused_convolution.hpp"
#include "nodes/fused_convolution_backprop_data.hpp"

using namespace ngraph;

using ActivationMode = CUDAPlugin::nodes::ActivationMode;
using FusedConv = CUDAPlugin::nodes::FusedConvolution;
using FusedConvBackpropData = CUDAPlugin::nodes::FusedConvBackpropData;

template <class A, class B>
std::pair<std::shared_ptr<A>, std::shared_ptr<B>> parse_eltwise_inputs(std::shared_ptr<ngraph::Node> node) {
    auto eltwise = std::dynamic_pointer_cast<A>(node->input(0).get_source_output().get_node_shared_ptr());
    auto constant = std::dynamic_pointer_cast<B>(node->input(1).get_source_output().get_node_shared_ptr());

    if (!eltwise) {
        eltwise = std::dynamic_pointer_cast<A>(node->input(1).get_source_output().get_node_shared_ptr());
        constant = std::dynamic_pointer_cast<B>(node->input(0).get_source_output().get_node_shared_ptr());
    }

    if (!eltwise || !constant) {
        return {nullptr, nullptr};
    }

    return {eltwise, constant};
}

bool fuse_convolution_with_biasadd(ngraph::pattern::Matcher &m) {
    auto eltwise = m.get_match_root();
    auto [m_conv, m_const] = parse_eltwise_inputs<ngraph::opset1::Convolution, ngraph::opset1::Constant>(eltwise);
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

    constexpr auto conv_bias_rank_min{3};
    constexpr auto conv_bias_rank_max{5};
    const auto &bias_shape = bias.get_shape();
    const auto bias_rank = bias_shape.size();
    if (bias_rank < conv_bias_rank_min || bias_rank > conv_bias_rank_max) {
        return false;
    }

    const auto num_spatial_dims = m_conv->get_output_shape(0).size() - 2;
    const auto nchw_channel_dim_reverse_offset = num_spatial_dims + 1;
    const auto output_shape = m_conv->get_output_shape(0);
    if (bias_shape.at(bias_shape.size() - nchw_channel_dim_reverse_offset) !=
        output_shape.at(output_shape.size() - nchw_channel_dim_reverse_offset)) {
        return false;
    }

    if (num_spatial_dims == 3) {
        return false;  // NOTE: 3D convolution fusing was disabled due to 3d_unet bad performance
    }

    auto fused_conv = std::make_shared<FusedConv>(data,                      //
                                                  filters,                   //
                                                  bias,                      //
                                                  m_conv->get_strides(),     //
                                                  m_conv->get_pads_begin(),  //
                                                  m_conv->get_pads_end(),    //
                                                  m_conv->get_dilations(),   //
                                                  m_conv->get_auto_pad(),    //
                                                  ActivationMode::NO_ACTIVATION);
    ngraph::Output<ngraph::Node> new_conv(fused_conv);

    fused_conv->set_friendly_name(eltwise->get_friendly_name());

    ngraph::copy_runtime_info({m_conv, eltwise}, new_conv.get_node_shared_ptr());

    const std::string originalLayers = eltwise->get_friendly_name() + "," + m_conv->get_friendly_name();
    fused_conv->get_rt_info()[ExecGraphInfoSerialization::ORIGINAL_NAMES] =
        std::make_shared<ngraph::VariantWrapper<std::string>>(originalLayers);

    ngraph::replace_node(m.get_match_root(), new_conv.get_node_shared_ptr());
    return true;
}

std::pair<std::shared_ptr<FusedConv>, std::shared_ptr<Node>> parse_fusedconv_inputs(std::shared_ptr<ngraph::Node> add) {
    std::shared_ptr<FusedConv> fused_conv = nullptr;
    std::shared_ptr<Node> node = nullptr;
    node = add->input(1).get_source_output().get_node_shared_ptr();
    fused_conv = std::dynamic_pointer_cast<FusedConv>(add->input(0).get_source_output().get_node_shared_ptr());
    if (!fused_conv) {
        node = add->input(0).get_source_output().get_node_shared_ptr();
        fused_conv = std::dynamic_pointer_cast<FusedConv>(add->input(1).get_source_output().get_node_shared_ptr());
    }

    if (!fused_conv) {
        return {nullptr, nullptr};
    }

    return {fused_conv, node};
}

bool sink_add_to_fused_convolution(ngraph::pattern::Matcher &m) {
    auto add = std::dynamic_pointer_cast<opset1::Add>(m.get_match_root());
    auto [fused_conv, node] = parse_fusedconv_inputs(m.get_match_root());

    if (fused_conv->has_add_node() || fused_conv->get_activation() != ActivationMode::NO_ACTIVATION) {
        return false;
    }

    const ngraph::Output<Node> &data = fused_conv->input(0).get_source_output();
    const ngraph::Output<Node> &filters = fused_conv->input(1).get_source_output();
    const ngraph::Output<Node> &bias = fused_conv->input(2).get_source_output();

    auto fused_conv_add = std::make_shared<FusedConv>(data,                          //
                                                      filters,                       //
                                                      bias,                          //
                                                      node,                          //
                                                      fused_conv->get_strides(),     //
                                                      fused_conv->get_pads_begin(),  //
                                                      fused_conv->get_pads_end(),    //
                                                      fused_conv->get_dilations(),   //
                                                      fused_conv->get_auto_pad(),    //
                                                      ActivationMode::NO_ACTIVATION);
    ngraph::Output<ngraph::Node> fused_conv_add_output(fused_conv_add);

    fused_conv_add->set_friendly_name(add->get_friendly_name());
    ngraph::copy_runtime_info({node, fused_conv}, fused_conv_add);

    auto &rt_info = fused_conv->get_rt_info();
    if (rt_info.count(ExecGraphInfoSerialization::ORIGINAL_NAMES) > 0) {
        auto &rt_info_layer_names = rt_info[ExecGraphInfoSerialization::ORIGINAL_NAMES];
        const auto original_names = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(rt_info_layer_names);
        const std::string original_names_with_activation = add->get_friendly_name() + "," + original_names->get();
        rt_info_layer_names = std::make_shared<ngraph::VariantWrapper<std::string>>(original_names_with_activation);
    }

    ngraph::replace_node(fused_conv, fused_conv_add);
    ngraph::replace_node(m.get_match_root(), fused_conv_add);

    return true;
}

bool sink_activation_to_fused_convolution(ngraph::pattern::Matcher &m, ActivationMode activation) {
    auto activationNode = m.get_match_root();
    auto fused_conv =
        std::dynamic_pointer_cast<FusedConv>(activationNode->input(0).get_source_output().get_node_shared_ptr());
    if (fused_conv->get_activation() != ActivationMode::NO_ACTIVATION) {
        return false;
    }

    fused_conv->set_activation(activation);
    fused_conv->set_friendly_name(activationNode->get_friendly_name());

    auto &rt_info = fused_conv->get_rt_info();
    if (rt_info.count(ExecGraphInfoSerialization::ORIGINAL_NAMES) > 0) {
        auto &rt_info_layer_names = rt_info[ExecGraphInfoSerialization::ORIGINAL_NAMES];
        const auto original_names = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(rt_info_layer_names);
        const std::string original_names_with_activation =
            activationNode->get_friendly_name() + "," + original_names->get();
        rt_info_layer_names = std::make_shared<ngraph::VariantWrapper<std::string>>(original_names_with_activation);
    }

    ngraph::replace_node(m.get_match_root(), fused_conv);

    return true;
}

bool fuse_convolution_backprop_data_with_add(ngraph::pattern::Matcher &m) {
    auto add = std::dynamic_pointer_cast<opset1::Add>(m.get_match_root());
    auto [conv_backprop_data, add_constant] =
        parse_eltwise_inputs<ngraph::opset1::ConvolutionBackpropData, ngraph::opset1::Constant>(add);

    const auto conv_element_type = conv_backprop_data->get_input_element_type(0);
    const auto add_element_type = add_constant->get_output_element_type(0);
    if (conv_element_type != add_element_type) {
        return false;
    }

    const auto conv_output_shape = dynamic_cast<ngraph::Node *>(conv_backprop_data.get())->get_output_shape(0);
    const auto add_output_shape = add_constant->get_output_shape(0);
    const auto size = ngraph::element::Type(conv_element_type).size();
    const auto conv_in_bytes = size * ngraph::shape_size(conv_output_shape);
    const auto add_in_bytes = size * ngraph::shape_size(add_output_shape);
    if (add_in_bytes > conv_in_bytes) {
        return false;
    }

    const ngraph::Output<Node> &data = conv_backprop_data->input(0).get_source_output();
    const ngraph::Output<Node> &filters = conv_backprop_data->input(1).get_source_output();
    std::shared_ptr<FusedConvBackpropData> fused_conv_backprop_data_add;

    if (3 == conv_backprop_data->get_input_size()) {
        auto output_shape = conv_backprop_data->input(2).get_source_output();
        fused_conv_backprop_data_add =
            std::make_shared<FusedConvBackpropData>(data,                                  //
                                                    filters,                               //
                                                    output_shape,                          //
                                                    add_constant,                          //
                                                    conv_backprop_data->get_strides(),     //
                                                    conv_backprop_data->get_pads_begin(),  //
                                                    conv_backprop_data->get_pads_end(),    //
                                                    conv_backprop_data->get_dilations(),   //
                                                    conv_backprop_data->get_auto_pad(),    //
                                                    conv_backprop_data->get_output_padding());
    } else {
        fused_conv_backprop_data_add =
            std::make_shared<FusedConvBackpropData>(data,                                  //
                                                    filters,                               //
                                                    add_constant,                          //
                                                    conv_backprop_data->get_strides(),     //
                                                    conv_backprop_data->get_pads_begin(),  //
                                                    conv_backprop_data->get_pads_end(),    //
                                                    conv_backprop_data->get_dilations(),   //
                                                    conv_backprop_data->get_auto_pad(),    //
                                                    conv_backprop_data->get_output_padding());
    }

    ngraph::Output<ngraph::Node> fused_conv_backprop_data_add_output(fused_conv_backprop_data_add);

    fused_conv_backprop_data_add->set_friendly_name(add->get_friendly_name());
    ngraph::copy_runtime_info({conv_backprop_data, add}, fused_conv_backprop_data_add);

    auto &rt_info = conv_backprop_data->get_rt_info();
    if (rt_info.count(ExecGraphInfoSerialization::ORIGINAL_NAMES) > 0) {
        auto &rt_info_layer_names = rt_info[ExecGraphInfoSerialization::ORIGINAL_NAMES];
        const auto original_names = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(rt_info_layer_names);
        const std::string original_names_with_activation = add->get_friendly_name() + "," + original_names->get();
        rt_info_layer_names = std::make_shared<ngraph::VariantWrapper<std::string>>(original_names_with_activation);
    }

    ngraph::replace_node(add, fused_conv_backprop_data_add);
    ngraph::replace_node(conv_backprop_data, fused_conv_backprop_data_add);

    return true;
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::FuseConvolutionWithBiasAdd, "FuseConvolutionWithBiasAdd", 0);

ngraph::pass::FuseConvolutionWithBiasAdd::FuseConvolutionWithBiasAdd() {
    auto conv = ngraph::pattern::wrap_type<opset1::Convolution>(pattern::consumers_count(1));
    auto add = ngraph::pattern::wrap_type<opset1::Add>({conv, pattern::any_input()});

    matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) { return fuse_convolution_with_biasadd(m); };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, "FuseConvolutionWithBiasAdd");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::FuseConvolutionWithBiasaddAdd, "FuseConvolutionWithBiasAddAdd", 0);

pass::FuseConvolutionWithBiasaddAdd::FuseConvolutionWithBiasaddAdd() {
    auto fused_convolution = ngraph::pattern::wrap_type<FusedConv>(pattern::consumers_count(1));
    auto relu = ngraph::pattern::wrap_type<ngraph::op::v0::Relu>(pattern::consumers_count(2));
    auto add = ngraph::pattern::wrap_type<opset1::Add>({pattern::any_input(), fused_convolution});

    matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) { return sink_add_to_fused_convolution(m); };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, "FuseConvolutionWithBiasaddAdd");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::SinkReluToFusedConvolution, ngraph::pass::SinkReluToFusedConvolution::Name, 0);

pass::SinkReluToFusedConvolution::SinkReluToFusedConvolution() {
    auto fused_convolution = ngraph::pattern::wrap_type<FusedConv>(pattern::consumers_count(1));
    auto activation = ngraph::pattern::wrap_type<opset1::Relu>({fused_convolution});

    matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        return sink_activation_to_fused_convolution(m, ActivationMode::RELU);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(activation, pass::SinkReluToFusedConvolution::Name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::SinkSigmoidToFusedConvolution,
                       ngraph::pass::SinkSigmoidToFusedConvolution::Name,
                       0);

pass::SinkSigmoidToFusedConvolution::SinkSigmoidToFusedConvolution() {
    auto fused_convolution = ngraph::pattern::wrap_type<FusedConv>(pattern::consumers_count(1));
    auto activation = ngraph::pattern::wrap_type<opset1::Sigmoid>({fused_convolution});

    matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        return sink_activation_to_fused_convolution(m, ActivationMode::SIGMOID);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(activation, pass::SinkSigmoidToFusedConvolution::Name);
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::CudaFuseConvBiasAddActivation, "CudaFuseConvBiasAddActivation", 0);

ngraph::pass::CudaFuseConvBiasAddActivation::CudaFuseConvBiasAddActivation() {
    add_matcher<FuseConvolutionWithBiasAdd>();
    add_matcher<FuseConvolutionWithBiasaddAdd>();
    add_matcher<SinkReluToFusedConvolution>();
    add_matcher<SinkSigmoidToFusedConvolution>();
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::CudaFuseConvBackpropDataAdd, ngraph::pass::CudaFuseConvBackpropDataAdd::Name, 0);

ngraph::pass::CudaFuseConvBackpropDataAdd::CudaFuseConvBackpropDataAdd() {
    auto conv_backprop_data =
        ngraph::pattern::wrap_type<ngraph::op::v1::ConvolutionBackpropData>(pattern::consumers_count(1));
    auto add = ngraph::pattern::wrap_type<opset1::Add>({conv_backprop_data, pattern::any_input()});

    matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        return fuse_convolution_backprop_data_with_add(m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(add, ngraph::pass::CudaFuseConvBackpropDataAdd::Name);
    register_matcher(m, callback);
}
