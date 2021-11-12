// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>
#include <arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h>
#include "arm_converter/arm_converter.hpp"

namespace ArmPlugin {

enum ConvInput {Features, Weights, Bias};
template<typename Conv>
static auto ConvParameters(const Conv& node) {
    unsigned int pad_l    = node.get_pads_begin().at(D2::W);
    unsigned int pad_r    = node.get_pads_end().at(D2::W);
    unsigned int pad_t    = node.get_pads_begin().at(D2::H);
    unsigned int pad_b    = node.get_pads_end().at(D2::H);
    unsigned int stride_x = node.get_strides().at(D2::W);
    unsigned int stride_y = node.get_strides().at(D2::H);

    return std::make_pair(
        arm_compute::PadStrideInfo {stride_x, stride_y, pad_l, pad_r, pad_t, pad_b, arm_compute::DimensionRoundingType::FLOOR},
        arm_compute::Size2D {node.get_dilations().at(D2::W), node.get_dilations().at(D2::H)});
}

static arm_compute::ActivationLayerInfo GetActivationInfo(const ngraph::Node& node) {
    auto itInfo = node.get_rt_info().find("ActivationLayerInfo");
    if (itInfo != node.get_rt_info().end()) {
        return safe_cast<ngraph::VariantWrapper<arm_compute::ActivationLayerInfo>>(itInfo->second)->get();
    } else {
        return {};
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmConvolution& node) {
    arm_compute::PadStrideInfo conv_info;
    arm_compute::Size2D dilation;
    std::tie(conv_info, dilation) = ConvParameters(node);
    if (node.get_input_size() == 3) {
        return MakeConversion<arm_compute::NEConvolutionLayer>(
            node.input(Features), node.input(Weights), node.input(Bias), node.output(0),
            conv_info, arm_compute::WeightsInfo{}, dilation, GetActivationInfo(node));
    } else {
        return MakeConversion<arm_compute::NEConvolutionLayer>(
            node.input(Features), node.input(Weights), nullptr, node.output(0),
            conv_info, arm_compute::WeightsInfo{}, dilation, GetActivationInfo(node));
    }
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::ArmGroupConvolution& node) {
    arm_compute::PadStrideInfo conv_info;
    arm_compute::Size2D dilation;
    std::tie(conv_info, dilation) = ConvParameters(node);
    auto ngraphWeightsShape = node.input(Weights).get_shape();
    _layers.at(node.get_instance_id())._inputs.at(node.input(Weights))->_tensor->info()->set_tensor_shape(ShapeCast({
        ngraphWeightsShape[1],
        ngraphWeightsShape[0]*ngraphWeightsShape[2],
        ngraphWeightsShape[3],
        ngraphWeightsShape[4]
    }));
    if (node.get_input_size() == 3) {
        return MakeConversion<arm_compute::NEDepthwiseConvolutionLayer>(
            node.input(Features), node.input(Weights), node.input(Bias), node.output(0),
            conv_info, 1u, GetActivationInfo(node), dilation);
    } else {
        return MakeConversion<arm_compute::NEDepthwiseConvolutionLayer>(
            node.input(Features), node.input(Weights), nullptr, node.output(0),
            conv_info, 1u, GetActivationInfo(node), dilation);
    }
}
}  //  namespace ArmPlugin
