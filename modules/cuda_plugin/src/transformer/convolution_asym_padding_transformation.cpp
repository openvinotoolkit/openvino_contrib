// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_asym_padding_transformation.hpp"

#include <gsl/gsl_assert>
#include <ngraph/node.hpp>
#include <ngraph/op/convolution.hpp>
#include <ngraph/op/pad.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

namespace {

ngraph::CoordinateDiff add_two_zero_pads(const ngraph::CoordinateDiff &pad) {
    ngraph::CoordinateDiff result = pad;
    result.insert(result.begin(), 0);
    result.insert(result.begin(), 0);

    return result;
}

template <typename TBaseConvolution>
bool convolution_with_padding(ngraph::pattern::Matcher &m) {
    static_assert(std::is_same_v<TBaseConvolution, ngraph::op::v1::Convolution> ||
                      std::is_same_v<TBaseConvolution, ngraph::op::v1::GroupConvolution>,
                  "TBaseConvolution should be either Convolution or GroupConvolution");

    auto convolution = std::dynamic_pointer_cast<TBaseConvolution>(m.get_match_root());
    if (!convolution || convolution->inputs().size() != 2) {
        return false;
    }

    const auto pads_begin = add_two_zero_pads(convolution->get_pads_begin());
    const auto pads_end = add_two_zero_pads(convolution->get_pads_end());

    if (pads_begin == pads_end) {
        return false;
    }
    Expects(pads_begin.size() == pads_end.size());

    const ngraph::Output<ngraph::Node> &data = convolution->input(0).get_source_output();
    const ngraph::Output<ngraph::Node> &filters = convolution->input(1).get_source_output();

    const auto pads_begin_node = std::make_shared<ngraph::op::Constant>(
        ngraph::element::i64, ngraph::Shape{pads_begin.size()}, pads_begin.data());
    const auto pads_end_node =
        std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{pads_end.size()}, pads_end.data());
    const auto padding = std::make_shared<ngraph::op::v1::Pad>(
        data,
        pads_begin_node,
        pads_end_node,
        ngraph::op::Constant::create(data.get_element_type(), ngraph::Shape{}, {0}),
        ngraph::op::PadMode::CONSTANT);

    const ngraph::CoordinateDiff zero_pads(convolution->get_pads_begin().size(), 0);
    auto new_convolution = std::make_shared<TBaseConvolution>(padding->output(0),
                                                              filters,
                                                              convolution->get_strides(),
                                                              zero_pads,
                                                              zero_pads,
                                                              convolution->get_dilations(),
                                                              ngraph::op::PadType::EXPLICIT);

    new_convolution->set_friendly_name(convolution->get_friendly_name());

    ngraph::copy_runtime_info(convolution, new_convolution);

    ngraph::replace_node(convolution, new_convolution);

    return true;
}
}  // namespace

namespace ngraph::pass {

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvolutionAsymPaddingTransformation, "ConvolutionAsymPaddingTransformation", 0);

ConvolutionAsymPaddingTransformation::ConvolutionAsymPaddingTransformation() {
    const auto conv = ngraph::pattern::wrap_type<opset1::Convolution>();

    matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        return convolution_with_padding<ngraph::op::v1::Convolution>(m);
    };

    const auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "ConvolutionAsymPaddingTransformation");
    register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ngraph::pass::GroupConvolutionAsymPaddingTransformation,
                       "GroupConvolutionAsymPaddingTransformation",
                       0);
GroupConvolutionAsymPaddingTransformation::GroupConvolutionAsymPaddingTransformation() {
    const auto conv = ngraph::pattern::wrap_type<opset1::GroupConvolution>();

    matcher_pass_callback callback = [](ngraph::pattern::Matcher &m) {
        return convolution_with_padding<ngraph::op::v1::GroupConvolution>(m);
    };
    const auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "GroupConvolutionAsymPaddingTransformation");
    register_matcher(m, callback);
}

}  // namespace ngraph::pass
