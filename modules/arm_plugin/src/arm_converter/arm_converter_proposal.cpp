// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <details/ie_exception.hpp>

#include "arm_converter/arm_converter.hpp"
#include <ngraph/runtime/reference/proposal.hpp>

namespace ArmPlugin {
template <typename T1, typename T2>
void wrap_proposal_v4(const T1* class_probs,
                             const T1* bbox_deltas,
                             const T2* image_shape,
                             T1* output,
                             T1* out_probs,
                             const ngraph::Shape& class_probs_shape,
                             const ngraph::Shape& bbox_deltas_shape,
                             const ngraph::Shape& image_shape_shape,
                             const ngraph::Shape& output_shape,
                             const ngraph::Shape& out_probs_shape,
                             const ngraph::op::ProposalAttrs& attrs) {
    std::vector<T1> image_shape_vec(image_shape, image_shape + ngraph::shape_size(image_shape_shape));
    ngraph::runtime::reference::proposal_v4<T1>(class_probs,
                                                bbox_deltas,
                                                image_shape_vec.data(),
                                                output,
                                                out_probs,
                                                class_probs_shape,
                                                bbox_deltas_shape,
                                                image_shape_shape,
                                                output_shape,
                                                out_probs_shape,
                                                attrs);
}

template<> Converter::Conversion::Ptr Converter::Convert(const opset::Proposal& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(2),
                                    node.output(0),
                                    node.output(1),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_input_shape(2),
                                    node.get_output_shape(0),
                                    node.get_output_shape(1),
                                    node.get_attrs());
    };
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::f16 :
            if (node.get_input_element_type(2) == ngraph::element::f32) {
                return make(wrap_proposal_v4<ngraph::float16, float>);
            }
            return make(wrap_proposal_v4<ngraph::float16, ngraph::float16>);
        case ngraph::element::Type_t::f32 :
            if (node.get_input_element_type(2) == ngraph::element::f32) {
                return make(wrap_proposal_v4<float, float>);
            }
            return make(wrap_proposal_v4<float, ngraph::float16>);
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

template <typename T1, typename T2>
void wrap_proposal_v0(const T1* class_probs,
                      const T1* bbox_deltas,
                      const T2* image_shape,
                      T1* output,
                      const ngraph::Shape& class_probs_shape,
                      const ngraph::Shape& bbox_deltas_shape,
                      const ngraph::Shape& image_shape_shape,
                      const ngraph::Shape& output_shape,
                      const ngraph::op::ProposalAttrs& attrs) {
    std::vector<T1> image_shape_vec(image_shape, image_shape + ngraph::shape_size(image_shape_shape));
    ngraph::runtime::reference::proposal_v0<T1>(class_probs,
                                                bbox_deltas,
                                                image_shape_vec.data(),
                                                output,
                                                class_probs_shape,
                                                bbox_deltas_shape,
                                                image_shape_shape,
                                                output_shape,
                                                attrs);
}

template<> Converter::Conversion::Ptr Converter::Convert(const ngraph::op::v0::Proposal& node) {
    auto make = [&] (auto refFunction) {
        return this->MakeConversion(refFunction,
                                    node.input(0),
                                    node.input(1),
                                    node.input(2),
                                    node.output(0),
                                    node.get_input_shape(0),
                                    node.get_input_shape(1),
                                    node.get_input_shape(2),
                                    node.get_output_shape(0),
                                    node.get_attrs());
    };
    switch (node.get_input_element_type(0)) {
        case ngraph::element::Type_t::f16 :
            if (node.get_input_element_type(2) == ngraph::element::f32) {
                return make(wrap_proposal_v0<ngraph::float16, float>);
            }
            return make(wrap_proposal_v0<ngraph::float16, ngraph::float16>);
        case ngraph::element::Type_t::f32 :
            if (node.get_input_element_type(2) == ngraph::element::f32) {
                return make(wrap_proposal_v0<float, float>);
            }
            return make(wrap_proposal_v0<float, ngraph::float16>);
        default: IE_THROW() << "Unsupported Type: " << node.get_input_element_type(0); return {};
    }
}

}  //  namespace ArmPlugin
