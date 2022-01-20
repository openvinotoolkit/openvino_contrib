// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <ngraph/op/convolution.hpp>
#include <ngraph/type/element_type.hpp>

#include "cuda_plugin_custom_node_types.hpp"
#include "ngraph/attribute_adapter.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/type.hpp"

namespace CUDAPlugin::nodes {

// TODO: Try to use BasicFusedConvolution or derive from ngraph::op::v1::ConvolutionBackpropData
class FusedConvBackpropData : public ngraph::op::Op {
public:
    explicit FusedConvBackpropData(const ngraph::Output<Node>& data_batch,
                                   const ngraph::Output<Node>& filters,
                                   const ngraph::Output<Node>& add,
                                   const ngraph::Strides& strides,
                                   const ngraph::CoordinateDiff& pads_begin,
                                   const ngraph::CoordinateDiff& pads_end,
                                   const ngraph::Strides& dilations,
                                   const ngraph::op::PadType& auto_pad,
                                   const ngraph::CoordinateDiff& output_padding);
    explicit FusedConvBackpropData(const ngraph::Output<Node>& data_batch,
                                   const ngraph::Output<Node>& filters,
                                   const ngraph::Output<Node>& outputShape,
                                   const ngraph::Output<Node>& add,
                                   const ngraph::Strides& strides,
                                   const ngraph::CoordinateDiff& pads_begin,
                                   const ngraph::CoordinateDiff& pads_end,
                                   const ngraph::Strides& dilations,
                                   const ngraph::op::PadType& auto_pad,
                                   const ngraph::CoordinateDiff& output_padding);

    inline static constexpr type_info_t type_info{"FusedConvBackpropData", 0};
    const type_info_t& get_type_info() const override { return type_info; }

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override;

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;

    void validate_and_infer_types() override;
    void conv_validate_and_infer_types();
    void infer_conv_backprop_output_spatial_shape(const std::vector<ngraph::Dimension>& input_data_shape,
                                                  const std::vector<ngraph::Dimension>& filters_shape,
                                                  const ngraph::Strides& strides,
                                                  const ngraph::Strides& dilations,
                                                  const ngraph::CoordinateDiff& pads_begin,
                                                  const ngraph::CoordinateDiff& pads_end,
                                                  const ngraph::CoordinateDiff& output_padding,
                                                  std::vector<ngraph::Dimension>& output_spatial_shape);
    ngraph::PartialShape get_output_shape() const;

    const auto& get_strides() const { return strides_; }
    const auto& get_dilations() const { return dilations_; }
    const auto& get_pads_begin() const { return pads_begin_; }
    const auto& get_pads_end() const { return pads_end_; }
    const auto& get_auto_pad() const { return auto_pad_; }
    const auto& get_output_padding() const { return output_padding_; }

private:
    /// Used for the shape validation
    ngraph::Strides strides_;
    ngraph::CoordinateDiff pads_begin_;
    ngraph::CoordinateDiff pads_end_;
    ngraph::Strides dilations_;
    ngraph::op::PadType auto_pad_;
    ngraph::CoordinateDiff output_padding_;
    ngraph::Shape add_shape_;
    ngraph::element::Type add_type_;
};

}  // namespace CUDAPlugin::nodes
