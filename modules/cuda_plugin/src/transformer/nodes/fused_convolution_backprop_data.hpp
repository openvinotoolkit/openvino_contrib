// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <ngraph/type/element_type.hpp>
#include <openvino/op/convolution.hpp>

#include "cuda_plugin_custom_node_types.hpp"
#include "ngraph/attribute_adapter.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/type.hpp"

namespace CUDAPlugin::nodes {

// TODO: Try to use BasicFusedConvolution or derive from ov::op::v1::ConvolutionBackpropData
class FusedConvBackpropData : public ov::op::Op {
public:
    explicit FusedConvBackpropData(const ov::Output<Node>& data_batch,
                                   const ov::Output<Node>& filters,
                                   const ov::Output<Node>& add,
                                   const ov::Strides& strides,
                                   const ov::CoordinateDiff& pads_begin,
                                   const ov::CoordinateDiff& pads_end,
                                   const ov::Strides& dilations,
                                   const ov::op::PadType& auto_pad,
                                   const ov::CoordinateDiff& output_padding);
    explicit FusedConvBackpropData(const ov::Output<Node>& data_batch,
                                   const ov::Output<Node>& filters,
                                   const ov::Output<Node>& outputShape,
                                   const ov::Output<Node>& add,
                                   const ov::Strides& strides,
                                   const ov::CoordinateDiff& pads_begin,
                                   const ov::CoordinateDiff& pads_end,
                                   const ov::Strides& dilations,
                                   const ov::op::PadType& auto_pad,
                                   const ov::CoordinateDiff& output_padding);

    inline static constexpr type_info_t type_info{"FusedConvBackpropData", 0ul};
    const type_info_t& get_type_info() const override { return type_info; }

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void validate_and_infer_types() override;
    void conv_validate_and_infer_types();
    void infer_conv_backprop_output_spatial_shape(const std::vector<ov::Dimension>& input_data_shape,
                                                  const std::vector<ov::Dimension>& filters_shape,
                                                  const ov::Strides& strides,
                                                  const ov::Strides& dilations,
                                                  const ov::CoordinateDiff& pads_begin,
                                                  const ov::CoordinateDiff& pads_end,
                                                  const ov::CoordinateDiff& output_padding,
                                                  std::vector<ov::Dimension>& output_spatial_shape);
    ov::PartialShape get_output_shape() const;

    const auto& get_strides() const { return strides_; }
    const auto& get_dilations() const { return dilations_; }
    const auto& get_pads_begin() const { return pads_begin_; }
    const auto& get_pads_end() const { return pads_end_; }
    const auto& get_auto_pad() const { return auto_pad_; }
    const auto& get_output_padding() const { return output_padding_; }

private:
    /// Used for the shape validation
    ov::Strides strides_;
    ov::CoordinateDiff pads_begin_;
    ov::CoordinateDiff pads_end_;
    ov::Strides dilations_;
    ov::op::PadType auto_pad_;
    ov::CoordinateDiff output_padding_;
    ov::Shape add_shape_;
    ov::element::Type add_type_;
};

}  // namespace CUDAPlugin::nodes
