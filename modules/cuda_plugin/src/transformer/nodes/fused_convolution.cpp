// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_convolution.hpp"

#include <gsl/gsl_assert>
#include <ngraph/validation_util.hpp>

namespace CUDAPlugin::nodes {

FusedConvolution::FusedConvolution(
    const ngraph::Output<Node>& data_batch,
    const ngraph::Output<Node>& filters,
    const ngraph::Output<Node>& bias,
    const ngraph::Strides& strides,
    const ngraph::CoordinateDiff& pads_begin,
    const ngraph::CoordinateDiff& pads_end,
    const ngraph::Strides& dilations,
    const ngraph::op::PadType& auto_pad,
    ActivationMode activation)
    : ngraph::op::Op(ngraph::OutputVector{data_batch, filters, bias}),
      strides_(strides),
      pads_begin_(pads_begin),
      pads_end_(pads_end),
      dilations_(dilations),
      auto_pad_(auto_pad),
      bias_shape_{bias.get_shape()},
      bias_type_(bias.get_element_type()),
      activation_{activation} {
  constructor_validate_and_infer_types();
}

FusedConvolution::FusedConvolution(
    const ngraph::Output<Node>& data_batch,
    const ngraph::Output<Node>& filters,
    const ngraph::Output<Node>& bias,
    const ngraph::Output<Node>& addArgNode,
    const ngraph::Strides& strides,
    const ngraph::CoordinateDiff& pads_begin,
    const ngraph::CoordinateDiff& pads_end,
    const ngraph::Strides& dilations,
    const ngraph::op::PadType& auto_pad,
    ActivationMode activation)
    : ngraph::op::Op(ngraph::OutputVector{data_batch, filters, bias, addArgNode}),
      strides_(strides),
      pads_begin_(pads_begin),
      pads_end_(pads_end),
      dilations_(dilations),
      auto_pad_(auto_pad),
      bias_shape_{bias.get_shape()},
      bias_type_(bias.get_element_type()),
      has_add_node_(true),
      activation_{activation} {
    constructor_validate_and_infer_types();
}

bool FusedConvolution::visit_attributes(ngraph::AttributeVisitor& visitor) {
  visitor.on_attribute("strides", strides_);
  visitor.on_attribute("dilations", dilations_);
  visitor.on_attribute("pads_begin", pads_begin_);
  visitor.on_attribute("pads_end", pads_end_);
  visitor.on_attribute("auto_pad", auto_pad_);
  return true;
}

std::shared_ptr<ngraph::Node> FusedConvolution::clone_with_new_inputs(
    const ngraph::OutputVector& new_args) const {
  check_new_args_count(this, new_args);
    if (new_args.size() == 3) {
        return std::make_shared<FusedConvolution>(
          new_args.at(0), new_args.at(1), new_args.at(2),
          strides_, pads_begin_, pads_end_, dilations_, auto_pad_, activation_);
    } else {
        return std::make_shared<FusedConvolution>(
          new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
          strides_, pads_begin_, pads_end_, dilations_, auto_pad_, activation_);
    }
}

void FusedConvolution::conv_validate_and_infer_types() {
    const ngraph::PartialShape& data_batch_shape = get_input_partial_shape(0);
    ngraph::element::Type data_batch_et = get_input_element_type(0);
    const ngraph::PartialShape& filters_shape = get_input_partial_shape(1);
    ngraph::element::Type filters_et = get_input_element_type(1);

    ngraph::PartialShape result_shape = ngraph::PartialShape::dynamic();
    if (data_batch_shape.rank().is_static()) {
        result_shape =
            std::vector<ngraph::Dimension>(data_batch_shape.rank().get_length(), ngraph::Dimension::dynamic());

        if (data_batch_shape.rank().get_length() > 1) {
            result_shape[0] = data_batch_shape[0]; // batch size
        }
        if (filters_shape.rank().is_static() && filters_shape.rank().get_length() > 1) {
            result_shape[1] = filters_shape[0]; // filter channel size
        }
    }

    ngraph::element::Type result_et;
    NODE_VALIDATION_CHECK(
        this,
        ngraph::element::Type::merge(result_et, data_batch_et, filters_et),
        "Element types for data batch and filters do not match (data batch element type: ",
        data_batch_et,
        ", filters element type: ",
        filters_et,
        ").");

    if (strides_.size() == 0) {
        strides_ = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (dilations_.size() == 0) {
        dilations_ = conv_default_strides(this, data_batch_shape, filters_shape);
    }

    if (pads_begin_.size() == 0 || auto_pad_ == ngraph::op::PadType::VALID) {
        pads_begin_ = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (pads_end_.size() == 0 || auto_pad_ == ngraph::op::PadType::VALID) {
        pads_end_ = conv_default_padding(this, data_batch_shape, filters_shape);
    }

    if (auto_pad_ == ngraph::op::PadType::SAME_UPPER || auto_pad_ == ngraph::op::PadType::SAME_LOWER) {
        bool auto_padding_applied = false;
        if (filters_shape.is_static()) {
            pads_begin_.clear();
            pads_end_.clear();
            auto filter_shape = filters_shape.to_shape();
            filter_shape.erase(filter_shape.begin(), filter_shape.begin() + 2); // Remove {O,I}
            auto_padding_applied = try_apply_auto_padding(
                data_batch_shape,
                filter_shape,
                strides_,
                dilations_,
                auto_pad_,
                pads_end_,
                pads_begin_);
        }
        if (!auto_padding_applied) {
            set_output_type(0, result_et, result_shape);
            return;
        }
    }

    result_shape = ngraph::infer_convolution_forward(
        this,
        data_batch_shape,
        ngraph::Strides(strides_.size(), 1), // dummy data dilations
        pads_begin_,
        pads_end_,
        filters_shape,
        strides_,
        dilations_);

    set_output_type(0, result_et, result_shape);
}

void FusedConvolution::validate_and_infer_types() {
  conv_validate_and_infer_types();
  const auto& conv_out_shape = get_output_shape(0);
  const auto& element_type = get_output_element_type(0);

  const size_t num_spatial_dims = conv_out_shape.size() - 2;
  const size_t nchw_channel_dim_offset = num_spatial_dims + 1;
  constexpr size_t conv_output_rank_max{5};
  constexpr size_t conv_bias_rank_min{3};
  Expects(conv_out_shape.size() <= conv_output_rank_max);
  Expects(bias_shape_.size() >= conv_bias_rank_min);
  const size_t conv_channel_dim_size = conv_out_shape.at(conv_out_shape.size() - nchw_channel_dim_offset);
  const size_t bias_channel_dim_size = bias_shape_.at(bias_shape_.size() - nchw_channel_dim_offset);

  Expects(conv_channel_dim_size == bias_channel_dim_size);
  Expects(bias_type_ == element_type);

  set_output_type(0, element_type, conv_out_shape);
}

bool FusedConvolution::has_add_node() const {
  return has_add_node_;
}

void FusedConvolution::set_activation(ActivationMode act) {
  activation_ = act;
}

ActivationMode FusedConvolution::get_activation() const {
  return activation_;
}

}  // namespace CUDAPlugin::nodes
