// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution2d_biasadd_activation.hpp"

#include <gsl/gsl_assert>

namespace CUDAPlugin::nodes {

Conv2DBiasAddActivation::Conv2DBiasAddActivation(
    const ngraph::Output<Node>& data_batch, const ngraph::Output<Node>& filters,
    const ngraph::Output<Node>& bias, const ngraph::Strides& strides,
    const ngraph::CoordinateDiff& pads_begin,
    const ngraph::CoordinateDiff& pads_end, const ngraph::Strides& dilations,
    const ngraph::op::PadType& auto_pad, ActivationMode activation)
    : ngraph::op::Op(ngraph::OutputVector{data_batch, filters, bias}),
      conv_op_(data_batch, filters, strides, pads_begin, pads_end, dilations,
               auto_pad),
      bias_shape_{bias.get_shape()},
      bias_type_(bias.get_element_type()),
      activation_{activation} {
  constructor_validate_and_infer_types();
}

bool Conv2DBiasAddActivation::visit_attributes(
    ngraph::AttributeVisitor& visitor) {
  // TODO: visitor.on_attribute("activation", m_activation); ?
  return conv_op_.visit_attributes(visitor);
}

std::shared_ptr<ngraph::Node>
Conv2DBiasAddActivation::clone_with_new_inputs(
    const ngraph::OutputVector& new_args) const {
  check_new_args_count(this, new_args);
  return std::make_shared<Conv2DBiasAddActivation>(
      new_args.at(0), new_args.at(1), new_args.at(2), conv_op_.get_strides(),
      conv_op_.get_pads_begin(), conv_op_.get_pads_end(),
      conv_op_.get_dilations(), conv_op_.get_auto_pad(), activation_);
}

void Conv2DBiasAddActivation::validate_and_infer_types() {
  // Re-using the Convolution shape inferrer and validator.
  // As the BiasAdd does not change neither the element type, nor
  // the output shape, we can just set the Convolution's ones to this op.
  conv_op_.validate_and_infer_types();
  const auto& conv_out_shape = conv_op_.get_output_shape(0);
  const auto& element_type = conv_op_.get_output_element_type(0);

  constexpr size_t nchw_channel_dim_offset = 3;
  constexpr size_t conv2_output_rank{4};
  constexpr size_t conv2_bias_rank_min{3};
  Expects(conv_out_shape.size() == conv2_output_rank);
  Expects(bias_shape_.size() >= conv2_bias_rank_min);
  const size_t conv_channel_dim_size =
      conv_out_shape.at(conv_out_shape.size() - nchw_channel_dim_offset);
  const size_t bias_channel_dim_size =
      bias_shape_.at(bias_shape_.size() - nchw_channel_dim_offset);

  Expects(conv_channel_dim_size == bias_channel_dim_size);
  Expects(bias_type_ == element_type);

  set_output_type(0, element_type, conv_out_shape);
}

void Conv2DBiasAddActivation::set_activation(ActivationMode act) {
  activation_ = act;
}

ActivationMode
Conv2DBiasAddActivation::get_activation() const {
  return activation_;
}

const ngraph::op::v1::Convolution& Conv2DBiasAddActivation::conv_op() const {
    return conv_op_;
}

}  // namespace CUDAPlugin::nodes
