// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gsl/span_ext>

#include "convolution_components.hpp"

#include "openvino/core/except.hpp"
#include "openvino/op/group_conv.hpp"

namespace ov::nvidia_gpu::Convolution::Details {

constexpr int CONV_1D_DIMS_NUMBER = NON_SPATIAL_DIMS_NUMBER + 1;

template <typename TConvNode>
ConvolutionParams::ConvolutionParams(const TConvNode& node)
    : element_type_{node.get_input_element_type(ConvArgIndices::input)},
      input_shape_{node.get_input_shape(ConvArgIndices::input)},
      filter_shape_{node.get_input_shape(ConvArgIndices::filter)},
      output_shape_{node.get_output_shape(ConvArgIndices::output)},
      strides_{node.get_strides()},
      dilations_{node.get_dilations()},
      padding_before_{node.get_pads_begin()},
      padding_after_{node.get_pads_end()},
      groups_{1U} {
    OPENVINO_ASSERT(input_shape_.size() > NON_SPATIAL_DIMS_NUMBER);
    OPENVINO_ASSERT(element_type_ == node.get_input_element_type(ConvArgIndices::filter));
    OPENVINO_ASSERT(element_type_ == node.get_output_element_type(ConvArgIndices::output));

    if constexpr (std::is_same_v<TConvNode, ov::op::v1::GroupConvolution> ||
                  std::is_same_v<TConvNode, nodes::FusedGroupConvolution>) {
        groups_ = node.get_input_shape(1)[0];
        OPENVINO_ASSERT(input_shape_[1] % groups_ == 0);
        filter_shape_.erase(filter_shape_.begin());
    }
    OPENVINO_ASSERT(groups_ >= 1U);
    filter_shape_[0] *= groups_;

    if (input_shape_.size() == CONV_1D_DIMS_NUMBER) ConvertConv1DToConv2D();

    const size_t dims_number = NumberOfDims();
    OPENVINO_ASSERT(input_shape_.size() == dims_number);
    OPENVINO_ASSERT(filter_shape_.size() == dims_number);
    OPENVINO_ASSERT(output_shape_.size() == dims_number);

    const size_t spatial_dims_number = NumberOfSpatialDims();
    // Convolution dimension according to op spec is 1D, 2D or 3D.
    // 1D is already turned into 2D at this point.
    OPENVINO_ASSERT((spatial_dims_number == 2) || (spatial_dims_number == 3));
    OPENVINO_ASSERT(strides_.size() == spatial_dims_number);
    OPENVINO_ASSERT(dilations_.size() == spatial_dims_number);
    OPENVINO_ASSERT(padding_before_.size() == spatial_dims_number);
    OPENVINO_ASSERT(padding_after_.size() == spatial_dims_number);
}
template ConvolutionParams::ConvolutionParams(const ov::op::v1::GroupConvolution& node);
template ConvolutionParams::ConvolutionParams(const ov::op::v1::Convolution& node);
template ConvolutionParams::ConvolutionParams(const nodes::FusedConvolution& node);
template ConvolutionParams::ConvolutionParams(const nodes::FusedGroupConvolution& node);

void ConvolutionParams::ConvertConv1DToConv2D() {
    if (input_shape_.size() != CONV_1D_DIMS_NUMBER) return;

    OPENVINO_ASSERT(input_shape_.size() == CONV_1D_DIMS_NUMBER);
    input_shape_.insert(input_shape_.begin() + NON_SPATIAL_DIMS_NUMBER, 1);
    OPENVINO_ASSERT(filter_shape_.size() == CONV_1D_DIMS_NUMBER);
    filter_shape_.insert(filter_shape_.begin() + NON_SPATIAL_DIMS_NUMBER, 1);
    OPENVINO_ASSERT(output_shape_.size() == CONV_1D_DIMS_NUMBER);
    output_shape_.insert(output_shape_.begin() + NON_SPATIAL_DIMS_NUMBER, 1);
    strides_.insert(strides_.begin(), 1);
    dilations_.insert(dilations_.begin(), 1);
    padding_before_.insert(padding_before_.begin(), 0);
    padding_after_.insert(padding_after_.begin(), 0);
}

template <typename TConvNode>
ConvolutionBackwardDataParams::ConvolutionBackwardDataParams(const TConvNode& node)
    : element_type_{node.get_input_element_type(ConvBackArgIndices::doutput)},
      doutput_shape_{node.get_input_shape(ConvBackArgIndices::doutput)},
      filter_shape_{node.get_input_shape(ConvBackArgIndices::filter)},
      dinput_shape_{static_cast<const ov::Node&>(node).get_output_shape(0)},
      strides_{node.get_strides()},
      dilations_{node.get_dilations()},
      pads_begin_{node.get_pads_begin()},
      pads_end_{node.get_pads_end()},
      auto_pad_{node.get_auto_pad()},
      output_padding_{node.get_output_padding()},
      groups_{1U} {
    OPENVINO_ASSERT(doutput_shape_.size() > NON_SPATIAL_DIMS_NUMBER);
    OPENVINO_ASSERT(element_type_ == node.get_input_element_type(ConvBackArgIndices::filter));
    OPENVINO_ASSERT(element_type_ == node.get_output_element_type(ConvBackArgIndices::dinput));

    if constexpr (std::is_same_v<TConvNode, ov::op::v1::GroupConvolutionBackpropData>) {
        groups_ = node.get_input_shape(1)[0];
        OPENVINO_ASSERT(groups_ >= 1U);
        OPENVINO_ASSERT(dinput_shape_[1] % groups_ == 0);
        filter_shape_.erase(filter_shape_.begin());
        filter_shape_[0] *= groups_;
    }

    if (doutput_shape_.size() == CONV_1D_DIMS_NUMBER) {
        ConvertConv1DToConv2D();
    }

    const size_t dims_number = NumberOfDims();
    OPENVINO_ASSERT(doutput_shape_.size() == dims_number);
    OPENVINO_ASSERT(filter_shape_.size() == dims_number);
    OPENVINO_ASSERT(dinput_shape_.size() == dims_number);

    const size_t spatial_dims_number = NumberOfSpatialDims();
    // Convolution dimension according to op spec is 1D, 2D or 3D.
    // 1D is already turned into 2D at this point.
    OPENVINO_ASSERT((spatial_dims_number == 2) || (spatial_dims_number == 3));
    OPENVINO_ASSERT(strides_.size() == spatial_dims_number);
    OPENVINO_ASSERT(dilations_.size() == spatial_dims_number);
    OPENVINO_ASSERT(pads_begin_.size() == spatial_dims_number);
    OPENVINO_ASSERT(pads_end_.size() == spatial_dims_number);
}
template ConvolutionBackwardDataParams::ConvolutionBackwardDataParams(
    const ov::op::v1::GroupConvolutionBackpropData& node);
template ConvolutionBackwardDataParams::ConvolutionBackwardDataParams(const ov::op::v1::ConvolutionBackpropData& node);
template ConvolutionBackwardDataParams::ConvolutionBackwardDataParams(const nodes::FusedConvBackpropData& node);

void ConvolutionBackwardDataParams::ConvertConv1DToConv2D() {
    if (doutput_shape_.size() != CONV_1D_DIMS_NUMBER) return;

    OPENVINO_ASSERT(doutput_shape_.size() == CONV_1D_DIMS_NUMBER);
    doutput_shape_.insert(doutput_shape_.begin() + NON_SPATIAL_DIMS_NUMBER, 1);
    OPENVINO_ASSERT(filter_shape_.size() == CONV_1D_DIMS_NUMBER);
    filter_shape_.insert(filter_shape_.begin() + NON_SPATIAL_DIMS_NUMBER, 1);
    OPENVINO_ASSERT(dinput_shape_.size() == CONV_1D_DIMS_NUMBER);
    dinput_shape_.insert(dinput_shape_.begin() + NON_SPATIAL_DIMS_NUMBER, 1);
    strides_.insert(strides_.begin(), 1);
    dilations_.insert(dilations_.begin(), 1);
    pads_begin_.insert(pads_begin_.begin(), 0);
    pads_end_.insert(pads_end_.begin(), 0);
}

template <typename TConvNode>
FusedConvolutionParams::FusedConvolutionParams(const TConvNode& node)
    : conv_{node},
      bias_shape_{node.get_input_shape(FusedConvolutionIndices::bias)},
      activation_{node.get_activation()} {
    OPENVINO_ASSERT(conv_.NumberOfSpatialDims() == 2 || conv_.NumberOfSpatialDims() == 3);
    OPENVINO_ASSERT(conv_.element_type_ == node.get_input_element_type(FusedConvolutionIndices::bias));
    if (node.inputs().size() == 4) {
        add_shape_ = node.get_input_shape(FusedConvolutionIndices::add);
    }

    if (conv_.output_shape_.size() == CONV_1D_DIMS_NUMBER + 1 && bias_shape_.size() == CONV_1D_DIMS_NUMBER) {
        bias_shape_.insert(bias_shape_.begin() + NON_SPATIAL_DIMS_NUMBER, 1);
    }
    if (add_shape_ && conv_.output_shape_.size() == CONV_1D_DIMS_NUMBER + 1 &&
        add_shape_->size() == CONV_1D_DIMS_NUMBER) {
        add_shape_->insert(add_shape_->begin() + NON_SPATIAL_DIMS_NUMBER, 1);
    }
}
template FusedConvolutionParams::FusedConvolutionParams(const nodes::FusedConvolution& node);
template FusedConvolutionParams::FusedConvolutionParams(const nodes::FusedGroupConvolution& node);

FusedConvolutionBackwardDataParams::FusedConvolutionBackwardDataParams(
    const ov::nvidia_gpu::nodes::FusedConvBackpropData& node)
    : conv_{node} {
    OPENVINO_ASSERT(conv_.NumberOfSpatialDims() == 2 || conv_.NumberOfSpatialDims() == 3);
    if (node.inputs().size() == 4) {
        add_shape_ = node.get_input_shape(FusedConvolutionBackwardDataIndices<4>::add);
        OPENVINO_ASSERT(conv_.element_type_ == node.get_input_element_type(FusedConvolutionBackwardDataIndices<4>::add));
    } else {
        add_shape_ = node.get_input_shape(FusedConvolutionBackwardDataIndices<3>::add);
        OPENVINO_ASSERT(conv_.element_type_ == node.get_input_element_type(FusedConvolutionBackwardDataIndices<3>::add));
    }
}

}  // namespace ov::nvidia_gpu::Convolution::Details
