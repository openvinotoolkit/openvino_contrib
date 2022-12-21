// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv_arm.hpp"

using namespace ov;
using namespace ArmPlugin;

opset::ArmConvolution::ArmConvolution(const ov::Output<ov::Node>& data_batch,
                                      const ov::Output<ov::Node>& filters,
                                      const ov::Strides& strides,
                                      const ov::CoordinateDiff& pads_begin,
                                      const ov::CoordinateDiff& pads_end,
                                      const ov::Strides& dilations,
                                      const ov::op::PadType& auto_pad,
                                      const DataLayout& layout)
    : Op({data_batch, filters}),
      m_strides{strides},
      m_pads_begin{pads_begin},
      m_pads_end{pads_end},
      m_dilations{dilations},
      m_auto_pad{auto_pad},
      m_layout{layout} {
    constructor_validate_and_infer_types();
}

opset::ArmConvolution::ArmConvolution(const ov::Output<ov::Node>& data_batch,
                                      const ov::Output<ov::Node>& filters,
                                      const ov::Output<ov::Node>& bias,
                                      const ov::Strides& strides,
                                      const ov::CoordinateDiff& pads_begin,
                                      const ov::CoordinateDiff& pads_end,
                                      const ov::Strides& dilations,
                                      const ov::op::PadType& auto_pad,
                                      const DataLayout& layout)
    : Op({data_batch, filters, bias}),
      m_strides{strides},
      m_pads_begin{pads_begin},
      m_pads_end{pads_end},
      m_dilations{dilations},
      m_auto_pad{auto_pad},
      m_layout{layout} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> opset::ArmConvolution::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    auto num_args = new_args.size();
    if (num_args == 2) {
        return std::make_shared<ArmConvolution>(new_args.at(0),
                                                new_args.at(1),
                                                m_strides,
                                                m_pads_begin,
                                                m_pads_end,
                                                m_dilations,
                                                m_auto_pad,
                                                m_layout);
    } else if (num_args == 3) {
        return std::make_shared<ArmConvolution>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                m_strides,
                                                m_pads_begin,
                                                m_pads_end,
                                                m_dilations,
                                                m_auto_pad,
                                                m_layout);
    } else {
        throw Exception("Unsupported number of arguments for ArmConvolution operation");
    }
}

void opset::ArmConvolution::validate_and_infer_types() {
    const auto& data_batch_type = get_input_element_type(0);
    const auto& filters_type = get_input_element_type(1);
    element::Type output_type;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(output_type, data_batch_type, filters_type),
                          "Element types for data batch and filters do not match (data batch element type: ",
                          data_batch_type,
                          ", filters element type: ",
                          filters_type,
                          ").");
    NODE_VALIDATION_CHECK(this,
                          output_type.is_real() || output_type.is_integral_number(),
                          "Element types must be numeric. Got: ",
                          output_type);

    auto input_shape = get_input_partial_shape(0);
    auto filters_shape = get_input_partial_shape(1);

    int64_t num_non_spatial_dims = 2;
    m_num_spatial = ov::op::v1::calculate_num_spatial(this, input_shape, filters_shape, num_non_spatial_dims, num_non_spatial_dims);
    NODE_VALIDATION_CHECK(this,
                          m_num_spatial != -1,
                          "Convolution shape_infer should be provided with correct num_spatial attribute");
    ov::op::v1::update_and_validate_attributes(this, m_num_spatial);

    if (input_shape.rank().is_dynamic())
        input_shape.resize(m_num_spatial + num_non_spatial_dims);
    if (filters_shape.rank().is_dynamic())
        filters_shape.resize(m_num_spatial + num_non_spatial_dims);

    if (get_input_size() > 2) {
        const auto& bias_shape = get_input_partial_shape(2);
        if (bias_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(this,
                          bias_shape[0].compatible(filters_shape[0]),
                          "Bias shape (", bias_shape[0], ") does not match filter output ",
                          "channel count (", filters_shape[0], ").");
        }
    }

    ov::PartialShape output_shape(std::vector<ov::Dimension>(m_num_spatial + num_non_spatial_dims, ov::Dimension::dynamic()));
    output_shape[0] = input_shape[0];

    int64_t spatial_dims_start = -1;
    size_t input_channels_idx = 0;
    if (m_layout == DataLayout::NCHW) {
        spatial_dims_start = 2;
        input_channels_idx = 1;
        output_shape[1] = filters_shape[0];
    } else if (m_layout == DataLayout::NHWC) {
        spatial_dims_start = 1;
        input_channels_idx = filters_shape.size() - 1;
        output_shape[output_shape.size() - 1] = filters_shape[0];
    } else {
        NODE_VALIDATION_CHECK(this, false, "Layout not supported");
    }

    NODE_VALIDATION_CHECK(this,
                          input_shape[input_channels_idx].compatible(filters_shape[input_channels_idx]),
                          "Data batch channel count (", input_shape[input_channels_idx], ") does not match filter input ",
                          "channel count (", filters_shape[input_channels_idx], ").");

    ov::op::v1::resolve_auto_pad_for_shape(this, m_pads_begin, m_pads_end,
                                           std::vector<ov::PartialShape>{input_shape, filters_shape},
                                           spatial_dims_start, spatial_dims_start);
    ov::op::v1::calculate_output_spatial_dims_for_convolution(this,
                                                              input_shape,
                                                              filters_shape,
                                                              output_shape,
                                                              m_num_spatial,
                                                              m_strides,
                                                              m_dilations,
                                                              m_pads_begin,
                                                              m_pads_end,
                                                              spatial_dims_start,
                                                              spatial_dims_start);
    set_output_type(0, output_type, output_shape);
}
