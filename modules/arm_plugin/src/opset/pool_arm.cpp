// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/validation_util.hpp"
#include "pool_arm.hpp"

using namespace ngraph;
using namespace ArmPlugin;

opset::util::PoolBase::PoolBase(const ov::Output<Node>& arg,
                                const ov::Strides& strides,
                                const ov::Shape& pads_begin,
                                const ov::Shape& pads_end,
                                const ov::Shape& kernel,
                                const ov::op::RoundingType& rounding_type,
                                const ov::op::PadType& auto_pad,
                                const DataLayout& layout)
        : Op({arg}),
          m_strides{strides},
          m_pads_begin{pads_begin},
          m_pads_end{pads_end},
          m_kernel{kernel},
          m_rounding_type{rounding_type},
          m_auto_pad{auto_pad},
          m_layout{layout} {
}

PartialShape opset::util::PoolBase::infer_shape(const ov::Strides& dilations, bool exclude_pad) {
    NODE_VALIDATION_CHECK(this, m_layout == DataLayout::NCHW || m_layout == DataLayout::NHWC, "Layout must be either NCHW or NHWC");

    size_t num_spatial_dims = m_kernel.size();
    NODE_VALIDATION_CHECK(this, num_spatial_dims > 0, "Kernel must have more than zero elements");

    if (m_strides.size() > 0) {
        NODE_VALIDATION_CHECK(this, num_spatial_dims == m_strides.size(),
                              "strides size must be equal to kernel size. Got: ",
                              m_strides.size());
    } else {
        m_strides = Strides(num_spatial_dims, 1);
    }

    if (m_auto_pad == ov::op::PadType::EXPLICIT) {
        NODE_VALIDATION_CHECK(this, num_spatial_dims == m_pads_begin.size(),
                              "pads_begin size must be equal to kernel size. Got: ",
                              m_pads_begin.size());
        NODE_VALIDATION_CHECK(this, num_spatial_dims == m_pads_end.size(),
                              "pads_end size must be equal to kernel size. Got: ",
                              m_pads_end.size());
    }

    const auto& input_shape = get_input_partial_shape(0);
    const auto& input_rank = input_shape.rank();

    NODE_VALIDATION_CHECK(
        this,
        input_rank.compatible(3) || input_rank.compatible(4) || input_rank.compatible(5),
        "Expected a 3D, 4D or 5D tensor for the input. Got: ",
        input_shape);

    if (input_rank.is_dynamic()) {
        return PartialShape::dynamic();
    }

    NODE_VALIDATION_CHECK(this,
                          num_spatial_dims == input_shape.size() - 2,
                          "kernel size must be equal to input rank - 2. Got: ",
                          num_spatial_dims);

    bool update_auto_padding_succeed = true;
    if (m_auto_pad == ov::op::PadType::SAME_UPPER || m_auto_pad == ov::op::PadType::SAME_LOWER) {
        CoordinateDiff pads_begin;
        CoordinateDiff pads_end;
        const auto filter_dilations = dilations.empty() ? Strides(num_spatial_dims, 1) : dilations;
        update_auto_padding_succeed = ngraph::try_apply_auto_padding(input_shape, m_kernel, m_strides, filter_dilations,
                                                                     m_auto_pad, pads_end, pads_begin);
        m_pads_end = ov::Shape(pads_end.begin(), pads_end.end());
        m_pads_begin = ov::Shape(pads_begin.begin(), pads_begin.end());
    } else if (m_auto_pad == ov::op::PadType::VALID) {
        m_pads_begin = Shape(num_spatial_dims, 0);
        m_pads_end = Shape(num_spatial_dims, 0);
    }

    PartialShape output_shape{std::vector<Dimension>(input_shape.size(), Dimension::dynamic())};

    if (input_shape[0].is_static()) {
        NODE_VALIDATION_CHECK(this, input_shape[0].get_length() > 0, "Batch size must be greater than zero");
        output_shape[0] = input_shape[0];
    }

    size_t channels_index = m_layout == DataLayout::NCHW ? 1 : input_shape.size() - 1;
    if (input_shape[channels_index].is_static()) {
        NODE_VALIDATION_CHECK(this, input_shape[channels_index].get_length() > 0, "Number of channels must be greater than zero");
        output_shape[channels_index] = input_shape[channels_index];
    }

    if (update_auto_padding_succeed) {
        size_t spatial_dims_start = m_layout == DataLayout::NCHW ? 2 : 1;
        PartialShape input_spatial_dims{std::vector<Dimension>(input_shape.begin() + spatial_dims_start,
                                                               input_shape.begin() + spatial_dims_start + num_spatial_dims)};
        PartialShape output_spatial_dims{PartialShape::dynamic(num_spatial_dims)};

        Strides input_dilation(num_spatial_dims, 1);
        Strides filter_dilations = dilations;
        if (filter_dilations.empty()) {
            filter_dilations = Strides(num_spatial_dims, 1);
        }

        CoordinateDiff pads_begin(m_pads_begin.begin(), m_pads_begin.end());
        CoordinateDiff pads_end(m_pads_end.begin(), m_pads_end.end());

        output_spatial_dims = ngraph::infer_windowed_reduction_output_shape(this,
                                                                            input_spatial_dims,
                                                                            input_dilation,
                                                                            pads_begin,
                                                                            pads_end,
                                                                            m_kernel,
                                                                            m_strides,
                                                                            filter_dilations,
                                                                            !exclude_pad,
                                                                            m_rounding_type == op::RoundingType::CEIL);

        for (size_t i = 0; i < num_spatial_dims; i++) {
            output_shape[i + spatial_dims_start] = output_spatial_dims[i];
        }
    }

    return output_shape;
}

opset::v1::ArmMaxPool::ArmMaxPool(const ov::Output<Node>& arg,
                                  const ov::Strides& strides,
                                  const ov::Shape& pads_begin,
                                  const ov::Shape& pads_end,
                                  const ov::Shape& kernel,
                                  const ov::op::RoundingType& rounding_type,
                                  const ov::op::PadType& auto_pad,
                                  const DataLayout& layout)
        : util::PoolBase{arg, strides, pads_begin, pads_end,
                         kernel, rounding_type, auto_pad, layout} {
    constructor_validate_and_infer_types();
}


std::shared_ptr<ov::Node> opset::v1::ArmMaxPool::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<ArmMaxPool>(new_args.at(0), m_strides, m_pads_begin, m_pads_end, m_kernel, m_rounding_type, m_auto_pad, m_layout);
}

void opset::v1::ArmMaxPool::validate_and_infer_types() {
    const auto output_shape = infer_shape({}, false);
    set_output_type(0, get_input_element_type(0), output_shape);
}

opset::v8::ArmMaxPool::ArmMaxPool(const ov::Output<Node>& arg,
                                  const ov::Strides& strides,
                                  const ov::Strides& dilations,
                                  const ov::Shape& pads_begin,
                                  const ov::Shape& pads_end,
                                  const ov::Shape& kernel,
                                  const ov::op::RoundingType& rounding_type,
                                  const ov::op::PadType& auto_pad,
                                  const ov::element::Type& index_element_type,
                                  int64_t axis,
                                  const DataLayout& layout)
        : util::PoolBase{arg, strides, pads_begin, pads_end,
                         kernel, rounding_type, auto_pad, layout},
          m_dilations{dilations},
          m_index_element_type{index_element_type},
          m_axis{axis} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> opset::v8::ArmMaxPool::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<ArmMaxPool>(new_args.at(0), m_strides, m_dilations, m_pads_begin, m_pads_end, m_kernel, m_rounding_type, m_auto_pad, m_index_element_type, m_axis, m_layout);
}

void opset::v8::ArmMaxPool::validate_and_infer_types() {
    if (m_dilations.size() > 0) {
        NODE_VALIDATION_CHECK(this, m_kernel.size() == m_dilations.size(),
                              "dilations size must be equal to kernel size. Got: ",
                              m_dilations.size());
    } else {
        m_dilations = Strides(m_kernel.size(), 1);
    }
    const auto output_shape = infer_shape(m_dilations, false);
    set_output_type(0, get_input_element_type(0), output_shape);
    set_output_type(1, m_index_element_type, output_shape);
}

opset::v1::ArmAvgPool::ArmAvgPool(const ov::Output<Node>& arg,
                                  const ov::Strides& strides,
                                  const ov::Shape& pads_begin,
                                  const ov::Shape& pads_end,
                                  const ov::Shape& kernel,
                                  bool exclude_pad,
                                  const ov::op::RoundingType& rounding_type,
                                  const ov::op::PadType& auto_pad,
                                  const DataLayout& layout)
        : util::PoolBase{arg, strides, pads_begin, pads_end,
                         kernel, rounding_type, auto_pad, layout},
          m_exclude_pad{exclude_pad} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> opset::v1::ArmAvgPool::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<ArmAvgPool>(new_args.at(0), m_strides, m_pads_begin, m_pads_end, m_kernel, m_exclude_pad, m_rounding_type, m_auto_pad, m_layout);
}

void opset::v1::ArmAvgPool::validate_and_infer_types() {
    const auto output_shape = infer_shape({}, m_exclude_pad);
    set_output_type(0, get_input_element_type(0), output_shape);
}
