// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "ngraph_opset.hpp"
#include "utils.hpp"
#include "quantize.hpp"
#include "convolution_shape_inference.hpp"

namespace ArmPlugin {
namespace opset {

class ArmConvolution : public ov::op::Op {
public:
    OPENVINO_OP("ArmConvolution", "arm_opset");
    ArmConvolution(const ov::Output<ov::Node>& data_batch,
                   const ov::Output<ov::Node>& filters,
                   const ov::Strides& strides,
                   const ov::CoordinateDiff& pads_begin,
                   const ov::CoordinateDiff& pads_end,
                   const ov::Strides& dilations,
                   const ov::op::PadType& auto_pad,
                   const DataLayout& layout = DataLayout::NCHW);

    ArmConvolution(const ov::Output<ov::Node>& data_batch,
                   const ov::Output<ov::Node>& filters,
                   const ov::Output<ov::Node>& bias,
                   const ov::Strides& strides,
                   const ov::CoordinateDiff& pads_begin,
                   const ov::CoordinateDiff& pads_end,
                   const ov::Strides& dilations,
                   const ov::op::PadType& auto_pad,
                   const DataLayout& layout = DataLayout::NCHW);

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    void validate_and_infer_types() override;

    const ov::Strides& get_strides() const {
        return m_strides;
    }

    void set_strides(const ov::Strides& strides) {
        m_strides = strides;
    }

    const ov::Strides& get_dilations() const {
        return m_dilations;
    }

    void set_dilations(const ov::Strides& dilations) {
        m_dilations = dilations;
    }

    const ov::CoordinateDiff& get_pads_begin() const {
        return m_pads_begin;
    }

    void set_pads_begin(const ov::CoordinateDiff& pads_begin) {
        m_pads_begin = pads_begin;
    }

    const ov::CoordinateDiff& get_pads_end() const {
        return m_pads_end;
    }

    void set_pads_end(const ov::CoordinateDiff& pads_end) {
        m_pads_end = pads_end;
    }

    const ov::op::PadType& get_auto_pad() const {
        return m_auto_pad;
    }

    void set_auto_pad(const ov::op::PadType& auto_pad) {
        m_auto_pad = auto_pad;
    }

    const DataLayout& get_layout() const {
        return m_layout;
    }

    void set_layout(const DataLayout& layout) {
        m_layout = layout;
    }

private:
    ov::Strides m_strides;
    ov::Strides m_dilations;
    ov::CoordinateDiff m_pads_begin;
    ov::CoordinateDiff m_pads_end;
    ov::op::PadType m_auto_pad;
    DataLayout m_layout;
    int m_num_spatial = -1;

    template <class ConvType>
    friend int64_t ov::op::v1::calculate_num_spatial(const ConvType* op,
                                                     const ov::PartialShape& input_shape,
                                                     const ov::PartialShape& filters_shape,
                                                     const int64_t& num_non_spatial_data_dims,
                                                     const int64_t& num_non_spatial_filter_dims);

    template <class ConvType, class ShapeType>
    friend void ov::op::v1::calculate_output_spatial_dims_for_convolution(const ConvType* op,
                                                       const ShapeType& input_shape,
                                                       const ShapeType& filters_shape,
                                                       ShapeType& output_shape,
                                                       const int64_t& num_spatial,
                                                       const ov::Strides& strides,
                                                       const ov::Strides& dilations,
                                                       const ov::CoordinateDiff& pads_begin,
                                                       const ov::CoordinateDiff& pads_end,
                                                       const int64_t& num_non_spatial_data_dims,
                                                       const int64_t& num_non_spatial_filter_dims);

    template <class ConvType>
    friend void ov::op::v1::update_and_validate_attributes(ConvType* op, int64_t num_spatial);

    template <class ConvType, class ShapeType>
    friend bool ov::op::v1::resolve_auto_pad_for_shape(const ConvType* op,
                                    ov::CoordinateDiff& pads_begin,
                                    ov::CoordinateDiff& pads_end,
                                    const std::vector<ShapeType>& input_shapes,
                                    const int64_t& num_non_spatial_data_dims,
                                    const int64_t& num_non_spatial_filter_dims);
};

}  // namespace opset
}  // namespace ArmPlugin
