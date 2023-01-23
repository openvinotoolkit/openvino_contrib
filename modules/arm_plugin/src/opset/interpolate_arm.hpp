// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph_opset.hpp"
#include "utils.hpp"
#include "ngraph/coordinate_transform.hpp"

namespace ArmPlugin {
namespace opset {

class ArmInterpolate : public Interpolate {
public:
    OPENVINO_OP("ArmInterpolate", "arm_opset", Interpolate);

    ArmInterpolate(const ngraph::Output<ngraph::Node>& image,
                   const ngraph::Output<ngraph::Node>& output_shape,
                   const ngraph::Output<ngraph::Node>& scales,
                   const Interpolate::InterpolateAttrs& attrs);

    ArmInterpolate(const ngraph::Output<ngraph::Node>& image,
                   const ngraph::Output<ngraph::Node>& output_shape,
                   const ngraph::Output<ngraph::Node>& scales,
                   const ngraph::Output<ngraph::Node>& axes,
                   const Interpolate::InterpolateAttrs& attrs);

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override;
private:
    Interpolate::InterpolateAttrs m_attrs;
};

class CoordinateTransform : protected ngraph::CoordinateTransformBasic {
public:
    using Iterator = ngraph::CoordinateIterator;

    CoordinateTransform(const ov::Shape& source_shape,
                        const ov::Coordinate& source_start_corner,
                        const ov::Coordinate& source_end_corner,
                        const ov::Strides& source_strides,
                        const ov::AxisVector& source_axis_order,
                        const ov::CoordinateDiff& target_padding_below,
                        const ov::CoordinateDiff& target_padding_above,
                        const ov::Strides& source_dilation_strides);

    CoordinateTransform(const ov::Shape& source_shape);

    /// \brief The tensor element index calculation by given coordinate.
    /// \param c tensor element coordinate
    size_t index(const ov::Coordinate& c) const;

    /// \brief Convert a target-space coordinate to a source-space coordinate.
    /// \param c tensor element coordinate
    ov::Coordinate to_source_coordinate(const ov::Coordinate& c) const;

    /// \brief Returns an iterator to the first coordinate of the tensor.
    ngraph::CoordinateIterator begin() const noexcept;

    /// \brief Returns an iterator to the coordinate following the last element of the tensor.
    const ngraph::CoordinateIterator& end() const noexcept;

private:
    ov::Coordinate m_source_start_corner;
    ov::Coordinate m_source_end_corner;
    ov::Strides m_source_strides;
    ov::AxisVector m_source_axis_order;
    ov::CoordinateDiff m_target_padding_below;
    ov::CoordinateDiff m_target_padding_above;
    ov::Strides m_target_dilation_strides;

    ov::Shape m_target_shape;
    size_t m_n_axes;
};
}  // namespace opset
}  // namespace ArmPlugin
